"""GPU pair scorers and the SAM mask-on-white crop upgrade (Task 7).

Mirrors Ring Forcing's scoring trio across the appear/reappear crops:
  - SIFT texture  -> scorers.SiftTextureScorer (CPU, separate module)
  - DINO semantic -> DinoSemanticScorer (here)
  - LoFTR geometry-> LoftrGeometryScorer (here, needs kornia)

All conform to the PairScorer protocol (``name`` + ``__call__(a, b) -> float`` in
[0, 1]). torch/kornia/segment_anything are imported lazily, so importing this
module never requires them; only constructing a given adapter does. Adapters run
on CUDA when available, else CPU.
"""
from __future__ import annotations

import numpy as np


def _pick_device(device):
    import torch
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


class DinoSemanticScorer:
    """Semantic consistency = cosine similarity of DINO CLS embeddings, clamped
    to [0, 1]. Uses the cached DINO ViT-B/16 hub model (loads offline)."""

    name = "dino"

    def __init__(self, device: "str | None" = None, model: str = "dino_vitb16"):
        import os

        import torch
        import torchvision.transforms as T

        self._torch = torch
        self.device = _pick_device(device)
        hub_repo = os.path.join(torch.hub.get_dir(), "facebookresearch_dino_main")
        if os.path.isdir(hub_repo):
            net = torch.hub.load(hub_repo, model, source="local")
        else:
            net = torch.hub.load("facebookresearch/dino:main", model)
        self._model = net.to(self.device).eval()
        self._tf = T.Compose([
            T.ToPILImage(),
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _embed(self, img):
        torch = self._torch
        x = self._tf(np.asarray(img).astype("uint8")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self._model(x)
        return f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def __call__(self, crop_a: np.ndarray, crop_b: np.ndarray) -> float:
        fa, fb = self._embed(crop_a), self._embed(crop_b)
        cos = float((fa * fb).sum().item())
        return max(0.0, min(1.0, cos))


class LoftrGeometryScorer:
    """Geometry consistency = confident LoFTR correspondences between the two
    crops, normalized to [0, 1]. Requires kornia (+ LoFTR weights)."""

    name = "loftr"

    def __init__(self, device: "str | None" = None, pretrained: str = "outdoor",
                 conf: float = 0.5, denom: int = 200):
        import kornia
        import torch

        self._torch = torch
        self._kornia = kornia
        self.device = _pick_device(device)
        self._matcher = kornia.feature.LoFTR(pretrained=pretrained).to(self.device).eval()
        self._conf = conf
        self._denom = denom

    def _gray_t(self, img):
        torch = self._torch
        g = img
        if g.ndim == 3:
            import cv2
            g = cv2.cvtColor(np.asarray(img).astype("uint8"), cv2.COLOR_RGB2GRAY)
        t = torch.from_numpy(np.asarray(g).astype("float32") / 255.0)
        return t[None, None].to(self.device)

    def __call__(self, crop_a: np.ndarray, crop_b: np.ndarray) -> float:
        torch = self._torch
        with torch.no_grad():
            out = self._matcher({"image0": self._gray_t(crop_a), "image1": self._gray_t(crop_b)})
        conf = out["confidence"]
        n = int((conf >= self._conf).sum().item()) if conf.numel() else 0
        return max(0.0, min(1.0, n / float(self._denom)))


def build_sam_predictor(checkpoint: "str | None" = None, model_type: str = "vit_h",
                        device: "str | None" = None):
    """Construct a SAM predictor. Needs segment_anything and a checkpoint whose
    host is currently unreachable from the lab server, so this is exercised at
    the integration run once the weights are staged locally."""
    from segment_anything import SamPredictor, sam_model_registry

    device = _pick_device(device)
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return SamPredictor(sam)


def sam_mask_on_white(frame: np.ndarray, box, predictor) -> np.ndarray:
    """Segment the boxed subject with SAM and composite it on white — Ring
    Forcing's crop normalization. Falls back to the plain box region if SAM
    returns no mask. Returns an RGB image the same shape as ``frame``."""
    x, y, w, h = (int(round(v)) for v in box)
    predictor.set_image(np.asarray(frame).astype("uint8"))
    box_xyxy = np.array([x, y, x + w, y + h], dtype=float)
    masks, scores, _ = predictor.predict(box=box_xyxy, multimask_output=False)
    out = np.full_like(frame, 255)
    if masks is not None and len(masks):
        m = masks[0].astype(bool)
        out[m] = np.asarray(frame)[m]
    else:
        out[y:y + h, x:x + w] = np.asarray(frame)[y:y + h, x:x + w]
    return out
