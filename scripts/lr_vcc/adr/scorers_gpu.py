"""GPU pair scorers and the SAM mask-on-white crop upgrade.

Mirrors Ring Forcing's scoring trio across the appear/reappear crops:
  - SIFT texture   -> scorers.SiftTextureScorer (CPU, separate module)
  - DINO semantic  -> DinoSemanticScorer (here; backbone = DINOv2 ViT-B/14)
  - RoMa geometry  -> RomaGeometryScorer (here, needs romatch) + RomaPoseGate
                      (selection gate); LoftrGeometryScorer is the kornia fallback.

The planned RoMa-v2 / DINOv3 upgrade was downgraded to RoMa v1 (``romatch``, which
pulls a DINOv2 backbone) + DINOv2 because the gated ``facebook/dinov3-*`` weights are
unreachable from the lab server (hf-mirror returns 403, huggingface.co is blocked),
and RoMa v2 depends on DINOv3. The mirror-available fallbacks are still a real upgrade
over LoFTR + DINOv1.

All scorers conform to the PairScorer protocol (``name`` + ``__call__(a, b) -> float``
in [0, 1], optional ``reliability``). torch/romatch/kornia/segment_anything are imported
lazily, so importing this module never requires them; only constructing a given adapter
does. Adapters run on CUDA when available, else CPU.
"""
from __future__ import annotations

import numpy as np

from .config import ROMA_MATCH_FLOOR


def _pick_device(device):
    import torch
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def roma_mean_certainty(certainty) -> float:
    """RoMa geometry consistency / pose-match score: the mean dense-match certainty,
    clamped to [0, 1]. Empty -> 0.0. Pure numpy (no torch) so it is unit-tested on the
    M1; the romatch forward pass that produces ``certainty`` is exercised on the server."""
    c = np.asarray(certainty, dtype=float).ravel()
    if c.size == 0:
        return 0.0
    return float(np.clip(c.mean(), 0.0, 1.0))


def roma_match_coverage(certainty, floor: float) -> float:
    """GT-side geometry signal = fraction of the dense certainty map at or above
    ``floor`` -- how much of the crop pair RoMa can match at all. Weights the geometry
    axis in the composite so a GT pair with no matchable structure down-weights the axis
    instead of free-passing on c_gt ~ 0. Empty -> 0.0."""
    c = np.asarray(certainty, dtype=float).ravel()
    if c.size == 0:
        return 0.0
    return float((c >= floor).mean())


class DinoSemanticScorer:
    """Semantic consistency = cosine similarity of DINOv2 CLS embeddings, clamped to
    [0, 1]. Backbone = DINOv2 ViT-B/14 via torch.hub (weights from fbaipublicfiles,
    reachable from the lab server; the gated DINOv3 is not). ``name`` stays "dino" so the
    anchor-metric / reliability wiring is unchanged -- the axis is just a stronger model."""

    name = "dino"

    def __init__(self, device: "str | None" = None, model: str = "dinov2_vitb14"):
        import os

        import torch
        import torchvision.transforms as T

        self._torch = torch
        self.device = _pick_device(device)
        hub_repo = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
        if os.path.isdir(hub_repo):
            net = torch.hub.load(hub_repo, model, source="local")
        else:
            net = torch.hub.load("facebookresearch/dinov2", model)
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


class RomaMatcher:
    """Shared RoMa v1 dense-matcher backend (``romatch``). Loads the outdoor model once;
    the geometry axis (RomaGeometryScorer) and the selection pose gate (RomaPoseGate) both
    read its dense certainty map -- one model, both jobs. romatch pulls a DINOv2 backbone
    (mirror-available); the RoMa-v2 / DINOv3 path is gated-blocked on the lab server."""

    def __init__(self, device: "str | None" = None, weights: str = "outdoor"):
        import torch
        from romatch import roma_indoor, roma_outdoor

        self._torch = torch
        self.device = _pick_device(device)
        loader = roma_indoor if weights == "indoor" else roma_outdoor
        self._model = loader(device=self.device)
        self._model.eval()

    def certainty(self, crop_a: np.ndarray, crop_b: np.ndarray) -> np.ndarray:
        """Dense match-certainty map in [0, 1] between two RGB crops, as a numpy array.
        romatch matches two PIL images; we hand it the in-memory crops directly."""
        from PIL import Image

        torch = self._torch
        a = Image.fromarray(np.asarray(crop_a).astype("uint8"))
        b = Image.fromarray(np.asarray(crop_b).astype("uint8"))
        with torch.no_grad():
            _warp, certainty = self._model.match(a, b, device=self.device)
        return certainty.detach().float().cpu().numpy()


class RomaGeometryScorer:
    """Geometry consistency = mean RoMa dense-match certainty between the crops, in
    [0, 1]. ``reliability`` is the GT-side matchable coverage (fraction of the certainty
    map above a floor), so a GT pair with no matchable structure down-weights -- rather
    than free-passes -- the geometry axis in the composite."""

    name = "roma"

    def __init__(self, device: "str | None" = None, matcher: "RomaMatcher | None" = None,
                 match_floor: float = ROMA_MATCH_FLOOR):
        self._m = matcher or RomaMatcher(device=device)
        self._floor = match_floor

    def __call__(self, crop_a: np.ndarray, crop_b: np.ndarray) -> float:
        return roma_mean_certainty(self._m.certainty(crop_a, crop_b))

    def reliability(self, crop_a: np.ndarray, crop_b: np.ndarray) -> float:
        return roma_match_coverage(self._m.certainty(crop_a, crop_b), self._floor)


class RomaPoseGate:
    """Selection pose gate (decision (a): RoMa gates, CLIP ranks). Returns a pose-match
    score in [0, 1] = mean RoMa certainty between two GT crops; run_adr admits a candidate
    pair when this is >= ROMA_POSE_GATE_THRESH, ensuring the pair is pose-stable enough to
    review and that the geometry axis has GT-side signal. Shares a RoMa backend with the
    geometry scorer so the weights load once."""

    def __init__(self, device: "str | None" = None, matcher: "RomaMatcher | None" = None):
        self._m = matcher or RomaMatcher(device=device)

    def __call__(self, crop_a: np.ndarray, crop_b: np.ndarray) -> float:
        return roma_mean_certainty(self._m.certainty(crop_a, crop_b))


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
