"""CLIP selection embedder (Task 7).

The embedding used to pick the matched cross-gap pair is a *different* model
from the scoring metrics, so "closest appearance" is judged independently of
what we later score. torch/clip/PIL are imported lazily inside the class so this
module imports without them and run_adr stays torch-free at top level.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class ClipEmbedder:
    """OpenAI CLIP image encoder → L2-normalized appearance embeddings.

    Default ViT-B/32; its weights are cached on the lab server, so it loads
    offline. ``embed`` takes a list of RGB uint8 crops and returns (N, D) float.
    """

    name = "clip"

    def __init__(self, model_name: str = "ViT-B/32", device: "str | None" = None,
                 batch_size: int = 256):
        import warnings

        import torch

        with warnings.catch_warnings():  # silence CLIP's noisy pkg_resources import warning
            warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
            import clip

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model, self._preprocess = clip.load(model_name, device=self.device)
        self._model.eval()
        self._dim = int(self._model.visual.output_dim)
        self._batch_size = int(batch_size)

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.zeros((0, self._dim), dtype=float)
        from PIL import Image

        torch = self._torch
        tensors = [
            self._preprocess(Image.fromarray(np.asarray(c).astype("uint8")))
            for c in crops
        ]
        # Encode in fixed-size chunks so a long clip's thousands of present-frame
        # crops never form one oversized batch that could OOM a shared GPU.
        outs = []
        with torch.no_grad():
            for k in range(0, len(tensors), self._batch_size):
                batch = torch.stack(tensors[k:k + self._batch_size]).to(self.device)
                feats = self._model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                outs.append(feats.cpu().numpy().astype(float))
        return np.concatenate(outs, axis=0)
