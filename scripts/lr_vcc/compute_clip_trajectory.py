"""Sub-metric D'' — anchor-window CLIP-trajectory divergence.

Same formula as D' (mean distance from first-N-frame anchor) but in CLIP-image
embedding space instead of Lab-histogram space. CLIP knows what "same scene"
means in a way colour histograms can't — robust to camera motion (which
trips up D'), but expensive (one CLIP forward pass per sampled frame).

A consistent video stays close in CLIP-space; background_drift walks away
because the scene literally changes. Score = exp(-beta * mean cosine distance
from anchor for t > anchor_len).

Server usage (env vsr or vbench, needs OpenAI clip package — pre-cached on
the lab server at ~/.cache/clip/ViT-B-32.pt, no network needed):
    python -m scripts.lr_vcc.compute_clip_trajectory \
        --videos_dir results/synthetic_artefacts/background_drift \
        --output_path results/lr_vcc/clip_trajectory/background_drift \
        --stride 8 --anchor_len 60 --beta 5.0

Stride 8 is the recommended default — every 8th frame at 30 fps is 4 samples
per second, plenty for a smooth trajectory and ~10x faster than full-rate.

Implementation note: we use OpenAI's original `clip` package (PyPI: `clip`)
instead of `open_clip` because open_clip insists on calling HuggingFace Hub
even for pretrained="openai", and HF Hub is unreachable from the lab server.
The `clip` package downloads from OpenAI's own CDN and caches at
~/.cache/clip/. Same architecture, same embedding dim (512), same weights as
"openai" tag in open_clip.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


_DEFAULT_STRIDE = 8        # sample every Nth frame
_DEFAULT_ANCHOR_LEN = 60   # ~2 s of frames at 30 fps
_DEFAULT_BETA = 5.0
_DEFAULT_BATCH = 32


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 − cosine similarity. a, b: 1D vectors."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - float(np.dot(a, b)))


def _load_sampled_frames(video_path: str, stride: int):
    cap = cv2.VideoCapture(video_path)
    frames, idx = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(fr)
        idx += 1
    cap.release()
    return frames


def _embed_frames(frames, model, preprocess, device, batch_size=_DEFAULT_BATCH):
    """BGR uint8 numpy frames → (N, D) float32 embeddings on CPU.

    Works with both `clip` (OpenAI's original) and `open_clip` models — the
    preprocess transform and model.encode_image API are identical.
    """
    import torch
    from PIL import Image
    embs = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            chunk = frames[i:i + batch_size]
            batch = torch.stack([
                preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                for f in chunk
            ]).to(device)
            emb = model.encode_image(batch).float().cpu().numpy()
            embs.append(emb)
    return np.concatenate(embs, axis=0) if embs else np.zeros((0,), dtype=np.float32)


def trajectory_score(video_path: str, model, preprocess, device,
                     anchor_len: int = _DEFAULT_ANCHOR_LEN,
                     beta: float = _DEFAULT_BETA,
                     stride: int = _DEFAULT_STRIDE) -> dict:
    """Compute D'' for one video. anchor_len is in *original* frame units; we
    convert to embeddings via anchor_len // stride."""
    frames = _load_sampled_frames(video_path, stride)
    if len(frames) < anchor_len // stride + 1:
        return {"score": 0.0, "reliability": 0.0,
                "details": {"error": "video too short", "n_sampled": len(frames)}}
    embs = _embed_frames(frames, model, preprocess, device)
    anchor_count = max(1, anchor_len // stride)
    anchor = embs[:anchor_count].mean(axis=0)
    post = embs[anchor_count:]
    dists = np.array([cosine_distance(e, anchor) for e in post])
    mean_dist = float(dists.mean())
    score = float(np.exp(-beta * mean_dist))
    score = max(0.0, min(1.0, score))
    return {
        "score": score,
        "reliability": 1.0,
        "details": {
            "anchor_len_frames": anchor_len,
            "anchor_count_sampled": anchor_count,
            "post_anchor_count_sampled": int(len(post)),
            "stride": stride,
            "beta": beta,
            "mean_cos_dist_to_anchor": mean_dist,
            "max_cos_dist_to_anchor": float(dists.max()),
            "trajectory_mean_per_quarter": [
                float(np.mean(c)) for c in np.array_split(dists, 4)
            ],
        },
    }


def main():
    import clip  # OpenAI's original CLIP package — cached at ~/.cache/clip/
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--anchor_len", type=int, default=_DEFAULT_ANCHOR_LEN)
    ap.add_argument("--beta", type=float, default=_DEFAULT_BETA)
    ap.add_argument("--stride", type=int, default=_DEFAULT_STRIDE)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading OpenAI CLIP ViT-B/32 (device={device}) ...", flush=True)
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=None)
    model.eval()

    os.makedirs(args.output_path, exist_ok=True)
    videos = sorted(f for f in os.listdir(args.videos_dir) if f.endswith(".mp4"))
    for vname in videos:
        base = vname[:-4]
        out_file = os.path.join(args.output_path, base + "_clip_trajectory.json")
        if os.path.isfile(out_file):
            print(f"[skip] {out_file}")
            continue
        vpath = os.path.join(args.videos_dir, vname)
        print(f"=== {vname} ===", flush=True)
        out = trajectory_score(vpath, model, preprocess, device,
                               anchor_len=args.anchor_len, beta=args.beta,
                               stride=args.stride)
        payload = {"video_path": vpath, **out}
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        det = out.get("details", {})
        print(f"  score={out['score']:.4f} mean_dist={det.get('mean_cos_dist_to_anchor', 0):.4f}",
              flush=True)


if __name__ == "__main__":
    main()
