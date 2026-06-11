"""Pick a reference scene frame for background_drift that is CLIP-distant
from the base video — prevents the 7WHI inversion (reference too colour-similar).

Server usage (conda env vbench, needs open_clip):
    python scripts/synthetic_artefacts/select_reference_scene.py \
        --base_video results/mgld_synthetic_mp4/7WHI2L_FDNg.mp4 \
        --candidates results/mgld_synthetic_mp4/BrRLKMbBTYQ.mp4:500 \
                     results/mgld_synthetic_mp4/KZ8p6b1zJ9U.mp4:200 \
        --tau 0.25 --out results/synthetic_artefacts/_references/ref_bg_for_7WHI.png

A candidate is "video_path:frame_index". The base video must never be its own
candidate. Prints all distances, saves the most distant frame if it clears tau.
"""
import argparse

import cv2
import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - float(np.dot(a, b)))


def pick_most_distant(candidates: dict, base_embedding: np.ndarray, tau: float):
    """candidates: {name: embedding}. Returns (name, distance) of the farthest
    candidate; raises ValueError when even the farthest is below tau."""
    best_name, best_d = None, -1.0
    for name, emb in candidates.items():
        d = cosine_distance(emb, base_embedding)
        if d > best_d:
            best_name, best_d = name, d
    if best_name is None or best_d < tau:
        raise ValueError(f"no candidate clears tau={tau}; best={best_name} d={best_d:.3f}")
    return best_name, best_d


def read_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_index} of {video_path}")
    return frame


def sample_frames(video_path: str, n: int = 8):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return [read_frame(video_path, i) for i in np.linspace(0, total - 1, n, dtype=int)]


def _embed_bgr_frames(frames, model, preprocess, device):
    import torch
    from PIL import Image
    with torch.no_grad():
        batch = torch.stack([
            preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames
        ]).to(device)
        emb = model.encode_image(batch).float().cpu().numpy()
    return emb.mean(axis=0)


def main():
    import open_clip
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--base_video", required=True)
    ap.add_argument("--candidates", nargs="+", required=True, help="video_path:frame_index")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)

    base_emb = _embed_bgr_frames(sample_frames(args.base_video), model, preprocess, device)

    cand_embs, cand_frames = {}, {}
    for spec in args.candidates:
        path, idx = spec.rsplit(":", 1)
        frame = read_frame(path, int(idx))
        cand_frames[spec] = frame
        cand_embs[spec] = _embed_bgr_frames([frame], model, preprocess, device)

    for name, emb in sorted(cand_embs.items(), key=lambda kv: -cosine_distance(kv[1], base_emb)):
        print(f"{cosine_distance(emb, base_emb):.3f}  {name}")

    name, dist = pick_most_distant(cand_embs, base_emb, args.tau)
    cv2.imwrite(args.out, cand_frames[name])
    print(f"selected {name} (d={dist:.3f}) -> {args.out}")


if __name__ == "__main__":
    main()
