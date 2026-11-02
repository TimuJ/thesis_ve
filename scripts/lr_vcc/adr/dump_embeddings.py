"""Pre-dump CLIP selection embeddings per clip (Task 7).

The heavy model runs once on the GPU box and writes arrays; selection can then
reuse them without reloading CLIP. Pure glue — imported lazily, exercised at the
integration run, not in unit tests.
"""
from __future__ import annotations


def main(argv=None):  # pragma: no cover - IO glue, validated at integration
    import argparse
    import csv

    import numpy as np

    from .annotations import load_annotations, present_mask
    from .crops import bbox_crop
    from .embed_clip import ClipEmbedder
    from .run_adr import _decode_frames

    p = argparse.ArgumentParser(description="Dump CLIP selection embeddings per clip")
    p.add_argument("--manifest", required=True)
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--ann-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    import os

    os.makedirs(args.out_dir, exist_ok=True)
    embedder = ClipEmbedder()

    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            name = row["video"]
            stem = name.rsplit(".", 1)[0]
            frames = _decode_frames(f"{args.gt_dir}/{name}")
            ann = load_annotations(f"{args.ann_dir}/{stem}")
            present = present_mask(ann)
            idx = [i for i in range(ann.n)
                   if present[i] and bbox_crop(frames[i], tuple(ann.boxes[i])) is not None]
            crops = [bbox_crop(frames[i], tuple(ann.boxes[i])) for i in idx]
            emb = embedder.embed(crops)
            np.savez(os.path.join(args.out_dir, f"{stem}.npz"),
                     present_idx=np.array(idx), embeddings=emb)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
