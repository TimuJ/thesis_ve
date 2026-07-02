"""Task 6 driver: shift + stretch sweeps through the verified RoPE hook.

Per condition: install hook -> inference -> restore -> save output frames +
a stub JSON. Pipeline is initialised ONCE for the whole sweep (the Task-5
gate proved runs are bit-deterministic in-process, so ordering is immaterial).

Server usage (cwd must be ~/repos/FlashVSR/examples/WanVSR, env `flashvsr`):

  PYTHONPATH=/home/timur/thesis_ve CUDA_VISIBLE_DEVICES=1 \
  python -m scripts.rope_probe.run_probe \
      --input ~/synthetic_data/synthetic/hhszUXL1Cu8.mp4 --frames 85 \
      --shifts 0,2,8,32,128,512,996 --stretches 1.0 \
      --out ~/results/rope_probe/shift/hhsz85

The driver only PRODUCES frames (per-condition PNG dirs + stub JSONs);
authoritative scoring is scripts/rope_probe/score_conditions.py (pyiqa,
DOVE RGB convention, vsr env). Grid logic is pure and unit-tested locally.
"""
import argparse
import json
import os

from scripts.rope_probe.position_override import PositionOverride, is_noop


def expand_grid(shifts, stretches):
    seen, grid = set(), []
    grid.append(PositionOverride())
    seen.add((0, 1.0))
    for s in shifts:
        for st in stretches:
            key = (int(s), float(st))
            if key in seen:
                continue
            seen.add(key)
            grid.append(PositionOverride(shift=int(s), stretch=float(st)))
    return grid


def cond_id(ov):
    return f"shift{ov.shift}_stretch{ov.stretch}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--frames", type=int, default=85)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shifts", default="0")
    ap.add_argument("--stretches", default="1.0")
    args = ap.parse_args()

    # server-only imports below (torch, cv2, the stock infer script)
    import cv2
    import numpy as np
    import torch
    from scripts.rope_probe.consistency_metrics import write_condition_json
    from scripts.rope_probe.flashvsr_hook import (
        default_table_builder, install_position_hook)
    from scripts.rope_probe.flashvsr_runner import (
        load_infer_module, prepare, run_once, to_uint8_frames)

    os.makedirs(args.out, exist_ok=True)
    shifts = [int(x) for x in args.shifts.split(",")]
    stretches = [float(x) for x in args.stretches.split(",")]
    grid = expand_grid(shifts, stretches)
    print(f"{len(grid)} conditions: {[cond_id(o) for o in grid]}", flush=True)

    infer = load_infer_module()
    pipe = infer.init_pipeline()
    dit = pipe.denoising_model()
    t_dim = dit.freqs[0].shape[1] * 2
    LQ, F = prepare(os.path.expanduser(args.input), args.frames)
    print(f"clip ready: F={F} ({args.frames} real frames)", flush=True)

    for ov in grid:
        cid = cond_id(ov)
        restore = None
        if not is_noop(ov):
            restore = install_position_hook(dit, ov, default_table_builder(t_dim))
        try:
            video = run_once(pipe, LQ, F)
        finally:
            if restore:
                restore()
        frames = to_uint8_frames(video, args.frames)
        del video

        # frames-producer only: authoritative scoring is a separate pyiqa
        # pass (scripts/rope_probe/score_conditions.py, vsr env) so numbers
        # match the project's DOVE-convention pyiqa metrics.
        write_condition_json(os.path.join(args.out, cid + ".json"),
                             {"shift": ov.shift, "stretch": ov.stretch,
                              "frames": args.frames, "input": args.input},
                             None, None)
        fdir = os.path.join(args.out, cid)
        os.makedirs(fdir, exist_ok=True)
        for i, fr in enumerate(frames):
            cv2.imwrite(os.path.join(fdir, f"{i:04d}.png"),
                        cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        print(f"done {cid}: {len(frames)} frames saved", flush=True)

    print("SWEEP_OK", flush=True)


if __name__ == "__main__":
    main()
