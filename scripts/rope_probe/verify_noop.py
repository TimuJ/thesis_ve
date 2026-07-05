"""Task 5 faithfulness gate: the position hook must be invisible when no-op.

Five runs of the stock FlashVSR v1.1 tiny pipeline on the same short clip,
one process, seed fixed:

  A  unhooked            (reference)
  B  unhooked            -> floor = max|A-B|  (process nondeterminism)
  C  hook, no-op override -> drift = max|A-C|; PASS requires drift <= floor
  D  hook, shift=+1       -> must differ (max|A-D| >> floor) or the hook
                             isn't actually engaged — a silent-pass trap
  E  unhooked after restore() -> max|A-E| <= floor (restore works)

Run ON THE SERVER from ~/repos/FlashVSR/examples/WanVSR:
  conda activate flashvsr
  CUDA_VISIBLE_DEVICES=<gpu> python -m scripts.rope_probe.verify_noop \
      --input ~/synthetic_data/synthetic/hhszUXL1Cu8.mp4 --frames 85 \
      --out /tmp/rope_noop_verdict.json
(needs PYTHONPATH pointing at both thesis_ve and cwd; see runner in the note)

Comparisons are on the raw bf16 output tensors upcast to float32, before any
uint8 quantisation, so sub-quantum drift is visible.
"""
import argparse
import json
import os

import torch

from scripts.rope_probe.flashvsr_hook import default_table_builder, install_position_hook
from scripts.rope_probe.flashvsr_runner import load_infer_module, prepare, run_once
from scripts.rope_probe.position_override import PositionOverride


def max_abs_diff(a, b):
    return float((a - b).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--frames", type=int, default=85)
    ap.add_argument("--out", default="/tmp/rope_noop_verdict.json")
    args = ap.parse_args()

    infer = load_infer_module()
    pipe = infer.init_pipeline()
    dit = pipe.denoising_model()
    t_dim = dit.freqs[0].shape[1] * 2
    LQ, F, (th, tw) = prepare(os.path.expanduser(args.input), args.frames)
    print(f"clip ready: F={F} (from {args.frames} real frames)", flush=True)

    a = run_once(pipe, LQ, F, th, tw)
    b = run_once(pipe, LQ, F, th, tw)
    floor = max_abs_diff(a, b)
    print(f"[floor] unhooked rerun max|A-B| = {floor:.3e}", flush=True)

    restore = install_position_hook(dit, PositionOverride(),
                                    default_table_builder(t_dim))
    c = run_once(pipe, LQ, F, th, tw)
    drift = max_abs_diff(a, c)
    print(f"[noop ] hooked no-op  max|A-C| = {drift:.3e}", flush=True)
    restore()

    restore = install_position_hook(dit, PositionOverride(shift=1),
                                    default_table_builder(t_dim))
    d = run_once(pipe, LQ, F, th, tw)
    engaged = max_abs_diff(a, d)
    print(f"[shift] hooked shift1 max|A-D| = {engaged:.3e}", flush=True)
    restore()

    e = run_once(pipe, LQ, F, th, tw)
    restored = max_abs_diff(a, e)
    print(f"[rest ] after restore max|A-E| = {restored:.3e}", flush=True)

    eps = 1e-6
    ok_noop = drift <= floor + eps
    ok_engaged = engaged > max(10 * floor, 1e-3)
    ok_restored = restored <= floor + eps
    verdict = {
        "floor": floor, "noop_drift": drift, "shift1_diff": engaged,
        "restored_diff": restored,
        "pass_noop": ok_noop, "pass_engaged": ok_engaged,
        "pass_restored": ok_restored,
        "PASS": ok_noop and ok_engaged and ok_restored,
    }
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=2)
    print(("GATE_PASS" if verdict["PASS"] else "GATE_FAIL"), json.dumps(verdict),
          flush=True)


if __name__ == "__main__":
    main()
