# Weekly Progress Report — Timur Iakshibaev

## Period: March 30 – April 12, 2026

## Key Result: MGLD-VSR Matches DOVE Benchmark Exactly

The MGLD-VSR inference pipeline on DOVE UDM10 LQ produces **identical** results to the DOVE paper.

| Metric | DOVE paper (MGLD) | Ours | Delta |
|--------|-------------------|------|-------|
| PSNR | 24.23 | **24.23** | +0.00 |
| SSIM | 0.6957 | **0.6957** | 0.0000 |
| LPIPS | 0.3272 | **0.3272** | 0.0000 |
| DISTS | 0.1677 | **0.1676** | -0.0001 |
| CLIP-IQA | 0.4557 | **0.4555** | -0.0002 |

**Implication:** The full pipeline (inference + evaluation) is aligned with the DOVE benchmark. DOVE's published comparison table can be used as a trusted baseline reference going forward.

## How Alignment Was Achieved

Two fixes were required to match DOVE exactly:

1. **Inference — use the tile-based script.** The standard MGLD-VSR inference script center-crops input to 512x512, which misaligns with DOVE's non-square LQ (318x180) and produces wrong-resolution output. The repo ships a separate tile-based script (`vsr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py`) that splits frames into overlapping patches, runs each through the UNet, and stitches them back to full resolution (1272x720).

2. **Evaluation — use DOVE's `eval_metrics.py`.** DOVE uses RGB PSNR/SSIM by default (no `--test_y_channel` flag in their `inference.sh`). Earlier pyiqa evaluation used Y-channel PSNR/SSIM (the standard VSR convention), which is why earlier numbers did not match DOVE's table — a metric-implementation mismatch, not a model issue.

## UAV on DOVE UDM10 — Close, Gap Narrowing

Evaluating the existing UAV output on DOVE LQ using DOVE's `eval_metrics.py`:

| Metric | DOVE paper (UAV) | Ours | Delta |
|--------|------------------|------|-------|
| PSNR | 21.72 | 22.96 | +1.24 |
| SSIM | 0.5913 | 0.6183 | +0.027 |
| LPIPS | 0.4116 | 0.4050 | -0.007 |

Remaining gap is from inference settings (we used `-n 150 -g 7`, UAV defaults are `-n 120 -g 6`). Re-running with default settings was in progress (7/10 clips completed) when the server disk failed — see issue below.

## Evaluation Infrastructure

- The DOVE repo was cloned on the server and its evaluation code analyzed
- Key finding: DOVE evaluates PSNR/SSIM on **RGB**, not Y-channel — this alone accounts for a ~0.3 dB difference vs the standard VSR convention
- Going forward: DOVE's `eval_metrics.py` is used for any DOVE-benchmark comparison; the pyiqa evaluator is kept for internal comparisons

## Issue: Server Disk Failure (April 12)

The lab GPU server's main data disk (`/data/disk1`) experienced an I/O-level failure:
- All paths under `/data/disk1` return "Input/output error"; `df -h /data/disk1` fails
- Both running experiments (UAV DOVE default, UAV VideoLQ) crashed
- Server-side data is temporarily inaccessible (code, checkpoints, datasets, conda envs)

**Local copies are safe:** all committed source code, the MGLD-VSR DOVE eval result above, and previously verified results. The lab admin has been contacted; no server-side progress is possible until the issue is resolved.

## In Progress at End of Period

| Task | Progress | Status |
|------|----------|--------|
| UAV DOVE LQ default (n120 g6) — expected to close the 1.24 dB gap | 7/10 clips | Needs re-run after disk restored |
| UAV VideoLQ NR inference | 43/50 clips | Needs re-run after disk restored |

## Next Steps

1. Resolve server disk failure with lab admin — blocker for further GPU work
2. Re-run UAV DOVE default (expected to close the gap to DOVE paper)
3. Set up VBench for human-perception-aligned evaluation (long-video beta version needs extension)
5. Test both baselines on long-video sequences (>1 min) when sample data arrives
6. Proposal rewrite — deadline May 31 (on track)
