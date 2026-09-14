# VBench dimension audit — findings (temporal_flickering, motion_smoothness)

**Status:** temporal_flickering complete on both available families; motion_smoothness
blocked then unblocked, re-run pending GPU headroom. Runner committed
(`scripts/server_runners/run_vbench_reward_audit.sh`); results on the server under
`~/results/vbench_reward_audit/`.

## What prompted this

The reduced-reference probe produced an identity-axis reductio: a self-anchored
identity measure ranks the degraded LQ input as the *most* consistent video,
because self-similarity rewards the absence of detail. This audit asks whether
VBench's own temporal/motion consistency dimensions have the same disease on a
different axis — measured directly against the corruption battery, not argued from
construct validity. It gives the theoretical prediction in
`docs/plans/2026-04-28-metrics-and-vbench-validation.md` (that `temporal_flickering`
is blind to drift slower than two frames, because it averages adjacent-frame MAE)
empirical teeth over a graded severity ladder, and interrogates `motion_smoothness`,
which that plan had left uninterrogated.

## A terminology correction worth recording

`temporal_flickering` and `motion_smoothness` are **VBench 1.x** long-extension
dimensions (`vbench2_beta_long/eval_long.py`, class `VBenchLong`). They are *not*
VBench 2.0 dimensions — the genuine 2.0 package (`Camera_Motion`, `Human_Identity`,
…) contains neither. So this audit speaks to the VBench 1.x temporal/motion
family; the identity reductio speaks to the 2.0 consistency family. Both should be
labelled by version in any write-up.

## Method

- Dimensions: `temporal_flickering`, `motion_smoothness`, via `eval_long.py
  --mode long_custom_input --dev_flag`, pinned to GPU1 (`nice`d; GPU0 carried a
  collaborator's live job).
- Families: `color_drift`, `background_drift` — the two gradual-drift families
  present on the server. These are the correct probe for the blindness claim: the
  corruption is a slow progressive drift, which is exactly what an adjacent-frame
  statistic should miss. (identity_degradation / flicker / chunk_boundary are not
  generated on the server; see limitations.)
- Ladder: severities 0.02, 0.05, 0.10, 0.20, 0.40 (a 20× range), 5 bases each.
- eval_long splits every ~2.8-min video into ~84 two-second clips and scores per
  clip; the clip path carries its `base_sevX`, so per-clip scores aggregate
  cleanly to a per-(base, severity) mean. 1885 clips per family.

## Result — temporal_flickering is blind, and its residual trend is inverted

Mean `temporal_flickering` score per (base, severity). Higher = "less flicker" =
what VBench treats as better.

**color_drift** (overall aggregate 0.98417):

| base | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 | Δ(.40−.02) |
|---|---:|---:|---:|---:|---:|---:|
| 7WHI2L_FDNg | 0.9857 | 0.9857 | 0.9858 | 0.9858 | 0.9860 | +0.0003 |
| BrRLKMbBTYQ | 0.9886 | 0.9886 | 0.9886 | 0.9886 | 0.9888 | +0.0002 |
| KZ8p6b1zJ9U | 0.9813 | 0.9813 | 0.9814 | 0.9816 | 0.9822 | +0.0009 |
| hhszUXL1Cu8 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9949 | −0.0001 |
| mJog8DlRk_4 | 0.9753 | 0.9753 | 0.9753 | 0.9754 | 0.9757 | +0.0004 |

mean |Δ| = 0.0004, mean spread across the ladder = **0.0004**.

**background_drift** (overall aggregate 0.98456):

| base | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 | Δ(.40−.02) |
|---|---:|---:|---:|---:|---:|---:|
| 7WHI2L_FDNg | 0.9858 | 0.9860 | 0.9862 | 0.9866 | 0.9874 | +0.0015 |
| BrRLKMbBTYQ | 0.9885 | 0.9885 | 0.9883 | 0.9879 | 0.9872 | −0.0014 |
| KZ8p6b1zJ9U | 0.9815 | 0.9817 | 0.9821 | 0.9829 | 0.9843 | +0.0028 |
| hhszUXL1Cu8 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | +0.0000 |
| mJog8DlRk_4 | 0.9754 | 0.9756 | 0.9759 | 0.9765 | 0.9776 | +0.0022 |

mean |Δ| = 0.0016, mean spread across the ladder = **0.0016**.

Figure: `reports/figures/vbench_reward_audit_tf.png`.

**Reading.** A 20× increase in injected drift moves the score by ~0.0004
(color_drift) to ~0.0016 (background_drift) on a [0,1] scale where the scores sit
around 0.975–0.995 — flat to three decimals. `temporal_flickering` does not see
gradual drift, exactly as predicted.

The residual trend, such as it is, points the **wrong way**. Counting the
end-to-end change with a |Δ| > 0.0002 threshold: the score **rises on 7 of 10**
base×family cells, is flat on 2, and falls on only 1 — and that single fall
(BrRLK/background_drift) is 0.0014, noise-level. "Rises" means the metric rates
the *more corrupted* video as *less flickery*, i.e. better. This is the same
mechanism as the identity reductio: drift smooths adjacent-frame differences,
lowering MAE, which VBench reads as improved temporal consistency.

**The comparison that matters.** On the same `color_drift` family where
`temporal_flickering` is flat at 0.0004, LR-VCC's anchored-colour sub-metric (D′)
responds on 4/5 bases (calibration record). Same corruption, same videos: the
benchmark's sub-metric moves, VBench's dimension does not.

## motion_smoothness — blocked, unblocked, re-run pending

Both `motion_smoothness` runs exited 1 without producing scores. Cause: the
dimension needs the AMT frame-interpolation checkpoint `amt-s.pth`, which was not
cached, and VBench's `init_submodules` **hard-codes a `huggingface.co` wget URL**
that ignores the `HF_ENDPOINT` mirror the runner exports — so the download hit the
blocked host (`wget … exit 4`). Not a data or GPU problem; the clip-splitting for
both families completed first.

Fixed by fetching `amt-s.pth` (12 MB, verified) from the mirror into
`~/.cache/vbench/amt_model/`. The re-run is unblocked and only waits on GPU
headroom (both GPUs were ~99% with other jobs at audit's end). Re-run:
`bash ~/run_vbench_reward_audit.sh` — it skips the finished temporal_flickering
cells and resumes at motion_smoothness. Whether motion_smoothness is also
blind/inverted or actually responds is the open question this leaves; because it
was never interrogated, an honest result either way is worth having.

## Honest limitations

- **Two families, not the full battery.** Only color_drift and background_drift
  are generated on the server; identity_degradation, flicker, chunk_boundary are
  not. So this establishes **blindness to gradual drift**, which is the claim the
  prediction makes. It does **not** test the blur-inversion case (a family that
  globally smooths detail) on these dimensions — that needs those families
  generated, and is the natural extension.
- **Scores sit near the ceiling** (~0.98–0.99). A near-ceiling metric has little
  room to move, which is part of the point, but it also means "flat" and
  "saturated" are not fully separable here without a family that is known to move
  the statistic.
- **Single run, no seed variation.** eval_long is deterministic for
  temporal_flickering (MAE), so this is not a concern for the numbers reported,
  but motion_smoothness (learned interpolation) should be checked once it runs.
- **Aggregation is per-clip mean → per-video mean.** This matches how the
  dimension is meant to be read; no static-clip filtering was disabled or forced
  beyond `--dev_flag` defaults.

## What this establishes / next

- Establishes, with measurements: VBench `temporal_flickering` is blind to graded
  gradual drift across a 20× severity range, and its residual trend rewards the
  corruption — the temporal-axis analogue of the identity reductio.
- Next, in priority order: (1) re-run motion_smoothness now that the checkpoint is
  staged; (2) if the blur-inversion case is wanted on these dimensions, generate
  identity_degradation / flicker on the server and extend the audit; (3) fold both
  the temporal (this note) and identity (RR probe) reductios into the
  VBench-vs-LR-VCC comparison as measured evidence rather than construct-validity
  argument.
