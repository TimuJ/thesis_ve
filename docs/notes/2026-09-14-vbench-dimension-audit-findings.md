# VBench dimension audit — findings (temporal_flickering, motion_smoothness)

**Status:** complete on both dimensions across both available families. Runner
committed (`scripts/server_runners/run_vbench_reward_audit.sh`); results on the
server under `~/results/vbench_reward_audit/`; figure
`reports/figures/vbench_reward_audit.png`.

## What prompted this

The reduced-reference probe produced an identity-axis reductio: a self-anchored
identity measure ranks the degraded LQ input as the *most* consistent video,
because self-similarity rewards the absence of detail. This audit asks whether
VBench's own temporal/motion consistency dimensions have the same disease on a
different axis — measured directly against the corruption battery, not argued from
construct validity. It gives the prediction in
`docs/plans/2026-04-28-metrics-and-vbench-validation.md` (that `temporal_flickering`
is blind to drift slower than two frames, because it averages adjacent-frame MAE)
empirical teeth over a graded severity ladder, and interrogates `motion_smoothness`,
which that plan had left uninterrogated.

## A terminology correction worth recording

`temporal_flickering` and `motion_smoothness` are **VBench 1.x** long-extension
dimensions (`vbench2_beta_long/eval_long.py`, class `VBenchLong`). They are *not*
VBench 2.0 dimensions — the genuine 2.0 package (`Camera_Motion`, `Human_Identity`,
…) contains neither. So this audit speaks to the VBench 1.x temporal/motion family;
the identity reductio speaks to the 2.0 consistency family. Label both by version.

## Method

- Dimensions: `temporal_flickering`, `motion_smoothness`, via `eval_long.py
  --mode long_custom_input --dev_flag`, pinned to GPU1 (`nice`d; GPU0 carried a
  collaborator's live job; the box ran at load ~8–10 with ~35 users throughout).
- Families: `color_drift`, `background_drift` — the two gradual-drift families
  present on the server, the correct probe for the blindness claim (the corruption
  is a slow progressive drift, exactly what an adjacent-frame statistic should
  miss). identity_degradation / flicker / chunk_boundary are not generated on the
  server; see limitations.
- Ladder: severities 0.02, 0.05, 0.10, 0.20, 0.40 (a 20× range), 5 bases each.
- eval_long splits every ~2.8-min video into ~84 two-second clips and scores per
  clip; the clip path carries its `base_sevX`, so per-clip scores aggregate cleanly
  to a per-(base, severity) mean. 1885 clips per family.

## Result — both dimensions are blind, and the residual trend is inverted

Mean dimension score per (base, severity). Higher = what VBench treats as better
(less flicker / smoother motion). Figure: `reports/figures/vbench_reward_audit.png`.

**temporal_flickering / color_drift** (aggregate 0.98417):

| base | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 | Δ(.40−.02) |
|---|---:|---:|---:|---:|---:|---:|
| 7WHI2L_FDNg | 0.9857 | 0.9857 | 0.9858 | 0.9858 | 0.9860 | +0.0003 |
| BrRLKMbBTYQ | 0.9886 | 0.9886 | 0.9886 | 0.9886 | 0.9888 | +0.0002 |
| KZ8p6b1zJ9U | 0.9813 | 0.9813 | 0.9814 | 0.9816 | 0.9822 | +0.0009 |
| hhszUXL1Cu8 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9949 | −0.0001 |
| mJog8DlRk_4 | 0.9753 | 0.9753 | 0.9753 | 0.9754 | 0.9757 | +0.0004 |

mean spread **0.0004**; rise 4 / flat 1 / fall 0.

**temporal_flickering / background_drift** (aggregate 0.98456):

| base | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 | Δ(.40−.02) |
|---|---:|---:|---:|---:|---:|---:|
| 7WHI2L_FDNg | 0.9858 | 0.9860 | 0.9862 | 0.9866 | 0.9874 | +0.0015 |
| BrRLKMbBTYQ | 0.9885 | 0.9885 | 0.9883 | 0.9879 | 0.9872 | −0.0014 |
| KZ8p6b1zJ9U | 0.9815 | 0.9817 | 0.9821 | 0.9829 | 0.9843 | +0.0028 |
| hhszUXL1Cu8 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | +0.0000 |
| mJog8DlRk_4 | 0.9754 | 0.9756 | 0.9759 | 0.9765 | 0.9776 | +0.0022 |

mean spread **0.0016**; rise 3 / flat 1 / fall 1.

**motion_smoothness / color_drift** (aggregate 0.98870):

| base | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 | Δ(.40−.02) |
|---|---:|---:|---:|---:|---:|---:|
| 7WHI2L_FDNg | 0.9908 | 0.9908 | 0.9908 | 0.9908 | 0.9909 | +0.0001 |
| BrRLKMbBTYQ | 0.9925 | 0.9925 | 0.9925 | 0.9925 | 0.9926 | +0.0001 |
| KZ8p6b1zJ9U | 0.9849 | 0.9849 | 0.9850 | 0.9851 | 0.9855 | +0.0006 |
| hhszUXL1Cu8 | 0.9955 | 0.9955 | 0.9955 | 0.9955 | 0.9955 | −0.0000 |
| mJog8DlRk_4 | 0.9830 | 0.9830 | 0.9830 | 0.9831 | 0.9832 | +0.0002 |

mean spread **0.0002** (flatter than temporal_flickering); rise 1 / flat 4 / fall 0.

**motion_smoothness / background_drift** (aggregate 0.98869):

| base | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 | Δ(.40−.02) |
|---|---:|---:|---:|---:|---:|---:|
| 7WHI2L_FDNg | 0.9908 | 0.9909 | 0.9910 | 0.9911 | 0.9912 | +0.0004 |
| BrRLKMbBTYQ | 0.9924 | 0.9922 | 0.9918 | 0.9910 | 0.9896 | −0.0028 |
| KZ8p6b1zJ9U | 0.9850 | 0.9852 | 0.9855 | 0.9861 | 0.9869 | +0.0019 |
| hhszUXL1Cu8 | 0.9954 | 0.9954 | 0.9954 | 0.9953 | 0.9953 | −0.0002 |
| mJog8DlRk_4 | 0.9831 | 0.9832 | 0.9833 | 0.9835 | 0.9838 | +0.0008 |

mean spread **0.0012**; rise 3 / flat 1 / fall 1.

**Reading.** All four mean spreads are ≤ 0.0016 on a [0,1] scale where the scores
sit around 0.975–0.996. A 20× increase in injected drift moves either dimension by
essentially nothing — both are blind to gradual drift. `motion_smoothness` is even
flatter than `temporal_flickering` on color_drift (0.0002 vs 0.0004), which makes
mechanical sense: colour drift barely changes frame-to-frame *interpolability*,
which is what AMT/motion_smoothness measures.

The residual trend, where it exists, points the **wrong way**. Across all 20
base×dimension×family cells (|Δ| > 0.0002 threshold): **11 rise, 7 flat, 2 fall**
— the score **fails to fall on 18 of 20**, and "rises" means the metric rates the
*more corrupted* video as better. Same mechanism as the identity reductio.

**One honest exception.** `motion_smoothness / background_drift` on base BrRLK falls
0.0028 — the single largest move in the audit and the *correct* direction: here
motion_smoothness partially responds (background_drift on this base injects real
per-frame content churn that perturbs interpolation). It is one cell out of 20 and
still sub-0.003; it does not lift the dimension out of "effectively blind", but it
is recorded rather than smoothed over.

**The comparison that matters.** On the same `color_drift` family where both VBench
dimensions are flat at 0.0002–0.0004, LR-VCC's anchored-colour sub-metric (D′)
responds on 4/5 bases (calibration record). Same corruption, same videos: the
benchmark's sub-metric moves; VBench's dimensions do not.

## motion_smoothness needed two env fixes (recorded for reproducibility)

`motion_smoothness` failed twice at model-init before producing numbers — both in
the `vbench` conda env, not in the data or the GPU, and both only surfacing an hour
into a run (after clip-splitting):

1. **AMT checkpoint missing.** VBench's `init_submodules` hard-codes a
   `huggingface.co` wget URL for `amt-s.pth` and ignores the `HF_ENDPOINT` mirror
   the runner exports; the blocked host gave `wget … exit 4`. Fixed by fetching
   `amt-s.pth` (12 MB) from `https://hf-mirror.com/lalala125/AMT/resolve/main/amt-s.pth`
   into `~/.cache/vbench/amt_model/`.
2. **Missing python deps.** The env lacked `omegaconf` (imported directly by
   `vbench/motion_smoothness.py`) and `einops` (AMT network), raising
   `NotImplementedError: UnImplemented dimension motion_smoothness!, No module
   named 'omegaconf'`. Installed both (pure-python; numpy untouched). Verified the
   full init path (`import vbench.motion_smoothness`; `init_submodules(['motion_smoothness'])`)
   resolves before the successful re-run, rather than discovering a third gap
   mid-run. The `skimage` numpy-ABI warning in the env is unrelated —
   motion_smoothness does not import it.

Both runs then completed EXIT=0 (~3 h/family; AMT interpolation on a
100%-utilised shared GPU is the cost).

## Honest limitations

- **Two families, not the full battery.** Only color_drift and background_drift are
  generated on the server; identity_degradation, flicker, chunk_boundary are not.
  This establishes **blindness to gradual drift** — the claim the prediction makes.
  It does **not** test the blur-inversion case (a family that globally smooths
  detail) on these dimensions; that needs those families generated.
- **Scores sit near the ceiling** (~0.98–0.996). A near-ceiling metric has little
  room to move, which is part of the point, but "flat" and "saturated" are not
  fully separable without a family known to move the statistic (BrRLK/bg on
  motion_smoothness shows the statistic *can* move, which is mild evidence against
  pure saturation).
- **Single run.** temporal_flickering (MAE) is deterministic; motion_smoothness
  uses a learned interpolator but with a fixed checkpoint and no sampling, so the
  per-severity means are stable to the precision reported.
- **Aggregation is per-clip mean → per-video mean**, matching how the dimension is
  meant to be read; no static-clip filtering changed beyond `--dev_flag` defaults.

## What this establishes / next

- Establishes, with measurements: both VBench-1.x consistency dimensions
  (temporal_flickering, motion_smoothness) are blind to graded gradual drift across
  a 20× severity range, and their residual trend rewards the corruption — the
  temporal/motion-axis analogue of the identity reductio. motion_smoothness, never
  previously interrogated, is if anything the flatter of the two.
- Next: (1) if the blur-inversion case is wanted on these dimensions, generate
  identity_degradation / flicker on the server and extend the audit; (2) fold both
  the temporal/motion (this note) and identity (RR probe) reductios into the
  VBench-vs-LR-VCC comparison as measured evidence rather than construct-validity
  argument.
