# VBench dimension audit — findings (temporal_flickering, motion_smoothness)

**Status:** complete — both dimensions across **four** corruption families
(color_drift, background_drift, identity_degradation, flicker). Runner committed
(`scripts/server_runners/run_vbench_reward_audit.sh`, family-parameterized);
results on the server under `~/results/vbench_reward_audit/`; figure
`reports/figures/vbench_reward_audit.png`.

## What prompted this

The reduced-reference probe produced an identity-axis reductio: a self-anchored
identity measure ranks the degraded LQ input as the *most* consistent video,
because self-similarity rewards the absence of detail. This audit asks whether
VBench's own temporal/motion consistency dimensions have the same disease —
measured directly against the corruption battery over a graded severity ladder,
not argued from construct validity.

## A terminology correction worth recording

`temporal_flickering` and `motion_smoothness` are **VBench 1.x** long-extension
dimensions (`vbench2_beta_long/eval_long.py`, class `VBenchLong`). They are *not*
VBench 2.0 dimensions. Label both by version in any write-up.

## Method

- Dimensions: `temporal_flickering`, `motion_smoothness`, via `eval_long.py
  --mode long_custom_input --dev_flag`, on GPU1 (`nice`d).
- Families (5 bases × severities 0.02, 0.05, 0.10, 0.20, 0.40 — a 20× range):
  - **color_drift**, **background_drift** — gradual drift.
  - **identity_degradation** — face-local Gaussian blur (Haar-detected faces
    only; a passthrough where no face is present).
  - **flicker** — global sinusoidal brightness modulation, period 15 frames
    (≈0.5 s), amplitude ∝ severity.
- eval_long splits each ~2.8-min video into ~84 two-second clips and scores per
  clip; clip paths carry `base_sevX`, so per-clip scores aggregate cleanly to a
  per-(base, severity) mean (1885 clips per family).

## Result — blind to everything except fast global flicker

Mean spread of the dimension score across the 20× severity ladder (higher score
= what VBench treats as better; a corruption-sensitive dimension should *fall*):

| dimension | color_drift | background_drift | identity_degradation | flicker |
|---|---:|---:|---:|---:|
| temporal_flickering | 0.0004 | 0.0016 | 0.0001 | **0.0257** |
| motion_smoothness | 0.0002 | 0.0012 | 0.0001 | **0.0053** |

Seven of the eight cells are flat to ≤0.0016 on a scale where scores sit around
0.94–0.996 — **blind**. The exception is **flicker**, and there the response is
correct: the score **falls monotonically** as flicker worsens.

**temporal_flickering / flicker** (aggregate 0.97391) — the one strong response:

| base | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 7WHI2L_FDNg | 0.9850 | 0.9829 | 0.9790 | 0.9710 | 0.9555 | −0.0295 |
| BrRLKMbBTYQ | 0.9879 | 0.9867 | 0.9846 | 0.9801 | 0.9714 | −0.0166 |
| KZ8p6b1zJ9U | 0.9805 | 0.9783 | 0.9738 | 0.9642 | 0.9455 | −0.0351 |
| hhszUXL1Cu8 | 0.9947 | 0.9940 | 0.9925 | 0.9891 | 0.9822 | −0.0125 |
| mJog8DlRk_4 | 0.9743 | 0.9720 | 0.9675 | 0.9581 | 0.9395 | −0.0348 |

All five bases fall; `motion_smoothness/flicker` also falls on all five but ~5×
weaker (spread 0.0053). Full per-family curves in the figure.

## Reading — a precise, mechanism-based boundary (and why it is the stronger claim)

The dimensions are **not uniformly broken.** They respond to exactly what they
were designed for — **fast, global** brightness flicker — and are blind to
everything else tested: gradual colour drift, gradual background drift, and
face-local blur. The boundary is **high-frequency global (seen) vs
low-frequency / local (missed)**.

This is a stronger position than a blanket "VBench is blind":

1. **It is fair.** We tested a corruption VBench *can* see (flicker) and reported
   that it sees it. The critique cannot be dismissed as a strawman.
2. **The missed corruptions are the ones that matter for super-resolution.** SR
   models do not produce fast global flicker; they produce gradual drift and
   detail/identity loss. So these dimensions catch the corruption that is
   *irrelevant* to SR and miss the ones that are *central*. That is the crux.
3. **temporal_flickering is aptly named and narrowly scoped** — an adjacent-frame
   MAE statistic sees per-frame brightness oscillation and nothing slower.

**The comparison that matters.** On the same `color_drift` family where both
dimensions are flat (≤0.0004), LR-VCC's anchored-colour sub-metric (D′) responds
on 4/5 bases (calibration record). Same corruption, same videos: the benchmark's
sub-metric moves; VBench's dimensions do not — except where the corruption is
fast global flicker, which SR does not produce.

## motion_smoothness needed two env fixes (recorded for reproducibility)

`motion_smoothness` failed twice at model-init before producing numbers, both in
the `vbench` conda env:
1. **AMT checkpoint missing** — VBench's `init_submodules` hard-codes a
   `huggingface.co` wget URL, ignoring `HF_ENDPOINT`; the blocked host gave
   `wget … exit 4`. Fixed by fetching `amt-s.pth` (12 MB) from the mirror into
   `~/.cache/vbench/amt_model/`.
2. **Missing python deps** — `omegaconf` (imported by `vbench/motion_smoothness.py`)
   and `einops` (AMT network). pip-installed (pure-python; numpy untouched);
   init path verified before the successful re-run.

## Honest limitations

- **identity_degradation tests "blind to *local* blur", not "rewards *global*
  blur".** The blur is confined to detected faces, so a global frame metric
  barely sees it (spread 0.0001) — which is expected, not a strong inversion
  result. Testing whether the score *rises* as detail is globally destroyed needs
  a **global-blur family** (next step).
- **Scores sit near the ceiling** (~0.98–0.996) on the flat families; flicker
  shows the statistic *can* move (down to 0.94), which is mild evidence against
  pure saturation on that family at least.
- **Single run**; temporal_flickering (MAE) is deterministic, motion_smoothness
  uses a fixed AMT checkpoint with no sampling.

## What this establishes / next

- Establishes, with measurements across four families: VBench-1.x
  temporal/motion consistency dimensions respond **only to fast global flicker**
  and are blind to gradual drift and local degradation — the corruptions that
  dominate long-video SR. temporal_flickering catches its namesake; nothing else.
- Next: (1) a **global-blur family** to test the blur-*inversion* claim directly;
  (2) fold this (the "what VBench misses" half) together with the reduced-
  reference correspondence work (the "what LR-VCC adds" half) into the
  VBench-vs-LR-VCC comparison as measured evidence.
