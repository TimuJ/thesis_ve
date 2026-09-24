# VBench dimension audit — findings (temporal_flickering, motion_smoothness)

**Status:** complete — both dimensions across **five** corruption families
(color_drift, background_drift, identity_degradation, flicker, global_blur).
Runner committed (`scripts/server_runners/run_vbench_reward_audit.sh`,
family-parameterized); results on the server under `~/results/vbench_reward_audit/`;
figure `reports/figures/vbench_reward_audit.png`.

## What prompted this

The reduced-reference probe produced an identity-axis reductio: a self-anchored
identity measure ranks the degraded LQ input as the *most* consistent video,
because self-similarity rewards the absence of detail. This audit asks whether
VBench's own temporal/motion consistency dimensions have the same disease —
measured directly against the corruption battery over a graded severity ladder.

## A terminology correction worth recording

`temporal_flickering` and `motion_smoothness` are **VBench 1.x** long-extension
dimensions (`vbench2_beta_long/eval_long.py`). They are *not* VBench 2.0
dimensions. Label both by version in any write-up.

## Method

- Dimensions: `temporal_flickering`, `motion_smoothness`, via `eval_long.py
  --mode long_custom_input --dev_flag`, on GPU1.
- Families (5 bases × severities 0.02–0.40, a 20× range):
  - **color_drift**, **background_drift** — gradual drift.
  - **identity_degradation** — face-local Gaussian blur (Haar-detected faces
    only; passthrough where no face is present).
  - **flicker** — global sinusoidal brightness, period 15 frames (≈0.5 s).
  - **global_blur** — whole-frame Gaussian blur, sigma = severity × 10.
- eval_long splits each ~2.8-min video into ~84 clips; per-clip scores aggregate
  to a per-(base, severity) mean (1885 clips per family).

## Result — a four-way behaviour, and it is damning in the right way

Mean score-spread across the ladder, and end-to-end Δ direction (a
corruption-sensitive dimension should *fall*):

| dimension | color_drift | background_drift | identity_degradation | flicker | global_blur |
|---|---:|---:|---:|---:|---:|
| temporal_flickering | 0.0004 · blind | 0.0016 · blind | 0.0001 · blind | **0.0257 · falls ✓** | **0.0026 · RISES ✗** |
| motion_smoothness | 0.0002 · blind | 0.0012 · blind | 0.0001 · blind | **0.0053 · falls ✓** | **0.0021 · RISES ✗** |

Four regimes, all measured:
1. **Blind** to gradual colour/background drift and face-local blur (flat to
   ≤0.0016).
2. **Correct** on fast global flicker — the score falls monotonically
   (temporal_flickering strongly, spread 0.0257; motion_smoothness weakly).
3. **Inverted** on global blur — see below.

**global_blur — the reward inversion (the key result).** As whole-frame blur
destroys detail, *both* dimensions rate the video as **more** consistent, on
**all five bases**:

| base | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 | Δ (TF) | Δ (MS) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7WHI2L_FDNg | 0.9857 | 0.9860 | 0.9866 | 0.9876 | 0.9893 | +0.0036 | +0.0028 |
| BrRLKMbBTYQ | 0.9885 | 0.9886 | 0.9890 | 0.9895 | 0.9904 | +0.0018 | +0.0011 |
| KZ8p6b1zJ9U | 0.9813 | 0.9815 | 0.9818 | 0.9823 | 0.9833 | +0.0020 | +0.0028 |
| hhszUXL1Cu8 | 0.9950 | 0.9951 | 0.9953 | 0.9958 | 0.9964 | +0.0014 | +0.0009 |
| mJog8DlRk_4 | 0.9753 | 0.9755 | 0.9762 | 0.9774 | 0.9796 | +0.0043 | +0.0029 |

(temporal_flickering column shown; motion_smoothness rises identically.)
Direction is **unanimous — 10/10 base×dimension cells rise.** The magnitude is
modest (~0.002–0.004) but the *sign is wrong*: destroying detail improves the
score.

## Reading — the exact failure boundary, and why it indicts VBench for SR

- **Seen:** high-frequency global change (flicker). **Missed:** low-frequency /
  local change (drift, face-local blur). **Rewarded:** global detail loss (blur).
- This is the temporal/motion analogue of the identity reductio: a blur has no
  flicker and interpolates perfectly, so self-referential temporal/smoothness
  metrics rate it as *more* consistent.
- **Global blur / detail loss is exactly the dominant failure of a weak SR
  model.** So VBench's temporal and motion dimensions would rank a
  detail-destroying SR method as *more* temporally consistent and smoother than a
  detail-preserving one — the precise inversion the benchmark must avoid, now
  measured on the battery rather than argued.
- On the same `color_drift` family where both dimensions are flat, LR-VCC's
  anchored-colour sub-metric (D′) responds on 4/5 bases.

## motion_smoothness needed two env fixes (recorded for reproducibility)

`motion_smoothness` failed twice at model-init before producing numbers:
1. **AMT checkpoint missing** — VBench's `init_submodules` hard-codes a
   `huggingface.co` wget URL, ignoring `HF_ENDPOINT`; fixed by fetching
   `amt-s.pth` (12 MB) from the mirror into `~/.cache/vbench/amt_model/`.
2. **Missing deps** — `omegaconf`, `einops` pip-installed in the `vbench` env.

## Honest limitations

- **Modest magnitudes.** The inversion is unanimous in direction but small
  (~0.002–0.004) — the dimensions sit near a ceiling (~0.98–0.996). The *sign*
  is the finding; the magnitude is bounded by saturation, which is itself part of
  why these metrics discriminate poorly.
- **identity_degradation is face-local**, so its flat result is "blind to local
  blur", the weaker cousin of the global_blur inversion — both are now on record.
- **Single run**; temporal_flickering (MAE) is deterministic, motion_smoothness
  uses a fixed AMT checkpoint.

## What this establishes / next

- Establishes, across five families: VBench-1.x temporal/motion consistency
  dimensions are blind to the gradual/local corruptions that dominate long-video
  SR, correct only on fast global flicker (which SR does not produce), and
  **actively reward global detail loss** (which SR *does* produce). Measured, not
  argued.
- Feeds the VBench-vs-LR-VCC comparison (`2026-09-18-vbench-vs-lrvcc-comparison.md`)
  as the "what VBench misses / inverts" half.
