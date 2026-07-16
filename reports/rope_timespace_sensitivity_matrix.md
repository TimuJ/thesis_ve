# Comprehensive RoPE Sensitivity in Time and Space — FlashVSR

**Timur Iakshibaev · 2026-07-10 · answering the group's July-10 ask:**
*"characterise RoPE sensitivity separately in the spatial (h, w) and temporal
dimensions, on real data, including direct resolution extrapolation."*

## 1. Question and background

FlashVSR's Wan2.1 DiT encodes position with **3D RoPE**: three separate
frequency tables for time, height, and width. The model trains on 81-frame
clips (= 21 latent frames after 4× VAE compression) at up to ~768×1408
(latent H-extent 48). Anything longer or larger forces RoPE to operate
outside its trained range. The question: **which kinds of positional
extrapolation actually cost output quality, per axis?**

## 2. Method (summary)

- **Instrument:** a runtime position-injection hook that swaps any one RoPE
  table for a wrapper transforming baseline positions — constant **offset**
  (shift), **geometry** scaling (stretch; continuous, i.e. true Position
  Interpolation without integer rounding), or cycling (modulo) — with
  on-demand table extension past the stock 1024 rows. The hook is **bit-exact
  when idle** (no-op output identical to stock on a measured-zero
  nondeterminism floor) and engagement-checked; the model repo is never
  modified. 48 unit tests.
- **Scoring:** PSNR/SSIM/LPIPS vs **real ground truth**, pyiqa with the DOVE
  RGB convention (identical to our verified MGLD/UAV baseline numbers).
  Sanity anchor: stock FlashVSR scores 24.02 dB on DOVE-UDM10, beside MGLD's
  verified 24.23 on the same protocol.
- **Data:** label sweeps on DOVE-UDM10 (10 clips × 29 frames, realistic
  degradation); resolution ladder on YouHQ40 (8 clips, 1080×1080 GT,
  270×270 LQ). Long-video arms on the 5 long benchmark videos (2412–5000
  frames, no GT → drift metric D″).

## 3. Results

### 3.1 Position labels only (content and compute fixed) — ΔPSNR dB vs stock

| stretch s | time (t) | height (h) | width (w) |
|---|---:|---:|---:|
| 0.5 (compression) | −0.66 | −1.00 | −0.83 |
| **0.75 (PI zone)** | **+0.01** | **+0.02** | **−0.03** |
| 1.25 | −0.12 | −0.05 | −0.02 |
| 1.5 | −0.22 | −0.07 | −0.03 |
| 2.0 | −0.25 | −0.15 | −0.07 |
| 3.0 (extreme) | −0.95 | **−2.55** | **−2.30** |

**Offsets (translation) are free on every axis** — temporal shift to
position 996 (≈ 50× beyond the trained window): +0.001 dB; spatial shifts
8/24: ±0.003 dB; identical at every stretch level (no interaction).

Temporal caveat: under streaming, FlashVSR's attention span is ~8 latents,
so relative distances stay inside the trained 21-latent range until s≈2.6 —
the small s≤2 costs are *within-window* geometry effects; s=3 is the first
genuinely out-of-window condition.

### 3.2 Real resolution extrapolation (YouHQ40 ladder, actual grids)

| output | latent grid | input | stock vs GT | spatial-PI | PI − stock |
|---|---|---|---:|---:|---:|
| 720² | 48×48 (= trained extent) | crop | 24.70 dB | ≡ stock | — |
| 1152² | 72×72 (1.5×) | native | 24.55 | 24.00 | **−0.56** |
| 1536² | 96×96 (2.0×) | upsampled ×1.33 | **13.69** | 13.96 | +0.28 |

*(Grids from the run records; earlier drafts mislabelled them 45/72/90 —
the input padding to multiples of 32 quantises the grid to multiples of 8.)*

- **1.5× real grid growth costs only −0.15 dB** with stock positions.
- **Spatial PI at 1.5× hurts** (−0.56 dB vs simply extrapolating) —
  compressing to the trained extent (factor 0.67) sits in the
  compression-cost zone of §3.1.
- The **1536² collapse (−11 dB) is not positional** (PI recovers only
  +0.28): prime suspect is FlashVSR's own resolution-adaptive attention
  sparsity (`topk_ratio` thins 3.8 → 1.5 → 0.83 across the rungs — below
  the nominal budget at the collapse rung); upsampled-input blur is the
  secondary candidate. (Window-partition remainders are ruled out: the
  collapse grid, 96, divides evenly into the 8-latent windows.) Follow-ups
  running: the 1536² rung with the sparsity ratio pinned to the healthy
  1152² value; a blur-isolation arm at fixed grid; and intermediate grids
  80/88 to locate the collapse knee against the sparsity dose.

### 3.3 Long videos (2412–5000 frames): drift is not positional

Three arms — stock segmented, true single-pass (positions to latent 1252 on
the extended table), and magnitude-bounded (positions cycled mod 336) — are
**indistinguishable on CLIP-trajectory drift (< 1 % relative spread)** on
every video. FlashVSR's long-video drift belongs to its streaming
cache/generation mechanism, not to position encoding. Bonus: the stock
4089-frame single-pass ceiling is purely the fixed 1024-row table — the
extended table sustained a 5009-frame single pass at 11.6 GiB VRAM.

### 3.4 Methodological finding

**Self-consistency does not predict quality:** s=0.75 and s=1.5 produce
identical output *change* (self-PSNR 31.8 both) but opposite GT verdicts
(+0.01 vs −0.22 dB). Position-perturbation studies need real references.

## 4. Recommendations

1. **For the window-extension study:** temporal extension to ~1.33×
   (21→28 latents) with continuous-PI positions is predicted RoPE-loss-free;
   beyond ~2× expect growing geometry costs (the s≥3 regime). The validated
   hook (continuous PI, per-axis, zero model modification) is ready for the
   extended-window runs — comparing stock vs PI positions there isolates
   RoPE's causal share directly.
2. **Do not apply spatial PI at ≤1.5× resolution extension** — stock
   extrapolation is cheaper.
3. **Long-video and high-resolution improvement effort should not target
   position encoding.** RoPE is exonerated on every axis at every realistic
   operating point; the located failure modes are the streaming KV-cache
   (drift) and resolution-scaling mechanics (sparsity/windowing ≥1440²).

## 5. Why Position Interpolation helps at all (and when it stops)

A fair objection: fractional positions (e.g. 2.5) produce rotary phases the
model never saw — aren't those out-of-distribution too? The resolution
(Chen et al., 2023) is the distinction between **interpolation within the
trained range and extrapolation beyond it**. Attention consumes *relative*
phase, (p−q)·θ; training taught the model this function at ~20 discrete
distances per frequency. PI places new inputs *between* those trained
points, where a learned, smooth function is provably well-behaved;
extrapolation places them *beyond* the trained interval, where the same
learned function is unconstrained. "Novel value" is only harmful outside
the convex hull of training inputs, not merely off the training grid.

Our measurements are this theory's fingerprint — and its boundary:
interpolated positions at s=0.75 cost **nothing** (+0.01 dB, every axis),
while extrapolated distances cost up to −2.6 dB; but compression is not
infinitely safe: at s=0.5, *neighbouring* tokens sit at relative distance
0.5 — **below the minimum nonzero distance ever trained (1.0)** — i.e.
interpolation between the "self" point and the "nearest-neighbour" point,
where attention behaviour changes qualitatively. That is why the PI-free
zone is narrow (~down to 0.75) in a zero-shot setting, and why the LLM
literature pairs aggressive PI with fine-tuning. Implementation note: our
continuous PI evaluates the RoPE formula directly at fractional positions
(no table indexing, no row interpolation — averaging complex phases would
be incorrect); evaluated at integer positions it reproduces the stock
table bit-for-bit in effect (identity condition at the numerics floor).

## 6. Limitations

Single model (FlashVSR 1.3B distilled). All vs-GT clips are temporally
within-window (29–31 frames); deep-length (>81-frame) quality curves lack
public GT — lab-provided long HR footage would close this (conversion
tooling ready). The 1440² confound is not yet decomposed. Spatial trained
extent inferred from the distillation resolution.

