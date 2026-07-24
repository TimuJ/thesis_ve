# Chapter 6 (RoPE Study) — Writing Notes and Corrected Canonical Numbers

**2026-07-16.** Single source of truth for the Ch6 writing pass. Supersedes
any number that conflicts in earlier notes/reports — two corrections landed
after the capstone note (grid labels; the resolution-"collapse" resolution).

## Canonical numbers (final, post-correction)

| claim | number | source |
|---|---|---|
| Baseline FlashVSR vs DOVE-UDM10 GT | 24.02 dB (MGLD verified: 24.23) | udm10_gt |
| Temporal shift +996 (≈50× window) | **+0.001 dB — free** | udm10_gt |
| Spatial shifts 8/24, any stretch | ±0.003 dB — free | spatial h/w |
| PI zone s=0.75 (t / h / w) | +0.01 / +0.02 / −0.03 — free | all sweeps |
| Dilation t: s=1.25/1.5/2/3 | −0.12/−0.22/−0.25/−0.95 | udm10_gt |
| **Extreme dilation t: s=5/10/20/50/125/250** | **−1.34/−1.03/−0.31/−1.57/−1.33/−1.56 — bounded, no collapse** | udm10_extreme |
| Dilation h,w at s=3 | −2.55 / −2.30 | spatial h/w |
| Compression s=0.5 (t / h / w) | −0.66 / −1.00 / −0.83 | all sweeps |
| Continuous-PI identity validation | 53.4 dB = numerics floor | stretch_cont |
| Self-PSNR trap | s=0.75 vs 1.5: both 31.8 self, +0.01 vs −0.22 GT | udm10_gt |
| D″ causal arms (3) spread | <1 % relative — drift NOT positional | dpp_causal |
| Single-pass past stock ceiling | 5009 frames, 11.6 GiB VRAM | dpp_causal |
| **Resolution ladder (corrected)** | grids 48/72/80/88/96 → 24.70/24.55/24.66/24.71/**24.78** — flat to 2× | res_ladder2 |
| Spatial-PI at 1.5× grid | **−0.56 dB (hurts)** | res_ladder |
| Sparsity-pinning arm at 2× | Δ −0.03 — sparsity cleared | res_ladder2 |

Trained window: 81 frames = 21 latents (temporal); spatial extent 48 latents
(from distillation res 768×1408). Ladder grids are multiples of 8 (input
padding to multiples of 32) — early drafts said 45/90; wrong.

## The two corrections (write them into Methods honestly)

1. **Self-consistency ≠ quality.** Identical self-PSNR, opposite GT verdicts
   (see trap row). Consequence: all headline claims are vs real GT.
2. **The −11 dB "collapse" at the top ladder rung was a scoring artefact:**
   the padded output frame (content = 93.75 %) was resized against GT at
   mismatched magnification. Decomposition arms cleared every model-side
   suspect (sparsity Δ−0.03; blur flat; grids 80/88 flat). Both corrections
   were possible because raw frames + re-scorable JSONs were retained —
   worth one sentence as a reproducibility-practice point.

## PI-theory section (port from sensitivity report §5 / presentation §4)

Argument chain: attention consumes relative phase (p−q)·θ → training covers
~20 discrete distances per frequency → PI interpolates between trained
points (smooth learned function), dilation extrapolates beyond → the
**symmetry argument**: rotation-change magnitude depends only on |s−1|·d
(compression and dilation move phases equally far) yet measured quality is
asymmetric → *where* phases land decides damage, not how far they moved →
zero-shot PI boundary at s≈0.5 (neighbour distance drops below minimum
trained distance 1.0, the self-vs-neighbour transition; cf. Chen et al.
2023 pairing aggressive PI with fine-tuning).

## Figures for the chapter

`reports/figures/pi/fig{1..4}.png` (RoPE clock; similarity kernel with
PI-vs-dilation arrows; |s−1|·d symmetry; measured per-axis dose-response).
For LaTeX: regenerate as PDF at same size — script is
`make_pi_figs.py`/`fix_pi_figs.py` in the session scratchpad; parameters:
θ_k = 10000^(−2k/44), 22 freqs, window 21. Add fig5 candidate: the
resolution ladder bar/line (flat 48→96) once numbers are final in prose.

## Section map (rope-study.tex TODOs → sources)

Instrument → flashvsr-rope-site note (hook = duck-typed swap of
`dit.freqs[axis]`; gates). Design arms 1–7 → listed in the tex comments,
now including the decomposition arms. Results subsections → table above.
Implications → findings note + sensitivity report §4 (window-extension
guidance: PI free to ~1.33× temporal; spatial: extrapolate, don't
compress). Limitations → findings note (drop "1440² undecomposed" — done).
