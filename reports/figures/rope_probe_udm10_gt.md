# RoPE probe — vs-GT quality verdict on DOVE-UDM10 (2026-07-06)

First **quality-bearing** position-perturbation results: FlashVSR v1.1 tiny,
10 UDM10 clips × 29 frames × 22 conditions (full shift × stretch cross,
continuous positions), scored with pyiqa (DOVE RGB convention) against DOVE
ground truth. JSONs: `results/rope_probe/udm10_gt/<clip>/`.

**Baseline sanity:** FlashVSR vs GT = **24.02 dB PSNR** / 0.703 SSIM /
0.268 LPIPS — right beside MGLD's verified 24.23 dB on the same protocol.

## Shift axis (position magnitude beyond trained window)

| condition | GT-PSNR | Δ vs baseline | GT-SSIM | GT-LPIPS |
|---|---:|---:|---:|---:|
| baseline | 24.024 | — | 0.7031 | 0.2678 |
| shift 32 | 24.024 | +0.000 | 0.7030 | 0.2676 |
| shift 996 | 24.025 | +0.001 | 0.7030 | 0.2676 |

**H0 verdict, now in quality terms: absolute-position extrapolation is
FREE.** Parking the clip at positions 996–1020 (≫ trained 0–20) costs zero
GT quality. Confirmed at every stretch level too (interaction terms match
the shift-0 rows to ±0.002 dB).

## Stretch axis (relative-distance geometry), shift=0, continuous

| s | GT-PSNR | ΔPSNR | GT-SSIM | GT-LPIPS | self-PSNR |
|---|---:|---:|---:|---:|---:|
| 0.5 | 23.367 | −0.657 | 0.6895 | 0.2854 | 26.5 |
| 0.75 | 24.035 | **+0.011** | 0.7046 | 0.2681 | 31.8 |
| 1.0 (baseline) | 24.024 | — | 0.7031 | 0.2678 | — |
| 1.25 | 23.901 | −0.123 | 0.6987 | 0.2714 | 33.4 |
| 1.5 | 23.801 | −0.223 | 0.6957 | 0.2740 | 31.8 |
| 2.0 | 23.775 | −0.249 | 0.6941 | 0.2785 | 29.4 |
| 3.0 | 23.079 | −0.945 | 0.6798 | 0.2917 | 26.1 |

## Readings

1. **Mild position compression is quality-free:** s=0.75 costs *nothing*
   (+0.01 dB, SSIM up). Direct PI implication: extending the trained
   21-latent window to ~28 latents with PI-compressed positions should be
   loss-free from RoPE's side.
2. **Dilation damages monotonically** (−0.12 at 1.25 → −0.95 at 3.0);
   compression damage starts around s≤0.5 (−0.66). Both extremes hurt, but
   the near-1 region is asymmetric in compression's favour.
3. **Self-consistency ≠ quality, proven:** s=0.75 and s=1.5 have virtually
   identical self-PSNR (31.8) yet GT damage differs by 0.23 dB (nothing vs
   real loss). The earlier self-consistency sweeps measured *change*;
   only this run measures *harm*. Conclusions based on self-consistency
   alone would have been wrong.
4. **Shift-invariance is total** — across magnitude AND in interaction with
   every stretch level. FlashVSR's growing streaming positions are not, by
   themselves, a quality problem; the risk of window extension is entirely
   in the relative-distance geometry.

## Caveats

- 29-frame windows (~8 latents): distances up to ~8·s — s=3 reaches ~24
  (window edge), s>3 untested here. The extended-window experiment (other
  student) remains the decisive test for large-N windows.
- GT deltas are means over 10 clips; per-clip spread not yet analysed.
- UDM10 = DOVE synthetic degradation; real-world sets (RealVSR) untested.
