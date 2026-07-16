# Comprehensive RoPE sensitivity matrix — time × space (2026-07-10)

All numbers: ΔPSNR (dB) vs ground truth relative to each protocol's stock
baseline, mean over clips; pyiqa, DOVE RGB convention. Temporal + spatial
label sweeps: 10 UDM10 clips (baseline 24.02 dB). Resolution ladder:
8 YouHQ40 clips. JSONs under `results/rope_probe/`.

## 1. Position labels only (content and compute fixed)

| stretch | time (t) | height (h) | width (w) |
|---|---:|---:|---:|
| 0.5 | −0.66 | −1.00 | −0.83 |
| **0.75 (PI zone)** | **+0.01** | **+0.02** | **−0.03** |
| 1.25 | −0.12 | −0.05 | −0.02 |
| 1.5 | −0.22 | −0.07 | −0.03 |
| 2.0 | −0.25 | −0.15 | −0.07 |
| 3.0 | −0.95 | **−2.55** | **−2.30** |

**Offset (shift):** free on ALL three axes at every magnitude tested
(t: to 996; h/w: 8, 24) and in every shift×stretch interaction (±0.003 dB).

Readings:
- Translation invariance is universal across the 3D RoPE.
- The PI zone (mild compression free) exists on every axis.
- Sensitivity is axis-asymmetric: moderate dilation is *gentler* in space
  than time, but extreme geometry distortion (s=3) is ~2.5× more damaging
  spatially. W slightly more tolerant than H (wider trained extent, 88 vs 48).
- Temporal caveat (window-boundary framing): under streaming, distances stay
  in-range until s≈2.6; s=3 is the first out-of-window point.

## 2. Real resolution extrapolation (YouHQ40 ladder, actual grids)

| rung | grid | input | stock GT-PSNR | spatial-PI | PI − stock |
|---|---|---|---:|---:|---:|
| 720² | 48×48 (= trained extent) | crop | 24.70 | ≡ stock | — |
| 1152² | 72×72 (1.5×) | native | 24.55 | 24.00 | **−0.56** |
| 1536² | 96×96 (2.0×) | upsampled ×1.33 | **13.69** | 13.96 | +0.28 |

Readings (calibrated):

1. **Moderate real resolution extrapolation is nearly free:** growing the
   grid from 45² to 72² (1.5× the trained H-extent) costs only **−0.15 dB**
   with stock positions. RoPE again exonerated at moderate extension —
   consistent with the label-sweep dilation numbers.
2. **Spatial PI is counterproductive at moderate extension:** compressing
   positions 72→48 (factor 0.667, inside the compression-cost zone per §1)
   costs −0.56 dB vs just letting positions extrapolate. **Do not apply PI
   for ≤1.5× spatial extension.**
3. **The 1440² rung collapses (−11 dB) for reasons beyond positions:** PI
   recovers only +0.28 dB of an ~11 dB hole, so position encoding is a minor
   factor. Confounds to decompose before interpreting: (a) upsampled-input
   blur; (b) FlashVSR's *adaptive* sparsity — the stock
   `topk_ratio = 2·(768·1280)/(th·tw)` formula thins attention as resolution
   grows (3.8 → 1.5 → 0.83 across the rungs), so the 1536² run attends far
   more sparsely; (Window remainders ruled out: grid 96 divides evenly by 8.)
   Follow-up: rerun 1536² with topk_ratio pinned to the 1152² value.

## 3. Combined verdict for the group

Across every axis and manipulation tested — temporal shift to 996, spatial
shifts, temporal single-pass to 5009 frames, D″ long-video drift, and real
grid growth to 1.5× — **RoPE position extrapolation is consistently NOT the
bottleneck** in FlashVSR. Position-side interventions (PI) only pay off in
narrow regimes (temporal window extension ~1.3×, per §1) and can actively
hurt (spatial ≤1.5×). The real failure modes located so far live elsewhere:
streaming cache/generation (D″ drift) and resolution-scaling mechanics
(adaptive sparsity / windowing at ≥1440²).

Provenance: temporal tables `rope_probe_udm10_gt.md`; D″
`dpp_causal_verdict.md`; spatial + ladder JSONs `results/rope_probe/
{udm10_spatial_h,udm10_spatial_w,res_ladder}/`.
