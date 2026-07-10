# RoPE Extrapolation Probe — Findings (Task 10 capstone)

**Period:** 2026-07-02 → 2026-07-11. **Model:** FlashVSR v1.1 (Wan2.1 DiT,
3D RoPE, one-step streaming VSR). **Branch:** `rope-probe`.
**Question (Direction 4 / supervisor-confirmed):** how does RoPE extrapolate
beyond its trained window in time and space, and is it the cause of
long-video / high-resolution quality degradation?

## Instrument

A runtime position-injection hook (`scripts/rope_probe/`): swaps any of the
three RoPE frequency tables (t/h/w) for a wrapper that transforms baseline
positions (shift / stretch / continuous-PI / modulo), with on-demand table
extension past the stock 1024 rows. **Bit-exact when idle** (no-op drift 0.0
on a measured 0.0 nondeterminism floor), engagement-checked, 48 local unit
tests, zero modification of the pristine-tagged FlashVSR repo. Scoring:
pyiqa (DOVE RGB convention) vs real GT (DOVE-UDM10, YouHQ40), matching the
project's verified baseline numbers (FlashVSR 24.02 dB ≈ MGLD's 24.23 on
the same protocol).

## Verdicts

1. **H0 — absolute-position extrapolation is free.** Offsets cost nothing on
   any axis, at any magnitude tested (temporal to position 996 ≈ 50× the
   trained window; spatial 8/24), in every interaction. The streaming
   design's unbounded position growth is harmless per se; the "4089-frame
   ceiling" is an implementation artefact (fixed table), removed in
   production by the extended table (5009-frame single pass, 11.6 GiB).
2. **H1 — refined, not confirmed as stated.** What damages quality is
   relative-distance *geometry* distortion, not extrapolated magnitude:
   dilation costs monotonically (time: −0.12 dB @1.25 → −0.95 @3.0; space
   ~2.5× steeper at the extreme); mild compression (s=0.75, the PI zone) is
   quality-free on all axes. Under streaming, distances stay in-range until
   s≈2.6 — the small s≤2 costs are within-window geometry effects; s=3 is
   the first genuinely out-of-window point.
3. **Long-video drift (D″) is NOT positional.** Segmented vs single-pass vs
   magnitude-bounded arms are indistinguishable (<1 % spread) on 2412–5000
   frame videos. FlashVSR's weakest benchmark cell belongs to the streaming
   cache/generation mechanism, not position encoding.
4. **Real resolution extrapolation:** 1.5× grid growth costs only −0.15 dB
   stock; spatial-PI there *hurts* (−0.56) — do not apply PI at ≤1.5×
   spatial extension. The 1440² collapse (−11 dB) is non-positional
   (candidates: adaptive topk sparsity thinning with resolution, upsampled
   input, window remainders); follow-up defined (pin topk_ratio).
5. **Method finding: self-consistency does not predict quality.** Conditions
   with identical self-PSNR showed opposite GT verdicts (s=0.75 vs s=1.5).
   Position-perturbation studies need real references.

## Implications

- **Window-extension study (handoff):** temporal extension to ~1.33×
  (21→28 latents) with continuous-PI positions is predicted RoPE-loss-free;
  beyond that, geometry costs grow. The validated hook + predictions are
  ready for the extended-window runs.
- **FlashVSR improvement directions:** position encoding is exonerated —
  effort should target the streaming KV-cache/generation (drift) and
  resolution-scaling mechanics (sparsity/windowing), not RoPE.
- **Round 2 (SeedVR2 contrast)** remains interesting for the *spatial* axis
  specifically (window-local positions vs our measured spatial brittleness
  at extreme geometry).

## Limitations

- All vs-GT evidence is temporally within-window (29–31-frame GT clips);
  deep-length (>81-frame) quality-vs-position curves lack GT — YouTube
  re-acquisition of the 5 long sources is blocked by platform enforcement
  (all clients, all networks tried); the path forward is lab-provided long
  HR footage via `make_long_gt.py` (group ask pending).
- Single model (FlashVSR 1.3B distilled); 1440² confounds undecomposed;
  spatial trained-extent taken from the distillation resolution (48 latent).

## Artefact index

`reports/figures/rope_sensitivity_matrix.md` (comprehensive matrix) ·
`rope_probe_udm10_gt.md` (temporal vs-GT) · `dpp_causal_verdict.md` (D″) ·
`realmodel_v5_3method.md` (benchmark row) ·
`docs/notes/2026-07-02-flashvsr-rope-site.md` (architecture + gates) ·
JSONs under `results/rope_probe/` · code `scripts/rope_probe/` (48 tests).
