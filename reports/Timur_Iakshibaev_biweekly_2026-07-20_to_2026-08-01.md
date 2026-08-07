# Biweekly Report — Timur Iakshibaev

## Period: July 20 – August 1, 2026

## The period in three headlines

1. **The benchmark's headline table got its rigour package — and a corrected,
   canonical protocol.** A provenance audit of the real-model pipeline found
   three inconsistencies in the earlier 3-method table (close-up identity
   gating applied to only one method; stale fps-handling on another; one
   parked gate still active in old JSONs). Under the corrected uniform
   protocol the 4-method leaderboard reads **MGLD 0.622 > FlashVSR 0.610 >
   RealESRGAN 0.604 > UAV 0.589, with MGLD winning all five videos** — and
   the ranking is now backed by a 52-config hyperparameter sweep (mean order
   stable in 45/52; the per-video MGLD>UAV result in 50/52) and a
   leave-one-out ablation showing every sub-metric family carries unique
   signal. The new frame-wise RealESRGAN anchor behaves exactly as designed:
   worst on identity, temporal flow, and exposure slope.
2. **The closest published competitors were put through our own severity
   protocol — and detected nothing.** VBench's subject/background-consistency
   dimensions (the field's standard no-reference consistency measures) rank
   the *degraded input above both of its super-resolutions*, order UAV above
   MGLD (inverse of human judgement), and respond to **0 of 20** severity
   cells on the long-range artefact families — including background
   consistency on background drift, its namesake failure mode — where LR-VCC
   is clean on 7 of the same 10 conditions. Mechanism identified: the long
   -video mode zeroes the cross-clip weight, making the dimensions
   within-clip-only by construction.
3. **SeedVR2 was stood up from zero as the round-2 contrast model, and the
   extreme retrieval-distance sweep closed the RoPE arc.** SeedVR2's
   environment is green (full recipe documented for the group) but its
   benchmark row is deferred: a 7.9 GiB per-rank rotary-frequency tensor is
   resolution-bound and OOMs the shared 40 GB A100s even under 2-way
   sequence parallelism. The retrieval-distance experiment answered the
   meeting's extreme-stretch question: temporal RoPE extrapolation is
   bounded and graceful — a ~−1.5 dB plateau even at 100× the trained
   window, no collapse.

## Key numbers

| result | value |
|---|---|
| Canonical 4-method LR-VCC v5 mean | **MGLD 0.622** > FlashVSR 0.610 > RealESRGAN 0.604 > UAV 0.589 (MGLD 5/5 videos) |
| Hyperparameter sweep (52 configs) | mean order stable **45/52** (all 7 exceptions at the flattest temperature τ=0.5); MGLD>UAV per-video **50/52** |
| Leave-one-out ablation | identity is the only single-family drop that flips the method order; temporal drop changes **0/60** matrix cells; each colour-family drop flips its designed artefact cells |
| Verdict matrix (canonical recompose) | **29/60** clean (PASS+WEAK) |
| VBench dims, severity response | **0/20** cells respond (all FLAT); 14/20 drift in the *rewarding* direction; LR-VCC clean 7/10 on same clips |
| VBench dims, ranking | background: LQ 0.9333 > UAV 0.9317 > MGLD 0.9235; subject: UAV 0.9031 > LQ 0.8936 > MGLD 0.8927 |
| RealESRGAN anchor signature | worst I (0.448), worst T (0.930), worst E (0.328) — as designed for a frame-wise method |
| SeedVR2 standup | env green (py3.10 / torch 2.6 cu124 / flash-attn / apex shim); row deferred — 7.9 GiB per-rank rotary tensor OOMs 40 GB A100s |
| Extreme retrieval-distance sweep | temporal extrapolation bounded: ~−1.5 dB plateau at 100× the trained window |

## Decisions and framing this period

- **Canonical evaluation protocol (supersedes the earlier table):** fps-
  corrected identity inputs and the close-up gate applied uniformly to *all*
  methods, parked dispersion gate excluded. The earlier FlashVSR-first table
  was a mixed-gating artefact; the audit trail is public
  (`reports/figures/lr_vcc_provenance_check.md`) and the correction is
  reported transparently. FlashVSR keeps its signature either way: best
  identity, worst long-range CLIP-trajectory drift.
- **VBench long-video caveat documented:** the official release does not
  support long-video custom input; our comparison runs the
  `vbench2_beta_long` extension with documented adjustments (clip splitting,
  cross-clip weight settings), and this scope note ships with the results.
- **SeedVR2 deferred, not abandoned:** full standup recipe + failure analysis
  in `docs/notes/2026-07-19-seedvr2-standup.md`; the row needs an 80 GB GPU
  or a rotary-library patch. It remains the window-local-positions contrast
  for the next research round.

## Outcomes this period

- [x] Provenance audit → canonical uniform-gating table (4 methods × 5
      videos) + RealESRGAN frame-wise anchor row
      (`reports/figures/realmodel_v5_gated.md`).
- [x] 52-config β/τ/α sensitivity sweep and leave-one-out sub-metric
      ablation, recomposed bit-exactly from cached JSONs
      (`sensitivity_sweep_v5.md`, `loo_ablation_v5.md`).
- [x] SOTA head-to-head: both VBench consistency dimensions scored on the
      real-SR set + 50 bit-reproducibly regenerated artefact clips under the
      identical Δ(severity) protocol (`vbench_sota_verdicts.md`).
- [x] SeedVR2 environment stood up from zero; OOM wall diagnosed and
      documented; row deferred with a relaunch recipe.
- [x] Extreme retrieval-distance sweep (meeting follow-up) finalised and
      committed: temporal RoPE extrapolation is bounded and graceful —
      ~−1.5 dB plateau even at 100× the trained window, no collapse —
      closing the RoPE arc (`reports/rope_timespace_sensitivity_matrix.md`
      §3.5).
- [ ] Long-video HR footage ask — still pending with the group.

## One-line summary for the meeting

The benchmark's headline ranking now carries stability, necessity, and
adversarial-baseline evidence — LR-VCC orders methods where the closest
published consistency measures reward degradation and detect nothing — and
the RoPE arc is closed with its final verdict: temporal extrapolation is
bounded and graceful even at 100× the trained window.
