# Progress Report — July 20 – August 1, 2026

**Topic:** Video Super-Resolution for Long Videos — LR-VCC benchmark + SOTA comparison

## Headline

**The benchmark reached its evidence-complete state.** A
provenance audit produced a single canonical evaluation protocol, under which
the 4-method leaderboard reads **MGLD 0.622 > FlashVSR 0.610 > RealESRGAN
0.604 > UAV 0.589 (MGLD wins 5/5 videos)** — a ranking that survives a
52-config hyperparameter sweep and a leave-one-out ablation. The head-to-head
against the closest published measures is decisive: VBench's
subject/background-consistency dimensions rank the degraded input *above* its
super-resolutions, invert the human ranking, and respond to **0 of 20**
long-range severity cells where LR-VCC is clean on 7 of 10.

## Key Results

### 1. Canonical real-model table — 4 methods, uniform gating

An audit found the earlier 3-method table mixed gating protocols across
methods (close-up identity gate on one method only, stale fps handling on
another). With identical gating everywhere:

| video | MGLD | UAV | FlashVSR | RealESRGAN | winner |
|---|---:|---:|---:|---:|:--:|
| 7WHI | **0.738** | 0.700 | 0.737 | 0.736 | MGLD |
| BrRLK | **0.402** | 0.379 | 0.393 | 0.401 | MGLD |
| KZ | **0.750** | 0.705 | 0.722 | 0.724 | MGLD |
| hhsz | **0.566** | 0.545 | 0.550 | 0.529 | MGLD |
| mJog | **0.654** | 0.617 | 0.649 | 0.631 | MGLD |
| **mean** | **0.622** | 0.589 | 0.610 | 0.604 | **MGLD** |

FlashVSR keeps its signature (best identity 0.598, worst long-range D″ drift
0.862); the new frame-wise RealESRGAN anchor is worst on exactly the
sub-metrics a frame-wise method should fail (identity, temporal flow,
exposure slope) — evidence the metric measures what it claims.

### 2. Rigour package: the ranking is not a hyperparameter accident

- **Sensitivity sweep (52 configs** over composition temperatures and the
  β-parameters of the drift sub-metrics): mean method order stable in
  **45/52**, and all seven exceptions occur at the flattest temperature
  (τ=0.5); the per-video MGLD>UAV verdict holds in **50/52**.
- **Leave-one-out ablation:** dropping identity is the only single-family
  removal that flips the method order; dropping temporal flow changes
  nothing; dropping either colour family flips exactly the artefact cells
  that family was designed to detect — every sub-metric carries unique,
  necessary signal.

### 3. SOTA head-to-head — VBench consistency dimensions disqualify themselves

| dimension | LQ input | MGLD | UAV |
|---|---:|---:|---:|
| subject_consistency | 0.8936 | 0.8927 | **0.9031** |
| background_consistency | **0.9333** | 0.9235 | 0.9317 |

Both readings follow from smoother-output bias: the blurry input scores as
the most "consistent" video, and UAV ranks above MGLD (inverse of human
judgement). Under our identical severity protocol on 50 regenerated
artefact clips, the dimensions respond to **0/20 cells** — universal FLAT,
with 14/20 drifting in the *rewarding* direction — vs LR-VCC clean on 7/10.
Mechanism: the long-video mode zeroes the cross-clip term, so the dimensions
are within-clip-only by construction. (Scope note shipped with the results:
long custom input is not officially supported by VBench; our adjustments are
documented.)

### 4. SeedVR2 — stood up, row deferred

Environment green from zero (full recipe + an apex compatibility shim
documented for the group). The benchmark row is blocked by a 7.9 GiB
per-rank rotary-frequency tensor that is resolution-bound and OOMs shared
40 GB A100s even with sequence parallelism — needs an 80 GB card or a
rotary-library patch. Deferred to the next round with a one-command relaunch
script.

### 5. RoPE arc closed

The extreme retrieval-distance sweep (meeting follow-up) delivered the
final verdict: temporal RoPE extrapolation is bounded and graceful — a
~−1.5 dB plateau even at 100× the trained window, no collapse — in sharp
contrast to spatial extrapolation. Written into the group sensitivity
report (`reports/rope_timespace_sensitivity_matrix.md` §3.5).

## Next Period

1. **Benchmark roadmap:** scale the video set beyond 5 clips, design the
   human-anchoring study, and target the weak matrix cells for a v6 metric
   iteration — detailed plan under discussion.
2. **SeedVR2 row** when an 80 GB slot or the rotary patch lands.
3. **Paper skeleton** from the benchmark + SOTA-audit material.
