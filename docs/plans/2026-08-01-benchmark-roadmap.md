# Benchmark Roadmap — post-RoPE phase

*Drafted 2026-08-01. The RoPE extrapolation arc is closed (verdicts delivered,
instrument handed off); research effort returns to the LR-VCC benchmark. Goal:
a publication-grade benchmark — CCF-B/ECCV-level paper as the north star.*

## Gap analysis (why these phases)

The metric and validation machinery are solid (canonical 4-method table,
52-config stability sweep, leave-one-out necessity evidence, SOTA audit), but
three weaknesses dominate any serious review:

1. **The benchmark rests on 5 videos.** Every claim — leaderboard, verdict
   matrix, gate statistics — inherits n=5.
2. **No human anchoring.** LR-VCC beats VBench dims on construct validity
   arguments, but there is no quantitative human-correlation number.
3. **The verdict matrix is clean on 29/60 cells.** The designed-for drift
   families hit 7/10, but the overall figure invites "your own battery says
   the metric misses half the corruptions."

Phases A–E attack these in dependency order. v5 stays the frozen reference;
nothing changes retroactively.

## Phase A — Scale the video set (August) — BLOCKS B/C/D

- Close the lab HR-footage ask; target **20–30 long videos** (≥1 min, a few
  ≥5 min), deliberately diversified: faces / no-faces, static / moving
  camera, low-light, texture-heavy, scene cuts vs single-shot.
- Fallback sources if lab footage falls short: TBD (Phase-A brainstorm).
- Degradation side already exists (`make_long_gt.py`); regenerate LQ inputs
  and the artefact battery on the new bases (bit-reproducible generator —
  GPU-days, not new code).
- Payoff: verdict matrix 60 → ~300+ cells; reliability gates get real
  statistics; "n=5" stops being the first review question.

- Size and stratify the set anticipating the Phase-C calibration/validation
  split: every degradation family needs bases in both the calibration and
  the held-out validation subsets.

**Exit criterion:** ≥20 curated bases with LQ inputs + artefact battery
regenerated + at least the 4 existing method rows re-run.

## Phase B — Human anchoring study (September)

- Pairwise 2AFC protocol: "which video is more consistent over time" on
  (i) real SR outputs (4 methods × N videos) and (ii) a stratified sample of
  artefact clips at graded severities.
- Raters: labmate-scale first (~3–5 raters); crowdsource only if a venue
  demands it. **[OPEN: scope decision]**
- Deliverables: Spearman/Kendall of LR-VCC vs human ranking; 2AFC agreement;
  identical numbers for VBench dims on the same pairs (their ranking
  inversion becomes a quantified human-disagreement result).

## Phase C — Metric v6 (September–October, driven by A+B data)

- Failure analysis of every non-clean matrix cell: which sub-metric should
  have fired, and which stage lost the signal (measurement, normalisation,
  gate, composition).
- Known levers queued: parked identity-dispersion gate; τ choice (the sweep's
  only instability source was τ=0.5).
- **Sensitivity calibration (explicit protocol):** promote the severity
  battery from validator to calibration signal — fit each sub-metric's
  response parameters (α, slope-β, D′/D″ β, gate thresholds) so that
  severity 0.02→0.40 maps to a target monotone response curve on its
  designed artefact family. Hard rule: **calibrate and validate on disjoint
  base videos** (this is why Phase A blocks it — at n=5 there is nothing to
  split); verdict matrices are reported on held-out bases only, otherwise
  the PASS counts become self-fulfilling. The sign-flip control families
  guard against over-calibration (a metric tuned to respond to everything
  must still stay silent on them). Phase B's human severity ratings add a
  second, perceptual calibration target.
- Rule: changes must be justified by the enlarged battery and/or human data,
  and reported against frozen v5.

## Phase D — Leaderboard + competitor expansion (October, parallel with C)

- **SeedVR2 row** (recipe + relaunch script ready; blocked on an 80 GB GPU
  or a rotary-library patch) — also the window-local-positions contrast that
  links back to the RoPE study.
- 1–2 current-generation diffusion SR methods (STAR or whatever leads by
  autumn). **[OPEN: shortlist]**
- Widen the metric-vs-metric audit beyond the two VBench dims: tOF / warping
  error, FVD-family, DOVE's metric suite — all through the same severity
  battery, making the "existing measures don't detect long-range corruption"
  claim comprehensive rather than two-sample.

## Phase E — Paper + release (November)

- Benchmark paper: metric + battery + human anchoring + leaderboard +
  competitor audit. Working target: **CVPR 2027 (~mid-November deadline)**.
  **[OPEN: confirm venue with collaborators]**
- Code + data release: cleanup, reproduction scripts, leaderboard page,
  licensing check on released footage.

## Open decisions

| # | decision | affects |
|---|---|---|
| 1 | Venue/timeline (CVPR 2027 vs alternative) | hardness of Phase A deadline |
| 2 | Video sourcing mix + realistic count | Phase A (brainstorm scheduled) |
| 3 | Human study scope (labmates vs crowdsourced) | Phase B budget/duration |
| 4 | New method shortlist | Phase D |
