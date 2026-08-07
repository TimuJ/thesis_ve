# Weekly Progress Report — Timur Iakshibaev

## Period: August 3 – 7, 2026

## Headline

**The RoPE arc is fully delivered and research effort has returned to the
benchmark.** This week produced the **five-phase roadmap that takes LR-VCC
from a validated metric to a publication-grade benchmark** — plus the
complete design for Phase A, which scales the video set from 5 to 15+ bases
under a degradation-stratified sourcing plan.

## 1. Benchmark roadmap (the plan going forward)

Honest gap analysis first: the metric and validation machinery are strong
(canonical 4-method table, 52-config stability sweep, leave-one-out
necessity evidence, the VBench dimensions audit), but three weaknesses
dominate any serious review — the benchmark rests on **5 videos**, has **no
human anchoring**, and the verdict matrix is clean on **29/60 cells**.

The roadmap attacks these in dependency order
(`docs/plans/2026-08-01-benchmark-roadmap.md`):

| phase | window | content |
|---|---|---|
| **A — Scale the video set** | now | 5 → 15+ bases; blocks everything downstream |
| **B — Human anchoring** | next month | pairwise 2AFC study; LR-VCC vs human correlation, VBench dims as the baseline on identical pairs |
| **C — Metric v6** | following | failure analysis of every non-clean matrix cell; parked-gate and temperature levers; frozen v5 stays the reference |
| **D — Leaderboard + competitor expansion** | parallel with C | SeedVR2 row (recipe ready, needs an 80 GB card), 1–2 current diffusion SR methods, and widening the metric-vs-metric audit (tOF, FVD-family, DOVE suite) through the same severity battery |
| **E — Paper + release** | autumn | working target **CVPR 2027**; scripts + source-links data release |

## 2. Phase A designed — sourcing decisions made

Lab HR footage fell through, so the set is built entirely from downloadable
sources. Design decisions fixed this week:

- **Native-LQ only:** genuinely degraded real footage, like the existing 5
  — no synthetic downscaling of pristine 4K. The benchmark stays a pure
  no-reference, in-the-wild story.
- **Single-shot only:** long continuous takes, matching the anchored drift
  measures' assumptions — no cut-aware metric surgery needed this phase.
- **Degradation-stratified quotas** rather than scraping one source: ~3×
  bitrate-starved modern uploads, ~3× analog-era digitizations
  (VHS/camcorder home movies), ~2× low-light/sensor-noise, ~2× low-res
  broadcast/news archive; about half the set with persistent faces so the
  identity sub-metric stays exercised. Diversity across degradation
  *families* becomes a designed, defensible property of the set.
- **Source pool:** Internet Archive (home-movie/public-access collections —
  direct downloads, stable links), NASA archives, and Vimeo/Bilibili/VK for
  contemporary content. The YouTube acquisition path remains closed
  (platform enforcement); these routes are verified alternatives.
- **Staged scale-up:** +10 videos now (15 total — verdict matrix 60 → 180
  cells, all four method rows re-run), curation continuing toward 25+ in the
  background. Bounds the diffusion-SR GPU cost while tripling the
  validation statistics.
- **Release model:** scripts + source links (the standard video-benchmark
  model) — keeps the source pool broad without hosting obligations.

Underway now: the Phase A design spec and the candidate-harvest / QC
pipeline (duration, resolution, and single-shot filters with a cut detector
as the gatekeeper).

## 3. Reporting package delivered

- Biweekly written report for the previous period (canonical 4-method
  table + rigour package + SOTA audit).
- April–August research summary prepared for the lab-group report:
  four liftable outcomes — the LR-VCC benchmark, the SOTA-measure audit,
  the RoPE causal study, and the reproducibility/systems track.

## 4. Asks

- An 80 GB GPU slot (or scheduling window) would unblock the SeedVR2
  leaderboard row — environment is green, relaunch is one command.

## Next week

1. Phase A execution: harvest candidates per quota, run the QC gauntlet,
   first ingests into the evaluation tree.
2. Artefact-battery regeneration on accepted bases (bit-reproducible
   generator; CPU/GPU-days, fully scripted).
3. Draft the Phase B human-study protocol while battery jobs run.
