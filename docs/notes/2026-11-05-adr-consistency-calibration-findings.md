# ADR cross-gap consistency — calibration of good vs bad consistency (FlashVSR)

**Status:** Complete for a single SR model (FlashVSR) on the 18-clip starter set.
Pending: a second SR model (to show the benchmark *discriminates* between models),
and the approved scorer trio (SIFT + RoMa v2 + DINOv3; this run uses SIFT + DINO
v1, the two axes available offline). Artifacts:
- eval JSON `data/review/full18_gated.json` (per-gap, 42 scored pairs, all clips `ok`)
- metrics table `data/review/full18_gated_metrics.csv`
- distribution figure `data/review/full18_gated_hist.png`
- review gallery `data/review/full18_gated/index.html` (SR-diverges / SR-preserves / excluded)
- code on branch `adr-cross-gap-consistency` (uncommitted); design spec
  `docs/superpowers/specs/2026-11-02-lrvcc-cross-gap-matched-pair-consistency-design.md`

## What prompted this

The GT-anchored cross-gap consistency `s_M = max(0, 1 − |c_sr − c_gt|)` produces a
number per matched pair, but a number is only useful once its scale is calibrated:
**which values mean the SR preserved the ground-truth cross-gap relationship, and
which mean it failed?** A secondary question: does the automatic pair-selection
produce a clean review set, or does it surface pairs that are not reviewable
(blurred, occluded, too small, pose-mismatched)?

## Method

- **Inputs:** 18 LaSOT disappear–reappear clips with ground-truth boxes/flags;
  FlashVSR super-resolution predictions for each (provided, no local inference).
- **Selection (frozen on GT, reused for the SR):** per-gap Option-B matched pairs
  — one pair per qualifying gap plus, where the big object never sits at a gap, one
  long-range early-vs-late pair. Selection now carries reviewer-feedback gates: an
  appearance gate on the embedding distance (reject different-pose pairs,
  `MAX_APPEARANCE_DIST = 0.35`), a size-scaled edge penalty, and a within-clip blur
  preference. Pairs are routed out of the review set ("excluded") when a reliability
  factor is weak: object on the frame edge (`edge < 0.6`), weak GT anchor
  (`anchor < 0.6`, i.e. the GT pair itself barely matches), or too small
  (`size < 0.35`, area ≲ 1.7 %).
- **Scoring:** SIFT texture consistency and DINO (v1, ViT-B/16) semantic
  consistency, each as `s_M`; composite consistency is the reliability-weighted
  combination.
- **Calibration:** the composite consistency distribution over the kept pairs,
  anchored against a per-pair visual judgement (object clearly degraded by the SR
  vs. clearly preserved) made on the rendered panels.

## Result

42 pairs scored; **29 kept**, 13 routed to the excluded appendix. On the 29 kept
pairs:

| statistic | consistency |
|---|---|
| mean | 0.916 |
| median | 0.949 |
| std | 0.086 |
| p25 / p75 | 0.896 / 0.969 |
| min / max | 0.589 / 0.995 |

Banding (composite): **9 pairs diverge (< 0.90)**, 20 preserve (≥ 0.90), of which
**13 strongly preserve (≥ 0.96)**.

Visually-anchored separation is clean at **0.90**:

| visual judgement | n | consistency range |
|---|---|---|
| SR **degrades** the object (gecko, licenseplate, leopard) | 6 | 0.589 – **0.896** |
| SR **preserves** the object (bus, hippo, tank, electricfan, giraffe) | 7 | **0.898** – 0.985 |

Per-axis `s_M` and coverage:

| scorer | min | median | pairs with GT signal (`c_gt > 0.1`) |
|---|---|---|---|
| SIFT (texture) | 0.512 | 0.965 | **13 / 29** |
| DINO (semantic) | 0.679 | 0.948 | **29 / 29** |

The lowest-consistency pairs are genuine SR failures, each legible in the panels:
gecko (SIFT 0.67→0.18, DINO 0.95→0.63 — the SR reconstructs a different-looking
gecko), licenseplate (SIFT 0.32→0.13 — the readable plate text is degraded), leopard
(DINO 0.56→0.37 — the rosette pattern is smoothed).

## Reading / interpretation

- **Threshold ≈ 0.90 separates fail from preserve.** The visually-degraded set tops
  out at 0.896 and the visually-preserved set starts at 0.898, so the gallery uses
  0.90 as the fail/preserve band edge and ≥ 0.96 as "strong preserve". A pair at
  0.90–0.96 is a *mild* deviation (e.g. bus 0.898, cattle 0.910).
- **The two scorers are complementary by construction, which the coverage numbers
  make explicit.** SIFT only has a GT signal on textured objects (13/29), but where
  it fires it is very sensitive (gecko 0.67→0.18). DINO has a signal on every pair
  (29/29) at moderate sensitivity. So SIFT is a high-precision texture probe and
  DINO a broad-coverage semantic probe; they drive the low scores roughly evenly
  (≈ half each), which is why both axes are needed.
- **GT-anchoring matters:** because `s_M` penalises deviation *from GT's own*
  cross-gap similarity, it flags both drift (c_sr < c_gt) and fabricated
  over-consistency (c_sr > c_gt). One pair (fox p3) shows DINO rising 0.59→0.64
  while SIFT falls 0.24→0.06 — the SR is smoother-but-less-textured than the real
  footage, which a plain "higher-is-better" similarity would miss.

## Honest limitations

- **Single model.** Only FlashVSR was scored, so this calibrates the *scale* but
  does not yet show the benchmark **discriminates between models** — the property
  that makes it a benchmark. That requires a second SR model's outputs and is the
  top open item.
- **Small n and single-reviewer anchoring.** 29 kept pairs, and the 0.90 threshold
  is anchored on one reviewer's panel-by-panel judgement of 42 pairs. The gate
  thresholds (0.35 size, 0.6 edge/anchor, 0.35 appearance) are fit to that same
  judgement and risk over-fitting; they are provisional, read only from config.
- **Not the approved scorer trio.** The approved spec is SIFT + RoMa v2 + DINOv3;
  this run is SIFT + DINO v1 (RoMa v2 / DINOv3 weights are blocked from the server
  and deferred). Absolute `s_M` values will shift when the trio is completed.
- **Negative result — Laplacian sharpness is not a usable motion-blur gate on
  these crops.** Measured over all 42 crops, the pairs the reviewer called
  motion-blurred had the *highest* Laplacian variance (helmet 353–989, monkey
  356–547) while a crisp-looking small object had the lowest (gecko 7–17): the
  statistic tracks texture/clutter density, not perceived motion blur. Sharpness
  was therefore dropped as an exclusion criterion; motion blur remains a reviewer
  eye-call, and ~6 motion-blurred pairs (monkey ×3, helmet, cattle, lion) sit in
  the kept set awaiting that manual call.
- **Composite weighting.** The composite folds reliability (size/edge/anchor); a
  tiny-but-kept object contributes at reduced weight, so the composite is not a raw
  average and should be read with the per-axis `s_M`.

## What this establishes / next

The consistency scale is calibrated: **< 0.90 = SR changed the cross-gap
relationship, ≥ 0.90 = preserved, ≥ 0.96 = strongly preserved**, anchored to visual
judgement and backed by complementary texture/semantic axes. The automatic gates
(edge / weak-anchor / too-small) produce a review set whose exclusions all match the
reviewer's "bad" calls. Next: (1) score a second SR model to demonstrate
discrimination; (2) complete the approved trio (RoMa v2 + DINOv3) and re-calibrate;
(3) widen the visual anchoring beyond a single reviewer before fixing the gate
thresholds.
