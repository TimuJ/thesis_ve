# MLLM consistency judge vs human perception — correlation findings

**Status:** Complete for this round (single rater, n=21). Open-model judge
(Qwen3-VL-8B) validated against human ratings on the cross-gap pair set; the
matching reduced-reference comparison and bootstrap CIs are done. Pending:
additional raters (inter-rater agreement), a second SR model for model-level
discrimination, and an optional judge re-framing (see Limitations).

Artifacts:

- Human ratings: `data/review/human_ratings_cross_gap.json` (42 pairs, 1–5 per axis).
- MLLM scores: `data/review/full18_judged.json` (Qwen3-VL-8B over the 42 gated pairs).
- Reduced-reference: `data/review/full18_gated.json` (SIFT, DINO), `data/review/full18_3axis.json` (LoFTR geometry).
- Labels used for the discrimination check: `data/review/cross_gap_examples/cross_gap_pairs.json`.
- Rating tool: `data/review/cross_gap_rater.html` (standalone, blind, offline).

## What prompted this

We want to justify reporting an open-MLLM consistency judge (Qwen3-VL-8B) on the
cross-gap benchmark. A judge is only worth reporting if it agrees with something
trustworthy — either the existing reduced-reference metrics, or human perception.

The metric-agreement route came back too weak to stand on. Judging all 42 gated
pairs and correlating against the composite gave Pearson +0.09 / Spearman +0.22;
the only moderate tie was MLLM↔DINO `c_sr` (Pearson +0.48). A discrimination
check on the 17 previously eye-vetted SR-diverges/SR-preserves pairs gave MLLM-mean
AUC 0.54 (permutation p≈0.82) — chance. Root cause: the judge is shown
`REFERENCE = GT[i]`, `GROUND TRUTH = GT[j]`, `RECONSTRUCTION = SR[j]` and never
sees `SR[i]`, so it scores per-frame reconstruction fidelity, not the cross-gap
drift that those labels (and `s_M`) encode. So we tested the judge against human
perception of the task it *does* perform.

## Method

- **Stimuli:** the 42 gated cross-gap pairs (FlashVSR reconstructions), spanning
  composite consistency 0.59–1.00. Each pair shown as the exact three crops the
  judge receives: REFERENCE (GT pre-gap), GROUND TRUTH (GT post-gap),
  RECONSTRUCTION (SR post-gap).
- **Judge:** Qwen3-VL-8B-Instruct, bf16, greedy decoding. After a short
  chain-of-thought (reason, then score) it emits each axis score as the
  probability-weighted mean over the 1–5 digit-token logits. Deterministic (fixed
  seeds); the three crops are passed as REFERENCE / GROUND TRUTH / RECONSTRUCTION.
  Scores in `full18_judged.json`.
- **Task:** one rater scored *how faithfully the reconstruction matches the ground
  truth*, 1 (badly off) – 5 (matches GT), on the same three axes the judge uses
  (Texture, Geometry, Semantic). Blind to all model and metric scores; pairs in
  randomized order.
- **Usable set:** the rater flagged 21/42 as bad/unratable (all of goldfish-10 and
  monkey-19, most of kangaroo-16, elephant-17, and a few fox/helmet/lizard pairs),
  leaving **21 fully-rated pairs**.
- **Statistics:** Pearson (PLCC) and Spearman (SRCC); two-sided permutation p
  (20k label shuffles); percentile bootstrap 95% CI (10k resamples of the 21
  pairs). Reduced-reference axes joined by `(video, i, j)`: SIFT/DINO from the
  2-axis gated run, LoFTR from the 3-axis run, composite = the gated
  reliability-weighted geometric mean.

## Result

**Human ↔ MLLM, per axis (n=21):**

| Axis | Pearson [95% CI] | perm p | Spearman [95% CI] | H mean | M mean |
| --- | --- | --- | --- | --- | --- |
| Texture | +0.21 [−0.17, +0.60] | 0.37 | +0.09 [−0.27, +0.63] | 2.48 | 2.39 |
| Geometry | **+0.67 [+0.46, +0.86]** | **0.001** | **+0.64 [+0.38, +0.86]** | 3.57 | 3.79 |
| Semantic | +0.40 [−0.01, +0.74] | 0.07 | +0.43 [+0.08, +0.75] | 3.62 | 4.08 |
| **Overall** | **+0.59 [+0.34, +0.80]** | **0.004** | **+0.69 [+0.31, +0.85]** | 3.22 | 3.42 |

**Per axis — does the human track the MLLM or the matched reduced-reference metric?**

| Axis | human ↔ MLLM (r / ρ) | human ↔ metric (r, `c_sr` / `s_M`) |
| --- | --- | --- |
| Texture | +0.21 / +0.09 | SIFT −0.06 / +0.16 |
| Geometry | **+0.67 / +0.64** | LoFTR **+0.09** / +0.34 |
| Semantic | +0.40 / +0.43 | DINO +0.24 / +0.04 |
| Overall | **+0.59 / +0.69** | composite +0.20 / (ρ) +0.35 |

The MLLM's overall score is a moderate-to-strong proxy for the human judgment
(SRCC 0.69, PLCC 0.59, p=0.004), strongest on geometry (SRCC 0.64, CI clear of
zero) and moderate on semantic. It agrees with the human markedly better than any
reduced-reference metric does: on geometry 0.67 vs LoFTR 0.09; overall 0.69 vs
composite 0.35.

## Reading

Both the human and the judge are answering the same question — is `SR[j]` a
faithful rendering of `GT[j]`, with `GT[i]` for context — so their agreement is
the right thing to measure, and it is real. The reduced-reference axes instead
compare SR self-similarity across the gap (`c_sr`) to GT self-similarity (`c_gt`);
that is a different quantity, which is why they track the human fidelity judgment
only weakly. The clean contrast is geometry: the judge's geometry axis matches the
human (0.67) while LoFTR, computed on the same pairs, does not (0.09).

The judge runs slightly generous relative to the human (overall 3.42 vs 3.22;
widest on semantic, 4.08 vs 3.62) — a small positive offset, not a ranking
disagreement, consistent with the strong rank correlation.

## Honest limitations

- **n=21, single rater.** The rater flagged half the set as unratable (a correct
  call — those clips lack a clean matched pair), so the usable set is small and the
  CIs are wide. There is no inter-rater agreement yet.
- **Texture does not correlate** (PLCC 0.21, SRCC 0.09; CI straddles zero). Both
  sides rate texture low on average (2.48 vs 2.39) — FlashVSR rarely restores fine
  texture — but they disagree pair-to-pair. The validated agreement is on geometry
  and overall, not texture. Semantic is moderate but its Pearson CI touches zero.
- **One SR model (FlashVSR).** The correlation reflects quality variation across
  pairs, not across models. It does not yet show the judge *discriminates between
  models*; a second model would add that and widen the score range.
- **Construct boundary (negative result retained).** This validates the judge as a
  human-aligned *reconstruction-fidelity* scorer, not as a *cross-gap consistency*
  measure. The judge never sees `SR[i]`, and neither the human nor the judge
  fidelity scores separate the old SR-diverges/SR-preserves labels (human-truth AUC
  0.58, MLLM 0.48 — both chance), because those labels encode cross-gap drift, a
  different property. Measuring drift would require re-framing the judge to see
  `SR[i]` and `SR[j]` together.

## What this establishes / next

The human-perception validation succeeds where metric-agreement did not: the open
MLLM judge correlates with human perception of reconstruction fidelity (SRCC 0.69
overall, 0.64 geometry, p≈0.004) and does so better than the SIFT/DINO/LoFTR
reduced-reference metrics. That is a defensible basis for reporting the judge,
provided the n=21 / single-rater / texture caveats are stated.

Next, in priority order: (1) add 2–3 raters for inter-rater agreement and tighter
CIs; (2) score a second SR model to test model-level discrimination and add spread;
(3) if cross-gap drift (not fidelity) is the target, re-frame the judge to see both
SR frames and re-run the discrimination check. Related: `docs/notes/2026-11-05-adr-consistency-calibration-findings.md`.
