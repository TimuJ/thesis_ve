# A stronger judge recovers the human signal — Claude-as-judge vs human perception

**Status:** Diagnostic complete (single rater, n=21). A high-capacity closed
judge (Claude, Opus-class multimodal) was scored on the *identical* cross-gap
pairs and rubric used for the open Qwen3-VL-8B judge, and correlated against the
same human ratings. It tracks human perception markedly better than Qwen-8B on
every axis. Two caveats gate the reading — the judge is **closed** (an
upper-bound, not a reportable open judge) and its scoring was **not fully blind**
(see Limitations). A stronger *open* judge (Qwen3-VL-32B) is downloading now to
test how much of this ceiling an open model recovers.

Companion to `docs/notes/2026-11-15-mllm-judge-human-correlation-findings.md`
(the Qwen3-VL-8B ↔ human validation); this note reuses its stimuli, rubric, and
human ratings so the two judges are measured on exactly the same footing.

Artifacts:

- Human ratings: `data/review/human_ratings_cross_gap.json` (42 pairs, 1–5 per axis; 21 fully rated).
- Claude judge scores: `data/review/claude_ratings_cross_gap.json` (same 21 pairs, 1–5 per axis).
- Qwen3-VL-8B scores: `data/review/full18_judged.json` (baseline for contrast).

## What prompted this

The open Qwen3-VL-8B judge correlated with human perception only moderately
(overall PLCC 0.59 / SRCC 0.69) and sat at chance on texture (PLCC 0.21, SRCC
0.09). The open question was whether that ceiling is a property of the *task* —
the rubric, the three-crop stimulus, the human's own noise — or of the *judge's
capacity*. If a far stronger judge, given the identical pairs and rubric,
recovers the human signal Qwen-8B misses, the bottleneck is capacity, and a
larger open model is worth running. This note is that control.

## Method

- **Stimuli:** the same 21 fully-rated cross-gap pairs (FlashVSR
  reconstructions), each shown as the exact three crops the judges receive:
  REFERENCE (GT pre-gap), GROUND TRUTH (GT post-gap), RECONSTRUCTION (SR
  post-gap).
- **Judge:** Claude (Opus-class multimodal). For each pair it scored *how
  faithfully the reconstruction matches the ground truth*, 1 (badly off) – 5
  (matches GT), on the same three axes the human and Qwen use (Texture, Geometry,
  Semantic) — identical rubric and anchor images. Scores in
  `claude_ratings_cross_gap.json`.
- **Reference judge:** Qwen3-VL-8B-Instruct scores from the companion run, joined
  by `(video, i, j)`.
- **Statistics:** Pearson (PLCC) and Spearman (SRCC) of each judge against the
  human, plus a percentile bootstrap 95% CI (B=10000 resamples of the 21 pairs)
  on every Pearson coefficient. Axis scores and an unweighted 3-axis mean
  ("Overall").

## Result

**Human ↔ judge, per axis (n=21) — Claude vs the Qwen-8B baseline:**

| Axis | Human ↔ **Claude** PLCC [95% CI] | SRCC | Human ↔ Qwen-8B PLCC [95% CI] | SRCC |
| --- | --- | --- | --- | --- |
| Texture | **+0.72 [+0.36, +0.91]** | +0.57 | +0.21 [−0.17, +0.60] | +0.09 |
| Geometry | **+0.90 [+0.80, +0.97]** | +0.88 | +0.67 [+0.46, +0.86] | +0.64 |
| Semantic | **+0.78 [+0.57, +0.90]** | +0.64 | +0.40 [−0.01, +0.74] | +0.43 |
| **Overall** | **+0.87 [+0.74, +0.94]** | **+0.88** | +0.59 [+0.34, +0.80] | +0.69 |

**Mean score level (calibration), 1–5:**

| Axis | Human | Claude | Qwen-8B |
| --- | --- | --- | --- |
| Texture | 2.48 | 2.71 | 2.39 |
| Geometry | 3.57 | 3.71 | 3.79 |
| Semantic | 3.62 | 3.52 | 4.08 |
| Overall | 3.22 | **3.32** | 3.42 |

Claude agrees with the human more strongly than Qwen-8B on **every** axis, and
every Claude Pearson CI clears zero. The overall rank agreement is high (SRCC
0.88, PLCC 0.87 [0.74, 0.94]). The two largest gains are exactly where Qwen-8B
was weakest: **texture** (0.72 vs 0.21 — from chance to strong) and **semantic**
(0.78 vs 0.40). Geometry, already Qwen's best axis, is near-ceiling for Claude
(0.90 vs 0.67). Claude's mean score level also sits closest to the human's
(overall 3.32 vs 3.22), i.e. the smallest calibration offset of the three.

## Reading

The control answers the question it was built for: the human signal Qwen-8B
misses is **recoverable** — a stronger judge on the identical pairs and rubric
tracks the human closely, including on texture, where the task itself was
suspected of being the problem. So the ceiling was the judge's **capacity**, not
the rubric or the stimulus. The clean contrast is texture: same images, same 1–5
question, yet 0.72 vs 0.21. That is the central result for choosing a judge —
it says scaling the judge, not redesigning the task, is the lever.

This directly motivates the open Qwen3-VL-32B run (downloading now): the question
is no longer *whether* a capable judge aligns with the human, but *how much of
Claude's ceiling an openly reportable model recovers.*

## Honest limitations

- **Upper bound, not a reportable judge.** Claude is a closed model; this is a
  ceiling estimate for how far judge-capacity can carry human alignment, not a
  judge we can report as the method. Its value is as a target for the open
  Qwen-32B run, not as a result in itself.
- **Scoring was NOT fully blind (optimistic bias).** When Claude scored these
  pairs, the human and Qwen scores were already present in its working context.
  The human ↔ Claude correlation is therefore an **optimistic** estimate and may
  be inflated by that exposure; treat it as an upper bound on an upper bound. A
  clean, context-isolated re-score (judge sees only the crops) is the correct
  confirmation and is pending. This caveat must travel with every number above.
- **Single rater, n=21.** Same human set as the companion note; no inter-rater
  agreement yet, and the CIs are wide (e.g. texture [0.36, 0.91]). The ranking
  (Claude > Qwen on every axis) is robust to this; the exact coefficients are not.
- **One SR model (FlashVSR).** The correlation reflects quality variation across
  pairs, not across models; it does not show either judge discriminates *between*
  models.
- **Construct boundary (unchanged).** Like the companion note, this validates a
  *reconstruction-fidelity* judge, not a *cross-gap-drift* measure: the judge
  never sees `SR[i]`. The human↔Claude agreement is on the fidelity task both
  perform, not on the old SR-diverges/SR-preserves labels.
- **Judge-vs-judge, for context only (de-emphasized).** Claude and Qwen agree
  with each other moderately (overall r=0.74); this is secondary — the quantity
  of interest is agreement with the human, not between the two judges.

## What this establishes / next

A high-capacity judge recovers human-perceived reconstruction fidelity far better
than Qwen-8B (overall SRCC 0.88 vs 0.69; texture 0.72 vs 0.21 PLCC), on identical
pairs and rubric — evidence that **judge capacity, not task design, is the
limiting factor**, subject to the closed-model and non-blind caveats above. Next:
(1) the open Qwen3-VL-32B run, to measure how much of this ceiling a reportable
open judge recovers; (2) a context-isolated blind re-score to remove the
optimistic bias; (3) additional raters for inter-rater agreement and tighter CIs.
