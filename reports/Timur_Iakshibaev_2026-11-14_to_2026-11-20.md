# Lab-Meeting Report — Timur Iakshibaev (recap since the last meeting)

*Audience: lab members + lab lead. This period **produced new results**: the composite
free-pass from last meeting is fixed, the scorer stack is upgraded and run on a curated
54-pair set, and a **multimodal-LLM consistency judge** was built and **validated against
human ratings**. Last meeting's three-axis FlashVSR numbers, the free-pass diagnosis and
the 17-pair example pack are not repeated.*

## Where we left off

Last meeting: FlashVSR drifts on all three Ring-Forcing axes (texture worst). We reported
the axes individually because the composite had a verified no-signal "free-pass", and we
planned the per-axis fix, the RoMa v2 + DINOv3 upgrade and a second SR model.

## 1. Free-pass fixed; upgraded scorers run on a curated 54-pair set

- **Fix, and how it works.** Each axis now carries its **own GT-side reliability**, which
  multiplies the pair's scalar reliability inside the composite, so one shared weight no
  longer applies to every axis. For SIFT it is `min(#keypoints on the GT crops) / 20`,
  clamped to [0, 1]. It is computed on **GT, not SR**, on purpose. If it were keyed on SR,
  an over-smoothed SR (few keypoints) would down-weight its own axis and hide exactly the
  over-consistency failure `s_M` is meant to catch. Axes with no reliability hook default
  to 1.0.
- **Upgrade: fallback stack, not the approved one.** DINOv3 and RoMa v2 weights are gated
  and unreachable from the lab server (the mirror returns 403 and huggingface.co is
  blocked; RoMa v2 itself needs DINOv3). So the pre-approved fallback went in instead:
  **geometry = RoMa v1** (DINOv2 backbone, mean match certainty), **semantic = DINOv2
  ViT-B/14**, texture = SIFT. RoMa v1 also serves as the **pose gate** for pair
  selection.
- **Run.** We scored **66 pairs** (35 auto-selected + 31 hand-curated) and gated **12** out
  (too small, or weak pose/anchor), leaving **54 reviewable pairs over 13 clips**.
  Composite: **mean 0.891, median 0.910**. Bands: **23 diverge / 31 preserve**.

| Axis | n (signal) | `c_gt` | `c_sr` (Ring-Forcing) | `s_M` |
| --- | --- | --- | --- | --- |
| SIFT (texture) | 31 | 0.244 | 0.089 | 0.844 |
| DINOv2 (semantic) | 54 | 0.791 | 0.673 | 0.876 |
| RoMa v1 (geometry) | 54 | 0.057 | 0.054 | 0.971 |

- **Texture and semantics still drift** (SIFT −59 %, DINO −15 % in `c_sr` vs `c_gt`), the
  same direction as last meeting.
- **Open question about RoMa (my interpretation, not yet checked).** RoMa's cross-gap
  certainty is near zero on **both** sides (`c_gt` 0.057, `c_sr` 0.054), so `s_M` ≈ 0.97
  comes mostly from `1 − |small − small|`, not from a measured geometric match. It looks
  like the same no-signal shape the fix targets, but on the geometry axis. Its GT-side
  coverage reliability evidently does not down-weight it. LoFTR had a usable `c_gt` (0.386)
  on the earlier set. Until this is checked, **I would not read the 0.971 as "geometry is
  preserved"**.

## 2. An MLLM consistency judge, and why feature metrics weren't enough

- **Motivation: a one-scene concept check (report shared after the last meeting).** We ran
  a giraffe disappear-reappear sequence through five memory variants of the
  collaborator's model and scored each with DINO, SIFT and a three-axis MLLM judgment.
  Two problems showed up:
  - **DINO is gamed by construction.** A variant that assembles its output from
    DINO-retrieved features scores 0.829 against GT, within 0.007 of the best variant and
    above a variant humans prefer (0.761).
  - **SIFT prefers blur.** The LQ input scores 0.368 against GT, while every
    reconstruction sits near 0.07.

  The three-axis judge ranked the variants in the human-preferred order. It also caught
  the gaming as Semantic 3 / Texture 2, a split that one cosine number hides.
- **Reproducible open judge.** Qwen3-VL-8B-Instruct, bf16, greedy decoding with fixed
  seeds. It reasons briefly, then gives each axis score as the **probability-weighted mean
  over the 1–5 digit-token logits**, which yields a continuous score rather than an
  integer. The stimulus is three crops: GT pre-gap, GT post-gap, SR post-gap.
- **Agreement with the existing metrics failed, and that was informative.** Over 42 gated
  pairs the judge vs the composite gives Pearson +0.09 / Spearman +0.22. On the 17
  eye-vetted diverge/preserve labels its AUC is 0.54 (p≈0.82), which is chance.
  **Mechanism:** the judge never sees `SR[i]`, so it scores **reconstruction fidelity at
  `j`** and not cross-gap drift. The labels and `s_M` encode drift, a different quantity.
- **So we validated it against humans on the task it actually does.** One blind rater
  scored the same three crops on the same 1–5 rubric. They flagged 21/42 pairs as
  unratable, leaving **n=21**.

| Axis | human ↔ Qwen-8B PLCC [95% CI] | SRCC | human ↔ matched metric (r) |
| --- | --- | --- | --- |
| Texture | +0.21 [−0.17, +0.60] | +0.09 | SIFT −0.06 |
| Geometry | **+0.67 [+0.46, +0.86]** | +0.64 | LoFTR +0.09 |
| Semantic | +0.40 [−0.01, +0.74] | +0.43 | DINO +0.24 |
| **Overall** | **+0.59 [+0.34, +0.80]** (p=0.004) | **+0.69** | composite +0.20 |

The judge tracks human perception of fidelity much better than any of our feature metrics
(geometry 0.67 vs LoFTR 0.09). **Texture does not correlate**: both the human and the judge
rate texture low on average, but they disagree pair to pair.

## 3. Is the ceiling the task or the judge? A capacity control and a blind sweep

- **Capacity control (upper bound only).** A frontier closed model (Claude) scored the
  identical 21 pairs with the identical rubric. It reached overall **PLCC 0.87 / SRCC
  0.88**, and texture rose from 0.21 to **0.72**. That points to judge capacity, not
  rubric design, as the bottleneck. **Caveat: this scoring was not blind.** The human and
  Qwen scores were in its context, so 0.87 is an optimistic ceiling, not a result.
- **Blind re-score, the fix for that caveat.** Each pair went out as an isolated
  OpenRouter call, with no other scores and no other pairs in context. 42/42 scored, 0
  parse failures, about $0.40 total.

| Judge (blind) | Readout | Overall PLCC [95% CI] | SRCC |
| --- | --- | --- | --- |
| Gemini-4-argon (closed) | argmax 1–5 | **+0.71 [+0.46, +0.87]** | **+0.70** |
| Qwen3-VL-8B (open, local) | digit-logit | +0.59 [+0.34, +0.80] | +0.69 |
| Qwen3-VL-235B (open, API) | argmax 1–5 | +0.51 [+0.20, +0.73] | +0.56 |

- **Readout confound.** The API exposes no logprobs, so the API judges give a coarse
  argmax integer while the local 8B uses the continuous digit-logit score. The only clean
  comparison is therefore Gemini vs 235B (same readout). The table does **not** show that
  "235B < 8B". Texture stays weak for every blind judge (ρ 0.03–0.22).
- **Next:** run local **Qwen3-VL-32B** with the digit-logit readout (downloaded; waiting
  on free GPUs). That gives the fair, open, same-readout comparison.

## 4. Pair selection: from precision to recall (in progress, not committed)

The pose gate filtered too hard: some videos returned no pairs. And on large candidate
sets, running one RoMa forward per candidate made selection grind. Current working-tree
changes:

- **Pose budget = 50** RoMa verifications per gap, with an early stop once top-K is met.
- **Backfill to top-K = 5.** Pose-verified pairs still rank first, and the rest are filled
  with the best-by-cost diverse candidates, because manual review filters after
  selection.
- **Area floor relaxed from 1 % to 0.2 %.** Objects often shrink below 1 % just before a
  gap, which zeroed the pre-gap side.
- **Diversity separation cut from 1.0 s to 0.5 s.**
- **RoMa runs on the pure-torch `local_corr` fallback.** The compiled kernel is missing on
  the server; the fallback is correct, just slightly slower.

Unit tests cover the budget, the early stop and the backfill ordering.

## Key numbers

- Curated run: **54 reviewable / 66 scored pairs, 13 clips**; composite **mean 0.891 /
  median 0.910**; **23 diverge / 31 preserve**. `s_M`: SIFT **0.844**, DINOv2 **0.876**,
  RoMa v1 **0.971** (RoMa saturation suspected; see §1).
- Concept check: DINO gives the gamed variant **0.829** vs the human-preferred **0.761**;
  SIFT scores LQ **0.368** vs reconstructions **~0.07**.
- Qwen-8B vs metrics, measured: composite r **+0.09**, label AUC **0.54**, i.e. chance.
- Human vs judge, n=21, single rater, measured: Qwen-8B overall SRCC **0.69**
  (p=0.004), geometry **0.64**; best blind judge Gemini-4-argon PLCC **0.71** / SRCC
  **0.70**; non-blind Claude **0.87** (optimistic, upper bound only).
- Blind API sweep cost **≈ $0.40** for 2 models × 42 pairs.

## Next period (plan)

1. **Qwen3-VL-32B with the digit-logit readout**, for a reportable open judge on the same
   readout as the 8B.
2. **Check the RoMa geometry axis:** is cross-gap certainty genuinely near zero, and should
   its reliability gate it out?
3. **Rerun selection with the recall changes**, then manual review, then rescore.
4. **2–3 more raters**, for inter-rater agreement and tighter CIs.
5. **A second SR model** (the collaborator's), so the judge and the axes can show
   *model-level* discrimination.

## Decisions needed from the lab

1. **What the judge measures.** Report it as a **reconstruction-fidelity** judge (what is
   validated now), or re-frame it to see `SR[i]` and `SR[j]` together and measure drift,
   then re-validate? *Proposal: report fidelity now, prototype the drift framing next.*
2. **Closed vs open judge.** Gemini-4-argon is the best blind judge but closed. *Proposal:
   headline the open model if 32B closes the gap, and keep Gemini as a reference row.*
3. **Raters.** Can 2–3 labmates rate the 42 pairs? It takes about 20 minutes with the
   offline rating page.

## One-line summary

The composite free-pass is fixed and the curated 54-pair run is in (texture and semantics
still drift; the RoMa geometry axis looks saturated). An MLLM judge that **agrees with
human perception far better than SIFT/DINO/LoFTR** is validated (blind best 0.70 SRCC),
with the caveat that it measures reconstruction fidelity, not yet cross-gap drift.
