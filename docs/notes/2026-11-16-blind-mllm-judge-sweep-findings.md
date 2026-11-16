# Blind MLLM consistency judges vs human perception — context-isolated API sweep

**Status:** Complete (single rater, n=21). Two judges were run **blind** — each
pair scored in its own isolated API call, with no human/Qwen scores and no other
pairs in context — via OpenRouter: a closed model (`google/gemini-4-argon`) and a
large open model (`qwen/qwen3-vl-235b-a22b-instruct`). This is the
**context-isolated re-score** that the Claude note flagged as pending, so it
removes the non-blind caveat on that result. Headline: the best *blind* judge
(Gemini-4-argon) reaches overall PLCC 0.71 / SRCC 0.70 — the strongest blind
approximation to the (contaminated) 0.87 Claude ceiling, and clear of both Qwen
judges.

Companion to:
- `docs/notes/2026-11-15-mllm-judge-human-correlation-findings.md` (open Qwen3-VL-8B ↔ human)
- `docs/notes/2026-11-15-claude-judge-human-correlation-findings.md` (in-chat Claude ↔ human, non-blind)

All four judges here are correlated against the **same** 21 human-rated pairs,
with statistics computed identically (`analyze_sweep.py`), so the rows are
directly comparable except for the readout caveat called out below.

Artifacts:

- Scripts: `dump_crops.py`, `judge_pairs_or.py`, `analyze_sweep.py`; judge library `scripts/lr_vcc/mllm_judge/` (`prompt.py`, `scorer.py`, `judge.py`).
- Blind results: `api_scores_gemini4argon.json`, `api_scores_qwen235b.json` (records: `video,i,j,texture,geometry,semantic`).
- Inputs: `human_ratings_cross_gap.json`, `claude_ratings_cross_gap.json` (in-chat Claude), `full18_judged.json` (local Qwen-8B).

## What prompted this

The Claude note found a high-capacity judge tracks the human far better than
Qwen-8B (overall 0.87 vs 0.59), but with two caveats: Claude is closed, and its
scoring was **not blind** — the human and Qwen numbers were already in its
context, making 0.87 "an upper bound on an upper bound." The fix the note named
was to re-score each pair in an isolated context (judge sees only the crops).
Running the judge through an API, one independent call per pair, does exactly
that — and lets us swap in several strong models for the price of one key.

## Method

- **Dataset:** 18 LaSOT sequences (bbox annotations, `groundtruth.txt`), 42
  cross-gap pairs, 21 fully rated by the single human rater.
- **Stimuli:** for pair (i, j), `ref = GT[i]` (pre-gap), `gt = GT[j]`,
  `recon = SR[j]` (FlashVSR) — the identical crops the open judge receives,
  dumped by `dump_crops.py` (same crop assembly as `judge_pairs.py`).
- **Blind judges:** `judge_pairs_or.py` sends one isolated OpenRouter
  (OpenAI-compatible) `chat.completions` call per pair — no human/Qwen scores, no
  other pairs in context. The rubric and three-image framing come from `prompt.py`
  **unchanged**, so every model is asked the identical question. Models (pinned
  slugs): `google/gemini-4-argon` (closed, $2/$10 per MTok) and
  `qwen/qwen3-vl-235b-a22b-instruct` (open, 235B MoE / 22B active, $0.21/$1.90).
  Both: 42/42 scored, 0 failures, 0 parse misses; ≈ $0.4 total.
- **Readout (important):** the emitted **1–5 integer per axis (argmax)**. The
  Messages/chat API exposes no per-token logprobs, so the open judge's
  probability-weighted digit-logit expected value (`scorer.py`) is **not
  reproducible via API**. The local Qwen-8B row below uses that continuous
  digit-logit readout; the two API judges use the coarser argmax. See Limitations.
- **Reference rows:** local Qwen3-VL-8B (digit-logit, blind — `full18_judged.json`)
  and the in-chat Claude (argmax, **non-blind** — `claude_ratings_cross_gap.json`).
- **Statistics:** `analyze_sweep.py` — Pearson (PLCC), Spearman (SRCC), and a
  percentile bootstrap 95% CI (B=10000) on overall PLCC; joined on `(video, i, j)`.

## Result

**Human ↔ judge (n=21).** Per-axis values are SRCC (ρ); Overall shows PLCC
[bootstrap 95% CI] and SRCC.

| Judge | Blind? | Readout | Texture ρ | Geometry ρ | Semantic ρ | **Overall PLCC [95% CI]** | **SRCC** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| in-chat Claude | ✗ (contaminated) | argmax | +0.57 | +0.88 | +0.64 | **+0.87 [+0.74, +0.94]** | **+0.88** |
| **Gemini-4-argon** | ✓ | argmax | +0.22 | +0.37 | +0.43 | **+0.71 [+0.46, +0.87]** | **+0.70** |
| Qwen3-VL-8B (local) | ✓ | digit-logit | +0.09 | +0.64 | +0.43 | +0.59 [+0.34, +0.80] | +0.69 |
| Qwen3-VL-235B | ✓ | argmax | +0.03 | +0.68 | +0.32 | +0.51 [+0.20, +0.73] | +0.56 |

Gemini-4-argon, scored fully blind, is the strongest blind judge at overall PLCC
0.71 / SRCC 0.70 (CI clear of zero). Among the two blind **API** judges — the
only same-readout comparison in the table — it beats Qwen3-VL-235B decisively
(0.71 vs 0.51). The 235B blind score lands below the local Qwen-8B *on paper*,
but those two rows use different readouts (see Limitations). Texture stays the
weak axis for every blind judge (ρ 0.03–0.22); only the non-blind Claude recovers
it (0.57).

## Reading

The blind path does its job: isolating each pair removes the contamination that
made the Claude 0.87 "an upper bound on an upper bound," and a blind judge still
lands at 0.71/0.70 (Gemini-4-argon). That is a **reportable, GPU-free,
human-aligned judge** with no context caveat — the main thing the Claude note was
missing. No blind judge reaches the non-blind Claude ceiling, which is consistent
with that ceiling being optimistic.

The clean model comparison is between the two blind API judges, which share the
argmax readout: **Gemini-4-argon (0.71) > Qwen3-VL-235B (0.51)**. The 235B
underperforming is the surprise; the most likely contributors are the coarse
argmax readout (it compresses the score range relative to digit-logit) and
model-specific behavior. It is *not* evidence that "235B < 8B" — those rows are
not on the same readout.

## Honest limitations

- **Readout confound (API argmax vs local digit-logit).** The two API judges emit
  a single 1–5 integer (argmax); the local Qwen-8B uses the continuous
  probability-weighted digit-logit score. Argmax is coarser and correlates worse
  by construction, so **do not compare the API rows to the Qwen-8B row directly**
  — the clean comparison is blind-API-internal (Gemini vs 235B). A same-readout
  large-open number needs the digit-logit run (next section).
- **Single rater, n=21.** Wide CIs (e.g. Gemini overall [0.46, 0.87]). Rankings
  are more robust than the point estimates.
- **Closed vs open.** Gemini-4-argon is closed (reproducible only via the pinned
  slug + fixed prompt, not open weights); Qwen3-VL-235B is open-weights. For a
  paper's *reportable* judge, the open model is the one that can be fully specified.
- **OpenRouter routing.** A slug can route to different provider backends; version
  and behavior may drift. Not provider-pinned here — do so for strict reproduction.
- **One SR model (FlashVSR); construct boundary unchanged.** This validates a
  *reconstruction-fidelity* judge, not a *cross-gap-drift* measure: the judge
  never sees `SR[i]`.
- **In-chat Claude row is non-blind.** Kept only as the contaminated upper-bound
  reference; not a result.

## What this establishes / next

A **blind, GPU-free** judge (Gemini-4-argon) correlates with human perception at
0.70 SRCC / 0.71 PLCC — strong enough to report and free of the non-blind caveat
that gated the Claude result. Next:

1. **Digit-logit large-open judge:** run the downloaded local Qwen3-VL-32B (bf16,
   2-GPU; waiting on free GPUs) with the `scorer.py` digit-logit readout, for a
   fair same-readout comparison against the local 8B and a reportable open number.
2. **More raters** — inter-rater agreement and tighter CIs.
3. **A second SR model** — to show model-level discrimination, not just per-pair
   quality variation.
