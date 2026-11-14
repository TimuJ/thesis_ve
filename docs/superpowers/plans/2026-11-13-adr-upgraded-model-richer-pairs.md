# Upgraded-model pair selection + richer cross-gap pair set — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce ~5 reviewable cross-gap pairs per video on the 18 starter clips, selected by a RoMa-v2 pose gate (CLIP still ranks), scored on SIFT + RoMa v2 + DINOv3 with a GT-based per-axis reliability fix, surfacing genuine FlashVSR failures.

**Architecture:** Extend the existing GT-frozen selection to emit top-K diverse pairs per video and admit them through an injected RoMa pose gate; extend the PairScorer protocol with a GT-side reliability signal and feed per-axis reliabilities into the composite; add RoMa/DINOv3 scorer adapters and run on the server. Local pure-logic tasks (1–4) are TDD'd on the M1 with injected fakes; server tasks (5–8) are smoke/verification-based.

**Tech Stack:** Python, numpy, pytest (local); PyTorch 2.6 + RoMa v2 + DINOv3 via transformers, hf-mirror weights (server, env `lrvcc`).

**Spec:** `docs/superpowers/specs/2026-11-13-adr-upgraded-model-richer-pairs-design.md`

## Global Constraints

- **Selection frozen on GT.** The pose gate and ranking read GT crops only; the chosen pair is reused for GT and SR scoring. No selection signal ever reads SR.
- **Per-axis reliability is GT-based** — computed on the GT crop pair, never the SR crop pair.
- **No band recalibration.** The 0.90 preserve/diverge threshold is unchanged; do not re-fit it.
- **Decision (a):** RoMa is a *binary admissibility gate*; the existing CLIP cost keeps ranking admissible pairs.
- **Tunables live in `config.py`** only; never hard-code them elsewhere.
- **Branch:** work stays on `adr-cross-gap-consistency` (not main). Each commit message ends with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Server weights via hf-mirror** (huggingface.co is blocked). Fallbacks: RoMa v2→v1→LoFTR; DINOv3→DINOv2 (flag any fallback taken).
- **Server data:** GT `~/data/LRVCC/src/`, SR `~/data/LRVCC/pred/flashvsr/`, out `~/data/LRVCC/out/`. Server connection details stay out of git.

## Review Focus

- **SR-side reliability leak:** if per-axis reliability is ever computed on the SR crop, an over-smoothed SR (few keypoints/matches) down-weights and hides its own over-consistency failure. → Task 3 tests reliability is GT-sourced.
- **Mean dilution:** a per-video mean over ~5 pairs hides one collapsed pair. → Task 4 tests the summary surfaces the min-consistency pair.
- **Diversity-rule boundary:** two pairs within <1s on both endpoints must collapse; 1s apart on either endpoint must both survive. → Task 1 tests both sides of the boundary.
- **Pose-gate rejects everything:** a too-strict gate yields 0 gap pairs for a video; selection must return a well-formed result *gracefully* (no crash, and it does **not** force unreviewable pairs in — reviewability beats count; `ROMA_POSE_GATE_THRESH` is the tuning lever). → Task 2 tests empty-gate-result handling.
- **No-signal free-pass still gone:** an axis with GT `c_gt≈0` (no signal) must not dominate the composite. → Task 3 tests a zero-signal axis gets ~0 weight.

---

### Task 1: Top-K diverse pair selection (`choose_pairs`)

**Files:**
- Modify: `scripts/lr_vcc/adr/selection.py`
- Modify: `scripts/lr_vcc/adr/config.py`
- Test: `tests/lr_vcc/adr/test_selection.py`

**Interfaces:**
- Consumes: existing `selection_cost`, `_cos_dist`, `PairChoice` (fields `i, j, cost, n_admissible`) from `selection.py`.
- Produces:
  - `config.TOP_K_PER_VIDEO: int = 5`, `config.MIN_PAIR_SEP_S: float = 1.0`, `config.ROMA_POSE_GATE_THRESH: float` (provisional, e.g. 0.5).
  - `selection._pairs_too_close(ai, aj, bi, bj, min_sep: int) -> bool` — True iff `abs(ai-bi) < min_sep and abs(aj-bj) < min_sep`.
  - `selection.choose_pairs(pre_idx, post_idx, embeddings, areas, aspects, *, top_k: int, min_pair_sep: int, pose_gate_fn: "Callable[[int,int],bool] | None" = None, a_min=A_MIN, lambda_area=LAMBDA_AREA, lambda_aspect=LAMBDA_ASPECT, lambda_size=LAMBDA_SIZE, blur_pen=None, edge_pen=None, lambda_blur=0.0, lambda_border=0.0, lambda_appear=1.0, max_appearance_dist=None) -> list[PairChoice]` — admissible straddling pairs (area ≥ a_min, within `max_appearance_dist`) sorted by `selection_cost` ascending, greedily kept so no kept pair is `_pairs_too_close` to an earlier kept pair, each passing `pose_gate_fn(i,j)` when given, truncated to `top_k`. `n_admissible` on each is the total admissible count (same value as `choose_pair`). Empty list when none qualify.

- [ ] **Step 1: Write failing tests** in `test_selection.py`:
  - `test_choose_pairs_returns_topk_sorted` — with ≥K admissible pairs of distinct cost, returns exactly `top_k`, costs non-decreasing.
  - `test_choose_pairs_diversity_collapses_near_duplicates` — candidates (4,14) and (5,15) with `min_pair_sep=... ` for 1s both survive (1s apart on both ends ⇒ not `<min_sep`); (4,14) and (4,14±(min_sep-1)) collapse to one.
  - `test_choose_pairs_pose_gate_filters` — a `pose_gate_fn` rejecting a specific `(i,j)` drops exactly that pair.
  - `test_choose_pairs_empty_when_none_admissible` — all areas `< a_min` ⇒ `[]`.
- [ ] **Step 2: Run tests, verify they fail** — `pytest tests/lr_vcc/adr/test_selection.py -k choose_pairs -v` → FAIL (choose_pairs undefined).
- [ ] **Step 3: Add the three config constants** to `config.py` (PROVISIONAL section) and implement `_pairs_too_close` + `choose_pairs` in `selection.py`. Reuse the admissibility/cost loop from `choose_pair`; collect all admissible `(cost, i, j, n_admissible)`, sort by cost, greedily filter by `_pairs_too_close` and `pose_gate_fn`.
- [ ] **Step 4: Run tests, verify pass** — same command → PASS.
- [ ] **Step 5: Commit** — `git add scripts/lr_vcc/adr/selection.py scripts/lr_vcc/adr/config.py tests/lr_vcc/adr/test_selection.py && git commit -m "feat(adr): choose_pairs top-K diverse selection with pose-gate hook"`.

---

### Task 2: Video-level multi-pair collection (`select_all_pairs`) + wiring

**Files:**
- Modify: `scripts/lr_vcc/adr/weak_anchor.py`
- Modify: `scripts/lr_vcc/adr/run_adr.py` (`_prepare_clip`, `evaluate_clip_pairs`)
- Test: `tests/lr_vcc/adr/test_weak_anchor.py`, `tests/lr_vcc/adr/test_run_adr.py`

**Interfaces:**
- Consumes: `FallbackResult`, `anchor_reliability` (existing in `weak_anchor.py`); `choose_pairs`, `_pairs_too_close` (Task 1).
- Produces:
  - `weak_anchor.select_all_pairs(gaps_desc, choose_pairs_fn: "Callable[[gap], list[PairChoice]]", c_gt_fn: "Callable[[gap, PairChoice], float]", *, floor: float, top_k: int, min_pair_sep: int) -> list[FallbackResult]` — iterate gaps_desc, for each the list from `choose_pairs_fn`, skip a pair that is `_pairs_too_close` to any already-kept pair (video-level dedup across gaps), set `status/reliability` from `c_gt` vs `floor` exactly as `select_all_gaps` does, stop at `top_k` total. Returns kept list (≤ top_k).
  - `_prepare_clip` ctx gains `choose_pairs_fn` (wraps `choose_pairs` with `top_k=TOP_K_PER_VIDEO`, `min_pair_sep=round(MIN_PAIR_SEP_S*fps)`, `pose_gate_fn` from ctx) alongside the existing `choose_fn`; and `pose_gate_fn` built from an injected `pose_gate` over `gt_crops` (None when not injected).
  - `evaluate_clip_pairs(..., pose_gate=None, top_k=TOP_K_PER_VIDEO)` uses `select_all_pairs` instead of `select_all_gaps`; the long-range Option-B pair is still appended once.

- [ ] **Step 1: Write failing tests:**
  - `test_select_all_pairs_caps_at_top_k` (weak_anchor) — fake `choose_pairs_fn` yielding many pairs across 2 gaps ⇒ result length == `top_k`.
  - `test_select_all_pairs_cross_gap_dedup` — two gaps returning a pair within `<min_pair_sep` on both endpoints ⇒ only one kept.
  - `test_evaluate_clip_pairs_multi_per_gap` (run_adr) — fake embedder + fake scorers + a gap wide enough for ≥2 diverse pairs ⇒ `len(result["pairs"]) >= 2`.
  - `test_evaluate_clip_pairs_pose_gate_rejects_all_is_graceful` — a `pose_gate` rejecting every pair ⇒ result is well-formed with `pairs == []` (or only the long-range Option-B pair, which stays on the CLIP gate), no exception raised. *(Review Focus: pose-gate rejects everything — reviewability first, threshold is the lever.)*
- [ ] **Step 2: Run tests, verify fail.**
- [ ] **Step 3: Implement** `select_all_pairs` in `weak_anchor.py`; add `choose_pairs_fn`/`pose_gate_fn` to `_prepare_clip` ctx and switch `evaluate_clip_pairs` to `select_all_pairs`. Keep `select_all_gaps`/`evaluate_clip` untouched (single-pair path still used by `evaluate_clip`).
- [ ] **Step 4: Run tests + full adr suite** — `pytest tests/lr_vcc/adr -q` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(adr): select_all_pairs video-level top-K with RoMa pose-gate injection"`.

---

### Task 3: GT-based per-axis reliability in the composite

**Files:**
- Modify: `scripts/lr_vcc/adr/run_adr.py` (`_score_pair`)
- Modify: `scripts/lr_vcc/adr/scorers.py` (protocol doc + optional `reliability`)
- Create: helper `scripts/lr_vcc/adr/reliability_axes.py` (small)
- Test: `tests/lr_vcc/adr/test_run_adr.py`, `tests/lr_vcc/adr/test_adapters_contract.py`

**Interfaces:**
- Consumes: `compose_score(scores, reliabilities, ...)` (`scripts/lr_vcc/composite.py`); `consistency_scores(c_gt, c_sr)`; scorers with `.name` + `__call__`.
- Produces:
  - Optional protocol method `reliability(self, a, b) -> float` in `[0,1]` (GT-side signal). Scorers without it default to `1.0`.
  - `reliability_axes.axis_reliabilities(scorers, gci, gcj) -> dict[str, float]` — `{M.name: M.reliability(gci,gcj) if hasattr(M,"reliability") else 1.0}`.
  - `_score_pair` change: `axis_rel = axis_reliabilities(scorers, gci, gcj)`; `per_axis = [reliability * axis_rel[name] for name in s_M]`; `composite = compose_score(list(s_M.values()), per_axis)`; record gains `"axis_reliability": axis_rel`.

- [ ] **Step 1: Write failing tests:**
  - `test_per_axis_reliability_zero_signal_axis_gets_low_weight` — three fake scorers; one has `reliability→0` on the GT pair; assert its `composite["weights"]` entry ≪ the others (no free pass). *(Review Focus.)*
  - `test_reliability_is_gt_sourced` — a fake scorer whose `reliability` returns 1.0 for the GT crop pair but would return 0.0 for the SR crop pair; assert the recorded `axis_reliability` is 1.0 (proves GT crops were passed, not SR). *(Review Focus.)*
  - `test_scorer_without_reliability_defaults_one` (contract) — a scorer lacking `reliability` ⇒ `axis_reliabilities` returns 1.0 for it.
- [ ] **Step 2: Run tests, verify fail.**
- [ ] **Step 3: Implement** `reliability_axes.py`; wire into `_score_pair` (use `gci, gcj` — the GT crops already computed there); document the optional protocol method in `scorers.py`.
- [ ] **Step 4: Run tests + full adr suite** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "fix(adr): GT-based per-axis reliability in composite (kills no-signal free-pass)"`.

---

### Task 4: Cross-pair / per-video aggregation

**Files:**
- Modify: `scripts/lr_vcc/adr/export_pairs.py`
- Test: `tests/lr_vcc/adr/test_export_pairs.py`

**Interfaces:**
- Consumes: pair records (dicts with `consistency`, `clip`/`video`, `c_gt`, `c_sr`, `s_M`).
- Produces: `export_pairs.video_summary(pairs: list[dict]) -> dict` → `{"n_pairs", "min_consistency", "worst_pair": (i,j), "n_failures", "mean_consistency"}` where `n_failures = sum(p["consistency"] < 0.90)` and `worst_pair` is the argmin-consistency record's `(i,j)`. Used so headlines report the worst pair, never only a mean.

- [ ] **Step 1: Write failing test** `test_video_summary_surfaces_worst_pair` — 5 records, one at consistency 0.30, rest ≥0.95 ⇒ `min_consistency==0.30`, `n_failures==1`, `worst_pair` is the 0.30 record's `(i,j)`; `mean_consistency` > 0.30 (proving the mean alone would hide it). *(Review Focus: mean dilution.)*
- [ ] **Step 2: Run test, verify fail.**
- [ ] **Step 3: Implement** `video_summary`.
- [ ] **Step 4: Run test + full adr suite** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(adr): per-video summary surfaces worst/min-consistency pair"`.

---

### Task 5: RoMa v2 geometry scorer + pose gate (server env `lrvcc`)

**Files:**
- Modify: `scripts/lr_vcc/adr/scorers_gpu.py`
- (server) create env `lrvcc`, download RoMa weights via hf-mirror.
- Test: `tests/lr_vcc/adr/test_adapters_contract.py` (contract, local, mocked); server smoke (manual verification).

**Interfaces:**
- Produces:
  - `scorers_gpu.RomaGeometryScorer` — `name = "roma"`; `__call__(a, b) -> float` ∈ [0,1] (normalized confident-correspondence quality); `reliability(a, b) -> float` ∈ [0,1] (GT-side confident-match count / cap, clamped). Lazy torch/roma import.
  - `scorers_gpu.RomaPoseGate` — callable `__call__(a, b) -> float` ∈ [0,1] pose-match score; used by selection as `pose_gate` (gate admits when `≥ ROMA_POSE_GATE_THRESH`). May share the loaded RoMa model with the scorer.

- [ ] **Step 1: Contract test (local, no weights)** `test_roma_scorer_conforms` — monkeypatch/mocked matcher; assert `name=="roma"`, `__call__` returns float in [0,1], `reliability` present and in [0,1].
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `RomaGeometryScorer` + `RomaPoseGate` (lazy imports; CUDA when available). Add the install + weights steps to a server setup note; apply fallback chain if weights unreachable (RoMa v1, else keep `LoftrGeometryScorer` + CLIP gate — flag it).
- [ ] **Step 4a: Contract test** → PASS.
- [ ] **Step 4b: Server smoke** — on `lrvcc`, score gecko-12 GT pair vs SR pair: `c_gt` high, `c_sr` lower on the known failure, `reliability` > 0; record the numbers. Expected: in-range, failure visible.
- [ ] **Step 5: Commit** — `git commit -m "feat(adr): RoMa v2 geometry scorer + pose gate"`.

---

### Task 6: DINOv3 semantic scorer (server)

**Files:**
- Modify: `scripts/lr_vcc/adr/scorers_gpu.py`
- (server) download DINOv3 weights via hf-mirror (fallback DINOv2).
- Test: `tests/lr_vcc/adr/test_adapters_contract.py` (contract, mocked); server smoke.

**Interfaces:**
- Produces: `scorers_gpu.Dinov3SemanticScorer` — `name = "dino"` (keeps the anchor-metric name so `anchor_metric="dino"` and the c_gt floor keep working); `__call__(a, b) -> float` = cosine of DINOv3 features clamped [0,1]; no `reliability` (defaults to 1.0, matching semantic ≈ full signal).

- [ ] **Step 1: Contract test** `test_dinov3_scorer_conforms` (mocked backbone) — `name=="dino"`, `__call__` ∈ [0,1].
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `Dinov3SemanticScorer` (lazy transformers import; hf-mirror model id; DINOv2 fallback flagged).
- [ ] **Step 4a: Contract test** → PASS.
- [ ] **Step 4b: Server smoke** — gecko-12 GT/SR: `c_gt`/`c_sr` in range, cosine sane.
- [ ] **Step 5: Commit** — `git commit -m "feat(adr): DINOv3 semantic scorer (DINOv2 fallback)"`.

---

### Task 7: Full run on the 18 starter (server)

**Files:**
- Modify/create: a run entry wiring `evaluate_clip_pairs` with scorers `[SiftTextureScorer, RomaGeometryScorer, Dinov3SemanticScorer]`, `embedder=ClipEmbedder`, `pose_gate=RomaPoseGate` (the `_build_scorers` glue in `run_adr.py` / the CLI).
- Output: `~/data/LRVCC/out/starter_richer.json` + metrics CSV + headline.

**Interfaces:**
- Consumes: Tasks 1–6. CLI: `python -m scripts.lr_vcc.adr.run_adr --manifest <starter> --gt-dir ~/data/LRVCC/src --sr-dir ~/data/LRVCC/pred/flashvsr --ann-dir <ann> --out .../starter_richer.json --per-gap`.

- [ ] **Step 1: Locate the server annotation dir** (not `~/data/LRVCC/ann`; check `GT/`, LaSOT source) and confirm the 18-clip manifest.
- [ ] **Step 2: Wire `_build_scorers`** to the new trio + inject `pose_gate`; launch the run detached (lab-server-gpu-jobs skill: `setsid`, verify separately).
- [ ] **Step 3: Verify output** — `pairs` present for all 18 clips; **median pairs/video ≈ 5**; each pair has 3 axes + `axis_reliability`; failures (`consistency < 0.90`) surfaced; no clip crashed. Record pairs/video distribution and failure count.
- [ ] **Step 4: Pull** `starter_richer.json` + metrics locally (scp retry loop) into `data/review/` (gitignored).
- [ ] **Step 5: Commit** the run wiring (code only) — `git commit -m "feat(adr): wire upgraded trio + RoMa pose-gate run"`.

---

### Task 8: Rebuild the review pack (gap-sorted) + per-video summary

**Files:**
- Modify: `~/.lrvcc_helpers/build_package.py`, `build_metrics.py`, `build_pdf.py` (outside repo) to read `starter_richer.json`, sort/highlight by `c_gt − c_sr` gap, and emit the per-video `video_summary`.
- Output: `data/review/cross_gap_examples/` refreshed (gitignored).

**Interfaces:**
- Consumes: Task 4 `video_summary`; Task 7 output.

- [ ] **Step 1:** Point the build helpers at `starter_richer.json`; add RoMa/DINOv3 columns (`roma_*`, keep `dino_*`), sort pairs by the gap, include `video_summary` (worst pair per video) in the headline.
- [ ] **Step 2: Build** the pack (CSV/JSON/PDF/panels); verify counts (~5/video), gap-sorted order, failures on top, summary present.
- [ ] **Step 3:** Update `data/review/cross_gap_examples/README.md` + `COVER_NOTE.md` to name the upgraded trio (SIFT + RoMa v2 + DINOv3), the RoMa pose-gate selection, and the per-axis GT-based reliability; drop the LoFTR/DINO-v1 caveats that no longer apply.
- [ ] **Step 4: Verify** the pack opens and is reviewable (manual).
- [ ] **(No code commit — build helpers live outside the repo and `data/review/` is gitignored.)**

---

## Notes for the executor

- Tasks 1–4 run fully on the M1 with `pytest`; no GPU/weights needed (fakes injected). Do these first.
- Tasks 5–8 need the server (`lrvcc` env, weights, GPU1). Use the `lab-server-gpu-jobs` skill: launch detached, verify with separate read-only SSH, back off on sshd throttle.
- After Task 7, if pairs/video falls short of ~5, the first levers are `TOP_K_PER_VIDEO`, `MAX_WINDOW_OFFSET` (more offsets per gap), and `ROMA_POSE_GATE_THRESH` (gate strictness) — all in `config.py`, tuned on GT behaviour only, never on SR scores.
