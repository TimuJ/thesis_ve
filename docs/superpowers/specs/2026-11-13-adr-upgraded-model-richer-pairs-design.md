# Design — Upgraded-model pair selection + richer cross-gap pair set

**Status:** draft for approval (all decisions settled) · **Date:** 2026-11-13 ·
**Builds on:** [`2026-11-02-lrvcc-cross-gap-matched-pair-consistency-design.md`](2026-11-02-lrvcc-cross-gap-matched-pair-consistency-design.md)

## Intent and why

The cross-gap matched-pair benchmark (Ring-Forcing A-D-R adapted to real LaSOT
footage with GT) currently yields **too few reviewable pairs**: 42 pairs across 18
videos, with **7 videos at a single pair** and **8 with zero FlashVSR failures**.
For the benchmark to be convincing — and to be shown — we need a *richer* set:
**~5 reviewable pairs per video**, from which we **surface the genuine FlashVSR
failures** (cases where the SR fails to preserve the object's cross-gap relationship).

The lever is selection quality. The selection-signal probe
(`docs/notes/2026-11-06-adr-selection-signal-probe-findings.md`) found the dominant
reason a candidate pair is *rejected* is **pose** — the two frames show the object at
different angles, so the pair is not consistency-checkable. The current pose gate is a
**CLIP-embedding** cosine distance (`ClipEmbedder`, the deliberately-*independent*
selector from the predecessor design — see §Circularity), which is coarse. The
approved upgrade — **RoMa v2** (dense geometric matcher) for the geometry axis **and**
as the pose-aware selection gate, plus **DINOv3** for the semantic axis — is therefore
not just a scoring improvement: it is the enabler that lets us accept many more
genuinely-reviewable pairs per video.

**Decision (user):** pairs must be **selected by the upgraded model**. We stand up
RoMa v2 + DINOv3 first, then produce the richer pair set with them. There is no
"richer selection now with the current gate" interim.

## Goals

1. **~5 reviewable pairs per video** on the 18 starter videos, pose-vetted by RoMa v2.
2. **Surface the genuine FlashVSR failures that exist among the reviewable pairs**
   (`c_sr ≪ c_gt`), reported **per-pair** so a single collapsed pair is visible, never
   averaged away. Some videos may legitimately show no failure — we report the SR's
   actual behaviour, we do not manufacture one.
3. **Selection driven by the upgraded model** — RoMa v2 pose-match is the
   *reviewability admissibility gate*; the existing independent **CLIP** selector keeps
   *ranking* among admissible pairs (decision (a), see §Circularity).
4. **Scoring on the upgraded trio** — SIFT (texture, kept) + RoMa v2 (geometry) +
   DINOv3 (semantic) — reported **per-axis**.
5. **Per-axis reliability fix** so a no-signal axis no longer free-passes the composite
   (weight each axis by its *own* signal, **computed on the GT side** — see Constraints).

## Non-goals

- **Band recalibration.** The 0.90 preserve/diverge threshold stays where it is.
  Re-fitting it to our own visual labels is circular/biased (explicit user call).
- **Relying on a single composite number.** The composite is *fixed* (per-axis
  reliability) but reporting stays per-axis; we do not lean on one number.
- **88-video expansion in this phase.** Deferred — generating FlashVSR on new clips is
  slow. Starter first.

## Anti-goals (failure even if it technically works)

- **Forcing or cherry-picking failures** to inflate the failure count. A clean
  preserve is a real result; we report the SR's actual behaviour.
- **Un-reviewable pairs leaking in** (wrong pose, motion blur) because the gate was
  loosened to hit the count. Count is subordinate to reviewability.
- **Overfitting any threshold to our own labels.**

## How failures are surfaced (mechanics — resolves the "frozen on GT vs. weighted toward failures" tension)

Selection is **GT-frozen**: it picks pairs that are *reviewable* (GT object holds pose
across the gap), never reading SR. It does **not** steer toward failures — that would
require reading SR and risks cherry-picking. Instead:

1. Selection emits ~5 **reviewable** pairs/video (GT-only), spread across the gap/offsets
   for diversity (so we sample easy *and* harder reconstructions, not only the single
   best-matched pair).
2. Scoring then measures `c_sr` vs `c_gt` per pair; a **failure is identified**
   wherever `c_sr` collapses below `c_gt`.
3. The review pack and tables **sort / highlight by the `c_gt − c_sr` gap**, so the
   failures are front-and-centre without changing what was selected.

This means Goal 2 is "surface the failures that are there", **not** a guarantee that
every video yields one. Diversity in selection (not a failure-targeting objective) is
what raises the chance of catching failures where they exist.

## Constraints

- **Selection frozen on GT.** Reviewability is judged on the GT object; the chosen pair
  is reused for every model, so GT and SR are scored at the same frames/boxes. A
  *failure* is a GT-reviewable pair where SR's `c_sr` collapses below `c_gt`.
- **Per-axis reliability is GT-based.** The per-axis signal (SIFT ∝ GT keypoints,
  RoMa ∝ GT confident matches, DINOv3 ≈ full) is computed on the **GT** crop pair, i.e.
  "does this pair afford signal on this axis?" — **never** on the SR side. Keying it on
  SR would let an over-smoothed SR (few keypoints/matches) register as "no signal" and
  down-weight its *own* over-consistency failure, hiding exactly the case we want.
- **Server (facts from recon 2026-11-13):** data under `~/data/LRVCC/` — GT `src/`
  (18), FlashVSR SR `pred/flashvsr/` (18), outputs/manifests `out/`. Torch-2.6 envs
  available (`flashvsr`, `seedvr`, `seedvr310`; seedvr310 has transformers 5.14.1).
  Network: pypi + hf-mirror reachable, huggingface.co blocked (weights via mirror).
  GPU1 ~31 GB free. ADR pipeline present.
- **Server connection stays out of git.** Reports frame everything as research
  outcomes, no exact calendar dates in prose.

## Circularity (the one methodological decision — see §Open)

The predecessor design's central anti-goal was **not selecting with the metric we
score with**: it used **CLIP** as an *independent* selector precisely so the SIFT /
geometry / DINO scorers stay honest. Making RoMa v2 **both** the selection gate **and**
the scored geometry axis reintroduces that coupling — pairs admitted because RoMa says
the GT geometry matches, then scored on geometry by RoMa.

Key reframe: the coupling is only *harmful* if selection steers toward making the
**score** (`c_sr`/`s_M`) look good. Gating on **GT-signal availability** (does RoMa
find GT matches across the gap?) does **not** inflate `s_M` — it ensures the geometry
axis is *measurable* (c_gt out of the noise), which is the same thing the per-axis
reliability fix wants. The residual bias is on *coverage* (object types RoMa matches
well), which is a stated caveat, not a score inflation. Ranking pairs by RoMa (not just
gating) is where the geometry c_gt gets systematically maximised — that is the stronger
circularity.

**Resolved — decision (a):** RoMa is the pose *admissibility gate* only; the
independent **CLIP** selector keeps **ranking** admissible pairs. So the geometry axis
is gated for GT-signal (benign) but never ranked-to-maximise (the harmful case is
avoided), and the predecessor's independent-selector property is preserved.

## Current state (what exists)

- `scripts/lr_vcc/adr/` pipeline: `selection.py` (`choose_pair` = argmin one pair per
  gap; appearance cost + `max_appearance_dist` gate on **CLIP** embeddings via
  `embed_clip.ClipEmbedder`), `run_adr.py` (per-gap orchestration, `_score_pair`,
  `compose_score`; weak-anchor c_gt floor keys on `anchor_metric="dino"`),
  `scorers_gpu.py` (PairScorer protocol: `name` + `__call__(a,b)->float∈[0,1]`;
  `DinoSemanticScorer`, `LoftrGeometryScorer`), `scorers.py` (`SiftTextureScorer`),
  `weak_anchor.py` (`select_all_gaps`, reliability), `config.py` (all tunables).
- The composite free-pass bug (verified): `run_adr.py` feeds the *same* per-pair
  reliability to every axis, so a no-signal axis votes full weight.
- 120 LaSOT videos have a ≥3s gap (`data/lasot_work/full_lasot_scan.json`); several
  have multiple gaps — the Phase-3 expansion pool.

## Plan

### Phase 1 — Stand up the upgraded models (the enabler)

1. **Env.** Dedicated torch-2.6 conda env `lrvcc` on the server (clone of a 2.6 env),
   to keep RoMa/DINOv3 deps away from the FlashVSR env. *(Decided.)*
2. **RoMa v2.** Install the matcher; download weights via hf-mirror. Wire two uses of
   the one model: (a) a **PairScorer** `RomaGeometryScorer` (geometry axis; score =
   normalized confident-correspondence quality in [0,1]), and (b) a **selection
   pose-gate** exposing a GT pose-match score between two crops. **Fallback chain if
   weights are unreachable:** RoMa v2 → RoMa v1 → keep `LoftrGeometryScorer` as the
   geometry axis and the existing CLIP gate for selection (flag the degradation).
3. **DINOv3.** Install; download weights via hf-mirror. Wire `Dinov3SemanticScorer`
   (PairScorer; cosine of DINOv3 features, clamped [0,1]) to replace
   `DinoSemanticScorer`. **Fallback:** DINOv2 (flag it) if the gated DINOv3 weights are
   not served by the mirror. *(Decided.)*
4. **Smoke test** on one clip (e.g. gecko-12): models load, produce sane `c_gt`/`c_sr`
   on a known failure, numbers in-range.

### Phase 2 — Richer selection + scoring on the 18 starter

1. **`choose_pair` → `choose_pairs` (top-K).** Emit up to K reviewable pairs per gap
   (and across gaps), targeting ~5/video, instead of a single argmin — by sliding the
   pre/post frames within the existing adjacent-run windows.
2. **≥1s diversity rule.** A candidate pair is dropped only if an already-kept pair in
   the same video sits within **<1s on both endpoints** (pre *and* post). So (4s,14s)
   and (5s,15s) both survive; (4.0s,14.0s) and (4.2s,14.1s) collapse to one.
   *(My formalization of "1s different"; builder's call to tune.)*
3. **Selection gate** = RoMa pose-match as a **binary GT admissibility check** (is the
   pair reviewable?); the existing **CLIP** appearance cost keeps **ranking** among
   admissible pairs (decision (a)). Frozen on GT. Keep the existing weak-anchor c_gt
   floor; the RoMa gate is an *additional* admissibility check, not a replacement.
4. **Per-axis reliability fix.** In `_score_pair`/`compose_score`, weight each axis by
   its own **GT-side** signal (see Constraints), not a shared per-pair reliability.
   **No band recalibration** — 0.90 stays.
5. **Cross-pair aggregation.** Report **per-pair** (the pack is per-pair panels). For
   any per-video / headline summary, report a central value **and the worst (min-
   consistency) pair / failure count** — never a 5-pair mean that hides one collapse.
6. **Run** selection + scoring on the 18 starter (GT `src/`, SR `pred/flashvsr/`),
   surface failures (`c_sr ≪ c_gt`), rebuild the review pack (the
   `~/.lrvcc_helpers/build_*` + `data/review/cross_gap_examples/` pipeline) with the
   richer set, sorted/highlighted by the gap. **Human review stays in the loop** — the
   RoMa gate is a pre-filter that raises the hit-rate, not a replacement for the
   reviewer's eyeball on the new set.

### Phase 3 — 88-video expansion (deferred)

Generate FlashVSR SR on the next tranche of the 120-candidate pool (server GPU, slow),
then run the same selection + scoring to grow coverage. Prioritized after the starter
set is solid.

## Decisions already made (record)

| Decision | Source |
| --- | --- |
| Pairs selected by the upgraded model (RoMa v2 pose-match) | user |
| RoMa's role in selection = **(a)** gate reviewability only; CLIP keeps ranking | user |
| ~5 pairs/video; multiple per gap allowed; ≥1s diversity | user |
| Reviewability judged on GT; failure = GT-reviewable pair where `c_sr` collapses below `c_gt` | user |
| Failures surfaced by scoring + gap-sort, not by steering selection; no forced failures | user + builder-check |
| Starter-first; 88-expansion later (generation slow) | user |
| RoMa v2 = geometry axis + selection gate; DINOv3 = semantic axis; SIFT kept = texture | user-approved upgrade |
| Per-axis reliability fix, computed on the **GT** side | user + builder-check |
| Cross-pair: per-pair + worst-case; no diluting mean | builder-check |
| No band recalibration (biased) | user |
| Report per-axis (composite fixed but not leaned on) | user |
| Dedicated `lrvcc` conda env; DINOv3→DINOv2 fallback; RoMa v2→v1→LoFTR fallback | builder default |
| Human review stays in the loop on the richer set | builder default |

## Left to the builder

- Exact RoMa v2 repo/package + weights URL (via hf-mirror); the pose-match → [0,1]
  score definition and the selection-gate threshold.
- DINOv3 model id + transforms; the cosine → [0,1] mapping.
- Top-K value and how pre/post offsets are spread within a gap window.
- RoMa/DINOv3 GT-side confidence → per-axis reliability mapping.
- Annotation path on the server (not in `~/data/LRVCC/ann`; locate during build).
- PairScorer protocol extension to expose the GT-side per-axis reliability signal.

## Decision — RoMa's role in selection (resolved: (a))

**Chosen: (a).** RoMa is the pose *admissibility gate* only; the independent **CLIP**
selector keeps ranking among admissible pairs — preserves the independent selector,
residual coupling is benign (ensures the geometry axis has GT signal). The rejected
alternatives, for the record:

- ~~(b) RoMa pose-match as the whole selection cost (gate + rank)~~ — matches "one
  model, both jobs" most literally, but the geometry axis would run in a narrow
  high-match regime (circularity caveat).
- ~~(c) RoMa selects, geometry scored by LoFTR (selector ≠ scorer)~~ — zero geometry
  circularity, but RoMa would not be the scored geometry upgrade (departs from the
  Ring-Forcing trio).
