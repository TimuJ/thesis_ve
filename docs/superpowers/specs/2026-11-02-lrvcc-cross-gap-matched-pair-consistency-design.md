# LR-VCC cross-gap matched-pair consistency — design (A-D-R adaptation)

**Status:** approved — design complete; builder-check calls + weak-anchor policy folded in; next step writing-plans
**Date:** 2026-11-02
**Predecessors:** `2026-09-14-reduced-reference-reframe-design.md` (the RR principle, on the pre-pivot YouHQ bases); `docs/notes/2026-10-20-full-lasot-disappear-reappear.md` (the LaSOT disappear-reappear pool, acceptance criteria, scale-variation ranking); project memory *FlashVSR context window* (gap must exceed ~3 s)
**Decision owners:** Timur (research owner); collaborating PhD colleague

## Why this exists

The shipped LR-VCC measures within-clip self-similarity, which a blur maximises
(the reductio in the predecessor). The RR reframe fixed that by scoring against a
reference of *what should stay fixed* — on the YouHQ bases that reference had to
be the **LQ input**, because those bases had **no ground truth**.

The dataset has since pivoted. We now build the benchmark on **LaSOT
disappear-reappear clips** and synthesize the LQ ourselves (DOVE second-order
degradation) from the real footage. So the real footage **is** the ground truth,
at every frame. That is a strict strengthening: we no longer need the LQ as a
correspondence proxy — we have the true frame on both sides of every gap.

What we still lack is **where to measure**. Ring Forcing (ECCV 2026, *Towards
Precise Long-Term Memory for Autoregressive Video Diffusion*) poses exactly our
question — "when an object leaves the frame, does the same object reappear?" — and
grades it with their **A-D-R benchmark** (Appear / Disappear / Reappear). But they
*generate* the video and have no GT, so (a) they compare the reappear view only to
the appear view (self-consistency), and (b) they pick frames by "largest subject
mask area" per segment. We are restoring real footage with GT, so we can do the
thing they can't: measure whether the SR model **preserves the same cross-gap
relationship the real footage has**. To do that fairly and automatically we need
an automatic, principled way to choose *which* pre-gap/post-gap frame pair to
measure on. This document specifies that selection and the GT-anchored scoring
built on top of it.

## What we adopt from Ring Forcing, and where we diverge

Their A-D-R recipe (verified from the paper):

- 64 generated cases; appear/reappear segments fixed at 5 s; gap ∈ {0,1,5,15,30,60} s.
- A detector runs on the gap to **drop cases where the subject didn't vanish**.
- In each segment: SAM-3 segments the subject by keyword, **take the keyframe
  with the largest subject mask area**, crop, paste on white.
- Score the appear-crop vs reappear-crop with **SIFT** (texture), **LoFTR**
  (geometry), **DINOv3** (semantic). Aesthetic / X-CLIP / optical-flow dynamics
  are segment-level extras, not on the pair. PSNR/SSIM/LPIPS appear only in their
  ablations.

We adopt: the **disappear-reappear framing**, the **object-crop comparison**, and
the **SIFT / LoFTR / DINOv3 scoring trio** (so our consistency numbers sit in the
same conceptual table as theirs). We diverge on three points, each because we
have GT and they don't:

1. **Frame choice.** Not "largest area each side" (their two clearest views can be
   at different scales, so their metric partly measures *genuine* object change).
   Instead, the **most-comparable cross-gap pair** — see §Selection.
2. **Reference.** Not SR-vs-SR self-consistency. The honest target in real footage
   is **not "identical"** — the object genuinely changed scale/pose/light — so we
   score the SR pair's relationship **against the GT pair's own relationship**.
3. **Correspondence source.** Not SAM-by-keyword — LaSOT already ships a per-frame
   tracked box, so the object identity across the gap is given, not inferred.

## Two separated concerns (not a contradiction)

- **Clip-level difficulty — the stressor** (already designed, out of scope here):
  gap > SR context window (FlashVSR ~3 s; pool floor 3 s, headline 5 s),
  big→small→big scale swing, and degraded/differential LQ so the low-res frame
  *cannot* hand the model the answer. This governs which clips enter the pool and
  how we degrade them.
- **Pair selection — the measurement anchor** (this document): inside a given
  clip, pick the *fairest, most comparable* pre/post pair. A clean anchor is not a
  soft test: the difficulty lives in the clip, the anchor just isolates "did the
  model keep the object stable" from "the object really moved."

## Alignment & pipeline invariants (build first — free, prevents silent corruption)

The scoring freezes indices `(i*, j*)` on GT and re-indexes into the LQ and every
SR output. If that index map is ever wrong, every `c_gt`, `c_sr`, and fidelity
number silently compares the wrong frames — exactly the predecessor design's
Workstream 0 failure, where fps re-tagging mis-split two of five bases. This
design inherits that guard, adapted to the LaSOT+GT setup:

- **Single canonical GT = the source mp4 decode.** The *same* decoded frames serve
  both as the degradation input (`LQ = degrade(GT)`) and as the scoring reference.
  The benchmark is self-referential by construction — SR's job is to invert our
  degradation back to this GT — so the mp4 is a valid, self-consistent GT and no
  separate "pristine" reference is required. (The original LaSOT JPEG frames are an
  available swap for a higher GT ceiling; it changes only the input files, not the
  design, and is unnecessary given crf-18 is near-transparent and frame-exact.)
- **Per-clip alignment guard + test.** Assert for every clip that
  `decoded_frames == len(groundtruth) == len(out_of_view) == len(full_occlusion)
  == manifest.frames`, and that the stream is **constant-rate**
  (`r_frame_rate == avg_frame_rate`). Verified clean on bus-20 (all 5601 @ 30/1).
  Any clip that fails is quarantined, never silently scored.
- **Frame-index alignment only.** LQ↔GT↔SR are paired by **frame index**, never by
  timestamp or clip index. The DOVE second-order degradation runs an mpeg
  round-trip (`random_mpeg`); it and every SR model must emit exactly one frame per
  GT frame, in order.
- **One continuous pass over the uncut clip.** LQ and SR are each produced by a
  single continuous pass over the full clip, gap included — never per-window or
  per-segment with state reset. The whole stressor is that the object leaves the SR
  model's ~3 s cache during the gap; a per-segment pipeline voids it. The chunked
  long-video driver still TBD in `make_lq.py` must preserve global continuity
  across chunk boundaries and across the gap.

## The pipeline

### C0 — present frames and gaps (from LaSOT annotations)

Inputs per sequence: `groundtruth.txt` (per-frame axis-aligned bbox),
`out_of_view.txt`, `full_occlusion.txt`. **We have boxes, not masks.**

- A frame is **present** if the object is neither out-of-view nor fully occluded
  and the box is valid.
- A **gap** is a maximal run of absent frames whose duration exceeds the pool
  threshold (≥3 s). A clip may have several.
- For a gap, the **pre-gap window** is the present run **directly adjacent** to the
  gap's start and the **post-gap window** the present run directly adjacent to its
  end, each **capped at a maximum offset from the gap edge** (builder sets the cap;
  default: the adjacent present run, no jumping an intervening gap). The chosen pair
  must straddle the gap (one from each window). The cap keeps the **measured span ≈
  the annotated gap** (so the reported gap length matches what was scored) and makes
  the per-clip search comparable regardless of how long the flanking present runs
  are. A clip whose object is large just before the gap and tiny just after
  therefore cannot roam to a far-away scale match — by design; that case is the
  weak-anchor policy (Open decisions), not a silently widened span.

### C1 — selection embedding (separate from the scorers) + configuration match

**Decision (user): the selection embedding is a *different* model from the scoring
trio**, so we never grade on the curve we drew (see Anti-goals). Concretely:

- **Appearance embedding `E`:** a **general object-appearance embedding** of the
  object crop — **CLIP ViT image features** (recommended). *Not* the server's
  existing identity embedder: that is **ArcFace faces**, and LaSOT objects are
  buses, animals, license plates — not faces. We reuse the embedding-**dump/cache
  infrastructure** (`dump_identity_embeddings.py` machinery), not the face model.
  DINOv2 is avoided as the selector because it shares lineage with the DINOv3
  *scorer* — too close to count as "separate."
- **Configuration match (user decision — "same configuration", not "same
  instance"):** the pair should be genuinely alike, not merely the same object at
  different scales. So the selection cost adds geometric terms on top of embedding
  distance:

  ```
  cost(a, b) = cos_dist(E(a), E(b))
             + λ_area   · |log(area_a / area_b)|
             + λ_aspect · |log(aspect_a / aspect_b)|
  ```

  where `a` ranges over the pre-gap window, `b` over the post-gap window,
  `area`/`aspect` come from the GT box. Scale/aspect enter as log-ratios so the
  penalty is symmetric and scale-free.

### C2 — cross-gap pair selection (frozen on GT, shared across models)

- **Chosen pair** `(i*, j*) = argmin cost(a, b)` over the straddling windows,
  subject to an **area floor** (`area_a, area_b ≥ A_min`) so we never match two
  tiny/occluded views where the embedding is unreliable.
- **Frozen on GT, shared.** The pair is chosen **once, from the clean GT** (GT
  crops → `E`; GT boxes → geometry), then the **same indices `(i*, j*)` are reused
  to score GT and every SR model**. No per-model re-selection — otherwise each
  model cherry-picks its own easiest pair and the numbers are not comparable.
- **Multi-gap default:** evaluate the **longest qualifying gap** as the headline;
  optionally emit every qualifying gap as a separate scored item (builder call).
- **Weak-anchor fallback (decided).** If the best admissible straddling pair for
  the chosen gap is poorly matched (below a `c_gt` floor) or nothing clears
  `A_min`, fall back to the **next-longest qualifying gap**; if no gap yields a
  well-matched pair, **keep the clip but down-weight its consistency reliability**
  — never silently drop the clip or widen the span. Reliability-weighting follows
  the predecessor's composition (sparse/weak anchors contribute less, not a
  spurious confident score).

### C3 — crop normalization

- **Start with the GT bbox crop** (cheap; we have it). Background inside the box
  contaminates the embedding and the scorers, worst when the object is small.
- **Upgrade: run SAM once on the chosen frames only** to get a Ring-Forcing-style
  mask-on-white crop that strips background. SAM runs on the handful of selected
  frames, not every frame — cheap. Which SAM and whether this is v1 or a
  fast-follow is a builder call.
- Resize each crop to a fixed short side for the scorers (builder: size).
- **SR must match GT resolution exactly.** The GT bbox indexes GT pixels and we
  crop SR with that same box, so every benchmarked model must output exactly GT
  resolution (×4 = 1280×720, no padding/letterbox/odd rounding). A model whose
  output grid differs has its crop taken by rescaling the GT box onto that grid
  (documented per model); a size mismatch is asserted and flagged, never silently
  cropped at the wrong pixels.

### C4 — scoring (reduced-reference against GT; plus standard fidelity)

For the chosen pair `(i*, j*)`:

**Cross-gap consistency — the new axis.** For each consistency metric
`M ∈ {SIFT-texture, LoFTR-geometry, DINOv3-semantic}` (each a similarity in
[0,1] on the two crops):

```
c_gt = M(GT_i*, GT_j*)     # the true cross-gap similarity the footage has
c_sr = M(SR_i*, SR_j*)     # the SR output's cross-gap similarity
s_M  = 1 − |c_sr − c_gt|   # reduced-reference consistency (higher = better)
```

`c_gt` is the **reference for how consistent the pair should be** — not 1, because
the real object changed. `s_M` therefore penalises **both** directions:

- `c_sr < c_gt`: the model **drifted** — lost texture/identity across the gap.
- `c_sr > c_gt`: the model **fabricated consistency** — e.g. over-smoothed both
  sides into a bland match, scoring *more* consistent than the real footage. Ring
  Forcing's SR-vs-SR metric rewards this; ours penalises it. This is the single
  biggest reason to anchor on GT.

Because selection is configuration-matched, `c_gt` is high and tight, so `s_M`
reads mostly as drift — with over-consistency still caught when it occurs.

**Per-frame fidelity — the usual axis, kept orthogonal.** Standard full-reference
`PSNR / SSIM / LPIPS(SR_k, GT_k)` for `k ∈ {i*, j*}`. Fidelity catches the failure
consistency cannot: a model that is internally consistent but *wrong* (the same
hallucinated texture on both sides) passes consistency and fails fidelity. Report
both; do not fold them into one number here. Compute fidelity **both full-frame**
(comparable to the VSR literature) **and on the object crop** — a model strong on
background but weak on the object must not hide behind a full-frame number.

## Validation

1. **Selection is stable and sane.** On the 18-clip starter, the chosen pair's GT
   crops are visibly the same object in a comparable configuration; `c_gt` is high
   (configuration match worked). Spot-check a handful by eye against the auto pick.
2. **The over-consistency penalty bites.** A deliberately over-smoothed SR (or the
   LQ upscaled) scores `c_sr > c_gt` and is **penalised** by `s_M`, not rewarded —
   the reductio that killed the self-anchored metric must stay dead under GT
   anchoring.
3. **Drift registers.** A controlled identity/texture corruption on the post-gap
   side lowers `s_M` monotonically with severity.
4. **Frozen-selection invariance.** Re-running selection on two different SR
   outputs yields the **same** `(i*, j*)` (because selection reads GT only).
5. **Separation holds.** Selection embedding (CLIP) and scorers (SIFT/LoFTR/
   DINOv3) are disjoint; swapping the selection backbone does not move the ranking
   of methods by more than noise (the pick is a chooser, not a scorer).

## Non-goals

- **Not re-deciding clip selection or degradation.** The pool, gap floor, scale
  ranking, and differential LQ are fixed upstream.
- **Not a generation benchmark.** We restore real footage; Ring Forcing is prior
  art to position against, not a pipeline to reproduce.
- **Not the adjacent-frame temporal axis.** This is cross-gap identity/structure;
  frame-to-frame smoothness is a separate measurement.
- **Not fitting the selection cost weights to flatter a method.** Weights and
  floors are set on GT behaviour (configuration match), never tuned on SR scores.

## Anti-goals (failure even if it runs)

- **Circular metric** — selecting with the metric we score with. Avoided: separate
  CLIP selector vs SIFT/LoFTR/DINOv3 scorers.
- **Per-model cherry-picking** — each model choosing its own easiest pair. Avoided:
  selection frozen on GT, indices shared.
- **Scale-mismatched anchor** — a pair so different that `c_gt` is low and the
  metric mostly reflects genuine change, not model error. Avoided: configuration
  match (area/aspect penalty) + area floor.
- **Face assumption** — reusing the ArcFace identity model on non-face objects.
  Avoided: general object embedding (CLIP), face model explicitly dropped.

## Constraints

- **Server & data.** Data lives under `/home/timur/data/LRVCC/{src,GT,LQ}` (per-user
  home is the only writable store; root fs ~52 GB free). GT kept as source mp4,
  frames materialised per clip on demand; LQ frames are small. **Run server tasks
  in the background** (user instruction).
- **Reuse.** Embedding dump/cache harness (not its face model); SAM on selected
  frames only; the existing crop/metric plumbing where it fits.
- **Framing.** Research-outcome framing in all project docs; no thesis mentions.

## Decided vs left to the builder

**Decided (by the user / as my call, marked):**
- Selection embedding is *separate* from the scorers *(user)*.
- "Closest" = *same configuration*, not scale-invariant same-instance *(user)*.
- GT-anchored reduced-reference scoring (`s_M = 1 − |c_sr − c_gt|`) *(my recommendation, user's earlier direction)*.
- CLIP as the selection backbone; ArcFace identity model dropped for non-faces; bbox crop first, SAM-on-white as upgrade *(my calls)*.
- Scoring trio = SIFT / LoFTR / DINOv3; fidelity kept as a separate axis *(adopted from Ring Forcing)*.
- Single canonical GT = the source mp4 decode, one source for both degradation and scoring, with a per-clip frame-index alignment guard + test *(my call; builder-check Q1, verified clean on bus-20)*.
- SR output must match GT resolution exactly (×4 = 1280×720); GT box rescaled onto any off-grid model, size mismatch asserted *(my call; Q2)*.
- Selection windows bounded to the present run adjacent to each gap edge, capped at a max offset, so the measured span ≈ the annotated gap *(my call; Q3)*.
- LQ and SR each produced by one continuous pass over the uncut clip *(my call / project premise; Q5)*.
- Fidelity reported both full-frame and on the object crop *(my call; Q6)*.
- Weak-anchor policy: fall back to the next-longest qualifying gap, else keep the clip at reduced consistency-reliability; never silent-drop or span-widen *(user-approved)*.

**Left to the builder:**
- `λ_area`, `λ_aspect`, and `A_min` (area floor) — set on GT configuration-match behaviour, not SR scores.
- Single best pair vs **top-k** matched pairs (robustness vs one measurement) and how the per-metric values aggregate to a 0–1 like their Table 1.
- Multi-gap policy: longest gap only vs one item per qualifying gap.
- Exact `M` implementations and how SIFT/LoFTR raw outputs map to [0,1]; crop resize size.
- Whether SAM-on-white is v1 or a fast-follow, and which SAM.
- Signed vs absolute deviation in `s_M`, and whether over-consistency gets a heavier penalty than equal-magnitude drift.

## Open decisions (surfaced for review)

1. **Headline unit:** one score per clip (longest gap) vs one per qualifying gap —
   sets the report/table shape. Owner: research owner.
2. **CLIP vs a dedicated instance/ReID embedding** for the selector on non-face
   objects — CLIP is the safe default; a ReID backbone could match configuration
   better but adds a dependency. Confirm before writing-plans.
3. **How we present alongside Ring Forcing:** do we also report the GT-free
   SR-vs-SR consistency (their exact number) as a second column so the comparison
   is apples-to-apples, in addition to the GT-anchored `s_M`? Confirm.
