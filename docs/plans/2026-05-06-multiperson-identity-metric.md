# Multi-Person Identity Metric for VBench-2.0 (Long-Video SR)

**Date:** 2026-05-06
**Status:** Design — pending user approval, then writing-plans
**Goal:** Extend VBench-2.0 `Human_Identity` to handle multi-person long videos for VSR evaluation. Headline use is the MGLD-vs-UAV synthetic comparison; broader use is thesis-grade evidence on whether identity metrics are effective for long-video SR.

## Context

- VBench-2.0 `Human_Identity` tracks one reference identity (largest face) → artificial low scores on crowd scenes (~0.20 on our 5 videos).
- Existing slow-fast adapter in `scripts/vbench2_long/human_identity_long.py` already wraps the upstream metric in 2-sec clips (slow) + clip-first-frames concat (fast), producing 0.555 vs 0.463 (MGLD vs UAV) under single-identity scoring.
- All face detection (RetinaFace) + face embedding (ArcFace) infra is already wired and runs per frame.

## Decisions

1. **Scope:** B (thesis-grade) — pragmatic core + ablation suite to test metric effectiveness for long-video SR evaluation.
2. **What to measure:** C (both) — self-consistency (no LQ used) + LQ-reference (compares LQ ↔ SR identities).
3. **Granularity:** A (per-clip) — multi-person handled within each 2-sec clip; no cross-clip re-identification (open as future work).
4. **Algorithm:** Approach 1 (cluster purity) as headline; Approach 3 (anchor-based) noted as fast fallback / sanity-check column.

## Pipeline

```
LQ video ─┐                                    ┌─ slow: per-clip score_fn averaged across clips
          ├─ split into 2-sec clips ───────────┤
SR video ─┘                                    └─ fast: score_fn on concat-of-clip-first-frames

For each clip, two scoring paths:
  ├─ self-consistency:  score_clip_self(SR_clip)            → cluster purity
  └─ LQ-reference:      score_clip_lqref(LQ_clip, SR_clip)  → matched-pair sim × recall

Per-video output:
   self : { slow, fast, fused }
   lqref: { slow, fast, fused, evaluable_clips_pct }
```

Existing `human_identity_long.py` keeps its single-identity behavior for back-compat; new functions add `--mode {single, multi-self, multi-lqref, all}`.

## Self-consistency variant (Approach 1 — headline)

Within one clip of the SR video:

1. Detect faces every frame with RetinaFace (already running). Skip frames with no face.
2. Embed each detected face with `extract_face_features` (ArcFace; concat of original + horizontally-flipped feature vectors as in upstream).
3. Collect all embeddings → matrix `E ∈ ℝ^(N×D)` where `N` = total face detections in the clip and `D` = embedding dim from `extract_face_features`.
4. Cluster with **agglomerative clustering**, average linkage, cosine distance, threshold τ (default `0.4`, matching existing `IDTracker.similarity_threshold`).
5. Discard clusters of size < 2 from scoring (counted as "noise rate" diagnostic).
6. Per-cluster score = mean cosine sim of cluster members to centroid.
7. **Clip score** = `Σ (cluster_size × cluster_score) / Σ cluster_sizes` (face-weighted average across clusters).

Edge cases:
- No faces in clip → score = `NaN`, excluded from video-level mean.
- All clusters size 1 → score = `NaN`. Diagnostic: "% all-singleton clips" reported per video.
- Single cluster → degenerates to single-identity behavior (clean narrative for thesis).

## LQ-reference variant

Within one clip with both LQ (320×180) and SR (1280×720) at the same time indices:

1. Run RetinaFace on **both** LQ and SR frames.
2. Per frame, match LQ↔SR faces **one-to-one** by spatial overlap: rescale SR bbox by 1/4 to LQ space, build the LQ×SR IoU matrix, solve assignment via Hungarian (or greedy if Hungarian's overhead matters), and accept assignments with `IoU > 0.5`.
3. For each matched pair, compute cosine similarity of their ArcFace embeddings.
4. **Clip score** = `mean(pair_sims) × (matched_pairs / detected_LQ_faces)`. The recall multiplier penalizes SR for "smoothing someone away."
5. **Evaluable clip** = clip with at least one accepted LQ↔SR pair. `evaluable_clips_pct` = (evaluable clips) / (total clips), per video.
6. Clusters are **not** used in this variant — time-alignment + spatial overlap give the ground-truth pairing for free.

Edge cases:
- No LQ faces detected (face too small) → clip excluded from LQ-ref score; tracked as `evaluable_clips_pct` diagnostic per video.
- Hallucinated SR faces (more SR detections than LQ): unmatched SR faces don't directly hurt LQ-ref (recall is over LQ), but they show up in self-consistency as low cluster purity. **Self + LQ-ref together pick this up** — argument for keeping both.
- Bbox scaling drift: IoU > 0.5 tolerates ±2 px shift, generous at these sizes.

## Slow-fast integration & output schema

The slow-fast scaffold doesn't change shape — only the scoring function plugged in changes:

```python
slow_branch(video, score_fn):
    clips = split_into_2sec_clips(video)
    scores = [score_fn(clip) for clip in clips]
    return nanmean(scores)

fast_branch(video, score_fn):
    fast_video = concat([clip.first_frame for clip in clips])
    return score_fn(fast_video)

fused = w_slow * slow + w_fast * fast   # default 0.5/0.5
```

For LQ-ref, both branches accept paired `(LQ_clip, SR_clip)` inputs; the fast branch concatenates clip-first-frames in lockstep on both sides so frame index aligns.

Per-video output (one method, one video):
```json
{
  "self":  {"slow": 0.74, "fast": 0.62, "fused": 0.68},
  "lqref": {"slow": 0.81, "fast": 0.55, "fused": 0.68,
            "evaluable_clips_pct": 0.78}
}
```

## Approach 3 (anchor-based fallback)

Documented but not the primary path. Useful as a fast sanity-check column:

- First detected frame of clip → cluster its faces with the same agglomerative clustering at the same τ as Approach 1; the resulting clusters' centroids are the reference identities (K is whatever the clustering returns; typically 1–3 for our videos).
- For each subsequent frame, match each detected face to nearest reference identity (greedy NN, threshold τ; one-to-one).
- Clip score = mean similarity of accepted matches × (matched faces / detected faces in the clip excluding the reference frame).

Run on the headline videos only, as one column in T2 (discrimination ablation).

## Ablation plan (thesis evidence)

**T1 — Headline.** Methods (MGLD, UAV, LQ baseline) × 5 videos × 6 score columns: `slow_self, fast_self, fused_self, slow_lqref, fast_lqref, fused_lqref`. Main result of the run.

**T2 — Discrimination test.** Four metric variants compared as MGLD-vs-UAV rankers:
1. Single-identity slow-fast (existing 0.555 vs 0.463 baseline)
2. Multi-self fused
3. Multi-lqref fused
4. Approach 3 (anchor) — sanity column

Per-video winner + aggregate wins-per-metric. Answers: does multi-person change the conclusion?

**T3 — Threshold τ sensitivity.** τ ∈ {0.3, 0.4, 0.5, 0.6}, report `fused_self`. Tests metric robustness; ranking flip = fragile; no flip = robust. Either is a valid thesis finding.

**T4 — Slow/fast fusion weight.** Weight ∈ {0/100, 25/75, 50/50, 75/25, 100/0} for `fused_self`. Tests whether 50/50 is meaningful or arbitrary.

**T5 — Self-vs-LQ-ref correlation + coverage.** Per-video Pearson correlation between per-clip `self` and `lqref` scores; plus `evaluable_clips_pct`. Tells reviewers what the two variants buy over each other.

**Optional figure:** per-cluster consistency curves for one representative video — one line per detected identity over time. No extra runtime.

**Compute:** T1 is the only full run. T2–T5 reuse cached per-clip embeddings/clusters from T1, so post-hoc and essentially free.

## Risks & mitigations

- **LQ face detection coverage** — main empirical risk. If `evaluable_clips_pct < 50%` for most videos, LQ-ref becomes a sparse signal and self-consistency carries the thesis story. Mitigation: report coverage explicitly (T5); if too low, add a footnote and downplay the LQ-ref variant.
- **Threshold τ over-fitting** — picking τ to favor MGLD invalidates the comparison. Mitigation: T3 sweep is the safeguard; pick τ that produces stable rankings, not the highest delta.
- **Clustering noise on short clips** — 2-sec clips at 24 fps = 48 frames; with 5 people, that's only ~10 detections per identity, marginal for clustering quality. Mitigation: noise-rate diagnostic + the size-1 cluster discard rule. If diagnostic shows >20% all-singleton clips, revisit clip length (sensitivity table).
- **RetinaFace flicker** — detector may miss frames, fragmenting an identity into multiple clusters within one clip. Mitigation: in cluster scoring we already weight by size; small fragments matter little. If still problematic, add inter-frame IoU smoothing as a post-detection step.

## Out of scope (future work)

- Cross-clip re-identification (Granularity B from the design discussion). Reserved for if T1 results are too noisy and we need whole-video tracking.
- Detector replacement — sticking with RetinaFace + ArcFace from upstream VBench-2.0 to keep the metric "an extension of VBench."
- Multi-frame temporal embeddings (e.g., 3D ArcFace) — out of scope; we stay with per-frame embeddings.

## Implementation pointers

Files to add/modify:
- `scripts/vbench2_long/human_identity_long.py` — add `score_clip_self()`, `score_clip_lqref()`, `score_clip_anchor()` (Approach 3); add `--mode {single, multi-self, multi-lqref, all}` arg; default `single` for back-compat.
- New helper: `scripts/vbench2_long/multiperson_clustering.py` — agglomerative clustering wrapper (sklearn), threshold/linkage configurable.
- New helper: `scripts/vbench2_long/face_matching.py` — IoU spatial matching for LQ↔SR.
- Cache layer: introduce a per-frame face/embedding cache (keyed by `(video_id, frame_idx)`, stored under `results/cache/identity_long/`) so T2–T5 ablations reuse the expensive RetinaFace + ArcFace passes from T1.
- `results/` table writer: extend to emit the 6-column-per-video JSON shape described above.

Upstream patches (`vbench2/human_identity.py`) — keep the three existing patches (multi-face frames, late ref init, ZeroDivision guards). No new upstream changes needed for multi-person — all multi-person logic lives in our adapter.
