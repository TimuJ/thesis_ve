# Metric-Failure Diagnostic on KZ8p6b1zJ9U

**Date:** 2026-05-07
**Status:** Plan — pending implementation
**Goal:** Localize where VBench-2.0 metrics disagree with human visual judgment, on the one video where both `Human_Identity` and `Human_Anatomy` rank UAV above MGLD despite UAV being visually worse.

## Motivating finding

Across our 5 synthetic SR videos, MGLD wins 4/5 per-video on both `Human_Identity` (slow-fast fused) and `Human_Anatomy` (whole-video). The single exception is `KZ8p6b1zJ9U` — and **on that video the metric flips so hard it pulls the anatomy mean to a tie**:

| Metric on KZ8p6b1zJ9U | MGLD | UAV |
|-----------------------|------|-----|
| Identity (fused, slow-fast) | 0.657 | **0.751** |
| Anatomy (whole-video) | 0.144 | **0.435** |

Visual inspection of the SR outputs **clearly favors MGLD** on this video — UAV's output is described as "really ugly" by the user. So both metrics fail in the **same direction on the same video**, on a **single-person scene** (ruling out the multi-face / crowd hypothesis from earlier reports).

Two metrics failing together is unlikely to be random noise — there's a systematic property of MGLD's output on `KZ8p6b1zJ9U` that triggers both:

- the anomaly detectors to fire more often, **and**
- the ArcFace embeddings to drift away from each other across frames.

## Why this matters

This is the strongest-form thesis evidence for the "metric effectiveness for long-video SR" track:

- A real, reproducible, two-metric counterexample to the standard VBench-2.0 SR evaluation protocol.
- It contradicts the expectation that "MGLD wins identity → MGLD looks better" — at least on this video.
- It points at a concrete failure mode (we don't yet know which: anomaly-detector overconfidence on diffusion artifacts? RetinaFace mis-localization in MGLD's plate? ArcFace embedding drift on noisy SR output?). The diagnostic below will tell us which.

## Plan

### Step 1 — per-frame anatomy trace (cheap)

`compute_abnormality` already computes per-frame `{person_count, abnormal_count, per-person scores}` then drops it. We persist it.

- New script `scripts/vbench2_long/diagnose_anatomy_per_frame.py`:
  - Imports `Detector` and `Analyzer` from `vbench2.third_party.ViTDetector.detect`.
  - Runs on a **single** video (CLI arg).
  - Saves the full `analyzer.analyze(...)`'s `frame_results` to JSON: `[{frame, person_count, abnormal_count, persons:[{person_id, abnormal, scores}]}, ...]`.
- Run for MGLD + UAV on `KZ8p6b1zJ9U` only. Estimated runtime: ~14 min MGLD, ~10 min UAV based on last night's per-iteration timings.
- Output: `results/vbench2_anatomy/{mgld,uav}_KZ8p6b1zJ9U_per_frame.json`.

### Step 2 — per-clip identity trace (cheap, reuses existing adapter)

`human_identity_long.py` builds `clip_detail` (per-clip identity scores) then aggregates and discards.

- Add `--save-clip-detail` flag that persists `clip_detail` and `fast_detail` into the per-video output JSON. Keep default off (back-compat with the existing T1 results).
- Run for MGLD + UAV on `KZ8p6b1zJ9U` only. Estimated runtime: ~30 min.
- Output: per-clip identity scores at known time offsets — directly localizes which 2-second windows UAV "wins".

### Step 3 — analysis (local, post-hoc)

For each method on `KZ8p6b1zJ9U`:

1. **Anatomy timeline** — per-frame abnormal rate (`abnormal_count / person_count`) vs frame index.
2. **Anomaly category breakdown** — for the frames where MGLD scores worse, which detector fires (human / face / hand)?
3. **Identity timeline** — per-clip identity score vs clip index (each clip = ~2 sec).
4. **Disagreement frames** — locate the top-K frames/clips where MGLD's score is far below UAV's. Extract those frames as PNGs from both methods (`ffmpeg -ss <t> -i <video.mp4> -frames:v 1`).
5. **Visual confirmation** — eyeball the side-by-side PNGs to confirm: does MGLD genuinely have a perceivable artefact at those frames, or is the metric being fooled by an SR property that the human eye prefers (e.g., diffusion-style sharpness mistaken for "anomaly")?

### Step 4 — write-up

If the diagnostic localizes the failure cleanly (e.g., "MGLD's hand detector fires 4× more often on this video due to diffusion sharpening of texture"), document it as a thesis case study under `docs/plans/2026-05-07-metric-failure-diagnostic.md`.

If it doesn't localize cleanly (failure is spread uniformly across frames), that itself is a finding: the metric is uniformly biased on this video, not pointing at a specific scene moment.

## Extension (later, if signal is clear)

Run the same per-frame/per-clip trace on the **other 4 videos** to test the inverse claim:

> "Does the metric agree with human judgment on the videos where MGLD wins?"

If yes on all 4 + clear failure mode on `KZ8p6b1zJ9U`, that's a clean thesis story: VBench-2.0 metrics work on most SR content but break in `<failure-mode>`.

## Out of scope

- Fixing or replacing the metric. This plan only **diagnoses** it.
- Comparing more SR methods. This is a within-pair (MGLD vs UAV) localization study.
- Re-encoding videos. We keep the existing `mgld_synthetic_mp4/` and `uav_synthetic_mp4/` files.

## Implementation pointers

Files to add/modify:

- `scripts/vbench2_long/diagnose_anatomy_per_frame.py` — new, ~80 lines.
- `scripts/vbench2_long/human_identity_long.py` — add `--save-clip-detail` flag + persist `clip_detail` and `fast_detail` when set.
- `results/vbench2_anatomy/diagnostic_KZ8p6b1zJ9U/` — new dir for per-frame and per-clip JSONs and the side-by-side frame extracts.

Run script on server (manual, after pushing the diagnostic):

```bash
cd $DISK2/repos/VBench/VBench-2.0
export VBENCH2_CACHE_DIR=$DISK2/cache/vbench2
export PYTHONPATH="$PWD:$DISK2/repos/YOLO-World:${PYTHONPATH:-}"
conda activate vbench

for method in mgld uav; do
    CUDA_VISIBLE_DEVICES=0 python diagnose_anatomy_per_frame.py \
        --video $DISK2/results/${method}_synthetic_mp4/KZ8p6b1zJ9U.mp4 \
        --output $DISK2/results/vbench2_human_test/diagnostic/${method}_KZ8p6b1zJ9U_per_frame.json
done
```
