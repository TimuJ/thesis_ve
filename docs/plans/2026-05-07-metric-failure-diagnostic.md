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

## Step 1 results — anatomy per-frame trace (DONE 2026-05-07)

Both runs completed on GPU 0; outputs at `results/vbench2_anatomy/diagnostic_KZ8p6b1zJ9U/{mgld,uav}_KZ8p6b1zJ9U_per_frame.json`. Verifies cleanly: `1 - total_abnormal/total_people` from the per-frame trace reproduces the `eval_results.json` score (MGLD 0.1435, UAV 0.4346) exactly.

| | MGLD | UAV |
|---|---|---|
| Frames | 5000 | 5000 |
| Frames with detected people | 78.9% | 72.0% |
| Frames where **all** detected people flagged abnormal | **84.8%** | 53.2% |
| Per-frame abnormal-rate, mean | 0.863 | 0.557 |
| Per-frame abnormal-rate, median | **1.000** | 1.000 |
| Detector triggers (above threshold) — human / face / hand | 2179 / 2074 / 2714 | 858 / 973 / 1520 |

Two findings:

1. **All three detectors fire ~2× more on MGLD.** Not a single-detector artefact — human, face, and hand all light up uniformly more.
2. **Failure is uniform across the video, not localized.** Binned into 10-second windows, *every* window has MGLD's abnormal-rate strictly above UAV's, by +0.13 to +0.52. There is no specific "bad moment" — MGLD is flagged more consistently from `0s` through `200s`.

### Why MGLD wins on the other 4 videos but loses on this one

A "diffusion is uniformly penalized" hypothesis would predict MGLD < UAV on *all* videos. It doesn't — MGLD wins 4/5. So that one-line story is wrong. The corrected reading uses the absolute score levels:

| Video | MGLD | UAV | Detector regime (both methods) |
|-------|------|-----|--------------------------------|
| 7WHI2L_FDNg | 0.832 | 0.735 | low trigger — detector "stable" |
| BrRLKMbBTYQ | 0.522 | 0.437 | medium |
| **KZ8p6b1zJ9U** | **0.144** | **0.435** | **high trigger — detector "unstable"** |
| hhszUXL1Cu8 | 0.925 | 0.878 | low trigger |
| mJog8DlRk_4 | 0.577 | 0.541 | medium |

`KZ8p6b1zJ9U` is the only video where the anomaly detector is firing heavily on **both methods** (both scores < 0.5). On the other 4 videos the detector is in a low-fire regime and MGLD wins by small margins (the order most evaluators would expect). So:

- **Stable regime (4/5 videos):** detector fires rarely, MGLD's slightly cleaner output → MGLD wins as expected.
- **Unstable regime (`KZ8p6b1zJ9U`):** something about this scene content drives the detector into a high-fire state, and within that regime MGLD's diffusion output is asymmetrically more triggering than UAV's smooth output. The detector's behaviour flips.

The thesis-relevant claim is therefore narrower but still strong: `Human_Anatomy` **is not metrically stable** under diffusion-style SR. On most content it agrees with the expected ordering; on certain content it inverts hard enough to drag the per-video mean to a tie. This is a usability failure for the metric on long videos that contain even a single such scene.

What's distinctive about `KZ8p6b1zJ9U` is the next thing to check (close-up faces? specific lighting? motion-blurred hands?). The user has already looked and confirms it's a single-person scene without crowd, so the failure is not multi-person related.

### Step 1.5 — per-frame anatomy on a MGLD-wins video (queued)

Run the same per-frame diagnostic on **`hhszUXL1Cu8`** (MGLD's biggest win, 0.925 vs 0.878) the next time GPUs are free. Goal: confirm that in the stable-regime case both methods have near-zero abnormal rates and the MGLD/UAV ordering is small but consistent. If the detector triggers ~2× more on MGLD there too (just at very low absolute rates), then the "MGLD always trips it more" story is right and the per-video mean only flips when the absolute rate is high enough; if the detector triggers ~equally low for both, the failure is genuinely content-specific to KZ and we should look at scene properties.

### Step 2 — per-clip identity (queued, GPUs saturated as of May 7 17:30 CST)

Original Step 2 still planned: run `human_identity_long.py --save_clip_detail` on `KZ8p6b1zJ9U` for both methods, persist per-clip slow scores. Determines whether identity fails in the same uniform way or localizes to specific clip ranges.

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
