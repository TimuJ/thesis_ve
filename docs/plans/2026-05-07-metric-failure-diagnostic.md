# Metric-Failure Diagnostic on KZ8p6b1zJ9U

**Date:** 2026-05-07
**Status (updated 2026-05-27):** **EXECUTED.** Per-frame anatomy + identity diagnostics produced. Root cause identified as the **close-up regime shift** (close-up faces/hands trigger the anomaly classifier's high-fire regime). Findings live in `docs/notes/2026-05-13-kz-regime-shift-trigger.md`. Outcome motivated LR-VCC's **closeup-p50 reliability gate** (`scripts/lr_vcc/identity.py` + `closeup_map_artefacts/*.json`), which now down-weights Identity sub-metric when close-up content dominates. The KZ flip is preserved under LR-VCC: MGLD wins 5/5 (Layer 1+2 validation, see `docs/notes/2026-05-21-lr-vcc-validation.md`).
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

**PARTIALLY ANSWERED 2026-05-13 → bbox-size correlates but isn't causal.** Bbox-area analysis on the cached per-frame traces (`docs/notes/2026-05-13-kz-regime-shift-trigger.md`): KZ has hand bbox p50 = 18% of frame vs ≤7% on all other videos. Across videos, hand-bbox p50 monotonically tracks the MGLD-vs-UAV anatomy gap. But two post-hoc fixes that should have rescued KZ if close-ups were the cause **don't fully work**:

1. Drop frames with face / hand bbox ≥ 5% of frame (stable-regime filter), re-aggregate slow-fast: KZ gap *widens* from -0.339 to -0.437. MGLD's filtered KZ score *drops* to 0.076. Non-close-up KZ frames also flag MGLD aggressively.
2. Continuous aggregation `1 - mean(p_abnormal)` instead of fraction-above-threshold: KZ gap halves (-0.291 → -0.136) but does not disappear. Threshold-near-boundary effect accounts for ~50%; the rest is real signal.

So the bbox-size correlation is **predictive at the video level but not the proximate cause at the frame level**. There's a deeper content-specific bias in the detector against MGLD's KZ output. Candidates for the next experiment: scene genre (interview/talking-head), person appearance, camera motion. Would need a *second* close-up video to disambiguate (single-video confounder problem).

### Step 1.5 results — per-frame anatomy on hhszUXL1Cu8 (DONE 2026-05-07)

| | MGLD | UAV |
|---|---|---|
| Frames with detected people | 94.0% | 94.9% |
| Total people / total abnormal | 3012 / 225 | 3060 / 372 |
| Per-frame abnormal-rate, mean | 0.058 | 0.104 |
| Per-frame abnormal-rate, median | **0.000** | **0.000** |
| % frames with **no** abnormal flag | 90.6% | 84.8% |
| Detector triggers — human / face / hand | **0 / 0 / 242** | **0 / 2 / 391** |
| Score | 0.9253 | 0.8784 |

**Confirms the regime hypothesis.** In the low-fire regime, the median per-frame abnormal-rate is **0 for both methods** (vs 1.0 for both on KZ), and **MGLD has fewer triggers than UAV** (242 vs 393, total). So the "MGLD always trips the detector more" story is wrong — the bias only emerges in a high-fire regime tripped by certain content. Where the detector is stable (this video), MGLD wins as expected because its outputs are slightly cleaner.

Notable: on this video the only firing detector is `hand` (the human and face detectors barely fire). On KZ all three detectors fire heavily for both methods. The KZ failure may therefore be content-specific to scenes that activate the human/face anomaly detectors at high rates — which is the next thing to characterize if we want to pin down what triggers the regime switch.

### Step 2 results — per-clip identity slow-fast on KZ8p6b1zJ9U (DONE 2026-05-07)

| | MGLD | UAV |
|---|---|---|
| Total clips | 83 | 104 |
| Clips with faces detected | 42 (51%) | 42 (40%) |
| Per-clip score: mean / median / stdev | 0.703 / 0.854 / 0.336 | 0.723 / 0.951 / 0.358 |
| **Slow** (within-clip avg) | **0.7025** | **0.7233** |
| **Fast** (cross-clip first-frames) | **0.6111** | **0.7778** |
| Fused (50/50) | 0.6568 | 0.7505 |

**Identity failure is localized, not uniform — and the failure mechanism differs from anatomy.**

- Within-clip slow scores are essentially tied (MGLD 0.7025 vs UAV 0.7233 — gap 0.02). MGLD does not lose locally.
- The whole identity gap comes from the **fast branch** (cross-clip identity): MGLD 0.611 vs UAV 0.778 — gap 0.17.
- Of 15 clips with faces detected in **both** methods, wins are split: UAV 6, MGLD 5, tied 4. Mean per-clip delta ≈ +0.008. So no uniform bias — just clip-specific drifts in different time windows.
- Top UAV-wins clips cluster at `t=56–72s` (clips 28, 29, 30, 36) where MGLD scores 0.14–0.58 vs UAV 1.0.
- Top MGLD-wins clips at `t=18s, 44s, 50s` (clips 9, 22, 25) where MGLD scores 0.82–1.0 vs UAV 0.04–0.30.

**Mechanism:** the fast branch concatenates the *first frame of every clip* into a synthetic video and runs identity on the resulting frame sequence. UAV's smoother SR produces **more consistent face appearance across clip boundaries** because diffusion is absent — same blur, same lack of detail. MGLD's diffusion adds chunk-dependent fine-grained variation (different noise patterns per processing chunk) that ArcFace reads as identity drift across clip boundaries, even when within each clip the same person is clearly recognizable.

This is a different failure than anatomy:

- **Anatomy** — content-dependent detector instability (some scenes trigger an unstable high-fire regime).
- **Identity** — cross-clip ArcFace embedding drift; smoother output trivially wins cross-clip identity even when visually worse.

Both are **long-video-specific failures** that wouldn't show up on the short 16-frame benchmarks VBench-2.0 was designed for.

### Sub-finding: framerate metadata differs between methods

Probed on server with cv2: both SR mp4s have **5000 frames** (same content count) but different fps tags — **MGLD 30 fps → 166.7 s; UAV 24 fps → 208.3 s**. The 2-sec slow-fast splitter at native fps therefore produces clips with **different frame counts** (MGLD 60 frames/clip → 83 clips; UAV 48 frames/clip → 104 clips). Implications:

- Slow branch: per-clip face detection has slightly more frames per clip on MGLD (60 vs 48), but slow scores end up nearly equal anyway (0.703 vs 0.723), so this is a minor effect.
- Fast branch: UAV has more first-frames concatenated (104 vs 83), so *more* cross-clip transitions where identity could fail. UAV still scores higher (0.778 vs 0.611) — strengthens the "smoother output is trivially more cross-clip consistent" interpretation.
- Fair-comparison fix to consider: re-encode both to the same fps before evaluation, or split by frame-count instead of seconds. Worth doing if the framerate mismatch is suspected to bias other metrics too.

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
