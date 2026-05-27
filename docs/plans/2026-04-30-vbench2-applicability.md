# VBench-2.0 Applicability for Super-Resolution

**Date:** 2026-04-30
**Source:** `VBench/VBench-2.0/vbench2/` — 18 dimensions across 5 categories.

> **STATUS (2026-05-27):** Survey phase executed. Of the SR-applicable dimensions, **Human_Anatomy** and **Human_Identity** are now in production via slow-fast adapters (`scripts/vbench2_long/`) and integrated into LR-VCC (Identity = sub-metric I; Anatomy gated via closeup-p50 map and replaced by reliability-weighted Identity after the KZ regime-shift finding — see `docs/notes/2026-05-13-kz-regime-shift-trigger.md`). The applicability findings below remain accurate as a one-time survey; production behaviour now superseded by the notes and by `docs/plans/2026-05-21-lr-vcc-design.md`.

VBench-2.0 evaluates **intrinsic faithfulness** of generated videos: physics correctness, common sense, human anatomy, motion order. Most dimensions assume the video was generated from a text prompt and use VLMs (LLaVA-Video, Qwen) or auxiliary metadata to compare against the prompt's intent.

For SR evaluation, only dimensions that don't require text prompts can be repurposed.

## NOT applicable (require text prompt or auxiliary_info)

13 of 18 dimensions evaluate prompt-to-video alignment via VLMs:

| Dimension | What it does | Why not for SR |
|-----------|--------------|-----------------|
| camera_motion | Detects camera type, compares to expected label in `auxiliary_info` | No expected motion in SR |
| complex_landscape | Qwen judges if landscape from prompt appears | No prompt |
| complex_plot | Qwen judges if plot from prompt occurs | No prompt |
| composition | LLaVA-Video answers prompt's composition question | No prompt question |
| dynamic_attribute | LLaVA-Video answers attribute question | No prompt question |
| dynamic_spatial_relationship | LLaVA-Video answers spatial question | No prompt question |
| human_clothes | LLaVA-Video evaluates clothes from prompt | No prompt |
| human_interaction | Qwen judges interaction match with prompt | No prompt |
| material | LLaVA-Video answers material question | No prompt question |
| mechanics | LLaVA-Video answers mechanics question | No prompt question |
| motion_order_understanding | Qwen judges action order from prompt | No prompt |
| motion_rationality | LLaVA-Video answers rationality question | No prompt question |
| thermotics | LLaVA-Video answers thermal question | No prompt question |
| instance_preservation | Anomaly detection per prompt's instance list | No prompt instances |

## NOT applicable (other reasons)

| Dimension | Issue |
|-----------|-------|
| diversity | Compares **multiple generations** of the same prompt with different seeds. SR produces one output per input — N=1, no diversity to measure. |
| multi_view_consistency | Designed for multi-view 3D coherence (orbiting camera, multiple angles of same scene). Our SR videos are sequential single-view. Could potentially be adapted to measure long-range view drift but conceptually different. |

## Potentially APPLICABLE for SR

Only 2 dimensions don't require text prompts or auxiliary metadata:

### 1. human_anatomy

- **What:** Uses `ViTDetector` to detect human anatomical anomalies (extra limbs, deformed faces/hands, etc.) frame-by-frame.
- **No prompt needed.** Per-frame analysis aggregated to video score.
- **For SR:** Highly relevant — SR-induced face/hand artifacts (warping, melting, extra fingers) are exactly what this detects.
- **Caveat:** Only works on videos containing humans. Of our 5 synthetic videos, hhszUXL1Cu8 contains people; check others.
- **Effort:** Need ViTDetector weights. Code expects `vbench2/third_party/ViTDetector/`.

### 2. human_identity

- **What:** Uses RetinaFace to detect faces, ArcFace to extract features, tracks identity consistency across frames.
- **No prompt needed.** Measures consistency of face features over time.
- **For SR:** Directly relevant — SR can morph face identity across long videos (one of the documented issues in the validation plan).
- **Caveat:** Same as above, needs human faces in videos.
- **Effort:** Need RetinaFace + ArcFace weights, CoTracker for face tracking.

## Recommendation

For VBench-2.0 long-video extension targeting SR:

1. **Adapt `human_anatomy` and `human_identity`** to long-video mode (clip splitting + slow-fast aggregation, similar to `vbench2_beta_long`).
2. **Skip the other 16 dimensions** — they're fundamentally about T2V faithfulness, not SR.
3. **For multi_view_consistency**, consider a custom adaptation: replace multi-view assumption with long-range temporal view drift. Not a direct port.

## Comparison with VBench 1.0 Quality Score (already done)

The 7 Quality Score dims from VBench 1.0 long (`vbench2_beta_long`) remain the most useful VBench metrics for SR. VBench-2.0 adds value mainly via:
- **human_anatomy** — detects SR face/hand artifacts (new capability)
- **human_identity** — detects long-range identity drift (complements `subject_consistency`)

## Implementation plan for long-video adaptation

To adapt these 2 dimensions to long videos (similar to how `vbench2_beta_long` extended VBench 1.0):

1. Reuse VBench 1.0 long's clip splitting (`split_video_into_scenes`, `split_video_into_clips`)
2. Run `human_anatomy` per clip → average per video (anomaly is per-frame, easy aggregation)
3. Run `human_identity`:
   - Slow branch: identity consistency within each 2-second clip
   - Fast branch: identity consistency across clip first-frames (catches long-range drift)
   - Fuse with weighted average

## Effort estimate

- Download required model weights (ViTDetector, RetinaFace, ArcFace, CoTracker): ~2 GB, locally + SCP
- Adapt 2 dimensions to long-video mode: ~2 days
- Run on MGLD-SR + UAV synthetic videos: ~1 day eval
- Total: ~1 week

## Open questions

1. Do our 5 synthetic videos all contain humans? hhszUXL1Cu8 yes (people). Need to check others.
2. Is identity tracking meaningful for animation/non-photorealistic content?
3. Should we focus on these 2 dimensions, or prioritize the validation experiments (Tests A–E from the previous plan)?

---

## Decision (April 30): Unified SR Benchmark

**Goal:** combine VBench 1.0 long Quality Score (7 dims, done) + VBench-2.0 human dimensions (2 dims, to adapt) into one SR-focused long-video benchmark.

### Key finding from `vbench2/__init__.py`

Both `Human_Anatomy` and `Human_Identity` are **already supported in `custom_input` mode** — they're NOT in `dim_custom_not_supported`:

```python
dim_custom_not_supported = set(dimension_list) & set([
    'Composition', 'Dynamic_Attribute', 'Dynamic_Spatial_Relationship',
    'Instance_Preservation', 'Complex_Plot', 'Complex_Landscape',
    'Motion_Rationality', 'Motion_Order_Understanding', 'Mechanics',
    'Thermotics', 'Material', "Camera_Motion", "Human_Interaction"
])
```

So we can run `human_anatomy` and `human_identity` on our SR videos **out of the box** — they don't need prompts.

### Limitation: VBench-2.0 doesn't have a long-video mode

VBench-2.0 is designed for short generated clips (typically ≤6 seconds). For our 100–208s videos, we need to add the same slow-fast pattern that `vbench2_beta_long` introduced.

### Implementation Plan

#### Phase 1: Run VBench-2.0 directly on our SR videos (1 day)

Quick test to see if it works at all on long videos before building the long-mode wrapper.

1. Set up VBench-2.0 env on server (similar to `vbench` env, separate from VBench 1.0)
2. Download model weights:
   - ViTDetector (anatomy detection) — likely on HF or as release asset
   - RetinaFace + ArcFace (identity) — `model_zoo.load_url` from torch hub
   - CoTracker (face tracking) — torch.hub
3. Run `evaluate.py --mode custom_input --dimension Human_Anatomy Human_Identity` on our 5 MGLD synthetic videos
4. Check OOM behavior and runtime — both depend on per-frame processing

#### Phase 2: Verify human content in our videos (1 hour)

Do all 5 synthetic videos contain humans? Check with simple detection. Without humans, `human_anatomy` and `human_identity` produce no signal. We need to:
- Either select human-containing videos
- Or note as "N/A" for non-human content (anomaly detection might still flag artifacts)

#### Phase 3: Build long-video mode adapter (2–3 days)

Mirror `vbench2_beta_long`'s slow-fast architecture:
- **Slow branch** (within-clip): split video into 2-second clips, run VBench-2.0 per clip, aggregate
- **Fast branch** (across-clip): for `human_identity`, compare face embeddings of clip-first-frames across the whole video — catches long-range identity drift
- **Fusion**: weighted average (default 50/50 from `slow_fast_params.yaml`)

For `human_anatomy`, the slow branch is sufficient — anatomy abnormality is per-frame, doesn't benefit from cross-clip evaluation.

For `human_identity`, the fast branch is the key addition — VBench-2.0 already tracks identity within a clip, but our SR can morph identity over minutes (between clips). The fast branch is critical here.

#### Phase 4: Unified benchmark wrapper (1 day)

Create `scripts/sr_benchmark/run_full_eval.sh` that:
1. Runs VBench 1.0 long Quality Score (7 dims) — existing
2. Runs VBench-2.0 long-mode Human Anatomy + Identity (2 new dims)
3. Runs DOVER + E\*warp + NR metrics — existing
4. Aggregates everything into single JSON + markdown table

### Suggested code structure

```
scripts/
  sr_benchmark/
    __init__.py
    vbench1_quality.py      # wraps vbench2_beta_long
    vbench2_human_long.py   # NEW — long-video adapter for human_anatomy, human_identity
    dover_runner.py         # wraps DOVER eval
    ewarp_runner.py         # wraps E*warp eval
    nr_metrics.py           # pyiqa-based NR eval
    run_full_eval.py        # orchestrator
    aggregate_results.py    # produce unified report
```

## Phase 1 Progress (May 1)

### human_identity — WORKING

- All weights downloaded and SCP'd: ArcFace `resnet18_110.pth` (98MB), RetinaFace zip (97MB)
- Two algorithm patches applied:
  - Allow multi-face frames (pick largest face) — original required exactly 1 face
  - Allow late reference frame initialization — original required face in frame 0
- All 5 MGLD + 5 UAV videos evaluated successfully
- Mean: MGLD 0.200 vs UAV 0.203 (UAV very slightly better)
- Documented multi-person limitation (algorithm tracks single identity)

### human_anatomy — BLOCKED

Setup completed up to but not including CLIP weights:
- mmcv 2.2.0, mmdet 3.3.0, mmyolo 0.6.0 installed
- Patched mmdet/mmyolo version checks (accept mmcv 2.2.0)
- Patched timm `_pil_interp` import (use `str_to_interp_mode`)
- YOLO-World source SCP'd, syntax error patched (`text_feats, _` instead of `text_feats, None`)
- ViTDetector config patched (replaced `/mnt/petrelfs/...` with `openai/clip-vit-base-patch32`)
- Anomaly detector weights SCP'd: human/face/hand .pth (88MB each), YOLO-World 168MB
- **BLOCKER:** CLIP-ViT-Base-Patch32 weights (577MB pytorch_model.bin) — SCP keeps disconnecting after 3-5MB transferred. Server connection unstable today. Need to retry when network is calmer, or split into smaller chunks (10MB blocks also failing). Once weights present, anatomy should run.

### TODO: Multi-person adaptation for human_identity (deferred)

VBench-2.0 `human_identity` tracks a single reference identity (largest face per frame).
For our synthetic videos with crowds, this produces artificially low scores because the
"largest face" can belong to different people across frames.

Initial run on 5 MGLD + 5 UAV videos showed mean ~0.20 (very low) due to this limitation.

**Proposed fix** (to implement after human_anatomy):
- Cluster-based identity tracking: maintain a set of tracked identities, not one
- For each detected face: find best-matching cluster; if similarity > threshold, count as
  consistent and update cluster centroid; else register a new identity
- Score = fraction of detected faces that match an existing cluster (>= threshold)

This properly measures "are individual people preserved consistently" rather than "is
the single largest face the same across frames".

### Multi-view consistency (deferred)

`multi_view_consistency` is harder to repurpose — it expects multiple camera angles. Consider after the 2-dim adapter is working. Possible adaptation: measure long-range view drift in single-view videos by comparing first-frame to subsequent-clip first-frames in feature space.

### Next concrete steps

1. Set up VBench-2.0 env (Phase 1, step 1) — try running existing scripts first
2. Identify which synthetic videos have humans (Phase 2)
3. Once Phase 1 confirms it runs, design the long-mode adapter (Phase 3)
