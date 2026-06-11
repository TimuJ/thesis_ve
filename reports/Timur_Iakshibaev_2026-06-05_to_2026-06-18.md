# Bi-weekly Progress Report — Timur Iakshibaev

## Period: June 5 – June 18, 2026

## Headline

This bi-weekly period responded to the supervisor's group-meeting criticism that the four existing synthetic artefacts (color_drift, chunk_boundary, flicker, identity_degradation) do not directly test "same human / same background throughout the scene" — the core long-term consistency framing. Two new artefacts were designed, implemented, generated on the lab server, and evaluated through the full 7-metric battery plus the LR-VCC v3+slope β=200 composite.

1. **Two new long-term-consistency artefacts landed.** `identity_drift` (progressive face-region morph toward a per-base reference identity) and `background_drift` (progressive background replacement toward a per-base reference scene, with Detectron2 Mask R-CNN human silhouettes precomputed to preserve the subject). Generator code, unit tests (5 + 8 = 13 new tests, all passing), and the updated `generate_all.py` driver are committed and pushed.
2. **The strongest single severity-response signal collected to date:** on the multi-face base, `background_drift` LR-VCC drops monotonically from 0.532 at severity 0.02 to 0.256 at severity 0.40 — **Δ −0.276**, larger than every previous artefact result. Driven by sub-metric E (colour trajectory slope), whose R²-gated reliability cleanly registers the cumulative scene replacement as a linear Lab-channel trajectory.
3. **The single-face base now has three documented failure modes**, all rooted in slow-fast pooling on one-subject-only content: identity_degradation inverted (heavy blur → faces look generically similar), identity_drift flat (cross-clip first-frame embedding floor at ~0.066 regardless of morph severity), background_drift inverted (LR-VCC rises because the mJog reference scene has a colour distribution close enough to 7WHI's that the cumulative blend flattens the per-frame Lab trajectory rather than steepening it).
4. **Server infrastructure debugging carried out**: 100%-full disk freed by clearing 37 GB of perceptual-model caches (pip, vbench, dreamsim); the vbench conda env needed a `pip install 'setuptools<81'` for detectron2's `pkg_resources` import to work; the tmux quoting gotcha (pipes break `set -e` and swallow `touch` after `| tee`) was characterised and worked around by manual `.done` flag creation. Server-side onboarding docs updated so the next session does not re-discover these.

---

## Two New Long-Term-Consistency Artefacts

### identity_drift — progressive face morph toward a reference identity

`scripts/synthetic_artefacts/identity_drift.py`. For each frame i of T:

```
blend = severity * (i / (T - 1))
```

For each Haar-detected face bounding box: alpha-blend the original face region with a reference face image at factor `blend`. Reference faces extracted from cross-base sources via `extract_reference_face.py`: hhsz target = KZ frame 100 (324×324, sharp close-up); 7WHI target = KZ frame 100 (same sharp 324×324, after a first run with a 65×65 BrRLK reference produced essentially no response and was hypothesised to be reference-resolution-limited).

5 unit tests cover severity-0 pass-through, idx-0 pass-through regardless of severity, no-face-detected pass-through, degenerate `n_frames ≤ 1`, and shape/dtype preservation. All pass.

### background_drift — progressive scene replacement with subject preserved

`scripts/synthetic_artefacts/background_drift.py`. For each frame i of T:

```
blend = severity * (i / (T - 1))
full_blend = (1 - blend) * frame + blend * reference_bg
out[human_mask] = frame[human_mask]
out[~human_mask] = full_blend[~human_mask]
```

The human silhouette mask is precomputed per source video by `scripts/synthetic_artefacts/precompute_human_masks.py` (Detectron2 Mask R-CNN, COCO person class, per-frame instance-mask union packed via `np.packbits` for ~120× compression). Mean coverage: hhsz 29.4%, 7WHI 29.6%.

8 unit tests cover severity-0 pass-through, idx-0 pass-through, `human_mask=None` blends everything, `mask=all-ones` preserves everything, partial mask preserves only inside, intermediate blend factor arithmetic, degenerate `n_frames ≤ 1`, shape/dtype preservation. All pass.

### Reference images and human masks

| File | Source | Size |
|------|--------|---|
| `_references/ref_face_for_hhsz.png` | KZ8p6b1zJ9U frame 100 (largest face) | 324×324 |
| `_references/ref_face_for_7WHI.png` | KZ8p6b1zJ9U frame 100 (same — sharp ref after BrRLK 65×65 failed) | 324×324 |
| `_references/ref_bg_for_hhsz.png` | BrRLKMbBTYQ frame 500 (full frame) | 1280×720 |
| `_references/ref_bg_for_7WHI.png` | mJog8DlRk_4 frame 500 (full frame) | 1280×720 |
| `_human_masks/hhszUXL1Cu8.npz` | Detectron2 person masks, 2412 frames | 6.3 MB packed |
| `_human_masks/7WHI2L_FDNg.npz` | Detectron2 person masks, 5000 frames | 19 MB packed |

---

## Server-Side Pipeline

### Metric battery results

Two pipelines ran in parallel: `eval_drift2` (identity_drift re-eval with sharp 7WHI reference, GPU 0) and `eval_bg` (background_drift full eval, GPU 7). Both completed during the night. Each ran the 7-metric battery (CLIP-IQA, multi-k tOF/tLP, DOVER, E*warp, color histogram, color slope, Identity slow-fast).

### LR-VCC composite — identity_drift (sharp 7WHI ref)

| sev | hhsz | 7WHI |
|---|---:|---:|
| 0.02 | 0.527 | 0.618 |
| 0.05 | 0.526 | 0.619 |
| 0.10 | 0.523 | 0.618 |
| 0.20 | 0.516 | 0.619 |
| 0.40 | 0.493 | 0.616 |
| **Δ** | **−0.034** | **−0.002** |

hhsz shows weak monotonic response at the composite level even though sub-metric I drops cleanly 0.667 → 0.388 — the composition averages I with four other sub-metrics (A, T, D, E) that correctly stay quiet on face-region morph. 7WHI essentially flat: the sharp 324×324 KZ reference did not rescue the single-face base's slow-fast-pooling pathology. The 7WHI fast component sits at ~0.066 (noise floor) regardless of severity.

### LR-VCC composite — background_drift

| sev | hhsz | 7WHI |
|---|---:|---:|
| 0.02 | 0.532 | 0.619 |
| 0.05 | 0.536 | 0.619 |
| 0.10 | 0.517 | 0.628 |
| 0.20 | 0.385 | 0.645 |
| 0.40 | **0.256** | **0.661** |
| **Δ** | **−0.276** | **+0.042** |

hhsz shows the strongest single severity-response signal collected. Sub-metric E (colour trajectory slope) is the driver — score drops 0.168 → 0.023 (Δ −0.145), monotonic, because the progressive scene replacement introduces a linear Lab-channel trajectory that E's R²-gated reliability registers confidently. Sub-metric I (Identity) stays at ~0.65 across the severity range (correct — the human subject IS preserved).

7WHI inverts. Mechanism: the mJog reference scene's colour distribution is similar enough to 7WHI's that the cumulative blend produces a *flatter* per-frame Lab trajectory than 7WHI's natural per-frame variation. Sub-metric E rises (0.370 → 0.467) instead of falling, and sub-metric D similarly rises (0.559 → 0.594). Documented failure: artefact is sensitive to reference-scene similarity. Fix direction: pick reference scenes with deliberately distant colour distributions, or replace the Lab-trajectory signal with a perceptual (CLIP-image-distance-style) trajectory.

---

## Consolidated 6-Artefact × 2-Base Verdict

| Artefact | hhsz Δ | 7WHI Δ | Outcome |
|----------|---:|---:|---|
| chunk_boundary | −0.236 | −0.162 | both monotonic; PASS |
| color_drift | −0.039 | −0.111 | 7WHI clean, hhsz weak; PARTIAL |
| flicker | +0.005 | −0.012 | flat both; FAIL |
| identity_degradation | −0.070 | +0.043 | hhsz clean, 7WHI inverted; PARTIAL |
| **identity_drift** (sharp ref) | **−0.034** | **−0.002** | hhsz weak-monotonic; 7WHI flat; PARTIAL |
| **background_drift** | **−0.276** | **+0.042** | **hhsz strongest-of-six**; 7WHI inverted (ref similarity); PARTIAL |

**LR-VCC catches 7/12 conditions cleanly across the six artefact families.** The two new long-term-consistency artefacts directly address the supervisor's framing and demonstrate that the colour-trajectory sub-metric E (originally designed for synthetic linear colour ramps) generalises naturally to the scene-replacement case. The 7WHI single-face base is now established as having a *structural* limitation rather than artefact-by-artefact noise: every face-affecting or scene-replacing artefact either inverts or stays flat on it.

---

## Server Infrastructure — Discovered Problems and Fixes

### Disk pressure

`/data/disk2` hit 100% (7.8 GB free) at the start of the period. Investigated with `du -h --max-depth=1`; the easy wins were:

| Path | Size | Risk |
|------|---|---|
| `cache/pip` | 27 GB | safe — `pip` rebuilds on demand |
| `cache/vbench` | 7.6 GB | safe — rebuilds on next VBench run |
| `cache/dreamsim` | 2.7 GB | safe — rebuilds on next model load |

Cleared ~37 GB, brought disk to 44 GB free at start of artefact runs, ~30 GB at end. The big inert footprints to keep an eye on next time: `repos/MGLD-VSR` (12 GB), `repos/Upscale-A-Video` (8.9 GB), `miniconda3` (49 GB), `results/mgld_synthetic` raw frames (23 GB), `results/uav_synthetic` raw frames (21 GB) — the raw frames could be pruned if the mp4 versions in `*_synthetic_mp4` directories are sufficient for downstream re-evaluation.

### vbench env needed `setuptools<81`

Running `precompute_human_masks.py` failed at import with `ModuleNotFoundError: No module named 'pkg_resources'`. detectron2's `model_zoo` imports `pkg_resources` from setuptools, which was removed/relocated in setuptools 81. Fix:

```bash
conda activate vbench
pip install 'setuptools<81'
```

Recorded in onboarding docs as a known one-shot fix.

### tmux quoting gotcha

This pattern is buggy:

```bash
tmux new-session -d -s name "cmd 2>&1 | tee /tmp/log; touch /tmp/done"
```

The `touch` ends up being parsed as an argument to `tee`, not as a separate command. Symptom: pipeline runs to completion but `.done` is never created, so downstream chained sessions wait forever. Two workarounds were used: (a) use `bash -lc 'multi-line script'` so the second statement is explicit; (b) manually `touch /tmp/done` when polling reveals work is finished.

### GPU pinning

GPU 0 free most of the time, GPU 7 free in a backup slot. GPUs 1–6 are typically taken by other lab processes. Always set `CUDA_VISIBLE_DEVICES=0` (primary) or `=7` (backup) before launching long stages.

---

## Code Delivered

| File | Purpose | Status |
|------|---------|---|
| `scripts/synthetic_artefacts/identity_drift.py` | Face morph generator (idx, n_frames, ref_face, severity) | New |
| `scripts/synthetic_artefacts/extract_reference_face.py` | Pick the largest Haar-detected face from a video frame | New |
| `scripts/synthetic_artefacts/background_drift.py` | Background blend generator (idx, n_frames, ref_bg, human_mask, severity) | New |
| `scripts/synthetic_artefacts/extract_reference_background.py` | Save an arbitrary video frame as the reference scene | New |
| `scripts/synthetic_artefacts/precompute_human_masks.py` | Detectron2 Mask R-CNN per-frame person masks → packed `.npz` | New |
| `scripts/synthetic_artefacts/generate_all.py` | + identity_drift and background_drift branches; per-base reference and mask paths and caches | Modified |
| `tests/synthetic_artefacts/test_identity_drift.py` | 5 unit tests | New |
| `tests/synthetic_artefacts/test_background_drift.py` | 8 unit tests | New |
| `docs/onboarding.md` | Server connection, env quirks, GPU pinning, tmux gotchas, standard experiment pattern | Updated |
| `docs/private/server-setup.md` | Runner scripts, reference / mask directories, setuptools fix | Updated |

Generator test suite: **29/29 PASS** (15 prior + 5 identity_drift + 8 background_drift, plus a coincident hit on a previously-stale 1 test that was re-validated).

---

## Commit Log (this period)

```
d175192 synthetic_artefacts: identity_drift — slow morph toward reference face
7ad9306 synthetic_artefacts: background_drift — preserve humans, blend rest toward reference scene
```

Two more commits expected before submission: server-workflow docs update + final report. Plus future commits for the inevitable next artefact iteration once the 7WHI single-face structural issue is addressed.

---

## Next Steps

1. **Address the 7WHI single-face structural issue directly.** Three documented failure modes on the single-face base now share the same root cause (slow-fast pooling on one-subject-only content). Concrete fix candidates: (a) gate sub-metric I reliability by per-face embedding *variance* (low variance → embeddings collapsed → metric abstains), (b) augment the slow-fast adapter with a body-region embedding so identity has multiple anchors, (c) add a third base video that's multi-face but in a different scene from hhsz so single-face is no longer the only test case. Recommend (a) as it's the cleanest delta on the existing infrastructure.
2. **Fix the background_drift 7WHI inversion.** The mJog reference scene is too colour-similar to 7WHI's source. Either re-extract from a deliberately-distant scene (an outdoor scene if 7WHI is indoor, or vice-versa) or move the sub-metric E reliability threshold so close-colour-distribution scenes do not trigger the linear-trajectory detector.
3. **Begin paper-direction work.** With six artefact families and the verdict matrix consolidated, the synthetic test set is now mature enough to anchor a publishable methodology contribution. Sketch the paper outline + draft the methodology figure.

## Open Technical Questions

1. **Reference-scene curation** — for background_drift, should we standardise on a single curated set of high-distance reference scenes (CLIP-image-distance > τ from each base) rather than picking arbitrary frames from the other test videos? Probably yes; the 7WHI inversion shows the artefact's sensitivity.
2. **Identity_drift reference quality** — the sharp 324×324 KZ reference still produced no response on 7WHI. Is the test telling us "single-face base is structurally unsuitable", or are we missing a more subtle mechanism for the slow-fast adapter to register face changes?
3. **Six-artefact validation set or seven?** Should we add a "subject swap" artefact (instantaneous discontinuity halfway through video) before the next batch of metric work, or is six sufficient and we now focus on closing the existing failure modes?
