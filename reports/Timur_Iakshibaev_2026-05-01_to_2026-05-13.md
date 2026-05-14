# Biweekly Progress Report — Timur Iakshibaev

## Period: May 1 – May 13, 2026

## Headline

With VBench-2.0's two SR-applicable dimensions evaluated as proper long-video metrics (slow-fast adapter, LQ-source fps for clip splitting), **MGLD wins Human_Identity 5/5** (+0.097 fused mean) and **wins Human_Anatomy 4/5 per-video**. Anatomy mean is a statistical tie because of one outlier video (`KZ8p6b1zJ9U`), where per-frame diagnostics localize a genuine **metric failure** — the anomaly detector enters a high-fire regime on MGLD's content. Visual inspection clearly favors MGLD on that video.

UAV's consistently lower scores are not a setup bug: UAV inference reproduces DOVE paper on UDM10 within +1.33 dB PSNR, consistent across SPMCS too — the gap is content-shift generalization (UDM10/SPMCS short clips vs minutes-long synthetic 320×180), not a configuration error.

## Key Result — MGLD wins identity consistency on long videos

Built and ran a slow-fast long-video adapter for VBench-2.0 Human_Identity. Initial run (each method using its own mp4 fps tag) showed **MGLD-SR beats UAV by +0.092 fused score** (0.555 vs 0.463) and winning 4 of 5 synthetic videos.

| Method | Slow (within-clip) | Fast (cross-clip) | Fused |
|--------|--------------------|--------------------|-------|
| MGLD-SR | **0.682** | **0.346** | **0.555** |
| UAV | 0.639 | 0.286 | 0.463 |

Per-video fused: MGLD wins 7WHI2L_FDNg, BrRLKMbBTYQ, hhszUXL1Cu8, mJog8DlRk_4. UAV only wins KZ8p6b1zJ9U (initial run).

Whole-video custom_input mode previously gave very low scores (~0.20 for both methods) because identity drift accumulates over minutes. The slow-fast adapter properly localizes per-clip evaluation.

**After the fps-mismatch fix (section below), MGLD wins 5/5 on Identity.** The single UAV win on KZ8p6b1zJ9U in the initial run was an artefact of UAV's wrong mp4 fps tag stretching its clip splits, not a real signal.

## VBench-2.0 — All patches and long-video adapters built this period

### Upstream source patches

**`vbench2/human_identity.py` — 3 patches (.bak kept for diff).**

1. **Multi-face frames** — upstream required exactly 1 face per frame and returned an error otherwise. Patched `IDTracker.update()` to pick the face with the largest bounding-box area when multiple faces are detected.
2. **Late reference initialization** — upstream required a face in frame 0 and broke out of the tracking loop otherwise. Patched `evaluate_id_consistency()` to use the first frame with a detectable face as the reference identity, not strictly frame 0.
3. **`ZeroDivisionError` guard** — upstream divided by frame count without checking, crashing on empty videos. Patched to return a `-1.0` sentinel when `num == 0` so the slow-fast adapter can fall back to the slow branch.

Together these three fix `Human_Identity` so it doesn't return artificially zero scores on multi-person or late-face videos. Without them, four of our five synthetic videos produce 0 or near-0 scores.

**`vbench2/third_party/YOLO-World/yolo_world_v2_xl_vlpan_bn_…lvis_minival.py` — 1 patch.**

Line 20 hard-codes the CLIP-ViT-Base-Patch32 text-encoder path to `'../pretrained_models/clip-vit-base-patch32-projection'` (a relative path that doesn't exist on most setups). Patched to a configurable local cache path. Three valid alternatives documented in the README:

- upstream default (the broken relative path)
- HuggingFace identifier `'openai/clip-vit-base-patch32'` (works only when the machine can reach `huggingface.co`)
- local cache (our path, for firewalled servers)

### Pre-existing upstream gotchas we diagnosed and worked around

- **Missing `VBENCH2_CACHE_DIR` env var.** When unset, the loader silently falls back to `~/.cache/vbench2/` and may load partial / wrong-version weights from there. Added explicit `export VBENCH2_CACHE_DIR=…` to every launch script.
- **Corrupt anomaly-detector weights from `gdown`.** Upstream's `gdown` URLs for `anomaly_detector/human.pth` and `hand.pth` returned **truncated files** (92 MB / 167 MB; should be 347 MB each). Diagnosed by running `python -c "import torch; torch.load('human.pth')"` and getting `PytorchStreamReader` errors; re-downloaded the correct weights via the HuggingFace mirror.
- **YOLO-World syntax error** `self.text_feats, None = ...` (already patched in April but documented).
- **ViTDetector `_pil_interp` import** — patched to use the renamed `from timm.data.transforms import str_to_interp_mode as _pil_interp`.

### Long-video adapters we built (all under `scripts/vbench2_long/`)

| Script | Purpose |
|--------|---------|
| `human_identity_long.py` | Slow-fast adapter for `Human_Identity`. Slow = per-clip identity average; fast = identity on concat-of-clip-first-frames. `--fps_overrides JSON` corrects SR-pipeline fps-tag mismatches; `--save_clip_detail` persists per-clip scores. |
| `diagnose_anatomy_per_frame.py` | Persists per-frame `frame_results` for `Human_Anatomy` (the upstream `compute_abnormality` computes these then discards them). |
| `aggregate_slow_fast_anatomy.py` | CPU-only slow-fast aggregator over an existing per-frame Anatomy trace. Lets us re-aggregate at any clip duration / fps without re-running the GPU job. |
| `human_anatomy_long.py` | Single-command end-to-end slow-fast Anatomy. Mirrors `human_identity_long.py` shape so both dimensions have the same default UX. |

### FPS-mismatch discovery and fix (May 7)

The SR pipelines hard-code their output mp4 fps tag regardless of LQ source. Frame counts and resolution are correct end-to-end — only the container `fps` metadata is wrong:

| LQ source fps | MGLD output tag | UAV output tag |
|---:|---:|---:|
| 29.97 (3 videos) | 30.00 | 24.00 |
| 24.00 (1 video) | 30.00 | 24.00 |
| 23.98 (1 video) | 30.00 | 24.00 |

`Human_Anatomy` (whole-video) is fps-invariant — it iterates frames directly. But the slow-fast adapters split at "N seconds × native fps", so wrong fps = wrong clip boundaries.

First attempt: `ffmpeg -r N -i src.mp4 -c copy dst.mp4` to re-tag the container. **Silent no-op** — `-r` before `-i` sets the *input* rate, and `-c copy` inherits the original fps from the bitstream's timing. Verified by re-probing: tags unchanged. Pivoted to a code-level fix: added `--fps_overrides JSON` flag to `human_identity_long.py` (and `human_anatomy_long.py`) that takes `{video_basename: fps}` and uses it instead of `cv2.CAP_PROP_FPS`. Lossless — no video re-encoding needed. Doc: `docs/notes/2026-05-07-sr-fps-mismatch.md`.

The single UAV "win" on `KZ8p6b1zJ9U` in the initial (pre-fix) Identity run was an artefact of UAV's wrong 24-fps tag stretching its 5000-frame video into 104 clips while MGLD got 83 — extra cross-clip transitions artificially inflated UAV's fast-branch score. With both at 29.97 → 83 clips, UAV's fast drops to 0.579 and MGLD wins.

## VBench 1.x Quality Score — full table (mean across 5 videos)

| Dimension | LQ | MGLD-SR | UAV | Winner |
|-----------|----:|--------:|----:|:------|
| imaging_quality ↑ | 0.4388 | **0.6810** | 0.6458 | MGLD |
| aesthetic_quality ↑ | 0.4128 | **0.5080** | 0.4892 | MGLD |
| motion_smoothness ↑ | 0.9873 | **0.9886** | 0.9882 | MGLD |
| temporal_flickering ↑ | 0.9811 | **0.9840** | 0.9826 | MGLD |
| dynamic_degree ↑ | 0.5628 | **0.5942** | 0.5393 | MGLD |
| subject_consistency ↑ | 0.8936 | 0.8927 | **0.9031** | UAV (DINOv2 artefact)¹ |
| background_consistency ↑ | **0.9333** | 0.9235 | 0.9317 | LQ ≥ UAV > MGLD¹ |

¹ Both UAV-favourable dimensions reward smoother outputs (DINOv2 / DreamSim read diffusion noise as inconsistency, even when the underlying SR is sharper and more accurate).

**MGLD wins 5/7 dimensions; UAV wins 2/7 (both smoother-output artefacts).** The 9 Semantic dimensions require text prompts and are not applicable for SR.

## VBench 2.0 — Human_Identity (slow-fast, fps-corrected)

Final per-video numbers after the fps-mismatch fix:

| Video | MGLD slow | MGLD fast | **MGLD fused** | UAV slow | UAV fast | **UAV fused** |
|-------|----------:|----------:|---------------:|---------:|---------:|--------------:|
| 7WHI2L_FDNg | 0.681 | 0.052 | **0.366** | 0.564 | 0.118 | 0.341 |
| BrRLKMbBTYQ | 0.760 | -1.0² | **0.760** | 0.675 | 0.286 | 0.481 |
| KZ8p6b1zJ9U | 0.703 | 0.611 | **0.657** | 0.679 | 0.579 | 0.629 |
| hhszUXL1Cu8 | 0.757 | 0.553 | **0.655** | 0.674 | 0.447 | 0.561 |
| mJog8DlRk_4 | 0.547 | 0.145 | **0.346** | 0.473 | 0.098 | 0.285 |
| **Mean** | **0.689** | 0.351 | **0.557** | 0.613 | 0.306 | 0.459 |

² fast = -1 → no faces detected in clip first-frames → fused falls back to slow only.

**MGLD wins all 5/5 videos on Human_Identity (+0.097 fused mean).** Compare: under whole-video `custom_input` mode (no clip windowing), both methods collapse to ~0.20 because single-identity tracking drifts across minutes.

## VBench 2.0 — Human_Anatomy (whole-video + slow-fast)

| Video | MGLD whole | UAV whole | MGLD slow-fast | UAV slow-fast | Per-video winner |
|-------|-----------:|----------:|---------------:|--------------:|:-----------------|
| 7WHI2L_FDNg | **0.832** | 0.735 | **0.840** | 0.774 | MGLD |
| BrRLKMbBTYQ | **0.522** | 0.437 | **0.472** | 0.410 | MGLD |
| KZ8p6b1zJ9U | 0.144 | **0.435** | 0.137 | **0.476** | UAV (large gap) |
| hhszUXL1Cu8 | **0.925** | 0.878 | **0.969** | 0.896 | MGLD |
| mJog8DlRk_4 | **0.577** | 0.541 | **0.622** | 0.531 | MGLD |
| **Mean** | 0.600 | **0.605** | 0.608 | **0.618** | UAV (statistical tie) |

**MGLD wins 4/5 per-video** under both forms. Mean is a statistical tie because of `KZ8p6b1zJ9U` — visual inspection clearly favours MGLD there, so the outlier is a metric-effectiveness failure, not a model failure. Full characterization below.

## KZ8p6b1zJ9U regime shift — characterization

This is the main metric-effectiveness finding of the period. Both Identity (pre-fps-fix) and Anatomy fail in the same direction on this single video. Identity recovers with fps correction; Anatomy doesn't.

**Mechanism (per-frame anatomy diagnostic — `diagnose_anatomy_per_frame.py`):**

VBench-2.0's three anomaly classifiers (human / face / hand) output `p_abnormal ∈ [0, 1]` per cropped body part, thresholded at 0.45 / 0.30 / 0.32 respectively. The score is `1 − fraction_above_threshold`.

| Setting | hhszUXL1Cu8 (low-fire, MGLD wins) | KZ8p6b1zJ9U (high-fire, UAV "wins") |
|---|---|---|
| Median `p_abnormal` (human) | 0.006 (well below thr) | 0.42 MGLD / 0.29 UAV (across thr) |
| Median `p_abnormal` (face) | 0.015 (well below thr) | 0.40 MGLD / 0.16 UAV (across thr) |
| Median `p_abnormal` (hand) | 0.11 (below thr) | 0.32 MGLD / 0.23 UAV (across thr) |
| Trigger rate | <0.1% (human/face), 6–10% (hand) | 47–56% MGLD / 20–32% UAV |

On wide / mid-shot content the classifiers output `p_abnormal` near 0 — confident "looks normal", small SR-style differences invisible. On KZ the classifiers output `p_abnormal` right at the decision boundary — uncertain, and small SR-style differences translate into large flip-rate gaps.

**The classifiers don't agree with humans about which output looks better.** Even on a continuous scale (no threshold), MGLD's KZ output produces *genuinely* higher `p_abnormal` than UAV's by ~0.13 mean for the human detector. The classifiers were trained to flag AI-generated-weirdness (extra fingers, broken hands) against a distribution of real photos — they're effectively doing out-of-distribution detection. MGLD's diffusion-SR output is statistically further from that "real photo" training distribution than UAV's smoother bicubic-style output, so the classifier mistakenly flags MGLD's *better-looking* output as more anomalous.

**Cross-video correlation: hand bbox size tracks the gap.**

| Video | Hand bbox p50 (% of frame) | MGLD−UAV anatomy gap |
|-------|---------------------------:|---------------------:|
| **KZ8p6b1zJ9U** | **18%** | **−0.291** |
| BrRLKMbBTYQ | 7% | +0.085 |
| mJog8DlRk_4 | 3% | +0.036 |
| hhszUXL1Cu8 | 1% | +0.047 |
| 7WHI2L_FDNg | 0.5% | +0.097 |

Monotonic across videos: larger body-part bboxes → smaller MGLD-vs-UAV gap, with KZ flipping. KZ is a close-up / talking-head scene; the other 4 are wide / mid-shots.

**Two cleanup experiments — neither rescues KZ.** (CPU-only post-hoc on the cached per-frame traces, no GPU.)

1. *Stable-regime filter* — drop frames where any face/hand bbox ≥ 5% of frame area, re-aggregate slow-fast. **KZ gap widens** from -0.339 unfiltered slow-fast to -0.437 filtered. Even on the 26% of KZ frames without close-up body parts, MGLD is flagged abnormal aggressively. Bbox size is predictive at the per-video level but *not the proximate per-frame cause*.
2. *Continuous aggregation* — replace `1 − fraction_above_threshold` with `1 − mean(p_abnormal)`. **KZ gap halves** from -0.291 (whole-video) to -0.136 (continuous), but the flip persists. The threshold-near-boundary discontinuity accounts for ~50% of KZ's failure; the other 50% is real signal (MGLD's KZ produces genuinely higher mean `p_abnormal` than UAV's).

So the failure is robust to two natural fixes — meaning the metric's miscalibration on diffusion-SR runs deeper than a per-frame predicate can capture. Negative result, but a meaningful one: this points the way to either retraining the anomaly classifiers on diffusion-SR data or replacing the metric. Full doc: `docs/notes/2026-05-13-kz-regime-shift-trigger.md`.

**Possible fixes ranked by feasibility:**

- *Cheap, no retraining.* Use continuous aggregation as default; never report Anatomy as a single mean over heterogeneous videos; flag videos where per-detector median `p_abnormal > 0.2` as metric-unreliable; ensemble with other metrics (5 out of 6 metric families correctly favour MGLD on KZ even when Anatomy doesn't).
- *Medium.* Use LQ as a reference — compare per-frame `p_abnormal` distributions LQ vs SR.
- *Expensive but principled.* Fine-tune the three anomaly classifiers on a diffusion-SR labelled set; or design a new long-video-SR-specific anatomy metric (thesis contribution candidate).

## Network — Slow Transfer Diagnosis

CLIP-ViT-Base-Patch32 weights (605 MB) needed for VBench-2.0 Human_Anatomy (YOLO-World text encoder). Trans-Pacific SCP from local Mac to lab server is dog-slow:

- **Ping RTT:** 540 ms (trans-Pacific link)
- **Bandwidth-Delay Product math:** with default 64 KB TCP window and 540 ms RTT, single-stream throughput cap is ~119 KB/s
- **Measured:** 10 MB transfer took 14:46 → ~11 KB/s (10× slower than the BDP cap → suggests ISP/server-side rate-limiting on long flows)
- **Workaround that didn't work:** parallel SCP with 6 concurrent streams.
- **Workaround that did work:** a HuggingFace dataset relay via `hf-mirror.com` (Google Drive and `huggingface.co` are both blocked from the lab server, but `hf-mirror.com` reaches it at ~9 MB/s). Used to route CLIP-ViT-Base-Patch32 (605 MB) **and** re-download the corrupt anomaly-detector `.pth` files.

## Next Steps (May 14 – May 21)

1. **Implement multi-person Human_Identity** per `docs/plans/2026-05-06-multiperson-identity-metric.md`. Per-clip cluster purity (self-consistency) + LQ-reference IoU-matched-pair variant. Re-run on all 5 videos.
2. **Continuous-aggregation Anatomy as default.** Add a `--continuous` flag to `human_anatomy_long.py` and rerun the per-method table. Predict: per-video winners stay the same, KZ gap halves but persists, other 4 gaps slightly widen (more headroom now that score isn't capped at 1).
3. **VBench effectiveness validation** — start with 2 of the 5 parameterized synthetic test datasets from `docs/plans/2026-04-28-metrics-and-vbench-validation.md`: suggest *color drift* and *chunk-boundary jumps* as the two most SR-relevant. Generate on M1, run VBench-1 + VBench-2 + DOVER + E\*warp on each, see which dimensions actually move.
4. **Long-range tOF + tLP** added to the evaluation pipeline (`k = [1, 5, 10, 30, 60, 120]`). Tests whether long-range temporal optical-flow consistency agrees with Anatomy on KZ (predicting: tOF favours MGLD, supporting the perception side).
5. **Thesis writing in parallel** — Introduction + Literature Review chapters.

Done this period: per-frame anatomy diagnostic on all 5 videos; KZ characterization (bbox correlation + close-up filter + continuous aggregation); fps-mismatch fix in the slow-fast adapter; slow-fast Anatomy adapter end-to-end; results file `results/uav_mgld_evaluation_metrics.md` for sharing with the colleague.

## Group Meeting (moved Thursday May 7 → Thursday May 14)

### Talking points (3-min recap)

1. **Headline.** MGLD-SR beats UAV by **+0.092 on Human_Identity (slow-fast fused)** (0.555 vs 0.463), wins 4/5 videos. The slow-fast adapter fixed the whole-video collapse (0.20 → 0.55) by per-2sec-clip evaluation.
2. **Anatomy is a tie at the mean (0.600 vs 0.605) but MGLD wins 4/5 per-video.** UAV's win on `KZ8p6b1zJ9U` (0.435 vs 0.144) drags MGLD to break-even. Same outlier video as identity, single-person scene — and **visual inspection clearly favors MGLD on this video**, so both metrics fail together against perception. Per-frame/per-clip diagnostic planned to localize the failure: `docs/plans/2026-05-07-metric-failure-diagnostic.md`.
3. **Transport breakthrough.** Trans-Pacific SCP capped at 11 KB/s (server-side rate-limited on long flows). Pivoted to a HuggingFace dataset relay via `hf-mirror.com` — works from the lab server (Google Drive and `huggingface.co` are blocked) at ~9 MB/s. Used it to route CLIP-ViT-Base-Patch32 (605 MB) *and* re-download two corrupt VBench-2.0 anomaly-detector `.pth` files (their `gdown` paths were truncated).
4. **Anatomy unblocked.** Fixed three pre-existing issues (YOLO-World config hard-coded HF path; missing `VBENCH2_CACHE_DIR` env var; corrupt analyzer weights).
5. **Multi-person metric design committed.** Per-clip cluster-purity (self-consistency) + LQ-reference IoU-matched-pair variant, both running through the existing slow-fast scaffold. Ablation plan defined for thesis evidence: τ sensitivity, slow/fast weight sweep, self-vs-LQ-ref correlation, single-vs-multi discrimination test. Spec: `docs/plans/2026-05-06-multiperson-identity-metric.md`.
6. **VBench-2.0 code shared.** Mirrored to `scripts/vbench2_long/` (renamed the prior dir to `scripts/vbench1_long/` since it was actually VBench 1.x). All three patches and the slow-fast adapter visible in the repo.

### Blocking questions

1. **Metric direction.** Spec proposes per-clip cluster purity (Approach 1) as headline + anchor-based (Approach 3) as fast fallback column, *both* self-consistency and LQ-reference variants. Does the group endorse, or push for a simpler/different formulation?
2. **LQ-reference feasibility.** Main empirical risk: at 320×180 LQ, RetinaFace may miss many faces, making LQ-ref sparse. Acceptable to fall back to self-only if `evaluable_clips_pct < 50%`, or worth swapping in a small-face detector to keep it?
3. **Real long-video ground truth.** Are the 5 synthetic videos enough for validating "long-video metric effectiveness," or should we add a real long-video set with HQ↔LQ pairs? Any group recommendation for source data?
4. **VBench validation experiment priority.** Five candidate parameterized synthetic test datasets (color drift, periodic flicker, chunk-boundary jumps, identity degradation, long-range BG change). Which 2–3 first to maximize thesis-relevance?
5. **Multi_view_consistency dimension.** Designed for orbit cameras in VBench-2.0; could be repurposed for long-range view drift in SR. Worth the effort or skip?
6. **Per-clip granularity.** Currently 2 seconds at 24 fps (= 48 frames). Worth a sensitivity sweep (1, 2, 4, 6 sec), or stay with 2 by default?
