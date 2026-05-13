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

## Slow-Fast Adapter — Implementation

`scripts/vbench2_long/human_identity_long.py`:

- **Slow branch** — split video into 2-second clips at 24fps, run patched VBench-2.0 identity per clip, average across clips with detected faces
- **Fast branch** — concatenate the first frame of each clip into a synthetic "fast video", run identity on it (catches long-range identity drift)
- **Fusion** — weighted average (default 50/50, configurable via `slow_fast_params.yaml` style)

Three patches to original VBench-2.0 `human_identity.py`:

1. Multi-face frames — pick largest face instead of requiring exactly 1 face
2. Late reference initialization — allow first detected face to be the reference (not strictly frame 0)
3. ZeroDivisionError guards — return -1.0 sentinel when no faces in clip

## Network — Slow Transfer Diagnosis

CLIP-ViT-Base-Patch32 weights (605MB) needed for VBench-2.0 Human_Anatomy (YOLO-World text encoder). Transfer to server is dog-slow:

- **Ping RTT:** 540ms (trans-Pacific link)
- **Bandwidth-Delay Product math:** with default 64KB TCP window and 540ms RTT, single-stream throughput cap is ~119 KB/s
- **Measured:** 10MB transfer took 14:46 → ~11 KB/s (10× slower than the BDP cap → suggests ISP/server-side rate-limiting on long flows)
- **Workaround:** parallel SCP with 6 concurrent streams (each connection gets its own TCP window, total throughput multiplies). Currently running.

## Anatomy results (completed May 7 ~02:24 server time)

Both runs done. MGLD took 3h 30min, UAV 47min for the same 5 videos.

| Video | MGLD-SR | UAV | Winner |
|-------|---------|-----|--------|
| 7WHI2L_FDNg | **0.832** | 0.735 | MGLD |
| BrRLKMbBTYQ | **0.522** | 0.437 | MGLD |
| KZ8p6b1zJ9U | 0.144 | **0.435** | UAV (large gap) |
| hhszUXL1Cu8 | **0.925** | 0.878 | MGLD |
| mJog8DlRk_4 | **0.577** | 0.541 | MGLD |
| **Mean** | 0.600 | **0.605** | UAV (+0.005, tie) |

**MGLD wins 4/5 per-video on anatomy too**, but loses on the same outlier video as identity (`KZ8p6b1zJ9U`). The 0.144 there is enough to drag MGLD's mean to a statistical tie.

**Visual inspection contradicts the metric on `KZ8p6b1zJ9U`:** UAV's SR looks clearly worse than MGLD's — yet both VBench-2.0 metrics (Identity *and* Anatomy) score UAV higher on this video. Both metrics fail in the same direction, on a single-person scene, ruling out crowd/multi-face artefacts. This is exactly the failure mode the metric-effectiveness study is designed to surface — strong evidence that VBench-2.0 quality scores do not always track perceptual SR quality on long videos. Diagnostic plan to localize *where* in the video they fail: `docs/plans/2026-05-07-metric-failure-diagnostic.md`.

Side-by-side with identity:

| Metric | MGLD | UAV | Δ |
|--------|------|-----|---|
| Identity (slow-fast fused) | 0.555 | 0.463 | **+0.092 MGLD** |
| Anatomy (whole-video) | 0.600 | 0.605 | −0.005 (tie) |

## Next Steps

1. **Per-frame/per-clip metric-failure diagnostic on `KZ8p6b1zJ9U`** (`docs/plans/2026-05-07-metric-failure-diagnostic.md`). Persist the per-frame anatomy results (already computed, currently dropped) and per-clip identity scores; locate the time windows where MGLD scores worse than UAV; extract those frames and confirm visually. Strongest near-term thesis evidence on metric effectiveness.
2. Implement multi-person Human_Identity per `docs/plans/2026-05-06-multiperson-identity-metric.md`.
3. Start VBench effectiveness validation — generate test datasets with parameterized artifacts (color drift, periodic flicker, identity degradation, etc.).
4. Add long-range tOF + tLP metrics to evaluation pipeline.

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
