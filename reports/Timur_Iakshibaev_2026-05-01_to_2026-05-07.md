# Weekly Progress Report — Timur Iakshibaev

## Period: May 1 – May 7, 2026

## Key Result — MGLD wins identity consistency on long videos

Built and ran a slow-fast long-video adapter for VBench-2.0 Human_Identity. **MGLD-SR beats UAV by +0.092 fused score** (0.555 vs 0.463) and wins 4 of 5 synthetic videos.

| Method | Slow (within-clip) | Fast (cross-clip) | Fused |
|--------|--------------------|--------------------|-------|
| MGLD-SR | **0.682** | **0.346** | **0.555** |
| UAV | 0.639 | 0.286 | 0.463 |

Per-video fused: MGLD wins 7WHI2L_FDNg, BrRLKMbBTYQ, hhszUXL1Cu8, mJog8DlRk_4. UAV only wins KZ8p6b1zJ9U.

Whole-video custom_input mode previously gave very low scores (~0.20 for both methods) because identity drift accumulates over minutes. The slow-fast adapter properly localizes per-clip evaluation.

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

## Next Steps (May 6 – May 7)

1. Complete CLIP weights transfer (parallel SCP running now)
2. Run VBench-2.0 Human_Anatomy on MGLD + UAV synthetic videos
3. Multi-person Human_Identity adaptation (cluster-based identity tracking) — addresses crowd-scene limitation
4. Start VBench effectiveness validation — generate test datasets with parameterized artifacts (color drift, periodic flicker, identity degradation, etc.)
5. Add long-range tOF + tLP metrics to evaluation pipeline

## Open Questions for Group

- Is per-clip 2-second granularity reasonable, or should it be longer (e.g., 4–6 seconds) to capture more within-clip drift?
- Multi-person identity tracking — should "fraction of detected faces consistent with any tracked cluster" be the metric, or weighted by face area?
- For the validation experiment: which parameterized artifacts most need coverage (we have 5 candidate datasets — see `docs/plans/`)?
