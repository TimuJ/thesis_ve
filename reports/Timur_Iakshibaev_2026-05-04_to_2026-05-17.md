# Biweekly Progress Report — Timur Iakshibaev

## Period: May 4 – May 17, 2026

## Carryover Result (week 1) — MGLD wins long-video identity consistency

Built a slow-fast long-video adapter for VBench-2.0 Human_Identity. **MGLD-SR beats UAV** on the 5 synthetic videos:

| Method | Slow (within-clip) | Fast (cross-clip) | Fused |
|--------|--------------------|--------------------|-------|
| MGLD-SR | **0.682** | **0.346** | **0.555** |
| UAV | 0.639 | 0.286 | 0.463 |

MGLD wins 4/5 videos on fused score. The slow-fast adapter (per-clip + cross-clip-first-frames) replaces VBench-2.0's whole-video mode, which gave artificially low scores (~0.20) because identity drift accumulates over minutes.

Implementation in `scripts/vbench2_long/human_identity_long.py`. Three patches applied to upstream VBench-2.0 (multi-face frames, late reference initialization, ZeroDivision guards).

## Plan for Period

### 1. Complete VBench-2.0 long-mode benchmark

- **Human_Anatomy** — blocked on CLIP-ViT-Base-Patch32 weights transfer (605MB, single-stream SCP capped at ~11 KB/s due to 540ms trans-Pacific RTT). Switched to parallel SCP (6 streams) to bypass per-connection rate limit.
- Once anatomy works, integrate into the same slow-fast wrapper as identity.
- Run both dimensions on MGLD + UAV synthetic videos, add to unified results table.

### 2. Multi-person Human_Identity adaptation

VBench-2.0 tracks one reference identity (largest face). Crowd scenes get artificially low scores. Plan:

- Cluster-based identity tracking — maintain a set of identity centroids, not one
- Score = fraction of detected faces matching an existing cluster (similarity > threshold)
- Re-run on all 5 synthetic videos, compare to single-identity scores

### 3. VBench Effectiveness Validation

Source-code analysis of VBench identified 7 limitations for SR evaluation (no long-range temporal_flickering branch, fast branch samples only 2% of frames, DINOv2 color invariance, etc.). Plan:

- Generate 5 synthetic test datasets with parameterized artifacts:
  - Test A — color drift
  - Test B — periodic flicker
  - Test C — chunk-boundary jumps
  - Test D — identity degradation
  - Test E — long-range background change
- Run all VBench dimensions on each → measure metric sensitivity to known artifacts
- Quantify which dimensions actually catch SR-specific failures

### 4. Long-range temporal metrics

Add tOF (temporal optical flow consistency) and tLP (temporal LPIPS) to pipeline. Current E\*warp captures adjacent-frame consistency; tOF/tLP add multi-frame and perceptual variants.

### 5. Thesis writing — start

- Draft Introduction chapter (motivation, problem statement, contributions)
- Draft Literature Review chapter (VSR methods, long-video benchmarks, diffusion models, SSM in video)
- Bootstrap from previous-thesis structure but rewrite content for VSR

### 6. Proposal outline

Pending materials from Teme. Once received, draft proposal outline and align with thesis structure.

## Risks & Mitigations

- **Network bottleneck** for CLIP transfer — using parallel SCP. If still slow, fall back to scp via faster relay or download directly on server from HuggingFace mirror (server has no GitHub but HF mirror may work).
- **CLIP weights blocking human_anatomy** — every other component (mmcv/mmdet/mmyolo, YOLO-World, ViTDetector weights) is in place. Once CLIP lands, anatomy should run end-to-end same day.
- **Multi-person adaptation effort** — single-identity baseline already gives meaningful comparison; multi-person is upgrade, not blocker.

## Open Questions

- Is the slow-fast 50/50 weighting right for SR? VBench-1.0 long uses 50/50 for everything; might tune per-dimension.
- Are 5 synthetic videos enough for the validation study, or should we add real long-video samples (with full ground truth)?
- Multi_view_consistency (VBench-2.0 dimension) — designed for orbit cameras, not directly applicable, but could be repurposed for long-range view drift. Worth the effort or skip?
