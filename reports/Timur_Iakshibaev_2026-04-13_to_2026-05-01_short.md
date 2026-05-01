# Progress Report — April 13 – May 1, 2026

## Key Results

### 1. Synthetic Long-Video Benchmark — MGLD vs UAV

5 synthetic long videos (22,412 frames, 320×180 → 1280×720, 4× upscale) evaluated on both methods. **MGLD-SR wins 8 of 9 metrics**.

| Metric | LQ | MGLD-SR | UAV | Winner |
|--------|----|---------|------|--------|
| CLIP-IQA ↑ | — | **0.496** | 0.391 | MGLD |
| MUSIQ ↑ | — | **65.07** | 56.28 | MGLD |
| NIQE ↓ | — | **4.67** | 5.73 | MGLD |
| BRISQUE ↓ | — | **24.74** | 50.90 | MGLD |
| DOVER overall ↑ | 10.44 | **73.81** | 65.06 | MGLD |
| E\*warp ↓ | 0.0092 | **0.0114** | 0.0137 | MGLD (best SR) |
| VBench imaging_quality ↑ | 0.4388 | **0.6810** | 0.6458 | MGLD |
| VBench aesthetic ↑ | 0.4128 | **0.5080** | 0.4892 | MGLD |
| VBench subject_consistency ↑ | 0.8936 | 0.8927 | **0.9031** | UAV |

UAV's win on subject_consistency is likely a DINOv2 color-invariance artifact, not real superiority.

### 2. DOVE Benchmark Alignment

| Method | Dataset | DOVE Paper | Ours | Status |
|--------|---------|-----------|------|--------|
| MGLD-VSR | UDM10 | PSNR 24.23 | 24.2339 | **Identical match** |
| UAV | UDM10 | PSNR 21.72 | 23.05 | +1.33 dB gap |
| UAV | SPMCS | PSNR 18.81 | 20.49 | +1.68 dB gap |

UAV gap is consistent across datasets; ruled out input format, seed, settings, frame count, resolution, prompt text. Torch 2.5.1 test attempted (matching DOVE's environment) but blocked by server cuDNN incompatibility. Stopping further alignment attempts — none of the variations have closed the gap and we'll proceed with our own UAV evaluation rather than waste more time matching DOVE numbers.

### 3. VBench Quality Score — Complete

All 7 VBench Quality Score dimensions evaluated for MGLD-SR + UAV + LQ. MGLD shows largest improvement on imaging_quality (+55% vs LQ) and aesthetic_quality (+23%). The 9 Semantic Score dimensions require text prompts and are not applicable for SR evaluation.

### 4. New Metrics Added

- **DOVER** — no-reference video quality (aesthetic + technical), installed and weights deployed
- **E\*warp** — temporal warping error via RAFT optical flow, implemented from scratch
- **DISTS** — added to pyiqa pipeline for full-reference perceptual quality

### 5. VBench-2.0 Adaptation for SR

Identified that real VBench-2.0 (separate from `vbench2_beta_long`) has 18 dimensions for intrinsic faithfulness in T2V generation. Of these, only 2 are repurposable for SR:
- **Human_Anatomy** — anomaly detection (per-frame, no prompt needed)
- **Human_Identity** — face identity consistency (RetinaFace + ArcFace)

Human_Identity ran via VBench-2.0's standard `custom_input` mode (no long-video slow-fast adapter built yet — that's the next step). Patched the algorithm to handle multi-face frames and late reference initialization. Initial results across 5 videos:

| Method | Mean Identity Score |
|--------|---------------------|
| MGLD-SR | 0.200 |
| UAV | **0.203** |

Single-identity tracking limits scores in crowd scenes; multi-person extension noted as future work.

### 6. VBench Effectiveness Validation Plan

Source-code analysis identified 7 limitations of VBench for SR evaluation (no long-range temporal_flickering branch, fast branch samples 2% of frames, DINOv2 color-invariance, etc.). Designed 5 synthetic test datasets with parameterized artifacts (color drift, periodic flicker, chunk-boundary jumps, identity degradation, long-range background change) to validate metric sensitivity.

## Next Steps

1. Implement multi-person Human_Identity adaptation (cluster-based tracking)
2. Complete VBench-2.0 Human_Anatomy evaluation
3. Build long-video adapter for VBench-2.0 (slow-fast pattern)
4. Generate VBench validation test datasets and run sensitivity analysis
5. Add long-range tOF + tLP metrics to evaluation pipeline
6. Begin thesis writing — Introduction + Literature Review chapters
7. Draft proposal outline once Teme provides materials
