# Baseline Target Metrics & Experiment Log

## Degradation Pipeline
Both papers use **RealBasicVSR degradation** (blur + noise + JPEG + video compression + downscale) for ALL synthetic test sets.
- Code: `src/data/realbasicvsr_degrade.py` (extracted from `basicsr/data/realbasicvsr_dataset.py`)
- Params: from `configs/mgldvsr/mgldvsr_512_realbasicvsr_deg.yaml`
- Degradation is random — results won't exactly match papers but should be in the same ballpark
- VideoLQ is a real-world dataset (no synthetic degradation, no GT)

## Evaluation Tools
- **Full-reference** (synthetic datasets with GT): `experiments/baselines/evaluate_pyiqa.py` — PSNR(Y), SSIM(Y), LPIPS(RGB)
- **No-reference** (VideoLQ): `experiments/baselines/evaluate_pyiqa_nr.py` — NIQE, BRISQUE, MUSIQ, CLIPIQA
- All metrics via pyiqa library

## Upscale-A-Video (CVPR 2024) — arxiv 2312.06640

### Paper-reported targets
| Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ | E*warp ↓ |
|---------|--------|--------|---------|----------|
| YouHQ40 | 25.83 | 0.733 | 0.268 | 0.737 |
| UDM10 | 30.79 | 0.878 | 0.133 | 0.446 |
| SPMCS | 25.32 | 0.741 | 0.222 | 0.367 |
| REDS30 | 24.41 | 0.631 | 0.335 | 1.278 |

### Our results
| Experiment | Dataset | LQ source | PSNR | SSIM | LPIPS | Notes |
|------------|---------|-----------|------|------|-------|-------|
| UAV DOVE UDM10 (MP4) | UDM10 | DOVE LQ | 22.00 | 0.6101 | 0.4073 | INVALID — MP4 encoding destroyed quality |
| UAV DOVE UDM10 v2 | UDM10 | DOVE LQ | 23.22 | 0.6183 | 0.4050 | Direct frames, but wrong degradation |
| UAV YouHQ40 (bicubic) | YouHQ40 | PIL bicubic 4x | 24.47 | 0.6754 | 0.2597 | Wrong degradation (bicubic too different from RealBasicVSR) |
| UAV YouHQ40 (RealBasicVSR) | YouHQ40 | RealBasicVSR degrade | — | — | — | RUNNING in tmux `uav_youhq_rb` on GPU 6 |

### Key learnings
- MP4 encoding causes ~7 dB PSNR loss — always use direct frame I/O
- Paper uses RealBasicVSR degradation for ALL test sets, not bicubic
- YouHQ40 GT from: https://drive.google.com/file/d/1rkeBQJMqnRTRDtyLyse4k6Vg2TilvTKC/view

## MGLD-VSR (ECCV 2024) — arxiv 2312.00853

### Paper-reported targets (full-reference, synthetic datasets)
| Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | VMAF ↑ |
|---------|--------|--------|---------|---------|--------|
| UDM10 | 25.99 | 0.7548 | 0.3491 | 0.1369 | 39.39 |
| SPMCS | 22.66 | 0.5960 | 0.3990 | 0.1934 | 28.70 |
| REDS4 | 22.46 | 0.5723 | 0.3776 | 0.1151 | 34.32 |

### Paper-reported targets (no-reference, VideoLQ)
| NIQE ↓ | BRISQUE ↓ | MUSIQ ↑ | DOVER ↑ |
|--------|-----------|---------|---------|
| 3.5346 | 21.9839 | 52.7812 | 0.7481 |

### Our results
| Experiment | Dataset | LQ source | PSNR | SSIM | LPIPS | Notes |
|------------|---------|-----------|------|------|-------|-------|
| MGLD DOVE UDM10 | UDM10 | DOVE LQ | 24.60 | 0.6957 | 0.3272 | Wrong degradation |
| MGLD bicubic UDM10 | UDM10 | BIx4 (bicubic) | 29.66 | 0.8692 | 0.1584 | Too easy — wrong degradation |
| MGLD VideoLQ | VideoLQ | Real-world (no GT) | — | — | — | RUNNING on GPUs 2,4,5,7 (~12h remaining) |

## Datasets on Server
| Path | Description | Status |
|------|-------------|--------|
| `data/UDM10/` | DOVE benchmark (GT + DOVE LQ) | Keep for reference |
| `data/UDM10_bicubic/` | Original PFNL (GT + BIx4 + BDx4) | Keep, GT used for degradation |
| `data/UDM10-RealBasicVSR-LQ/` | RealBasicVSR degraded from UDM10_bicubic/GT | READY for inference |
| `data/YouHQ40-Test/` | GT frames from UAV GitHub | Keep |
| `data/YouHQ40-Test-RealBasicVSR-LQ/` | RealBasicVSR degraded from YouHQ40-Test | READY, UAV running on it |
| `data/VideoLQ/` | Real-world LQ (50 clips, no GT) | MGLD running on it |

## Currently Running (as of 2026-04-04)
| tmux session | Model | Dataset | GPU | Est. remaining |
|-------------|-------|---------|-----|----------------|
| `uav_youhq_rb` | UAV | YouHQ40 RealBasicVSR LQ | 6 | ~6-8 hours |
| `mgld_vlq_0` | MGLD-VSR | VideoLQ (clips 0,4,8,...) | 2 | ~12 hours |
| `mgld_vlq_1` | MGLD-VSR | VideoLQ (clips 1,5,9,...) | 4 | ~12 hours |
| `mgld_vlq_2` | MGLD-VSR | VideoLQ (clips 2,6,10,...) | 5 | ~12 hours |
| `mgld_vlq_3` | MGLD-VSR | VideoLQ (clips 3,7,11,...) | 7 | ~12 hours |

## Next Steps
1. Wait for UAV YouHQ40 (RealBasicVSR) to finish → evaluate with pyiqa → compare with paper (PSNR 25.83)
2. Wait for MGLD-VSR VideoLQ to finish → evaluate with NR metrics → compare with paper (NIQE 3.53, MUSIQ 52.78)
3. Run MGLD-VSR on UDM10-RealBasicVSR-LQ → compare with paper (PSNR 25.99)
4. Run UAV on UDM10-RealBasicVSR-LQ → compare with paper (PSNR 30.79)
5. Once metrics match → setup is verified → test on our own long-video dataset (>1 min)
