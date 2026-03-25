# Baseline Inference Infrastructure Design

**Date:** 2026-03-25
**Status:** Approved
**Goal:** Run reproducible baseline inference for two diffusion-based VSR methods on UDM10 (4x), producing consistent metrics that can be compared against future experiments on custom long-form datasets.

## Context

PhD mentor identified metric inconsistency across papers as a key problem — same datasets and metrics yield different numbers due to varying LQ degradation seeds. DOVE standardizes this by providing fixed LQ inputs. Our baseline task: run Upscale-A-Video and MGLD-VSR on DOVE's UDM10 dataset and compute metrics uniformly.

These baselines will serve as reference numbers when evaluating our own method on both UDM10 and a future long-form benchmark (>1000 frames).

## Approach

Self-contained inference scripts per model with a shared evaluation script. Each model gets its own conda environment and setup/inference scripts. Evaluation is centralized through `src/evaluation/` to ensure consistent metric computation across all models.

## Directory Structure

```
experiments/
├── PhD_mentor_reports/                    # moved from repo root
│   └── Week 1 Feb 23 - Mar 4/
└── baselines/
    ├── data/
    │   ├── download_dove.sh               # downloads UDM10 from Google Drive via gdown
    │   ├── .gitignore                     # ignores downloaded dataset files
    │   └── README.md                      # manual download fallback instructions
    ├── upscale_a_video/
    │   ├── setup.sh                       # clone, conda env, checkpoints
    │   └── run_inference.sh               # inference with --input/--output interface
    ├── mgld_vsr/
    │   ├── setup.sh                       # clone, conda env, checkpoints
    │   └── run_inference.sh               # inference with --input/--output interface
    ├── results/                           # gitignored, inference outputs land here
    │   ├── upscale_a_video/UDM10/
    │   └── mgld_vsr/UDM10/
    ├── evaluate.py                        # shared evaluation script
    └── README.md                          # end-to-end workflow documentation
```

## Dataset (DOVE UDM10)

- **Source:** DOVE benchmark Google Drive
- **Contents:** 10 video clips, each with GT (ground truth HR) and LQ (4x degraded) frames
- **Structure after download:**
  ```
  baselines/data/UDM10/
  ├── GT/          # per-clip subdirs with HR frames
  └── LQ/          # per-clip subdirs with LQ frames
  ```
- **Download script** uses `gdown` Python package for Google Drive access
- `.gitignore` ensures dataset files are never committed

## Model: Upscale-A-Video

- **Paper:** CVPR 2024, diffusion-based VSR with text prompts for temporal consistency
- **Repo:** https://github.com/sczhou/Upscale-A-Video
- **Environment:** conda `uav`, Python 3.9

### setup.sh
1. Clone repo into `baselines/upscale_a_video/repo/`
2. Create conda env `uav` (Python 3.9)
3. `pip install -r requirements.txt`
4. Download pretrained checkpoints from Google Drive into `repo/pretrained_models/`

### run_inference.sh
- **Interface:** `--input <LQ_dir> --output <output_dir>`
- Loops over each clip subdirectory in input
- Calls `inference_upscale_a_video.py` with params: 150 steps, guidance 7, seed 30
- Saves output frames to `--output` matching GT directory structure (per-clip subdirs)

## Model: MGLD-VSR

- **Paper:** ECCV 2024, motion-guided latent diffusion for real-world VSR
- **Repo:** https://github.com/IanYeung/MGLD-VSR
- **Environment:** conda `mgldvsr`

### setup.sh
1. Clone repo into `baselines/mgld_vsr/repo/`
2. Create conda env from `environment.yaml`
3. Install additional deps: xformers, mmcv, taming-transformers, CLIP
4. Download checkpoints from HuggingFace (`ianyeung/MGLD-VSR`)

### run_inference.sh
- **Interface:** `--input <LQ_dir> --output <output_dir>`
- Calls `vsr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py` with config for 4x SR
- Params: 50 DDPM steps, dec_w 1.0, adain color fix
- Saves output frames to `--output` matching GT directory structure

## Shared Evaluation (`baselines/evaluate.py`)

### Interface
```bash
python baselines/evaluate.py \
  --results baselines/results/<model>/UDM10 \
  --gt baselines/data/UDM10/GT \
  --output baselines/results/<model>/UDM10_metrics.json
```

### Behavior
1. Walks GT directory, finds per-clip subdirs with ordered frames
2. Matches against results directory (same clip names)
3. Loads frame pairs as numpy arrays
4. Computes per-frame metrics via `src.evaluation.metrics` (PSNR, SSIM) + LPIPS
5. Aggregates per-clip averages and overall dataset averages

### Metrics
- **PSNR** — via `src.evaluation.metrics.psnr` (already implemented)
- **SSIM** — via `src.evaluation.metrics.ssim` (already implemented)
- **LPIPS** — to be added to `src/evaluation/metrics.py` (requires `lpips` package, GPU)

### Output format
```json
{
  "model": "upscale_a_video",
  "dataset": "UDM10",
  "overall": {"PSNR": 27.13, "SSIM": 0.843, "LPIPS": 0.190},
  "per_clip": {
    "clip_001": {"PSNR": 28.1, "SSIM": 0.87, "LPIPS": 0.15},
    "clip_002": {"PSNR": 26.5, "SSIM": 0.82, "LPIPS": 0.21}
  }
}
```

## Code Changes to `src/`

- **`src/evaluation/metrics.py`** — add `lpips_score(pred, gt)` function, graceful degradation when GPU unavailable
- **`requirements-gpu.txt`** — add `lpips` package

## .gitignore Additions

```
baselines/data/UDM10/
baselines/results/
baselines/upscale_a_video/repo/
baselines/mgld_vsr/repo/
```

## End-to-End Workflow

1. **Download dataset:** `bash baselines/data/download_dove.sh`
2. **Setup models (one-time):**
   - `bash baselines/upscale_a_video/setup.sh`
   - `bash baselines/mgld_vsr/setup.sh`
3. **Run inference:**
   - `bash baselines/upscale_a_video/run_inference.sh --input baselines/data/UDM10/LQ --output baselines/results/upscale_a_video/UDM10`
   - `bash baselines/mgld_vsr/run_inference.sh --input baselines/data/UDM10/LQ --output baselines/results/mgld_vsr/UDM10`
4. **Evaluate:**
   - `python baselines/evaluate.py --results baselines/results/upscale_a_video/UDM10 --gt baselines/data/UDM10/GT --output baselines/results/upscale_a_video/UDM10_metrics.json`
   - `python baselines/evaluate.py --results baselines/results/mgld_vsr/UDM10 --gt baselines/data/UDM10/GT --output baselines/results/mgld_vsr/UDM10_metrics.json`

## Adding New Models/Datasets

- **New model:** Create `baselines/<model_name>/setup.sh` + `run_inference.sh` following the same `--input`/`--output` interface
- **New dataset:** Download into `baselines/data/<dataset>/` with `GT/` and `LQ/` subdirs, run same scripts pointing to new paths

## Target Environment

- **Machine:** ZJU lab, A800/A100 GPUs (80GB VRAM)
- **Access:** SSH from Ubuntu laptop
- **No Docker** available — conda environments for isolation
