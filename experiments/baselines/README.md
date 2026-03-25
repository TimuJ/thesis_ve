# VSR Baseline Inference

Reproducible baseline inference for diffusion-based VSR methods on DOVE benchmarks.

## Quick Start

### 1. Download dataset

```bash
bash experiments/baselines/data/download_dove.sh UDM10
```

### 2. Setup models (one-time per model)

```bash
bash experiments/baselines/upscale_a_video/setup.sh
bash experiments/baselines/mgld_vsr/setup.sh
```

### 3. Run inference

```bash
bash experiments/baselines/upscale_a_video/run_inference.sh \
    --input experiments/baselines/data/UDM10/LQ \
    --output experiments/baselines/results/upscale_a_video/UDM10

bash experiments/baselines/mgld_vsr/run_inference.sh \
    --input experiments/baselines/data/UDM10/LQ \
    --output experiments/baselines/results/mgld_vsr/UDM10
```

### 4. Evaluate

```bash
python experiments/baselines/evaluate.py \
    --results experiments/baselines/results/upscale_a_video/UDM10 \
    --gt experiments/baselines/data/UDM10/GT \
    --output experiments/baselines/results/upscale_a_video/UDM10_metrics.json

python experiments/baselines/evaluate.py \
    --results experiments/baselines/results/mgld_vsr/UDM10 \
    --gt experiments/baselines/data/UDM10/GT \
    --output experiments/baselines/results/mgld_vsr/UDM10_metrics.json
```

## Models

| Model | Paper | Env | Notes |
|-------|-------|-----|-------|
| Upscale-A-Video | CVPR 2024 | `uav` | Diffusion + text prompts, expects video input |
| MGLD-VSR | ECCV 2024 | `mgldvsr` | Motion-guided latent diffusion |

## Adding a new model

1. Create `experiments/baselines/<model_name>/setup.sh` and `run_inference.sh`
2. Follow the same `--input <LQ_dir> --output <output_dir>` interface
3. Output frames as per-clip subdirectories with PNG files

## Adding a new dataset

1. Download into `experiments/baselines/data/<dataset>/` with `GT/` and `LQ/` subdirs
2. Run the same inference and evaluate commands pointing to the new paths
