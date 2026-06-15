#!/bin/bash
set -eo pipefail

GPU_ID=${1:-1}
DISK2="/data/disk2/timur"
CONDA="$DISK2/miniconda3"
DOVER_REPO="$DISK2/repos/DOVER"

eval "$($CONDA/bin/conda shell.bash hook)"
conda activate vsr

cd "$DOVER_REPO"

echo "=== DOVER Evaluation ==="
echo "GPU: $GPU_ID"

# MGLD-SR
echo ""
echo "=== MGLD-SR Synthetic ==="
mkdir -p "$DISK2/results/mgld_synthetic_eval/dover"
CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_a_set_of_videos.py \
    --input_video_dir "$DISK2/results/mgld_synthetic_mp4" \
    --output_result_csv "$DISK2/results/mgld_synthetic_eval/dover/results.csv" 2>&1

# LQ
echo ""
echo "=== LQ Baselines ==="
mkdir -p "$DISK2/results/lq_synthetic_eval/dover"
CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_a_set_of_videos.py \
    --input_video_dir "$DISK2/synthetic_data/synthetic" \
    --output_result_csv "$DISK2/results/lq_synthetic_eval/dover/results.csv" 2>&1

echo ""
echo "=== DOVER Done ==="
