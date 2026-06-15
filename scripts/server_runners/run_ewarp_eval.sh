#!/bin/bash
# Run E*warp evaluation on MGLD synthetic videos + LQ baselines
set -eo pipefail

GPU_ID=${1:-0}
DISK2="/data/disk2/timur"
CONDA="$DISK2/miniconda3"
DOVE_SCRIPTS="$DISK2/repos/DOVE/finetune/scripts"

eval "$($CONDA/bin/conda shell.bash hook)"
conda activate vsr

cd "$DOVE_SCRIPTS"

echo "=== E*warp Evaluation ==="
echo "GPU: $GPU_ID"

# MGLD-SR synthetic
echo ""
echo "=== MGLD-SR Synthetic ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python eval_ewarp.py \
    --pred "$DISK2/results/mgld_synthetic_mp4" \
    --model models/raft-things.pth \
    --out "$DISK2/results/mgld_synthetic_eval/ewarp" 2>&1

# LQ baselines
echo ""
echo "=== LQ Baselines ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python eval_ewarp.py \
    --pred "$DISK2/synthetic_data/synthetic" \
    --model models/raft-things.pth \
    --out "$DISK2/results/lq_synthetic_eval/ewarp" 2>&1

echo ""
echo "=== Done ==="
