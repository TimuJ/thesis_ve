#!/bin/bash
# Evaluate UAV output on DOVE UDM10 using DOVE's eval_metrics.py (RGB PSNR)
# Usage: bash eval_dove_udm10.sh
set -euo pipefail

DISK2="/data/disk2/timur"
CONDA="$DISK2/miniconda3"
DOVE_REPO="$DISK2/repos/DOVE"
GT_DIR="$DISK2/data/UDM10/GT"
PRED_DIR="$DISK2/results/uav_dove_udm10_frames"

# Use vsr env (has pyiqa, torch cu121)
eval "$($CONDA/bin/conda shell.bash hook)"
conda activate vsr

cd "$DOVE_REPO"

echo "=== DOVE Evaluation: UAV on UDM10 ==="
echo "GT:   $GT_DIR"
echo "Pred: $PRED_DIR"
echo "Metrics: psnr,ssim,lpips,dists,clipiqa (RGB, no Y-channel)"
echo ""

# Verify directories
echo "GT clips: $(ls "$GT_DIR" | wc -l)"
echo "Pred clips: $(ls "$PRED_DIR" | wc -l)"
echo ""

python eval_metrics.py \
    --gt "$GT_DIR" \
    --pred "$PRED_DIR" \
    --metrics psnr,ssim,lpips,dists,clipiqa \
    --out "$DISK2/results/uav_dove_udm10_eval"

echo ""
echo "=== Target (DOVE paper UAV): PSNR=21.72, SSIM=0.5913, LPIPS=0.4116 ==="
echo "=== Results saved to: $DISK2/results/uav_dove_udm10_eval ==="
