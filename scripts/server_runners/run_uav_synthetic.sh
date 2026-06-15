#!/bin/bash
# Run UAV inference on synthetic long videos
# Settings: n120 g6 s30 (same as DOVE benchmark)
set -euo pipefail

GPU_ID=${1:-3}
DISK2="/data/disk2/timur"
UAV_REPO="$DISK2/repos/Upscale-A-Video"
CONDA="$DISK2/miniconda3"
INPUT_DIR="$DISK2/data/synthetic_frames"
OUTPUT_DIR="$DISK2/results/uav_synthetic"

eval "$($CONDA/bin/conda shell.bash hook)"
conda activate uav

cd "$UAV_REPO"

echo "=== UAV Synthetic Long-Video Inference ==="
echo "Settings: n120 g6 s30"
echo "GPU: $GPU_ID"
echo ""

# Process videos (shortest first)
for video in hhszUXL1Cu8 BrRLKMbBTYQ KZ8p6b1zJ9U 7WHI2L_FDNg mJog8DlRk_4; do
    frames_dir="$INPUT_DIR/$video"
    nframes=$(ls "$frames_dir" | wc -l)
    echo "[$(date)] Processing: $video ($nframes frames)"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python inference_upscale_a_video.py \
        -i "$frames_dir" \
        -o "$OUTPUT_DIR/$video" \
        -n 120 -g 6 -s 30 \
        --no_llava --save_image 2>&1 | tail -5
    
    echo "[$(date)] Done: $video"
    echo ""
done

echo "=== All synthetic videos complete ==="
