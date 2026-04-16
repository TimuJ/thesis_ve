#!/bin/bash
# Run UAV inference on DOVE UDM10 LQ with default settings (n120 g6 s30)
# to match DOVE benchmark exactly.
# Usage: bash run_uav_dove_disk2.sh [GPU_ID]
set -euo pipefail

GPU_ID=${1:-2}
DISK2="/data/disk2/timur"
UAV_REPO="$DISK2/repos/Upscale-A-Video"
CONDA="$DISK2/miniconda3"
LQ_DIR="$DISK2/data/UDM10/LQ"
OUTPUT_DIR="$DISK2/results/uav_dove_udm10"
FINAL_DIR="$DISK2/results/uav_dove_udm10_frames"

# Activate UAV env
eval "$($CONDA/bin/conda shell.bash hook)"
conda activate uav

cd "$UAV_REPO"

echo "=== UAV DOVE UDM10 Inference ==="
echo "Settings: n120 g6 s30 (DOVE defaults)"
echo "GPU: $GPU_ID"
echo "Input: $LQ_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Run inference for each clip
for clip_dir in "$LQ_DIR"/*/; do
    clip_name=$(basename "$clip_dir")
    echo "[$(date)] Processing clip: $clip_name"

    CUDA_VISIBLE_DEVICES=$GPU_ID python inference_upscale_a_video.py \
        -i "$clip_dir" \
        -o "$OUTPUT_DIR/$clip_name" \
        -n 120 -g 6 -s 30 \
        --no_llava --save_image 2>&1 | tail -5

    echo "[$(date)] Done: $clip_name"
    echo ""
done

echo "=== Inference complete ==="

# Restructure output for DOVE eval
# UAV saves to: {output}/{clip}/frame/{clip}_n120_g6_s30/*.png
# DOVE eval expects: {pred}/{clip}/*.png  (matching GT/{clip}/*.png)
echo "Restructuring output for DOVE eval..."
mkdir -p "$FINAL_DIR"
for clip_dir in "$OUTPUT_DIR"/*/; do
    clip_name=$(basename "$clip_dir")
    src="$clip_dir/frame/${clip_name}_n120_g6_s30"
    if [ -d "$src" ]; then
        cp -r "$src" "$FINAL_DIR/$clip_name"
        echo "  $clip_name: $(ls "$FINAL_DIR/$clip_name" | wc -l) frames"
    else
        echo "  WARNING: $src not found"
        # Try to find the actual output dir
        found=$(find "$clip_dir/frame/" -maxdepth 1 -type d | tail -1)
        if [ -n "$found" ] && [ "$found" != "$clip_dir/frame/" ]; then
            cp -r "$found" "$FINAL_DIR/$clip_name"
            echo "  $clip_name: copied from $found"
        fi
    fi
done

echo ""
echo "=== Output ready at: $FINAL_DIR ==="
echo "Run eval: bash eval_dove_udm10.sh"
