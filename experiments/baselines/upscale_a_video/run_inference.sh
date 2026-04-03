#!/bin/bash
# Run Upscale-A-Video inference on a dataset.
# Usage: bash experiments/baselines/upscale_a_video/run_inference.sh --input <LQ_dir> --output <output_dir>
#
# IMPORTANT: Uses direct frame I/O (--save_image), NOT MP4 encoding.
# MP4 encoding via ffmpeg libx264 causes ~7 dB PSNR loss due to lossy compression.
# The model natively supports image folder input and PNG output.

set -euo pipefail

# Parse args
INPUT=""
OUTPUT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --input) INPUT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --input <LQ_frames_dir> --output <output_dir>"
    echo "  LQ_frames_dir should contain per-clip subdirectories with frames"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"

# Activate conda env
if ! command -v conda &> /dev/null; then
    for p in "$HOME/miniconda3" "/data/disk1/timur/miniconda3"; do
        [[ -f "$p/bin/conda" ]] && eval "$("$p/bin/conda" shell.bash hook)" && break
    done
fi
conda activate uav

mkdir -p "$OUTPUT"

# Process each clip — feed image folders directly, save PNG frames
for clip_dir in "$INPUT"/*/; do
    clip_name=$(basename "$clip_dir")
    echo "Processing clip: $clip_name"

    cd "$REPO_DIR"
    python inference_upscale_a_video.py \
        -i "$clip_dir" \
        -o "$OUTPUT" \
        -n 150 -g 7 -s 30 --no_llava --save_image

    # UAV saves frames to $OUTPUT/frame/<clip_name>_n150_g7_s30/
    # Move to expected per-clip structure
    SAVED_DIR=$(find "$OUTPUT/frame" -maxdepth 1 -type d -name "${clip_name}*" 2>/dev/null | head -1)
    if [[ -n "$SAVED_DIR" ]]; then
        mv "$SAVED_DIR" "$OUTPUT/$clip_name"
        echo "  Saved ${clip_name} frames to $OUTPUT/$clip_name"
    else
        echo "  Warning: No frame output found for $clip_name"
    fi

    cd - > /dev/null
done

# Clean up frame dir if empty
rmdir "$OUTPUT/frame" 2>/dev/null || true

echo "Done. Results at: $OUTPUT"
