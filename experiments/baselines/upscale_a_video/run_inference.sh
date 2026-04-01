#!/bin/bash
# Run Upscale-A-Video inference on a dataset.
# Usage: bash experiments/baselines/upscale_a_video/run_inference.sh --input <LQ_dir> --output <output_dir>

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
TEMP_DIR="$SCRIPT_DIR/.tmp_videos"

# Activate conda env
if ! command -v conda &> /dev/null; then
    for p in "$HOME/miniconda3" "/data/disk1/timur/miniconda3"; do
        [[ -f "$p/bin/conda" ]] && eval "$("$p/bin/conda" shell.bash hook)" && break
    done
fi
conda activate uav

mkdir -p "$OUTPUT"
mkdir -p "$TEMP_DIR/input"
mkdir -p "$TEMP_DIR/output"

# Process each clip
for clip_dir in "$INPUT"/*/; do
    clip_name=$(basename "$clip_dir")
    echo "Processing clip: $clip_name"

    clip_output="$OUTPUT/$clip_name"
    mkdir -p "$clip_output"

    # Assemble frames into video (Upscale-A-Video expects video input)
    input_video="$TEMP_DIR/input/${clip_name}.mp4"
    ffmpeg -y -i "$clip_dir/%08d.png" -c:v libx264 -pix_fmt yuv420p "$input_video" 2>/dev/null

    # Run inference
    # Note: verify actual CLI flags from repo's argparse before first run.
    # Flags below are from the repo README; if they fail, check:
    #   python inference_upscale_a_video.py --help
    cd "$REPO_DIR"
    python inference_upscale_a_video.py \
        -i "$input_video" \
        -o "$TEMP_DIR/output" \
        -n 150 -g 7 -s 30

    # Extract output frames back to per-clip dir
    output_video=$(find "$TEMP_DIR/output" -name "*.mp4" -newer "$input_video" | head -1)
    if [[ -n "$output_video" ]]; then
        ffmpeg -y -i "$output_video" "$clip_output/%08d.png" 2>/dev/null
        echo "  Saved ${clip_name} frames to $clip_output"
    else
        echo "  Warning: No output video found for $clip_name"
    fi

    # Cleanup temp
    rm -f "$input_video"
    rm -f "$TEMP_DIR/output"/*.mp4
    cd - > /dev/null
done

rm -rf "$TEMP_DIR"
echo "Done. Results at: $OUTPUT"
