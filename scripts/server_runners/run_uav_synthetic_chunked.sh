#!/bin/bash
# Run UAV on synthetic videos in chunks to avoid OOM
# Splits input frames into temp dirs of CHUNK_SIZE, runs UAV on each, reassembles
set -eo pipefail

GPU_ID=${1:-7}
CHUNK_SIZE=${2:-2000}
DISK2="/data/disk2/timur"
UAV_REPO="$DISK2/repos/Upscale-A-Video"
CONDA="$DISK2/miniconda3"
INPUT_DIR="$DISK2/data/synthetic_frames"
OUTPUT_DIR="$DISK2/results/uav_synthetic"

eval "$($CONDA/bin/conda shell.bash hook)"
conda activate uav
cd "$UAV_REPO"

echo "=== UAV Synthetic Chunked Inference ==="
echo "Settings: n120 g6 s30, chunk=$CHUNK_SIZE"
echo "GPU: $GPU_ID"
echo ""

for video in BrRLKMbBTYQ KZ8p6b1zJ9U 7WHI2L_FDNg mJog8DlRk_4; do
    src="$INPUT_DIR/$video"
    nframes=$(ls "$src" | wc -l)
    echo "[$(date)] Processing: $video ($nframes frames, chunk=$CHUNK_SIZE)"

    # Create output dirs
    mkdir -p "$OUTPUT_DIR/$video/frame/${video}_n120_g6_s30"
    mkdir -p "$OUTPUT_DIR/$video/video"

    # Split into chunks
    all_frames=($(ls "$src" | sort))
    total=${#all_frames[@]}
    chunk_idx=0

    for ((start=0; start<total; start+=CHUNK_SIZE)); do
        chunk_idx=$((chunk_idx + 1))
        end=$((start + CHUNK_SIZE))
        if [ $end -gt $total ]; then end=$total; fi
        chunk_count=$((end - start))

        echo "  Chunk $chunk_idx: frames $start-$((end-1)) ($chunk_count frames)"

        # Create temp dir with chunk frames
        tmp_dir=$(mktemp -d)
        idx=0
        for ((i=start; i<end; i++)); do
            ln -s "$src/${all_frames[$i]}" "$tmp_dir/$(printf '%04d' $idx).png"
            idx=$((idx + 1))
        done

        # Run UAV on chunk
        CUDA_VISIBLE_DEVICES=$GPU_ID python inference_upscale_a_video.py \
            -i "$tmp_dir" \
            -o "$OUTPUT_DIR/$video/chunk_${chunk_idx}" \
            -n 120 -g 6 -s 30 \
            --no_llava --save_image 2>&1 | tail -3

        # Move output frames with correct numbering
        chunk_out="$OUTPUT_DIR/$video/chunk_${chunk_idx}/frame"
        chunk_subdir=$(find "$chunk_out" -maxdepth 1 -type d | tail -1)
        if [ -d "$chunk_subdir" ] && [ "$chunk_subdir" != "$chunk_out" ]; then
            for f in "$chunk_subdir"/*.png; do
                fname=$(basename "$f")
                # Convert chunk-local index back to global index
                local_idx=$(echo "$fname" | sed 's/\.png//' | sed 's/^0*//' )
                [ -z "$local_idx" ] && local_idx=0
                global_idx=$((start + local_idx))
                cp "$f" "$OUTPUT_DIR/$video/frame/${video}_n120_g6_s30/$(printf '%04d' $global_idx).png"
            done
        fi

        # Cleanup
        rm -rf "$tmp_dir" "$OUTPUT_DIR/$video/chunk_${chunk_idx}"

        echo "  Chunk $chunk_idx done"
    done

    echo "[$(date)] Done: $video ($(ls "$OUTPUT_DIR/$video/frame/${video}_n120_g6_s30/" | wc -l) frames)"
    echo ""
done

echo "=== All synthetic videos complete ==="
