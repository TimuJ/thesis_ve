#!/bin/bash
# Convert UAV synthetic frames to MP4 for evaluation
set -eo pipefail

OUTDIR="/data/disk2/timur/results/uav_synthetic_mp4"
mkdir -p "$OUTDIR"

for d in /data/disk2/timur/results/uav_synthetic/*/; do
    name=$(basename "$d")
    if [ -f "$OUTDIR/$name.mp4" ]; then
        echo "Skipping $name (already exists)"
        continue
    fi

    # Find frame dir
    frame_subdir=$(find "$d/frame" -maxdepth 1 -type d ! -path "$d/frame" | head -1)
    if [ -z "$frame_subdir" ]; then
        echo "No frames for $name, skipping"
        continue
    fi

    n_frames=$(ls "$frame_subdir" | wc -l)
    echo "Converting $name ($n_frames frames) -> $OUTDIR/$name.mp4"
    ffmpeg -y -framerate 24 -i "$frame_subdir/%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUTDIR/$name.mp4" 2>&1 | tail -3
done
echo "Done"
ls -lh "$OUTDIR"
