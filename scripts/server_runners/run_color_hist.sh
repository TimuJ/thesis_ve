#!/bin/bash
set -eo pipefail
DISK2=/data/disk2/timur
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd $DISK2
export PYTHONPATH="$DISK2:${PYTHONPATH:-}"
for src_dir in \
    "results/synthetic_artefacts/color_drift:results/lr_vcc/color_histogram/color_drift" \
    "results/synthetic_artefacts/chunk_boundary:results/lr_vcc/color_histogram/chunk_boundary" \
    "results/mgld_synthetic_mp4:results/lr_vcc/color_histogram/mgld" \
    "results/uav_synthetic_mp4:results/lr_vcc/color_histogram/uav"; do
    SRC=$(echo $src_dir | cut -d: -f1)
    OUT=$(echo $src_dir | cut -d: -f2)
    mkdir -p $DISK2/$OUT
    echo "=== $SRC -> $OUT ==="
    python -m scripts.lr_vcc.compute_color_histogram \
        --videos_path $DISK2/$SRC \
        --output_path $DISK2/$OUT
done
echo DONE
