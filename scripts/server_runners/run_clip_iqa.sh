#!/bin/bash
set -eo pipefail
DISK2=/data/disk2/timur
GPU=${1:-0}
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd $DISK2
for method in mgld uav; do
    OUT=$DISK2/results/lr_vcc/clip_iqa/${method}
    mkdir -p $OUT
    echo "=== ${method} ==="
    CUDA_VISIBLE_DEVICES=$GPU python compute_clip_iqa.py \
        --videos_path $DISK2/results/${method}_synthetic_mp4 \
        --output_path $OUT
done
echo DONE
