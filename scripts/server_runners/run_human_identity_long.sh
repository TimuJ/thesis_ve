#!/bin/bash
set -eo pipefail
export TMPDIR=/data/disk2/timur/tmp

GPU=${1:-0}
INPUT=${2:-/data/disk2/timur/results/mgld_synthetic_mp4}
OUT=${3:-/data/disk2/timur/results/vbench2_human_test/identity_long_mgld}

DISK2="/data/disk2/timur"
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vbench

CUDA_VISIBLE_DEVICES=$GPU python /data/disk2/timur/human_identity_long.py \
    --videos_path "$INPUT" \
    --output_path "$OUT" \
    --w_slow 0.5 --w_fast 0.5 \
    --clip_duration 2.0 2>&1
