#!/bin/bash
set -eo pipefail
export TMPDIR=/data/disk2/timur/tmp

GPU=${1:-0}
DISK2="/data/disk2/timur"
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vbench

cd $DISK2/repos/VBench/VBench-2.0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p $DISK2/results/vbench2_human_test/identity_all_mgld
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
    --videos_path $DISK2/results/mgld_synthetic_mp4 \
    --dimension Human_Identity \
    --mode custom_input \
    --output_path $DISK2/results/vbench2_human_test/identity_all_mgld 2>&1 | tail -10

echo "=== UAV ==="
mkdir -p $DISK2/results/vbench2_human_test/identity_all_uav
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
    --videos_path $DISK2/results/uav_synthetic_mp4 \
    --dimension Human_Identity \
    --mode custom_input \
    --output_path $DISK2/results/vbench2_human_test/identity_all_uav 2>&1 | tail -10
