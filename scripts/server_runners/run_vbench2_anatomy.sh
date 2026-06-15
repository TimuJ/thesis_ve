#!/bin/bash
set -eo pipefail
export TMPDIR=/data/disk2/timur/tmp
export VBENCH2_CACHE_DIR=/data/disk2/timur/cache/vbench2

GPU=${1:-0}
DISK2="/data/disk2/timur"
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vbench

cd $DISK2/repos/VBench/VBench-2.0
export PYTHONPATH="$PWD:$DISK2/repos/YOLO-World:${PYTHONPATH:-}"

echo "=== MGLD-SR ==="
mkdir -p $DISK2/results/vbench2_human_test/anatomy_all_mgld
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
    --videos_path $DISK2/results/mgld_synthetic_mp4 \
    --dimension Human_Anatomy \
    --mode custom_input \
    --output_path $DISK2/results/vbench2_human_test/anatomy_all_mgld 2>&1 | tail -15

echo "=== UAV ==="
mkdir -p $DISK2/results/vbench2_human_test/anatomy_all_uav
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
    --videos_path $DISK2/results/uav_synthetic_mp4 \
    --dimension Human_Anatomy \
    --mode custom_input \
    --output_path $DISK2/results/vbench2_human_test/anatomy_all_uav 2>&1 | tail -15
