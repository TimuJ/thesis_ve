#!/bin/bash
set -eo pipefail
GPU=${1:-1}
DISK2="/data/disk2/timur"
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd $DISK2/repos/DOVE/finetune/scripts

CUDA_VISIBLE_DEVICES=$GPU python eval_ewarp.py \
    --pred $DISK2/results/uav_synthetic_mp4 \
    --model models/raft-things.pth \
    --out $DISK2/results/uav_synthetic_eval/ewarp 2>&1
