#!/bin/bash
set -eo pipefail
GPU=${1:-6}
DISK2="/data/disk2/timur"
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd $DISK2/repos/DOVER

mkdir -p $DISK2/results/uav_synthetic_eval/dover
CUDA_VISIBLE_DEVICES=$GPU python evaluate_a_set_of_videos.py \
    --input_video_dir $DISK2/results/uav_synthetic_mp4 \
    --output_result_csv $DISK2/results/uav_synthetic_eval/dover/results.csv 2>&1
