#!/bin/bash
set -eo pipefail
GPU=${1:-7}
DISK2="/data/disk2/timur"
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vbench
export PYTHONPATH="$DISK2/repos/VBench:${PYTHONPATH:-}"

for dim in imaging_quality motion_smoothness temporal_flickering aesthetic_quality dynamic_degree subject_consistency background_consistency; do
    echo "=== $dim ==="
    CUDA_VISIBLE_DEVICES=$GPU python $DISK2/repos/VBench/vbench2_beta_long/eval_long.py \
        --output_path $DISK2/results/vbench2_uav_all/$dim \
        --full_json_dir $DISK2/repos/VBench/vbench2_beta_long/VBench_full_info.json \
        --videos_path $DISK2/results/uav_synthetic_mp4 \
        --dimension $dim \
        --mode long_custom_input 2>&1 | tail -5
    echo "$dim DONE"
done
echo "ALL DONE"
