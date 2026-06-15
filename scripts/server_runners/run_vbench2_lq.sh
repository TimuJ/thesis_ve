#!/bin/bash
eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"
conda activate vbench
cd /data/disk2/timur/repos/VBench
export PYTHONPATH=/data/disk2/timur/repos/VBench:$PYTHONPATH

mkdir -p /data/disk2/timur/results/vbench2_lq_input
cp /data/disk2/timur/synthetic_data/synthetic/*.mp4 /data/disk2/timur/results/vbench2_lq_input/

for dim in imaging_quality motion_smoothness temporal_flickering; do
    echo === VBench 2.0 LQ: $dim ===
    CUDA_VISIBLE_DEVICES=5 python vbench2_beta_long/eval_long.py         --videos_path /data/disk2/timur/results/vbench2_lq_input         --dimension $dim         --mode long_custom_input         --dev_flag         --output_path /data/disk2/timur/results/vbench2_lq_all/$dim 2>&1
    echo $dim DONE
done
echo ALL_VBENCH2_LQ_DONE
