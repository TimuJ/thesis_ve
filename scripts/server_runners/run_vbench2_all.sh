#!/bin/bash
eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"
conda activate vbench
cd /data/disk2/timur/repos/VBench
export PYTHONPATH=/data/disk2/timur/repos/VBench:$PYTHONPATH

# Copy all MGLD synthetic MP4s to vbench2 input dir
mkdir -p /data/disk2/timur/results/vbench2_all_input
cp /data/disk2/timur/results/mgld_synthetic_mp4/*.mp4 /data/disk2/timur/results/vbench2_all_input/

for dim in imaging_quality motion_smoothness temporal_flickering; do
    echo === VBench 2.0: $dim ===
    CUDA_VISIBLE_DEVICES=5 python vbench2_beta_long/eval_long.py         --videos_path /data/disk2/timur/results/vbench2_all_input         --dimension $dim         --mode long_custom_input         --dev_flag         --output_path /data/disk2/timur/results/vbench2_mgld_all/$dim 2>&1
    echo $dim DONE
done
echo ALL_VBENCH2_DONE
