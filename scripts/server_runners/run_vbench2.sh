#!/bin/bash
eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"
conda activate vbench
cd /data/disk2/timur/repos/VBench
export PYTHONPATH=/data/disk2/timur/repos/VBench:$PYTHONPATH
CUDA_VISIBLE_DEVICES=5 python vbench2_beta_long/eval_long.py     --videos_path /data/disk2/timur/results/vbench2_input     --dimension imaging_quality     --mode long_custom_input     --dev_flag     --output_path /data/disk2/timur/results/vbench2_mgld 2>&1
echo VBENCH2_DONE
