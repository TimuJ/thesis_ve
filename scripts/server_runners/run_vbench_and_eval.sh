#!/bin/bash
eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"

echo === Step 1: Fix setuptools for CLIP-IQA ===
conda activate vsr
pip install 'setuptools<81' 2>&1 | tail -2

echo === Step 2: MGLD synthetic NR eval ===
cd /data/disk2/timur/repos/DOVE
mkdir -p /data/disk2/timur/results/mgld_synthetic_eval
for vid in 7WHI2L_FDNg BrRLKMbBTYQ KZ8p6b1zJ9U hhszUXL1Cu8 mJog8DlRk_4; do
    echo NR eval: $vid
    python eval_metrics.py         --pred /data/disk2/timur/results/mgld_synthetic/$vid         --metrics clipiqa         --out /data/disk2/timur/results/mgld_synthetic_eval/$vid 2>&1 | tail -5
done

echo === Step 3: VBench 2.0 long test ===
conda activate vbench
pip install 'setuptools<81' 2>&1 | tail -2
pip install 'scenedetect[opencv]' 2>&1 | tail -2
cd /data/disk2/timur/repos/VBench
CUDA_VISIBLE_DEVICES=2 python vbench2_beta_long/eval_long.py     --videos_path /data/disk2/timur/results/mgld_synthetic_mp4/hhszUXL1Cu8.mp4     --dimension imaging_quality     --mode long_custom_input     --dev_flag     --output_path /data/disk2/timur/results/vbench2_mgld 2>&1

echo ALL_DONE
