#!/bin/bash
eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"
conda activate vbench
cd /data/disk2/timur/repos/VBench
export PYTHONPATH=/data/disk2/timur/repos/VBench:$PYTHONPATH

# Run remaining Quality Score dimensions on MGLD-SR
for dim in subject_consistency background_consistency aesthetic_quality dynamic_degree; do
    echo === MGLD-SR: $dim ===
    CUDA_VISIBLE_DEVICES=5 python vbench2_beta_long/eval_long.py         --videos_path /data/disk2/timur/results/vbench2_all_input         --dimension $dim         --mode long_custom_input         --dev_flag         --output_path /data/disk2/timur/results/vbench2_mgld_all/$dim 2>&1
    echo $dim DONE
done

# Run same on LQ
for dim in subject_consistency background_consistency aesthetic_quality dynamic_degree; do
    echo === LQ: $dim ===
    CUDA_VISIBLE_DEVICES=5 python vbench2_beta_long/eval_long.py         --videos_path /data/disk2/timur/results/vbench2_lq_input         --dimension $dim         --mode long_custom_input         --dev_flag         --output_path /data/disk2/timur/results/vbench2_lq_all/$dim 2>&1
    echo $dim DONE
done

echo ALL_QUALITY_DIMS_DONE
