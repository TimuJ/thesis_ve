#!/bin/bash
eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"
conda activate vbench
cd /data/disk2/timur/repos/VBench
export PYTHONPATH=/data/disk2/timur/repos/VBench:$PYTHONPATH

ALL_DIMS="subject_consistency background_consistency temporal_flickering motion_smoothness aesthetic_quality imaging_quality dynamic_degree object_class multiple_objects human_action color spatial_relationship scene appearance_style temporal_style overall_consistency"

for dim in $ALL_DIMS; do
    # Check if already done
    existing=$(find /data/disk2/timur/results/vbench2_mgld_all/$dim -name '*eval_results*' 2>/dev/null | wc -l)
    if [ "$existing" -gt 0 ]; then
        echo SKIP MGLD $dim — already done
        continue
    fi
    mkdir -p /data/disk2/timur/results/vbench2_mgld_all/$dim
    echo === MGLD-SR: $dim ===
    CUDA_VISIBLE_DEVICES=5 python vbench2_beta_long/eval_long.py         --videos_path /data/disk2/timur/results/vbench2_all_input         --dimension $dim         --mode long_custom_input         --dev_flag         --output_path /data/disk2/timur/results/vbench2_mgld_all/$dim 2>&1 | tail -5
    echo $dim DONE
done

for dim in $ALL_DIMS; do
    existing=$(find /data/disk2/timur/results/vbench2_lq_all/$dim -name '*eval_results*' 2>/dev/null | wc -l)
    if [ "$existing" -gt 0 ]; then
        echo SKIP LQ $dim — already done
        continue
    fi
    mkdir -p /data/disk2/timur/results/vbench2_lq_all/$dim
    echo === LQ: $dim ===
    CUDA_VISIBLE_DEVICES=5 python vbench2_beta_long/eval_long.py         --videos_path /data/disk2/timur/results/vbench2_lq_input         --dimension $dim         --mode long_custom_input         --dev_flag         --output_path /data/disk2/timur/results/vbench2_lq_all/$dim 2>&1 | tail -5
    echo $dim DONE
done

echo ALL_16_DIMS_DONE
