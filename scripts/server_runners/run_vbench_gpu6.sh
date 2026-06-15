#!/bin/bash
MGLD_IN=/data/disk2/timur/results/vbench2_all_input
MGLD_OUT=/data/disk2/timur/results/vbench2_mgld_all
LQ_IN=/data/disk2/timur/results/vbench2_lq_input
LQ_OUT=/data/disk2/timur/results/vbench2_lq_all

for dim in aesthetic_quality dynamic_degree color scene appearance_style temporal_style; do
    echo === GPU6 MGLD $dim ===
    bash /data/disk2/timur/run_vbench_dim.sh $dim $MGLD_IN $MGLD_OUT 6 2>&1 | tee /data/disk2/timur/vbench_mgld_${dim}.log
done

for dim in aesthetic_quality dynamic_degree color scene appearance_style temporal_style; do
    echo === GPU6 LQ $dim ===
    bash /data/disk2/timur/run_vbench_dim.sh $dim $LQ_IN $LQ_OUT 6 2>&1 | tee /data/disk2/timur/vbench_lq_${dim}.log
done

echo GPU6_ALL_DONE
