#!/bin/bash
eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"
conda activate vbench
cd /data/disk2/timur/repos/VBench
export PYTHONPATH=/data/disk2/timur/repos/VBench:$PYTHONPATH

MGLD_IN=/data/disk2/timur/results/vbench2_all_input
MGLD_OUT=/data/disk2/timur/results/vbench2_mgld_all
LQ_IN=/data/disk2/timur/results/vbench2_lq_input
LQ_OUT=/data/disk2/timur/results/vbench2_lq_all

for dim in subject_consistency background_consistency; do
    for label_in_out in "MGLD $MGLD_IN $MGLD_OUT" "LQ $LQ_IN $LQ_OUT"; do
        set -- $label_in_out
        label=$1; input=$2; output=$3
        echo === $label $dim ===
        bash /data/disk2/timur/run_vbench_dim.sh $dim $input $output 5 2>&1
    done
done
echo CONSISTENCY_ALL_DONE
