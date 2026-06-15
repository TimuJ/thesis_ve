#!/bin/bash
eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd /data/disk2/timur/repos/DOVE

echo === NR metrics on MGLD synthetic ===
for vid in 7WHI2L_FDNg BrRLKMbBTYQ KZ8p6b1zJ9U hhszUXL1Cu8 mJog8DlRk_4; do
    echo === $vid ===
    python eval_metrics.py         --pred /data/disk2/timur/results/mgld_synthetic/$vid         --metrics clipiqa,musiq,niqe,brisque         --out /data/disk2/timur/results/mgld_synthetic_eval/${vid}_nr 2>&1 | tail -8
done
echo NR_EVAL_DONE
