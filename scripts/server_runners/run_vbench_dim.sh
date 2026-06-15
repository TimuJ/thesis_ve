#!/bin/bash
DIM=$1
INPUT=$2
OUTPUT=$3
GPU=$4

eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"
conda activate vbench
cd /data/disk2/timur/repos/VBench
export PYTHONPATH=/data/disk2/timur/repos/VBench:$PYTHONPATH

existing=$(find $OUTPUT/$DIM -name '*eval_results*' 2>/dev/null | wc -l)
if [ "$existing" -gt 0 ]; then
    echo SKIP $DIM — already done
    exit 0
fi

mkdir -p $OUTPUT/$DIM
echo === $DIM ===
CUDA_VISIBLE_DEVICES=$GPU python vbench2_beta_long/eval_long.py     --videos_path $INPUT     --dimension $DIM     --mode long_custom_input     --dev_flag     --output_path $OUTPUT/$DIM 2>&1
echo $DIM DONE
