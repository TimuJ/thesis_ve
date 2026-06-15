#!/bin/bash
set -eo pipefail
export TMPDIR=/data/disk2/timur/tmp
export VBENCH2_CACHE_DIR=/data/disk2/timur/cache/vbench2

GPU=${1:-0}
DISK2="/data/disk2/timur"
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vbench

cd $DISK2/repos/VBench/VBench-2.0
export PYTHONPATH="$PWD:$DISK2/repos/YOLO-World:${PYTHONPATH:-}"

OUTDIR=$DISK2/results/vbench2_human_test/diagnostic
mkdir -p $OUTDIR

for method in mgld uav; do
    echo "=== $method ==="
    CUDA_VISIBLE_DEVICES=$GPU python diagnose_anatomy_per_frame.py \
        --video $DISK2/results/${method}_synthetic_mp4/KZ8p6b1zJ9U.mp4 \
        --output $OUTDIR/${method}_KZ8p6b1zJ9U_per_frame.json
done
echo "DONE"
