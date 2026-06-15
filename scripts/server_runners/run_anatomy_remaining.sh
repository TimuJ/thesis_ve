#!/bin/bash
set -eo pipefail
export TMPDIR=/data/disk2/timur/tmp
export VBENCH2_CACHE_DIR=/data/disk2/timur/cache/vbench2
DISK2=/data/disk2/timur
GPU=${1:-0}
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vbench
cd $DISK2/repos/VBench/VBench-2.0
export PYTHONPATH="$PWD:$DISK2/repos/YOLO-World:${PYTHONPATH:-}"
OUT=$DISK2/results/vbench2_human_test/diagnostic
mkdir -p $OUT

# 3 remaining videos x 2 methods = 6 runs
for v in 7WHI2L_FDNg BrRLKMbBTYQ mJog8DlRk_4; do
    for method in mgld uav; do
        if [ -f $OUT/${method}_${v}_per_frame.json ]; then
            echo "[skip] $OUT/${method}_${v}_per_frame.json exists"
            continue
        fi
        echo "=== ${method} ${v} ==="
        CUDA_VISIBLE_DEVICES=$GPU python diagnose_anatomy_per_frame.py \
            --video $DISK2/results/${method}_synthetic_mp4/${v}.mp4 \
            --output $OUT/${method}_${v}_per_frame.json
    done
done
echo DONE
