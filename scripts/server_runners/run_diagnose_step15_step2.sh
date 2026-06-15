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

ANATOMY_OUT=$DISK2/results/vbench2_human_test/diagnostic
IDENTITY_OUT=$DISK2/results/vbench2_human_test/diagnostic_identity
mkdir -p $ANATOMY_OUT $IDENTITY_OUT

echo "=== Step 1.5: per-frame anatomy on hhszUXL1Cu8 (MGLD-wins reference) ==="
for method in mgld uav; do
    echo "--- $method ---"
    CUDA_VISIBLE_DEVICES=$GPU python diagnose_anatomy_per_frame.py \
        --video $DISK2/results/${method}_synthetic_mp4/hhszUXL1Cu8.mp4 \
        --output $ANATOMY_OUT/${method}_hhszUXL1Cu8_per_frame.json
done

echo "=== Step 2: per-clip identity slow-fast on KZ8p6b1zJ9U ==="
mkdir -p $DISK2/results/diagnostic_identity_kz_mgld $DISK2/results/diagnostic_identity_kz_uav

# stage just the one video into per-method dirs (the adapter walks a videos_path dir)
for method in mgld uav; do
    STAGE=$DISK2/tmp/diag_kz_${method}
    rm -rf $STAGE && mkdir -p $STAGE
    cp $DISK2/results/${method}_synthetic_mp4/KZ8p6b1zJ9U.mp4 $STAGE/
    CUDA_VISIBLE_DEVICES=$GPU python human_identity_long.py \
        --videos_path $STAGE \
        --output_path $IDENTITY_OUT/${method}_kz \
        --save_clip_detail
    rm -rf $STAGE
done

echo "DONE"
