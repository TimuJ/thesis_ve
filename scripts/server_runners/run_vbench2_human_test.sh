#!/bin/bash
# Test VBench-2.0 human_identity + human_anatomy on our MGLD synthetic videos
set -eo pipefail

export TMPDIR=/data/disk2/timur/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR

GPU=${1:-0}
DISK2="/data/disk2/timur"
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vbench

cd $DISK2/repos/VBench/VBench-2.0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Test on hhszUXL1Cu8 only first (has people, shortest video)
mkdir -p $DISK2/test_input
cp $DISK2/results/mgld_synthetic_mp4/hhszUXL1Cu8.mp4 $DISK2/test_input/

OUT="$DISK2/results/vbench2_human_test"
mkdir -p $OUT

echo "=== Testing human_identity ==="
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
    --videos_path $DISK2/test_input \
    --dimension Human_Identity \
    --mode custom_input \
    --output_path $OUT/identity 2>&1 | tail -30
