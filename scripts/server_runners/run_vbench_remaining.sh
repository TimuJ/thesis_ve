#!/bin/bash
# Run remaining VBench dims on specified GPU
set -eo pipefail

GPU_ID=$1
INPUT_PATH=$2
OUTPUT_BASE=$3
shift 3
DIMS=("$@")

DISK2="/data/disk2/timur"
CONDA="$DISK2/miniconda3"
VBENCH_REPO="$DISK2/repos/VBench"

eval "$($CONDA/bin/conda shell.bash hook)"
conda activate vbench

export PYTHONPATH="$VBENCH_REPO:${PYTHONPATH:-}"

for dim in "${DIMS[@]}"; do
    echo "=== $dim ==="
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$VBENCH_REPO/vbench2_beta_long/eval_long.py" \
        --output_path "$OUTPUT_BASE/$dim" \
        --full_json_dir "$VBENCH_REPO/vbench2_beta_long/VBench_full_info.json" \
        --videos_path "$INPUT_PATH" \
        --dimension "$dim" \
        --mode "long_custom_input" 2>&1 | tail -10
    echo "$dim DONE"
    echo ""
done
echo "ALL DONE"
