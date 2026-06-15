#!/bin/bash
# GPU 0 batch: Identity slow-fast on 20 synthetic test videos
set -eo pipefail
export TMPDIR=/data/disk2/timur/tmp
DISK2=/data/disk2/timur
GPU=${1:-0}
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vbench
cd $DISK2/repos/VBench/VBench-2.0
export PYTHONPATH="$PWD:$DISK2/repos/YOLO-World:${PYTHONPATH:-}"

EVAL_BASE=$DISK2/results/synthetic_artefacts_eval

for artefact in color_drift chunk_boundary; do
    OUT=$EVAL_BASE/identity/${artefact}
    mkdir -p $OUT
    echo "=== Identity slow-fast on $artefact ==="
    CUDA_VISIBLE_DEVICES=$GPU python human_identity_long.py \
        --videos_path $DISK2/results/synthetic_artefacts/${artefact} \
        --output_path $OUT \
        --save_clip_detail
done

echo DONE
