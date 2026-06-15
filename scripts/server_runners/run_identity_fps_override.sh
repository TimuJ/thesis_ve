#!/bin/bash
set -eo pipefail
export TMPDIR=/data/disk2/timur/tmp
DISK2=/data/disk2/timur
GPU=${1:-6}
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vbench
cd $DISK2/repos/VBench/VBench-2.0
export PYTHONPATH="$PWD:$DISK2/repos/YOLO-World:${PYTHONPATH:-}"

for method in mgld uav; do
  OUT=$DISK2/results/vbench2_human_test/identity_long_fps_overrides/${method}
  mkdir -p $OUT
  echo "--- ${method} ---"
  CUDA_VISIBLE_DEVICES=$GPU python human_identity_long.py \
      --videos_path $DISK2/results/${method}_synthetic_mp4 \
      --output_path $OUT \
      --fps_overrides $DISK2/fps_overrides.json \
      --save_clip_detail
done
echo DONE
