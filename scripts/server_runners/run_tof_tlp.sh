#!/bin/bash
set -eo pipefail
export TMPDIR=/data/disk2/timur/tmp
DISK2=/data/disk2/timur
GPU=${1:-4}
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd $DISK2

for method in mgld uav; do
  OUT=$DISK2/results/long_range_temporal/${method}
  mkdir -p $OUT
  echo "=== ${method} ==="
  CUDA_VISIBLE_DEVICES=$GPU python eval_tof_tlp.py \
      --videos_path $DISK2/results/${method}_synthetic_mp4 \
      --output_path $OUT \
      --k_values 1,5,10,30,60,120 \
      --max_pairs 200
done
echo DONE
