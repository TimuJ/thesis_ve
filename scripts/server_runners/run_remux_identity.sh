#!/bin/bash
set -eo pipefail
export TMPDIR=/data/disk2/timur/tmp
DISK2=/data/disk2/timur
FFMPEG=$DISK2/miniconda3/envs/vbench/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2

declare -A FPS
FPS[7WHI2L_FDNg]=29.97
FPS[BrRLKMbBTYQ]=24
FPS[KZ8p6b1zJ9U]=29.97
FPS[hhszUXL1Cu8]=29.97
FPS[mJog8DlRk_4]=23.98

VIDEOS=(7WHI2L_FDNg BrRLKMbBTYQ KZ8p6b1zJ9U hhszUXL1Cu8 mJog8DlRk_4)

echo "=== STEP A: lossless re-mux to LQ fps ==="
for method in mgld uav; do
  src=$DISK2/results/${method}_synthetic_mp4
  dst=$DISK2/results/${method}_synthetic_mp4_fps_fixed
  mkdir -p $dst
  for v in "${VIDEOS[@]}"; do
    if [ -f $dst/$v.mp4 ]; then
      echo "[skip] $dst/$v.mp4 exists"
    else
      echo "[remux] $method/$v -> ${FPS[$v]} fps"
      $FFMPEG -y -loglevel error -r ${FPS[$v]} -i $src/$v.mp4 -c copy $dst/$v.mp4
    fi
  done
done

echo "=== STEP B: identity slow-fast on re-muxed videos ==="
GPU=${1:-1}
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vbench
cd $DISK2/repos/VBench/VBench-2.0
export PYTHONPATH="$PWD:$DISK2/repos/YOLO-World:${PYTHONPATH:-}"

for method in mgld uav; do
  OUT=$DISK2/results/vbench2_human_test/identity_long_fps_fixed/${method}
  mkdir -p $OUT
  echo "--- ${method} ---"
  CUDA_VISIBLE_DEVICES=$GPU python human_identity_long.py \
      --videos_path $DISK2/results/${method}_synthetic_mp4_fps_fixed \
      --output_path $OUT \
      --save_clip_detail
done

echo DONE
