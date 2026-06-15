#!/bin/bash
# B10: compute D'' (CLIP-image trajectory) on all artefact clips. GPU-bound.
# Usage: run_b10_dprime2.sh <gpu>
set -uo pipefail
DISK2=/data/disk2/timur
GPU=${1:-0}
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr  # OpenAI clip package + torch
cd $DISK2
export PYTHONPATH="$DISK2:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=$GPU
for ART in color_drift chunk_boundary flicker identity_degradation identity_drift background_drift \
           flip_horizontal flip_transpose flip_periodic flip_elastic flip_channel_shuffle flip_invert; do
  SRC=$DISK2/results/synthetic_artefacts/$ART
  OUT=$DISK2/results/lr_vcc/clip_trajectory/$ART
  [ -d "$SRC" ] || { echo "[skip] $ART"; continue; }
  echo "=== D-prime-prime $ART (GPU $GPU) ==="
  mkdir -p $OUT
  python -m scripts.lr_vcc.compute_clip_trajectory --videos_dir $SRC --output_path $OUT --stride 8
done
touch /tmp/b10_dprime2_gpu${GPU}.done
