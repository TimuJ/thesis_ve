#!/bin/bash
set -uo pipefail
DISK2=/data/disk2/timur
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd $DISK2
export PYTHONPATH="$DISK2:${PYTHONPATH:-}"
for ART in color_drift chunk_boundary flicker identity_degradation identity_drift background_drift \
           flip_horizontal flip_transpose flip_periodic flip_elastic flip_channel_shuffle flip_invert; do
  SRC=$DISK2/results/synthetic_artefacts/$ART
  OUT=$DISK2/results/lr_vcc/color_hist_anchor/$ART
  [ -d "$SRC" ] || { echo "[skip] $ART"; continue; }
  echo "=== D-prime $ART ==="
  mkdir -p $OUT
  python -m scripts.lr_vcc.color_histogram_anchor --videos_dir $SRC --output_path $OUT --stride 2
done
touch /tmp/b9_dprime.done
