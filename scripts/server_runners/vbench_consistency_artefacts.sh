#!/usr/bin/env bash
# SOTA-baseline severity response: VBench subject_consistency +
# background_consistency over synthetic artefact families, using the same
# long_custom_input protocol as the April real-SR runs.
#
# Usage: bash vbench_consistency_artefacts.sh [family ...]
#   default families: background_drift color_drift
# Waits for the RealESRGAN battery done-markers before touching the GPUs.
set -u -o pipefail
H=$HOME
FAMILIES=${@:-"background_drift color_drift"}
VB=$H/repos/VBench
PYB=$H/miniconda3/envs/vbench/bin/python
OUT=$H/results/vbench_consistency_artefacts
mkdir -p ~/logs "$OUT"

echo "waiting for artefact generation and battery..."
until [ -f ~/logs/gen_subset.done ]; do sleep 60; done
until [ -f ~/logs/bat_realesrgan_gpu0.done ] && [ -f ~/logs/bat_realesrgan_gpu1.done ]; do
  sleep 60
done
echo "generation + battery done — starting VBench consistency runs"

cd "$VB"
export PYTHONPATH=$VB:${PYTHONPATH:-}
GPU=0
for fam in $FAMILIES; do
  SRCDIR=$H/results/synthetic_artefacts/$fam
  if [ ! -d "$SRCDIR" ]; then
    echo "MISSING_FAMILY $fam ($SRCDIR)" | tee -a ~/logs/vbench_consistency.log
    continue
  fi
  for dim in background_consistency subject_consistency; do
    echo "=== $fam / $dim (gpu $GPU)" | tee -a ~/logs/vbench_consistency.log
    CUDA_VISIBLE_DEVICES=$GPU $PYB vbench2_beta_long/eval_long.py \
      --videos_path "$SRCDIR" \
      --dimension "$dim" \
      --mode long_custom_input \
      --dev_flag \
      --output_path "$OUT/$dim/$fam" \
      2>&1 | tee ~/logs/vbench_${fam}_${dim}.log
    echo "=== $fam / $dim EXIT=$?" | tee -a ~/logs/vbench_consistency.log
  done
done
touch ~/logs/vbench_consistency.done
echo ALL_DONE
