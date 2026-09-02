#!/usr/bin/env bash
# Recompute the LR-VCC metric stack for the two REGENERATED identity artefact
# families, writing to parallel *_regen dirs so the original committed baseline
# (which the frozen v5 tests and the published matrix depend on) is never
# touched. Sequential on one GPU — the box is shared and nearly full.
#
# Why: the original battery clips for these families were pruned and had to be
# regenerated; re-encoding shifted marginal face detections (BrRLK clips with
# faces 18 -> 4), so every cached metric JSON computed on the original pixels
# is inconsistent with the video files now on disk. This restores consistency
# on the regenerated pixels; the old numbers remain as the frozen reference.
#
# Usage: bash regen_identity_families_eval.sh [gpu]   (default 1)
# Waits for any running dump_identity_embeddings to finish first.
set -uo pipefail
GPU=${1:-1}
H=$HOME
REPO=$H/thesis_ve
LRV=$H/results/lr_vcc
EVAL=$H/results/synthetic_artefacts_eval
PYV=$H/miniconda3/envs/vsr/bin/python
PYI=$H/miniconda3/envs/identity/bin/python
export PYTHONPATH=$REPO

echo "waiting for any embedding dump to finish..."
while pgrep -f "dump_identity_embeddings" >/dev/null; do sleep 120; done
echo "starting at $(date -u +%H:%M:%SZ) on GPU $GPU"

for FAM in identity_degradation identity_drift; do
  SRC=$REPO/results/synthetic_artefacts/$FAM        # regenerated clips
  R=${FAM}_regen
  echo "########## $FAM -> ${R} ##########"
  mkdir -p "$EVAL/clip_iqa/$R" "$EVAL/tof_tlp/$R" "$EVAL/identity/$R" \
           "$LRV/color_histogram/$R" "$LRV/color_slope/$R" \
           "$LRV/color_hist_anchor/$R" "$LRV/clip_trajectory/$R"

  cd $REPO
  echo "=== CLIP-IQA ==="
  CUDA_VISIBLE_DEVICES=$GPU $PYV -m scripts.lr_vcc.compute_clip_iqa \
    --videos_path $SRC --output_path $EVAL/clip_iqa/$R
  echo "=== tOF/tLP ==="
  CUDA_VISIBLE_DEVICES=$GPU $PYV scripts/long_range_temporal/eval_tof_tlp.py \
    --videos_path $SRC --output_path $EVAL/tof_tlp/$R \
    --k_values 1,5,10,30,60,120 --max_pairs 200
  echo "=== colour histogram ==="
  $PYV -m scripts.lr_vcc.compute_color_histogram \
    --videos_path $SRC --output_path $LRV/color_histogram/$R
  echo "=== colour slope ==="
  $PYV -m scripts.lr_vcc.compute_color_slope \
    --videos_path $SRC --output_path $LRV/color_slope/$R
  echo "=== anchored histogram ==="
  $PYV -m scripts.lr_vcc.color_histogram_anchor \
    --videos_dir $SRC --output_path $LRV/color_hist_anchor/$R
  echo "=== CLIP trajectory ==="
  CUDA_VISIBLE_DEVICES=$GPU $PYV -m scripts.lr_vcc.compute_clip_trajectory \
    --videos_dir $SRC --output_path $LRV/clip_trajectory/$R --stride 8
  echo "=== identity slow-fast ==="
  cd $REPO/scripts/vbench2_long
  CUDA_VISIBLE_DEVICES=$GPU $PYI human_identity_long.py \
    --videos_path $SRC --output_path $EVAL/identity/$R --save_clip_detail
  echo "DONE_$FAM at $(date -u +%H:%M:%SZ)"
done
echo "ALL_FAMILIES_DONE"
