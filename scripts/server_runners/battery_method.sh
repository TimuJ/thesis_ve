#!/usr/bin/env bash
# LR-VCC 7-stage battery for one method's SR outputs on the SmartML server.
# Usage: bash battery_method.sh [method]   (default: realesrgan)
# Launches two tmux chains (bat0_<m>: GPU0, bat1_<m>: GPU1); done markers in
# ~/logs/bat_<m>_gpu{0,1}.done. Safe to re-run: stages skip existing JSONs.
set -u
M=${1:-realesrgan}
H=$HOME
REPO=$H/thesis_ve
SRC=$H/results/${M}_synthetic_mp4
LRV=$H/results/lr_vcc
EVAL=$H/results/synthetic_artefacts_eval
PYV=$H/miniconda3/envs/vsr/bin/python
PYI=$H/miniconda3/envs/identity/bin/python
mkdir -p ~/logs "$LRV/clip_iqa/$M" "$LRV/clip_trajectory_realmodels/$M" \
  "$LRV/color_histogram/$M" "$LRV/color_slope/$M" \
  "$LRV/color_hist_anchor_realmodels/$M" \
  "$H/results/long_range_temporal/$M" "$EVAL/identity/$M"

tmux new-session -d -s "bat0_$M" "
export PYTHONPATH=$REPO; cd $REPO
CUDA_VISIBLE_DEVICES=0 $PYV -m scripts.lr_vcc.compute_clip_iqa \
  --videos_path $SRC --output_path $LRV/clip_iqa/$M \
  2>&1 | tee ~/logs/bat_${M}_clipiqa.log
CUDA_VISIBLE_DEVICES=0 $PYV -m scripts.lr_vcc.compute_clip_trajectory \
  --videos_dir $SRC --output_path $LRV/clip_trajectory_realmodels/$M --stride 8 \
  2>&1 | tee ~/logs/bat_${M}_traj.log
$PYV -m scripts.lr_vcc.compute_color_histogram \
  --videos_path $SRC --output_path $LRV/color_histogram/$M \
  2>&1 | tee ~/logs/bat_${M}_hist.log
$PYV -m scripts.lr_vcc.compute_color_slope \
  --videos_path $SRC --output_path $LRV/color_slope/$M \
  2>&1 | tee ~/logs/bat_${M}_slope.log
$PYV -m scripts.lr_vcc.color_histogram_anchor \
  --videos_dir $SRC --output_path $LRV/color_hist_anchor_realmodels/$M \
  2>&1 | tee ~/logs/bat_${M}_anchor.log
touch ~/logs/bat_${M}_gpu0.done"

tmux new-session -d -s "bat1_$M" "
export PYTHONPATH=$REPO
export VBENCH2_CACHE_DIR=$H/.cache/vbench2
cd $REPO
CUDA_VISIBLE_DEVICES=1 $PYV scripts/long_range_temporal/eval_tof_tlp.py \
  --videos_path $SRC --output_path $H/results/long_range_temporal/$M \
  2>&1 | tee ~/logs/bat_${M}_tof.log
cd $REPO/scripts/vbench2_long
CUDA_VISIBLE_DEVICES=1 $PYI human_identity_long.py \
  --videos_path $SRC --output_path $EVAL/identity/$M --save_clip_detail \
  2>&1 | tee ~/logs/bat_${M}_identity.log
touch ~/logs/bat_${M}_gpu1.done"
echo "LAUNCHED battery for $M"
