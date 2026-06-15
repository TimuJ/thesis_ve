#!/bin/bash
# B6: metric battery on the 3 promoted bases' clips only (symlink subset dirs).
# Usage: run_b6_eval.sh <gpu> "<artefact list>"
# Stages: CLIP-IQA, tOF/tLP, color_hist, color_slope, Identity slow-fast.
# DOVER + E*warp skipped (not consumed by LR-VCC composite).
set -eo pipefail
DISK2=/data/disk2/timur
GPU=${1:-0}
ARTS=${2:-"color_drift chunk_boundary flicker identity_degradation identity_drift background_drift"}
EVAL=$DISK2/results/synthetic_artefacts_eval
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
cd $DISK2
export PYTHONPATH="$DISK2:${PYTHONPATH:-}"

for ART in $ARTS; do
  NSRC=$DISK2/results/synthetic_artefacts/_new5_$ART
  rm -rf $NSRC; mkdir -p $NSRC
  for f in $DISK2/results/synthetic_artefacts/$ART/*.mp4; do
    case $(basename $f) in
      KZ8p6b1zJ9U*|BrRLKMbBTYQ*|mJog8DlRk_4*) ln -s $f $NSRC/ ;;
    esac
  done
  n=$(ls $NSRC | wc -l)
  echo "=== $ART: $n new clips ==="
  [ "$n" -eq 15 ] || { echo "FATAL: expected 15 clips for $ART, got $n"; exit 1; }
done

for ART in $ARTS; do
  NSRC=$DISK2/results/synthetic_artefacts/_new5_$ART
  conda activate vsr
  cd $DISK2
  echo "=== $ART CLIP-IQA (GPU $GPU) ==="
  mkdir -p $EVAL/clip_iqa/$ART
  CUDA_VISIBLE_DEVICES=$GPU python compute_clip_iqa.py --videos_path $NSRC --output_path $EVAL/clip_iqa/$ART
  echo "=== $ART tOF/tLP ==="
  mkdir -p $EVAL/tof_tlp/$ART
  CUDA_VISIBLE_DEVICES=$GPU python eval_tof_tlp.py --videos_path $NSRC --output_path $EVAL/tof_tlp/$ART --k_values 1,5,10,30,60,120 --max_pairs 200
  echo "=== $ART color histogram ==="
  mkdir -p $DISK2/results/lr_vcc/color_histogram/$ART
  python compute_color_histogram_lean.py --videos_path $NSRC --output_path $DISK2/results/lr_vcc/color_histogram/$ART
  echo "=== $ART color slope ==="
  mkdir -p $DISK2/results/lr_vcc/color_slope/$ART
  python -m scripts.lr_vcc.compute_color_slope --videos_path $NSRC --output_path $DISK2/results/lr_vcc/color_slope/$ART
  echo "=== $ART Identity slow-fast ==="
  conda activate vbench
  cd $DISK2/repos/VBench/VBench-2.0
  export PYTHONPATH="$PWD:$DISK2/repos/YOLO-World:$DISK2:${PYTHONPATH:-}"
  mkdir -p $EVAL/identity/$ART
  rm -rf $EVAL/identity/$ART/_work_g$GPU 2>/dev/null
  CUDA_VISIBLE_DEVICES=$GPU python human_identity_long.py --videos_path $NSRC --output_path $EVAL/identity/$ART --save_clip_detail
  cd $DISK2
  touch /tmp/b6_${ART}.done
  echo "=== $ART COMPLETE ==="
done
touch /tmp/b6_gpu${GPU}.done
