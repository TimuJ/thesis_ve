#!/bin/bash
# B8: metric battery on the 6 flip artefacts.
# Usage: run_b8_flip_eval.sh <gpu> "<artefact list>"
set -eo pipefail
DISK2=/data/disk2/timur
GPU=${1:-0}
ARTS=${2:-"flip_horizontal flip_transpose flip_periodic flip_elastic flip_channel_shuffle flip_invert"}
EVAL=$DISK2/results/synthetic_artefacts_eval
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
cd $DISK2
export PYTHONPATH="$DISK2:${PYTHONPATH:-}"
for ART in $ARTS; do
  SRC=$DISK2/results/synthetic_artefacts/$ART
  n=$(ls $SRC/*.mp4 2>/dev/null | wc -l)
  echo "=== $ART: $n clips (GPU $GPU) ==="
  conda activate vsr
  cd $DISK2
  mkdir -p $EVAL/clip_iqa/$ART
  CUDA_VISIBLE_DEVICES=$GPU python compute_clip_iqa.py --videos_path $SRC --output_path $EVAL/clip_iqa/$ART
  mkdir -p $EVAL/tof_tlp/$ART
  CUDA_VISIBLE_DEVICES=$GPU python eval_tof_tlp.py --videos_path $SRC --output_path $EVAL/tof_tlp/$ART --k_values 1,5,10,30,60,120 --max_pairs 200
  mkdir -p $DISK2/results/lr_vcc/color_histogram/$ART
  python compute_color_histogram_lean.py --videos_path $SRC --output_path $DISK2/results/lr_vcc/color_histogram/$ART
  mkdir -p $DISK2/results/lr_vcc/color_slope/$ART
  python -m scripts.lr_vcc.compute_color_slope --videos_path $SRC --output_path $DISK2/results/lr_vcc/color_slope/$ART
  conda activate vbench
  cd $DISK2/repos/VBench/VBench-2.0
  export PYTHONPATH="$PWD:$DISK2/repos/YOLO-World:$DISK2:${PYTHONPATH:-}"
  mkdir -p $EVAL/identity/$ART
  rm -rf $EVAL/identity/$ART/_work_g$GPU 2>/dev/null
  CUDA_VISIBLE_DEVICES=$GPU python human_identity_long.py --videos_path $SRC --output_path $EVAL/identity/$ART --save_clip_detail
  cd $DISK2
  touch /tmp/b8_${ART}.done
  echo "=== $ART COMPLETE ==="
done
touch /tmp/b8_gpu${GPU}.done
