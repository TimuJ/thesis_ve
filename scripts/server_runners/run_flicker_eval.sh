#!/bin/bash
set -eo pipefail
DISK2=/data/disk2/timur
GPU=2
SRC=$DISK2/results/synthetic_artefacts/flicker
EVAL=$DISK2/results/synthetic_artefacts_eval

eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd $DISK2
export PYTHONPATH="$DISK2:${PYTHONPATH:-}"

# 1. CLIP-IQA
echo "=== CLIP-IQA ==="
mkdir -p $EVAL/clip_iqa/flicker
CUDA_VISIBLE_DEVICES=$GPU python compute_clip_iqa.py \
    --videos_path $SRC \
    --output_path $EVAL/clip_iqa/flicker

# 2. tOF/tLP
echo "=== tOF/tLP ==="
mkdir -p $EVAL/tof_tlp/flicker
CUDA_VISIBLE_DEVICES=$GPU python eval_tof_tlp.py \
    --videos_path $SRC --output_path $EVAL/tof_tlp/flicker \
    --k_values 1,5,10,30,60,120 --max_pairs 200

# 3. DOVER
echo "=== DOVER ==="
mkdir -p $EVAL/dover/flicker
cd $DISK2/repos/DOVER
CUDA_VISIBLE_DEVICES=$GPU python evaluate_a_set_of_videos.py \
    --input_video_dir $SRC --output_result_csv $EVAL/dover/flicker/results.csv

# 4. E*warp
echo "=== E*warp ==="
mkdir -p $EVAL/ewarp/flicker
cd $DISK2/repos/DOVE/finetune/scripts
CUDA_VISIBLE_DEVICES=$GPU python eval_ewarp.py \
    --pred $SRC --model models/raft-things.pth --out $EVAL/ewarp/flicker

# 5. Color histogram (CPU)
echo "=== Color histogram ==="
cd $DISK2
mkdir -p $DISK2/results/lr_vcc/color_histogram/flicker
python compute_color_histogram_lean.py \
    --videos_path $SRC \
    --output_path $DISK2/results/lr_vcc/color_histogram/flicker

# 6. Identity slow-fast
echo "=== Identity slow-fast ==="
conda activate vbench
cd $DISK2/repos/VBench/VBench-2.0
export PYTHONPATH="$PWD:$DISK2/repos/YOLO-World:${PYTHONPATH:-}"
mkdir -p $EVAL/identity/flicker
rm -rf $EVAL/identity/flicker/_work 2>/dev/null
CUDA_VISIBLE_DEVICES=$GPU python human_identity_long.py \
    --videos_path $SRC \
    --output_path $EVAL/identity/flicker \
    --save_clip_detail

echo DONE
