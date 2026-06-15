#!/bin/bash
# Per-metric eval for identity_drift synthetic artefact (10 videos).
set -eo pipefail
DISK2=/data/disk2/timur
GPU=${1:-0}
SRC=$DISK2/results/synthetic_artefacts/identity_drift
EVAL=$DISK2/results/synthetic_artefacts_eval
ART=identity_drift

eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd $DISK2
export PYTHONPATH="$DISK2:${PYTHONPATH:-}"

# 1. CLIP-IQA
echo "=== CLIP-IQA ==="
mkdir -p $EVAL/clip_iqa/$ART
CUDA_VISIBLE_DEVICES=$GPU python compute_clip_iqa.py \
    --videos_path $SRC \
    --output_path $EVAL/clip_iqa/$ART

# 2. tOF/tLP
echo "=== tOF/tLP ==="
mkdir -p $EVAL/tof_tlp/$ART
CUDA_VISIBLE_DEVICES=$GPU python eval_tof_tlp.py \
    --videos_path $SRC --output_path $EVAL/tof_tlp/$ART \
    --k_values 1,5,10,30,60,120 --max_pairs 200

# 3. DOVER
echo "=== DOVER ==="
mkdir -p $EVAL/dover/$ART
cd $DISK2/repos/DOVER
CUDA_VISIBLE_DEVICES=$GPU python evaluate_a_set_of_videos.py \
    --input_video_dir $SRC --output_result_csv $EVAL/dover/$ART/results.csv
cd $DISK2

# 4. E*warp
echo "=== E*warp ==="
mkdir -p $EVAL/ewarp/$ART
cd $DISK2/repos/DOVE/finetune/scripts
CUDA_VISIBLE_DEVICES=$GPU python eval_ewarp.py \
    --pred $SRC --model models/raft-things.pth --out $EVAL/ewarp/$ART
cd $DISK2

# 5. Color histogram (CPU)
echo "=== Color histogram ==="
mkdir -p $DISK2/results/lr_vcc/color_histogram/$ART
python compute_color_histogram_lean.py \
    --videos_path $SRC \
    --output_path $DISK2/results/lr_vcc/color_histogram/$ART

# 6. Color slope (CPU)
echo "=== Color slope ==="
mkdir -p $DISK2/results/lr_vcc/color_slope/$ART
python -m scripts.lr_vcc.compute_color_slope \
    --videos_path $SRC \
    --output_path $DISK2/results/lr_vcc/color_slope/$ART

# 7. Identity slow-fast
echo "=== Identity slow-fast ==="
conda activate vbench
cd $DISK2/repos/VBench/VBench-2.0
export PYTHONPATH="$PWD:$DISK2/repos/YOLO-World:${PYTHONPATH:-}"
mkdir -p $EVAL/identity/$ART
rm -rf $EVAL/identity/$ART/_work 2>/dev/null
CUDA_VISIBLE_DEVICES=$GPU python human_identity_long.py \
    --videos_path $SRC \
    --output_path $EVAL/identity/$ART \
    --save_clip_detail

echo DONE_EVAL_$ART
