#!/bin/bash
# GPU 1 batch: CLIP-IQA + tOF/tLP + DOVER + E*warp on 20 synthetic test videos
set -eo pipefail
export TMPDIR=/data/disk2/timur/tmp
DISK2=/data/disk2/timur
GPU=${1:-1}
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd $DISK2

EVAL_BASE=$DISK2/results/synthetic_artefacts_eval

# ---- CLIP-IQA ----
echo "=== CLIP-IQA ==="
for artefact in color_drift chunk_boundary; do
    OUT=$EVAL_BASE/clip_iqa/${artefact}
    mkdir -p $OUT
    echo "  --- $artefact ---"
    CUDA_VISIBLE_DEVICES=$GPU python compute_clip_iqa.py \
        --videos_path $DISK2/results/synthetic_artefacts/${artefact} \
        --output_path $OUT
done

# ---- tOF / tLP at k = 1, 5, 10, 30, 60, 120 ----
echo "=== tOF/tLP ==="
for artefact in color_drift chunk_boundary; do
    OUT=$EVAL_BASE/tof_tlp/${artefact}
    mkdir -p $OUT
    echo "  --- $artefact ---"
    CUDA_VISIBLE_DEVICES=$GPU python eval_tof_tlp.py \
        --videos_path $DISK2/results/synthetic_artefacts/${artefact} \
        --output_path $OUT \
        --k_values 1,5,10,30,60,120 \
        --max_pairs 200
done

# ---- DOVER ----
echo "=== DOVER ==="
cd $DISK2/repos/DOVER
for artefact in color_drift chunk_boundary; do
    OUT=$EVAL_BASE/dover/${artefact}
    mkdir -p $OUT
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_a_set_of_videos.py \
        --input_video_dir $DISK2/results/synthetic_artefacts/${artefact} \
        --output_result_csv $OUT/results.csv
done

# ---- E*warp ----
echo "=== E*warp ==="
cd $DISK2/repos/DOVE/finetune/scripts
for artefact in color_drift chunk_boundary; do
    OUT=$EVAL_BASE/ewarp/${artefact}
    mkdir -p $OUT
    CUDA_VISIBLE_DEVICES=$GPU python eval_ewarp.py \
        --pred $DISK2/results/synthetic_artefacts/${artefact} \
        --model models/raft-things.pth \
        --out $OUT
done

echo DONE
