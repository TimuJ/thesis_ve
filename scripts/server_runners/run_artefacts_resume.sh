#!/bin/bash
# Resume the artefact metric battery on a single GPU.
# tof_tlp script has skip-if-exists; identity slow-fast does not (re-runs everything).
set -eo pipefail
export TMPDIR=/data/disk2/timur/tmp
DISK2=/data/disk2/timur
GPU=${1:-4}
EVAL_BASE=$DISK2/results/synthetic_artefacts_eval

# ---- 1) finish tof_tlp (will skip already-done videos) ----
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd $DISK2
for artefact in color_drift chunk_boundary; do
    OUT=$EVAL_BASE/tof_tlp/${artefact}
    mkdir -p $OUT
    echo "=== tof_tlp $artefact ==="
    CUDA_VISIBLE_DEVICES=$GPU python eval_tof_tlp.py \
        --videos_path $DISK2/results/synthetic_artefacts/${artefact} \
        --output_path $OUT \
        --k_values 1,5,10,30,60,120 --max_pairs 200
done

# ---- 2) DOVER ----
cd $DISK2/repos/DOVER
for artefact in color_drift chunk_boundary; do
    OUT=$EVAL_BASE/dover/${artefact}; mkdir -p $OUT
    echo "=== DOVER $artefact ==="
    CUDA_VISIBLE_DEVICES=$GPU python evaluate_a_set_of_videos.py \
        --input_video_dir $DISK2/results/synthetic_artefacts/${artefact} \
        --output_result_csv $OUT/results.csv
done

# ---- 3) E*warp ----
cd $DISK2/repos/DOVE/finetune/scripts
for artefact in color_drift chunk_boundary; do
    OUT=$EVAL_BASE/ewarp/${artefact}; mkdir -p $OUT
    echo "=== E*warp $artefact ==="
    CUDA_VISIBLE_DEVICES=$GPU python eval_ewarp.py \
        --pred $DISK2/results/synthetic_artefacts/${artefact} \
        --model models/raft-things.pth --out $OUT
done

# ---- 4) Identity slow-fast (full re-run; ~3h) ----
conda activate vbench
cd $DISK2/repos/VBench/VBench-2.0
export PYTHONPATH="$PWD:$DISK2/repos/YOLO-World:${PYTHONPATH:-}"
for artefact in color_drift chunk_boundary; do
    OUT=$EVAL_BASE/identity/${artefact}; mkdir -p $OUT
    rm -f $OUT/results_*.json $OUT/_work 2>/dev/null
    echo "=== Identity slow-fast $artefact ==="
    CUDA_VISIBLE_DEVICES=$GPU python human_identity_long.py \
        --videos_path $DISK2/results/synthetic_artefacts/${artefact} \
        --output_path $OUT --save_clip_detail
done

echo DONE
