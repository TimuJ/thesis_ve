#!/bin/bash
# Regenerate 7WHI background_drift clips with the new curated reference and re-evaluate.
set -eo pipefail
DISK2=/data/disk2/timur
GPU=${1:-7}
EVAL=$DISK2/results/synthetic_artefacts_eval
ART=background_drift
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
cd $DISK2
export PYTHONPATH="$DISK2:${PYTHONPATH:-}"

conda activate vsr
rm -f $DISK2/results/synthetic_artefacts/$ART/7WHI2L_FDNg_sev*.mp4
python scripts/synthetic_artefacts/generate_all.py   # regenerates only the 5 missing
n=$(ls $DISK2/results/synthetic_artefacts/$ART/7WHI2L_FDNg_sev*.mp4 | wc -l)
[ "$n" -eq 5 ] || { echo "FATAL: expected 5 regenerated clips, got $n"; exit 1; }

NSRC=$DISK2/results/synthetic_artefacts/_7whibg
rm -rf $NSRC; mkdir -p $NSRC
ln -s $DISK2/results/synthetic_artefacts/$ART/7WHI2L_FDNg_sev*.mp4 $NSRC/

echo "=== CLIP-IQA ==="
CUDA_VISIBLE_DEVICES=$GPU python compute_clip_iqa.py --videos_path $NSRC --output_path $EVAL/clip_iqa/$ART
echo "=== tOF/tLP ==="
CUDA_VISIBLE_DEVICES=$GPU python eval_tof_tlp.py --videos_path $NSRC --output_path $EVAL/tof_tlp/$ART --k_values 1,5,10,30,60,120 --max_pairs 200
echo "=== color histogram ==="
python compute_color_histogram_lean.py --videos_path $NSRC --output_path $DISK2/results/lr_vcc/color_histogram/$ART
echo "=== color slope ==="
python -m scripts.lr_vcc.compute_color_slope --videos_path $NSRC --output_path $DISK2/results/lr_vcc/color_slope/$ART
echo "=== Identity slow-fast ==="
conda activate vbench
cd $DISK2/repos/VBench/VBench-2.0
export PYTHONPATH="$PWD:$DISK2/repos/YOLO-World:$DISK2:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=$GPU python human_identity_long.py --videos_path $NSRC --output_path $EVAL/identity/$ART --save_clip_detail
cd $DISK2
touch /tmp/b6_7whibg.done
echo "=== 7WHI background_drift refresh COMPLETE ==="
