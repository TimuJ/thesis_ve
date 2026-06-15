#!/bin/bash
set -eo pipefail
DISK2=/data/disk2/timur
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd $DISK2
export PYTHONPATH="$DISK2:${PYTHONPATH:-}"

echo "=== Layer 1+2 non-regression UNIFORM: 5 SR videos x MGLD + UAV ==="
for m in mgld uav; do
    OUT=$DISK2/results/lr_vcc/composite_v2_uniform/$m
    mkdir -p $OUT
    python -m scripts.lr_vcc.run_lr_vcc \
        --method $m \
        --clip_iqa_dir $DISK2/results/lr_vcc/clip_iqa/$m \
        --tof_dir $DISK2/results/long_range_temporal/$m \
        --identity_results $(ls $DISK2/results/vbench2_human_test/identity_long_fps_overrides/$m/results_*.json | head -1) \
        --closeup_p50_map $DISK2/results/lr_vcc/closeup_map/$m.json \
        --color_hist_dir $DISK2/results/lr_vcc/color_histogram/$m \
        --temporal_weight uniform \
        --output_path $OUT
done

echo
echo "=== Synthetic artefacts UNIFORM: color_drift + chunk_boundary + flicker ==="
for a in color_drift chunk_boundary flicker; do
    OUT=$DISK2/results/lr_vcc/composite_artefacts_v2_uniform/$a
    mkdir -p $OUT
    python -m scripts.lr_vcc.run_lr_vcc \
        --method $a \
        --clip_iqa_dir $DISK2/results/synthetic_artefacts_eval/clip_iqa/$a \
        --tof_dir $DISK2/results/synthetic_artefacts_eval/tof_tlp/$a \
        --identity_results $(ls $DISK2/results/synthetic_artefacts_eval/identity/$a/results_*.json | head -1) \
        --closeup_p50_map $DISK2/results/lr_vcc/closeup_map_artefacts/$a.json \
        --color_hist_dir $DISK2/results/lr_vcc/color_histogram/$a \
        --temporal_weight uniform \
        --output_path $OUT
done
echo DONE
