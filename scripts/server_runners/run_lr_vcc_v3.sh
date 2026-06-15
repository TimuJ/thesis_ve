#!/usr/bin/env bash
# LR-VCC v3 — recalibrated D (alpha=0.394) + uniform temporal weight.
# Re-runs SR (mgld, uav) and 3 artefacts × 2 base videos × 5 severities.

set -euo pipefail

eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"
conda activate vsr

cd /data/disk2/timur

ALPHA=0.394

R=/data/disk2/timur/results
OUT_SR=$R/lr_vcc/composite_v3
OUT_ART=$R/lr_vcc/composite_artefacts_v3

mkdir -p "$OUT_SR" "$OUT_ART"

ID_MGLD=$R/vbench2_human_test/identity_long_mgld/results_2026-05-02-00:58:01_eval_results.json
ID_UAV=$R/vbench2_human_test/identity_long_uav/results_2026-05-02-04:08:15_eval_results.json
ID_ART=$R/synthetic_artefacts_eval/identity/flicker/results_2026-05-27-03:00:54_eval_results.json
# (color_drift and chunk_boundary share the same identity payload since the artefact only modifies pixels of the SAME source video set — but we should pick the right one per condition. Inspect what exists.)

echo "===== SR runs ====="
for METHOD in mgld uav; do
    if [ "$METHOD" = "mgld" ]; then ID=$ID_MGLD; else ID=$ID_UAV; fi
    python -m scripts.lr_vcc.run_lr_vcc \
        --method "$METHOD" \
        --clip_iqa_dir   $R/lr_vcc/clip_iqa/$METHOD \
        --tof_dir        $R/long_range_temporal/$METHOD \
        --identity_results "$ID" \
        --color_hist_dir $R/lr_vcc/color_histogram/$METHOD \
        --color_hist_alpha $ALPHA \
        --temporal_weight uniform \
        --output_path    $OUT_SR/$METHOD
done

echo "===== Artefact runs ====="
# Identity dirs per condition.
for COND in flicker color_drift chunk_boundary; do
    ID_PATH=$(ls $R/synthetic_artefacts_eval/identity/$COND/*.json | head -1)
    CLOSEUP=$R/lr_vcc/closeup_map_artefacts/$COND.json
    python -m scripts.lr_vcc.run_lr_vcc \
        --method "$COND" \
        --clip_iqa_dir   $R/synthetic_artefacts_eval/clip_iqa/$COND \
        --tof_dir        $R/synthetic_artefacts_eval/tof_tlp/$COND \
        --identity_results "$ID_PATH" \
        --closeup_p50_map "$CLOSEUP" \
        --color_hist_dir $R/lr_vcc/color_histogram/$COND \
        --color_hist_alpha $ALPHA \
        --temporal_weight uniform \
        --output_path    $OUT_ART/$COND
done

echo "===== DONE ====="
