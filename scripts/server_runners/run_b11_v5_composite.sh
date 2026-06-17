#!/usr/bin/env bash
# B11: LR-VCC v5 composite for 6 existing artefacts.
# Integrates D' (color_hist_anchor) and D'' (clip_trajectory) alongside A/T/I/D/E.
# Runs on the FULL 5-base x 5-sev = 25 clips per artefact.
# NO GPU needed (python-only scoring, no inference).
set -euo pipefail

eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"
conda activate vsr

DISK2=/data/disk2/timur
EVAL=$DISK2/results/synthetic_artefacts_eval
LRVCC=$DISK2/results/lr_vcc
export PYTHONPATH="$DISK2:${PYTHONPATH:-}"
cd $DISK2

mkdir -p $LRVCC/composite_artefacts_v5

# Step 1: merge identity JSONs into a single file per artefact (all 5 bases).
# background_drift already has _merged_v4.json (uploaded June 14).
# Others have 2 separate JSON files that together cover all 25 clips.
echo "=== Merging identity JSONs ==="
python3 - <<'PYEOF'
import json, glob, os

BASE="/data/disk2/timur/results/synthetic_artefacts_eval/identity"

artefacts = [
    "color_drift",
    "chunk_boundary",
    "flicker",
    "identity_degradation",
    "identity_drift",
]
for art in artefacts:
    files = sorted(glob.glob("%s/%s/*.json" % (BASE, art)))
    merged = {}
    for f in files:
        d = json.load(open(f))
        merged.update(d["per_video"])
    out_path = "%s/%s/_merged_v5.json" % (BASE, art)
    json.dump({"per_video": merged}, open(out_path, "w"), indent=2)
    print("  %s: %d videos -> %s" % (art, len(merged), out_path))

print("background_drift: using _merged_v4.json (pre-uploaded)")
PYEOF
echo "=== Identity merge done ==="

# Step 2: run composite for each artefact
ARTS="color_drift chunk_boundary flicker identity_degradation identity_drift background_drift"

for ART in $ARTS; do
    echo ""
    echo "===== $ART ====="

    clip_iqa_dir=$EVAL/clip_iqa/$ART
    tof_dir=$EVAL/tof_tlp/$ART
    color_hist_dir=$LRVCC/color_histogram/$ART
    color_slope_dir=$LRVCC/color_slope/$ART
    color_hist_anchor_dir=$LRVCC/color_hist_anchor/$ART
    clip_trajectory_dir=$LRVCC/clip_trajectory/$ART
    output_dir=$LRVCC/composite_artefacts_v5/$ART

    # Pick identity JSON
    if [ "$ART" = "background_drift" ]; then
        identity_results=$EVAL/identity/background_drift/_merged_v4.json
    else
        identity_results=$EVAL/identity/$ART/_merged_v5.json
    fi

    mkdir -p $output_dir

    python3 -m scripts.lr_vcc.run_lr_vcc \
        --method "$ART" \
        --clip_iqa_dir "$clip_iqa_dir" \
        --tof_dir "$tof_dir" \
        --identity_results "$identity_results" \
        --color_hist_dir "$color_hist_dir" \
        --color_hist_alpha 0.394 \
        --color_slope_dir "$color_slope_dir" \
        --color_slope_beta 200 \
        --color_hist_anchor_dir "$color_hist_anchor_dir" \
        --dprime_beta 0.5 \
        --clip_trajectory_dir "$clip_trajectory_dir" \
        --dprime2_beta 3.0 \
        --temporal_weight uniform \
        --output_path "$output_dir"

    echo "=== $ART DONE ==="
done

touch /tmp/b11_v5.done
echo ""
echo "===== ALL 6 ARTEFACTS COMPLETE (v5) ====="
