#!/bin/bash
# Finalize the flip_invert v5 row once the server's identity slow-fast run completes.
# Run from repo root:  bash scripts/finalize_flip_invert.sh
#
# Prereq: the server (instance-xzujqxam) has finished the identity run
#   (~/flip_identity.done present, results_*.json written).
# All other flip_invert stage JSONs are already local.

set -uo pipefail
SSH="ssh -p 11007 -o ConnectTimeout=12 -i $HOME/.ssh/id_ed25519_timuj timur@instance-xzujqxam.yc.smartml.cn"
REPO="$HOME/Desktop/Timur/thesis_ve"
cd "$REPO"

echo "=== [1] check all 8 parallel identity batches finished ==="
NDONE=$($SSH 'ls ~/results/synthetic_artefacts_eval/identity/flip_invert/batch_*/results_*.json 2>/dev/null | wc -l' 2>/dev/null)
if [ "${NDONE:-0}" -lt 8 ]; then
  echo "only ${NDONE:-0}/8 batches finished. Aborting — wait for the rest."
  $SSH 'ps aux | grep human_identity | grep -v grep | wc -l | xargs echo "workers still running:"' 2>/dev/null
  exit 1
fi
echo "all 8 batches complete."

echo "=== [2] merge 8 parallel batch identity JSONs (server-side) + pull ==="
$SSH 'bash -lc "
BASE=~/results/synthetic_artefacts_eval/identity/flip_invert
python3 - <<PYEOF
import json, glob, os
base=os.path.expanduser(\"~/results/synthetic_artefacts_eval/identity/flip_invert\")
merged={\"per_video\":{}}
for bj in sorted(glob.glob(base+\"/batch_*/results_*.json\")):
    d=json.load(open(bj))
    for v,pv in d.get(\"per_video\",{}).items():
        merged[\"per_video\"][v]=pv
out=base+\"/results_merged_flip_invert.json\"
json.dump(merged, open(out,\"w\"))
print(\"merged videos:\", len(merged[\"per_video\"]), \"->\", out)
PYEOF
"' 2>/dev/null
mkdir -p results/synthetic_artefacts_eval/identity/flip_invert
for try in 1 2 3 4 5 6; do
  rsync -a --partial -e "ssh -p 11007 -o ServerAliveInterval=15 -i $HOME/.ssh/id_ed25519_timuj" \
    timur@instance-xzujqxam.yc.smartml.cn:'~/results/synthetic_artefacts_eval/identity/flip_invert/results_merged_flip_invert.json' \
    results/synthetic_artefacts_eval/identity/flip_invert/ 2>/dev/null && break
  echo "  rsync retry $try"; sleep 4
done
ID=results/synthetic_artefacts_eval/identity/flip_invert/results_merged_flip_invert.json

echo "identity JSON: $ID"
python3 -c "import json; d=json.load(open('$ID')); print('  videos in JSON:', len(d.get('per_video',{})))"

echo "=== [3] compute flip_invert v5 composite ==="
python3 -m scripts.lr_vcc.run_lr_vcc \
  --method flip_invert \
  --clip_iqa_dir   results/synthetic_artefacts_eval/clip_iqa/flip_invert \
  --tof_dir        results/synthetic_artefacts_eval/tof_tlp/flip_invert \
  --identity_results "$ID" \
  --color_hist_dir results/lr_vcc/color_histogram/flip_invert \
  --color_hist_alpha 0.394 \
  --color_slope_dir results/lr_vcc/color_slope/flip_invert \
  --color_slope_beta 200 \
  --color_hist_anchor_dir results/lr_vcc/color_hist_anchor/flip_invert \
  --dprime_beta 0.5 \
  --clip_trajectory_dir results/lr_vcc/clip_trajectory/flip_invert \
  --dprime2_beta 3.0 \
  --temporal_weight uniform \
  --output_path results/lr_vcc/composite_artefacts_v5/flip_invert
n=$(ls results/lr_vcc/composite_artefacts_v5/flip_invert/*sev*.json 2>/dev/null | wc -l | tr -d ' ')
echo "  flip_invert composites written: $n / 25"
[ "$n" -eq 25 ] || { echo "expected 25 composites, got $n. Aborting."; exit 1; }

echo "=== [4] rebuild the full 12x5 verdict matrix ==="
python3 scripts/lr_vcc/build_verdict_matrix.py \
  --composites_dir results/lr_vcc/composite_artefacts_v5 \
  --out reports/figures/verdict_matrix_v5.md
echo ""
echo "=== [5] show the completed matrix ==="
cat reports/figures/verdict_matrix_v5.md

echo ""
echo "=== [6] commit ==="
git add reports/figures/verdict_matrix_v5.md
git commit -q -m "reports: complete v5 verdict matrix — flip_invert row filled (12/12)

flip_invert identity slow-fast finally run on the new SmartML server
(the row killed by the June-15 server decommission). Full 12-artefact x
5-base matrix now complete.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push 2>&1 | tail -2
echo "DONE — 12/12 matrix committed."
