#!/usr/bin/env bash
# VBench reward-degradation / blindness audit.
# Runs VBench-1.x long-extension temporal_flickering + motion_smoothness over the
# gradual-drift severity ladders (color_drift, background_drift) already present
# on the server, to test empirically whether these dimensions respond to graded
# drift (documented prediction: temporal_flickering is blind to drift slower than
# 2 frames; motion_smoothness previously uninterrogated).
#
# Pinned to GPU1 (GPU0 carries a collaborator's live job). Considerate footprint.
# Usage: bash run_vbench_reward_audit.sh
set -u -o pipefail
H=$HOME
VB=$H/repos/VBench
PYB=$H/miniconda3/envs/vbench/bin/python
OUT=$H/results/vbench_reward_audit
GPU=1
FAMILIES="${*:-color_drift background_drift}"
# temporal_flickering first (cheap MAE), motion_smoothness second (AMT interp, heavy)
DIMS="temporal_flickering motion_smoothness"

mkdir -p ~/logs "$OUT"
cd "$VB"
export PYTHONPATH=$VB:${PYTHONPATH:-}
export HF_ENDPOINT=https://hf-mirror.com   # huggingface.co blocked from this host

echo "reward-audit start pid=$$ gpu=$GPU $(date -u +%FT%TZ)" | tee -a ~/logs/vbench_reward_audit.log
for dim in $DIMS; do
  for fam in $FAMILIES; do
    SRC=$H/results/synthetic_artefacts/$fam
    ODIR=$OUT/$dim/$fam
    if [ ! -d "$SRC" ]; then
      echo "MISSING_FAMILY $fam ($SRC)" | tee -a ~/logs/vbench_reward_audit.log; continue
    fi
    if find "$ODIR" -name '*eval_results*' 2>/dev/null | grep -q .; then
      echo "SKIP $dim/$fam (already done)" | tee -a ~/logs/vbench_reward_audit.log; continue
    fi
    mkdir -p "$ODIR"
    echo "=== $dim / $fam (gpu $GPU) $(date -u +%FT%TZ)" | tee -a ~/logs/vbench_reward_audit.log
    CUDA_VISIBLE_DEVICES=$GPU nice -n 10 $PYB vbench2_beta_long/eval_long.py \
      --videos_path "$SRC" \
      --dimension "$dim" \
      --mode long_custom_input \
      --dev_flag \
      --output_path "$ODIR" \
      2>&1 | tee ~/logs/vbench_reward_${dim}_${fam}.log
    echo "=== $dim / $fam EXIT=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a ~/logs/vbench_reward_audit.log
  done
done
touch ~/logs/vbench_reward_audit.done
echo "ALL_DONE $(date -u +%FT%TZ)" | tee -a ~/logs/vbench_reward_audit.log
