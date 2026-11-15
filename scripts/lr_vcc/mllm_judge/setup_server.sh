#!/usr/bin/env bash
# Stand up the Qwen3-VL consistency judge on the lab server (A100). Run ON the
# server. Keeps the existing lrvcc/romatch env untouched by using a DEDICATED env.
#
# Constraints baked in:
#   - huggingface.co is BLOCKED on the box; only https://hf-mirror.com works.
#   - Qwen3-VL weights are NON-gated Apache-2.0, so the mirror serves them (no 403).
#   - transformers >= 4.57.0 is required for Qwen3VLForConditionalGeneration.
#   - Use bf16, NOT FP8 (FP8 weights are not loadable via transformers yet; the
#     FP8 path is vLLM-only, and vLLM is nondeterministic -> wrong for a metric).
set -euo pipefail

ENV=mllmjudge
CONDA=~/miniconda3
export HF_ENDPOINT=https://hf-mirror.com

# 1) Dedicated env cloned from a known-good torch 2.6.0+cu124 / py3.10 base.
if [ ! -d "$CONDA/envs/$ENV" ]; then
  conda create -y -n "$ENV" --clone seedvr310
fi
PY="$CONDA/envs/$ENV/bin/python"
PIP="$CONDA/envs/$ENV/bin/pip"

# 2) Qwen3-VL runtime deps. transformers>=4.57 from PyPI (not GitHub: github.com
#    may be blocked like huggingface.co). If the pinned version is unavailable,
#    fall back to the newest transformers PyPI offers and re-check the class import.
$PIP install -U "transformers>=4.57.0" "qwen-vl-utils" "accelerate"
$PY - <<'PYEOF'
import transformers
print("transformers", transformers.__version__)
from transformers import Qwen3VLForConditionalGeneration  # import sanity check
print("Qwen3VLForConditionalGeneration import OK")
PYEOF

# 3) Download weights via the mirror. 8B first for dev/iteration; 32B for the
#    reported metric. hf (new) or huggingface-cli (older) — try hf first.
dl() { "$CONDA/envs/$ENV/bin/hf" download "$1" --local-dir "$HOME/models/$(basename "$1")" \
       || "$CONDA/envs/$ENV/bin/huggingface-cli" download "$1" --local-dir "$HOME/models/$(basename "$1")"; }
dl Qwen/Qwen3-VL-8B-Instruct
# dl Qwen/Qwen3-VL-32B-Instruct   # uncomment for the metric run (~65 GB, bf16 one A100-80GB)

echo "SETUP_DONE env=$ENV models under ~/models/"
