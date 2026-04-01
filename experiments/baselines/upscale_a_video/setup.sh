#!/bin/bash
# Setup Upscale-A-Video: clone repo, create conda env, download checkpoints.
# Run from repo root: bash experiments/baselines/upscale_a_video/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"

# Auto-detect conda (support local miniconda installs)
if ! command -v conda &> /dev/null; then
    for p in "$HOME/miniconda3" "/data/disk1/timur/miniconda3"; do
        if [[ -f "$p/bin/conda" ]]; then
            eval "$("$p/bin/conda" shell.bash hook)"
            break
        fi
    done
fi

# 1. Clone repo
if [[ -d "$REPO_DIR" ]]; then
    echo "Repo already cloned at $REPO_DIR"
else
    echo "Cloning Upscale-A-Video..."
    git clone https://github.com/sczhou/Upscale-A-Video.git "$REPO_DIR"
fi

# 2. Create conda env
if conda env list | grep -q "^uav "; then
    echo "Conda env 'uav' already exists"
else
    echo "Creating conda env 'uav'..."
    conda create -n uav python=3.9 -y
fi

echo "Installing dependencies..."
eval "$(conda shell.bash hook)"
conda activate uav
pip install -r "$REPO_DIR/requirements.txt"

# 3. Download checkpoints
CKPT_DIR="$REPO_DIR/pretrained_models/upscale_a_video"
if [[ -d "$CKPT_DIR/unet" ]]; then
    echo "Checkpoints already downloaded"
else
    echo "Downloading checkpoints..."
    echo "Please download pretrained models from the Google Drive link in the repo README:"
    echo "  https://github.com/sczhou/Upscale-A-Video#pretrained-models"
    echo "Place them at: $CKPT_DIR/"
    echo ""
    echo "Expected subdirs: low_res_scheduler/ propagator/ scheduler/ text_encoder/ tokenizer/ unet/ vae/"
fi

echo ""
echo "Setup complete. Activate with: conda activate uav"
