#!/bin/bash
# Setup MGLD-VSR: clone repo, create conda env, download checkpoints.
# Run from repo root: bash experiments/baselines/mgld_vsr/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"

# 1. Clone repo
if [[ -d "$REPO_DIR" ]]; then
    echo "Repo already cloned at $REPO_DIR"
else
    echo "Cloning MGLD-VSR..."
    git clone https://github.com/IanYeung/MGLD-VSR.git "$REPO_DIR"
fi

# 2. Create conda env
if conda env list | grep -q "^mgldvsr "; then
    echo "Conda env 'mgldvsr' already exists"
else
    echo "Creating conda env 'mgldvsr'..."
    cd "$REPO_DIR"
    conda env create --file environment.yaml
    cd - > /dev/null
fi

echo "Installing additional dependencies..."
eval "$(conda shell.bash hook)"
conda activate mgldvsr
conda install xformers -c xformers/label/dev -y || echo "Warning: xformers install failed, continuing..."
pip install mim && mim install mmcv
pip install -e "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers"
pip install -e "git+https://github.com/openai/CLIP.git@main#egg=clip"

# 3. Download checkpoints from HuggingFace
CKPT_DIR="$REPO_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

if [[ -f "$CKPT_DIR/mgldvsr_unet.ckpt" ]]; then
    echo "Checkpoints already downloaded"
else
    echo "Downloading checkpoints from HuggingFace..."
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download Iceclear/MGLD-VSR --local-dir "$CKPT_DIR"
    else
        echo "huggingface-cli not found. Install with: pip install huggingface_hub"
        echo "Or manually download from: https://huggingface.co/Iceclear/MGLD-VSR"
        echo "Place files in: $CKPT_DIR/"
        echo "Required: mgldvsr_unet.ckpt, DAPE.pth, raft-things.pth"
    fi
fi

echo ""
echo "Setup complete. Activate with: conda activate mgldvsr"
echo "Required checkpoints in $CKPT_DIR/:"
echo "  - mgldvsr_unet.ckpt"
echo "  - DAPE.pth"
echo "  - raft-things.pth"
echo "  - Video VAE checkpoint"
