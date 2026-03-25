#!/bin/bash
# Run MGLD-VSR inference on a dataset.
# Usage: bash experiments/baselines/mgld_vsr/run_inference.sh --input <LQ_dir> --output <output_dir>

set -euo pipefail

# Parse args
INPUT=""
OUTPUT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --input) INPUT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --input <LQ_frames_dir> --output <output_dir>"
    echo "  LQ_frames_dir should contain per-clip subdirectories with frames"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"
CKPT_DIR="$REPO_DIR/checkpoints"
LATENT_DIR="$SCRIPT_DIR/.tmp_latents"

# Verify checkpoints exist
for ckpt in mgldvsr_unet.ckpt; do
    if [[ ! -f "$CKPT_DIR/$ckpt" ]]; then
        echo "Error: checkpoint not found: $CKPT_DIR/$ckpt"
        echo "Run setup.sh first."
        exit 1
    fi
done

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate mgldvsr

mkdir -p "$OUTPUT"
mkdir -p "$LATENT_DIR"

# MGLD-VSR expects --seqs-path pointing to the LQ directory
# and outputs to --outdir. It processes all clips found in seqs-path.
cd "$REPO_DIR"
python scripts/vsr_val_ddpm_text_T_vqganfin_w_latent.py \
    --config configs/mgldvsr/mgldvsr_512_realbasicvsr_deg.yaml \
    --ckpt "$CKPT_DIR/mgldvsr_unet.ckpt" \
    --vqgan_ckpt "$CKPT_DIR/vqgan_ckpt.ckpt" \
    --seqs-path "$INPUT" \
    --outdir "$OUTPUT" \
    --latent-dir "$LATENT_DIR" \
    --ddpm_steps 50 \
    --dec_w 1.0 \
    --colorfix_type adain \
    --select_idx 0 \
    --n_gpus 1

cd - > /dev/null

# Cleanup temp latents
rm -rf "$LATENT_DIR"

echo "Done. Results at: $OUTPUT"
