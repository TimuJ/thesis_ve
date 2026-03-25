#!/bin/bash
# Download DOVE test datasets from Google Drive.
# Requires: pip install gdown
#
# Usage: bash experiments/baselines/data/download_dove.sh [dataset_name]
# If no dataset specified, downloads UDM10 by default.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# DOVE Google Drive folder: https://drive.google.com/drive/folders/1yNKG6rtTNtZQY8qL74GoQwA0jgjBUEby
# Individual dataset IDs (from DOVE README)
declare -A DATASET_IDS=(
    ["UDM10"]="1AmGVSCwMm_OFPd3DKgNyTwj0GG2H-tG4"
    ["SPMCS"]="1b2uktCFPKS-R1fTecWcLFcOnmUFIBNWT"
    ["YouHQ40"]="1zO23UCStxL3htPJQcDUUnUeMvDrysLTh"
    ["RealVSR"]="1wr4tTiCvQlqdYPeU1dmnjb5KFY4VjGCO"
    ["MVSR4x"]="16sesBD_9Xx_5Grtx18nosBw1w94KlpQt"
    ["VideoLQ"]="1lh0vkU_llxE0un1OigJ0DWPQwt1i68Vn"
)

DATASET="${1:-UDM10}"

if [[ ! -v "DATASET_IDS[$DATASET]" ]]; then
    echo "Error: Unknown dataset '$DATASET'. Available: ${!DATASET_IDS[*]}"
    exit 1
fi

if [[ -d "$DATASET" ]]; then
    echo "Dataset '$DATASET' already exists at $SCRIPT_DIR/$DATASET, skipping."
    exit 0
fi

echo "Downloading $DATASET from DOVE Google Drive..."

# Check gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Error: gdown not found. Install with: pip install gdown"
    exit 1
fi

GDRIVE_ID="${DATASET_IDS[$DATASET]}"

# Download archive and extract
ARCHIVE="${DATASET}.zip"
gdown "$GDRIVE_ID" -O "$ARCHIVE" || {
    echo ""
    echo "gdown failed (Google Drive quota limit). Manual download:"
    echo "  1. Go to: https://drive.google.com/drive/folders/1yNKG6rtTNtZQY8qL74GoQwA0jgjBUEby"
    echo "  2. Download the $DATASET folder"
    echo "  3. Extract to: $SCRIPT_DIR/$DATASET/"
    echo "  Expected structure: $DATASET/GT/ and $DATASET/LQ/"
    rm -f "$ARCHIVE"
    exit 1
}

echo "Extracting $ARCHIVE..."
unzip -q "$ARCHIVE" -d .
rm -f "$ARCHIVE"

echo "Done. Dataset at: $SCRIPT_DIR/$DATASET/"
echo "Expected structure:"
echo "  $DATASET/GT/   (ground truth HR frames, per-clip subdirs)"
echo "  $DATASET/LQ/   (low-res 4x input frames, per-clip subdirs)"
