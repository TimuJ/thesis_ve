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
    ["UDM10"]="PLACEHOLDER_UDM10_ID"
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

# Download and extract
gdown --folder "$GDRIVE_ID" -O "$DATASET" || {
    echo ""
    echo "gdown failed (Google Drive quota limit). Manual download:"
    echo "  1. Go to: https://drive.google.com/drive/folders/1yNKG6rtTNtZQY8qL74GoQwA0jgjBUEby"
    echo "  2. Download the $DATASET folder"
    echo "  3. Extract to: $SCRIPT_DIR/$DATASET/"
    echo "  Expected structure: $DATASET/GT/ and $DATASET/LQ/"
    exit 1
}

echo "Done. Dataset at: $SCRIPT_DIR/$DATASET/"
echo "Expected structure:"
echo "  $DATASET/GT/   (ground truth HR frames, per-clip subdirs)"
echo "  $DATASET/LQ/   (low-res 4x input frames, per-clip subdirs)"
