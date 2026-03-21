"""
Central path configuration. Edit DATA_ROOT and PROJECT_ROOT for your machine.
"""
from pathlib import Path
import os

# Auto-detect: override with environment variables if set
PROJECT_ROOT = Path(os.environ.get("VSR_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA_ROOT = Path(os.environ.get("VSR_DATA_ROOT", PROJECT_ROOT / "data"))

# Dataset paths — common VSR benchmarks
REDS_ROOT = DATA_ROOT / "REDS"
REDS_TRAIN_LR = REDS_ROOT / "train" / "train_sharp_bicubic" / "X4"
REDS_TRAIN_HR = REDS_ROOT / "train" / "train_sharp"
REDS_VAL_LR = REDS_ROOT / "val" / "val_sharp_bicubic" / "X4"
REDS_VAL_HR = REDS_ROOT / "val" / "val_sharp"

VIMEO90K_ROOT = DATA_ROOT / "vimeo_septuplet"
VIMEO90K_LR = VIMEO90K_ROOT / "sequences_LR"
VIMEO90K_HR = VIMEO90K_ROOT / "sequences"

VID4_ROOT = DATA_ROOT / "Vid4"
UDM10_ROOT = DATA_ROOT / "UDM10"

# Model checkpoints
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
