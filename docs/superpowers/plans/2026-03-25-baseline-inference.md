# Baseline Inference Infrastructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create reproducible baseline inference scripts for Upscale-A-Video and MGLD-VSR on DOVE UDM10 (4x), with shared evaluation producing consistent PSNR/SSIM/LPIPS metrics.

**Architecture:** Self-contained shell scripts per model (setup + inference) with separate conda envs. A shared Python evaluation script loads output frames vs GT and computes metrics via `src/evaluation/metrics.py`. All large files (data, checkpoints, cloned repos, results) are gitignored; only scripts get committed.

**Tech Stack:** Bash scripts, conda, gdown, Python (numpy, Pillow, lpips, scikit-image), ffmpeg

**Spec:** `docs/superpowers/specs/2026-03-25-baseline-inference-design.md`

---

## File Map

**Create:**
- `experiments/baselines/data/download_dove.sh` — dataset download script
- `experiments/baselines/data/.gitignore` — ignore downloaded data
- `experiments/baselines/data/README.md` — manual download fallback
- `experiments/baselines/upscale_a_video/setup.sh` — clone, env, checkpoints
- `experiments/baselines/upscale_a_video/run_inference.sh` — inference wrapper
- `experiments/baselines/mgld_vsr/setup.sh` — clone, env, checkpoints
- `experiments/baselines/mgld_vsr/run_inference.sh` — inference wrapper
- `experiments/baselines/evaluate.py` — shared metric computation
- `experiments/baselines/README.md` — end-to-end workflow docs
- `tests/test_evaluate.py` — tests for evaluate.py and lpips_score

**Modify:**
- `src/evaluation/metrics.py` — add `lpips_score()`, integrate into `evaluate_sequence()`/`evaluate_dataset()`
- `.gitignore` — add baselines exclusions

Note: `requirements-gpu.txt` already includes `lpips` (line 29) — no changes needed.

**Move:**
- `PhD mentor reports/` → `experiments/PhD_mentor_reports/`

---

### Task 1: Directory structure and move PhD mentor reports

**Files:**
- Move: `PhD mentor reports/` → `experiments/PhD_mentor_reports/`
- Modify: `.gitignore`

- [ ] **Step 1: Create experiments directory and move PhD mentor reports**

```bash
mkdir -p experiments/baselines/data
mkdir -p experiments/baselines/upscale_a_video
mkdir -p experiments/baselines/mgld_vsr
mkdir -p experiments/baselines/results
git mv "PhD mentor reports" experiments/PhD_mentor_reports
```

- [ ] **Step 2: Update .gitignore with baselines exclusions**

Add to `.gitignore`:
```
# Baselines — large files
experiments/baselines/data/UDM10/
experiments/baselines/results/
experiments/baselines/upscale_a_video/repo/
experiments/baselines/mgld_vsr/repo/
```

- [ ] **Step 3: Commit**

```bash
git add experiments/ .gitignore
git commit -m "feat: create experiments/ structure, move PhD mentor reports"
```

---

### Task 2: Dataset download script

**Files:**
- Create: `experiments/baselines/data/download_dove.sh`
- Create: `experiments/baselines/data/.gitignore`
- Create: `experiments/baselines/data/README.md`

- [ ] **Step 1: Create data .gitignore**

Create `experiments/baselines/data/.gitignore`:
```
# Ignore downloaded datasets — download via download_dove.sh
UDM10/
SPMCS/
YouHQ40/
RealVSR/
MVSR4x/
VideoLQ/
```

- [ ] **Step 2: Create download script**

Create `experiments/baselines/data/download_dove.sh`:
```bash
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
```

Note: The `PLACEHOLDER_UDM10_ID` must be replaced with the actual Google Drive file/folder ID when first tested on the GPU machine. The DOVE README links to `https://drive.google.com/drive/folders/1yNKG6rtTNtZQY8qL74GoQwA0jgjBUEby` — the exact subfolder ID for UDM10 can be found there.

- [ ] **Step 3: Create README with manual download fallback**

Create `experiments/baselines/data/README.md`:
```markdown
# Baseline Datasets

## Download

```bash
bash experiments/baselines/data/download_dove.sh UDM10
```

Requires `gdown`: `pip install gdown`

## Manual Download (if gdown hits quota)

1. Go to [DOVE Google Drive](https://drive.google.com/drive/folders/1yNKG6rtTNtZQY8qL74GoQwA0jgjBUEby)
2. Download the UDM10 folder
3. Place at `experiments/baselines/data/UDM10/`

Expected structure:
```
UDM10/
├── GT/          # per-clip subdirs with HR frames
└── LQ/          # per-clip subdirs with LQ frames
```
```

- [ ] **Step 4: Commit**

```bash
git add experiments/baselines/data/
git commit -m "feat: add DOVE dataset download script and README"
```

---

### Task 3: Add LPIPS to src/evaluation/metrics.py

**Files:**
- Modify: `src/evaluation/metrics.py`
- Create: `tests/test_lpips.py`

- [ ] **Step 1: Write failing test for lpips_score**

Create `tests/test_lpips.py`:
```python
"""Tests for LPIPS metric."""
import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.evaluation.metrics import lpips_score


@pytest.mark.skipif(not HAS_TORCH, reason="LPIPS requires torch")
def test_lpips_identical():
    """Identical images should give LPIPS close to 0."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = lpips_score(img, img)
    assert result < 0.01


@pytest.mark.skipif(not HAS_TORCH, reason="LPIPS requires torch")
def test_lpips_different():
    """Different images should give positive LPIPS."""
    img1 = np.zeros((64, 64, 3), dtype=np.uint8)
    img2 = np.full((64, 64, 3), 255, dtype=np.uint8)
    result = lpips_score(img1, img2)
    assert result > 0.0


def test_lpips_disabled_excludes_key():
    """When compute_lpips=False, evaluate_sequence should not include LPIPS."""
    from src.evaluation.metrics import evaluate_sequence
    frames = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
    gts = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
    result = evaluate_sequence(frames, gts, compute_lpips=False)
    assert "LPIPS_mean" not in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lpips.py -v`
Expected: FAIL — `lpips_score` not importable

- [ ] **Step 3: Implement lpips_score and update evaluate_sequence/evaluate_dataset**

Modify `src/evaluation/metrics.py` — add at top after existing imports:
```python
try:
    import torch
    import lpips as lpips_lib
    _lpips_net = None

    def _get_lpips_net():
        global _lpips_net
        if _lpips_net is None:
            _lpips_net = lpips_lib.LPIPS(net="alex")
            if torch.cuda.is_available():
                _lpips_net = _lpips_net.cuda()
        return _lpips_net

    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
```

Add new function after `temporal_consistency`:
```python
def lpips_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Learned Perceptual Image Patch Similarity. Lower = more similar.
    Requires torch and lpips packages. Images should be (H, W, C) uint8."""
    if not HAS_LPIPS:
        raise RuntimeError("lpips not available — install torch and lpips")
    net = _get_lpips_net()
    device = next(net.parameters()).device

    def _to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t = t * 2.0 - 1.0  # normalize to [-1, 1]
        return t.unsqueeze(0).to(device)

    with torch.no_grad():
        score = net(_to_tensor(pred), _to_tensor(gt))
    return float(score.item())
```

Replace the entire `evaluate_sequence` function (lines 36-62 of the current file) with:
```python
def evaluate_sequence(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    compute_lpips: bool = True,
) -> dict:
    """
    Evaluate a single video sequence.

    Args:
        preds: List of predicted HR frames (H, W, C)
        gts: List of ground-truth HR frames (H, W, C)
        compute_lpips: Whether to compute LPIPS (requires torch + lpips packages)

    Returns:
        Dict with PSNR_mean, SSIM_mean, per-frame scores, temporal_consistency,
        and optionally LPIPS_mean if compute_lpips=True and lpips is available.
    """
    assert len(preds) == len(gts), f"Mismatch: {len(preds)} preds vs {len(gts)} gts"

    psnr_scores = [psnr(p, g) for p, g in zip(preds, gts)]
    ssim_scores = [ssim(p, g) for p, g in zip(preds, gts)]
    t_consist = temporal_consistency(preds)

    result = {
        "PSNR_mean": float(np.mean(psnr_scores)),
        "SSIM_mean": float(np.mean(ssim_scores)),
        "PSNR_per_frame": psnr_scores,
        "SSIM_per_frame": ssim_scores,
        "temporal_consistency": t_consist,
    }

    if compute_lpips and HAS_LPIPS:
        lpips_scores = [lpips_score(p, g) for p, g in zip(preds, gts)]
        result["LPIPS_mean"] = float(np.mean(lpips_scores))
        result["LPIPS_per_frame"] = lpips_scores

    return result
```

Replace the entire `evaluate_dataset` function (lines 65-101 of the current file) with:
```python
def evaluate_dataset(
    all_preds: dict[str, list[np.ndarray]],
    all_gts: dict[str, list[np.ndarray]],
    compute_lpips: bool = True,
) -> dict:
    """
    Evaluate across all sequences in a dataset.

    Args:
        all_preds: {sequence_name: [pred_frames]}
        all_gts: {sequence_name: [gt_frames]}
        compute_lpips: Whether to compute LPIPS

    Returns:
        Dict with per-sequence and overall metrics
    """
    per_sequence = {}
    psnr_means, ssim_means, lpips_means = [], [], []

    for seq_name in all_gts:
        if seq_name not in all_preds:
            print(f"Warning: sequence '{seq_name}' missing from predictions, skipping")
            continue
        result = evaluate_sequence(all_preds[seq_name], all_gts[seq_name], compute_lpips=compute_lpips)
        per_sequence[seq_name] = result
        psnr_means.append(result["PSNR_mean"])
        ssim_means.append(result["SSIM_mean"])
        if "LPIPS_mean" in result:
            lpips_means.append(result["LPIPS_mean"])

    overall = {
        "PSNR_mean": float(np.mean(psnr_means)) if psnr_means else 0.0,
        "SSIM_mean": float(np.mean(ssim_means)) if ssim_means else 0.0,
        "num_sequences": len(per_sequence),
    }
    if lpips_means:
        overall["LPIPS_mean"] = float(np.mean(lpips_means))

    return {
        "overall": overall,
        "per_sequence": per_sequence,
    }
```

- [ ] **Step 4: Run all metrics tests**

Run: `pytest tests/test_metrics.py tests/test_lpips.py -v`
Expected: All pass (LPIPS tests may skip if no torch on M1 local)

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/test_lpips.py
git commit -m "feat: add LPIPS metric to evaluation pipeline"
```

---

### Task 4: Shared evaluation script (evaluate.py)

**Files:**
- Create: `experiments/baselines/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write test for evaluate.py logic**

Create `tests/test_evaluate.py`:
```python
"""Tests for baselines evaluate.py frame loading and metric aggregation."""
import json
import os
import tempfile
import numpy as np
from PIL import Image
import pytest


def _create_test_dataset(root, clip_names, num_frames=3, size=(64, 64)):
    """Create a fake dataset with GT and results frame dirs."""
    gt_dir = os.path.join(root, "gt")
    res_dir = os.path.join(root, "results")
    for clip in clip_names:
        os.makedirs(os.path.join(gt_dir, clip), exist_ok=True)
        os.makedirs(os.path.join(res_dir, clip), exist_ok=True)
        for i in range(num_frames):
            img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(gt_dir, clip, f"{i:08d}.png"))
            Image.fromarray(img).save(os.path.join(res_dir, clip, f"{i:08d}.png"))
    return gt_dir, res_dir


def test_evaluate_produces_json():
    """evaluate.py should produce a valid JSON with expected keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_dir, res_dir = _create_test_dataset(tmpdir, ["clip_a", "clip_b"])
        out_json = os.path.join(tmpdir, "metrics.json")

        # Import and run main
        import importlib.util
        from pathlib import Path
        eval_path = str(Path(__file__).resolve().parents[1] / "experiments" / "baselines" / "evaluate.py")
        spec = importlib.util.spec_from_file_location(
            "evaluate", eval_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mod.main(["--results", res_dir, "--gt", gt_dir, "--output", out_json, "--no-lpips"])

        assert os.path.exists(out_json)
        with open(out_json) as f:
            data = json.load(f)
        assert "overall" in data
        assert "per_clip" in data
        assert "PSNR_mean" in data["overall"]
        assert "SSIM_mean" in data["overall"]
        assert len(data["per_clip"]) == 2


def test_evaluate_identical_frames_high_psnr():
    """Identical GT and results should produce high PSNR."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_dir, res_dir = _create_test_dataset(tmpdir, ["clip_a"])
        out_json = os.path.join(tmpdir, "metrics.json")

        import importlib.util
        from pathlib import Path
        eval_path = str(Path(__file__).resolve().parents[1] / "experiments" / "baselines" / "evaluate.py")
        spec = importlib.util.spec_from_file_location(
            "evaluate", eval_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mod.main(["--results", res_dir, "--gt", gt_dir, "--output", out_json, "--no-lpips"])

        with open(out_json) as f:
            data = json.load(f)
        assert data["overall"]["PSNR_mean"] > 50 or data["overall"]["PSNR_mean"] == float("inf")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL — `experiments/baselines/evaluate.py` does not exist

- [ ] **Step 3: Implement evaluate.py**

Create `experiments/baselines/evaluate.py`:
```python
#!/usr/bin/env python3
"""Shared evaluation script for baseline VSR models.

Computes PSNR, SSIM, and optionally LPIPS between model output frames
and ground truth frames. Outputs JSON with per-clip and overall metrics.

Usage:
    python experiments/baselines/evaluate.py \
        --results experiments/baselines/results/upscale_a_video/UDM10 \
        --gt experiments/baselines/data/UDM10/GT \
        --output experiments/baselines/results/upscale_a_video/UDM10_metrics.json
"""
import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
from PIL import Image

# Add repo root to path so we can import src/
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import evaluate_sequence


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_frames(clip_dir: str) -> list[np.ndarray]:
    """Load all image frames from a directory, sorted by filename."""
    clip_path = Path(clip_dir)
    files = sorted(
        f for f in clip_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No image files found in {clip_dir}")
    return [np.array(Image.open(f).convert("RGB")) for f in files]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate VSR baseline results")
    parser.add_argument("--results", required=True, help="Path to model output frames")
    parser.add_argument("--gt", required=True, help="Path to ground truth frames")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--no-lpips", action="store_true", help="Skip LPIPS computation")
    args = parser.parse_args(argv)

    gt_path = Path(args.gt)
    results_path = Path(args.results)
    compute_lpips = not args.no_lpips

    # Find clips — subdirectories in GT
    clip_dirs = sorted(d for d in gt_path.iterdir() if d.is_dir())
    if not clip_dirs:
        print(f"Error: No clip subdirectories found in {gt_path}")
        sys.exit(1)

    per_clip = {}
    psnr_means, ssim_means, lpips_means = [], [], []

    for clip_dir in clip_dirs:
        clip_name = clip_dir.name
        results_clip = results_path / clip_name

        if not results_clip.exists():
            print(f"Warning: clip '{clip_name}' missing from results, skipping")
            continue

        print(f"Evaluating clip: {clip_name}")
        gt_frames = load_frames(str(clip_dir))
        pred_frames = load_frames(str(results_clip))

        # Truncate to shorter length if mismatch
        n = min(len(gt_frames), len(pred_frames))
        if len(gt_frames) != len(pred_frames):
            print(f"  Warning: frame count mismatch ({len(pred_frames)} vs {len(gt_frames)} GT), using first {n}")
        gt_frames = gt_frames[:n]
        pred_frames = pred_frames[:n]

        result = evaluate_sequence(pred_frames, gt_frames, compute_lpips=compute_lpips)
        clip_metrics = {
            "PSNR_mean": result["PSNR_mean"],
            "SSIM_mean": result["SSIM_mean"],
        }
        if "LPIPS_mean" in result:
            clip_metrics["LPIPS_mean"] = result["LPIPS_mean"]
            lpips_means.append(result["LPIPS_mean"])

        per_clip[clip_name] = clip_metrics
        psnr_means.append(result["PSNR_mean"])
        ssim_means.append(result["SSIM_mean"])

    overall = {
        "PSNR_mean": float(np.mean(psnr_means)) if psnr_means else 0.0,
        "SSIM_mean": float(np.mean(ssim_means)) if ssim_means else 0.0,
    }
    if lpips_means:
        overall["LPIPS_mean"] = float(np.mean(lpips_means))

    output = {
        "overall": overall,
        "per_clip": per_clip,
    }

    # Write JSON
    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"Overall — PSNR: {overall['PSNR_mean']:.2f}, SSIM: {overall['SSIM_mean']:.4f}", end="")
    if "LPIPS_mean" in overall:
        print(f", LPIPS: {overall['LPIPS_mean']:.4f}")
    else:
        print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/baselines/evaluate.py tests/test_evaluate.py
git commit -m "feat: add shared evaluation script for baselines"
```

---

### Task 5: Upscale-A-Video setup and inference scripts

**Files:**
- Create: `experiments/baselines/upscale_a_video/setup.sh`
- Create: `experiments/baselines/upscale_a_video/run_inference.sh`

- [ ] **Step 1: Create setup.sh**

Create `experiments/baselines/upscale_a_video/setup.sh`:
```bash
#!/bin/bash
# Setup Upscale-A-Video: clone repo, create conda env, download checkpoints.
# Run from repo root: bash experiments/baselines/upscale_a_video/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"

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
```

- [ ] **Step 2: Create run_inference.sh**

Create `experiments/baselines/upscale_a_video/run_inference.sh`:
```bash
#!/bin/bash
# Run Upscale-A-Video inference on a dataset.
# Usage: bash experiments/baselines/upscale_a_video/run_inference.sh --input <LQ_dir> --output <output_dir>

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
TEMP_DIR="$SCRIPT_DIR/.tmp_videos"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate uav

mkdir -p "$OUTPUT"
mkdir -p "$TEMP_DIR/input"
mkdir -p "$TEMP_DIR/output"

# Process each clip
for clip_dir in "$INPUT"/*/; do
    clip_name=$(basename "$clip_dir")
    echo "Processing clip: $clip_name"

    clip_output="$OUTPUT/$clip_name"
    mkdir -p "$clip_output"

    # Assemble frames into video (Upscale-A-Video expects video input)
    input_video="$TEMP_DIR/input/${clip_name}.mp4"
    ffmpeg -y -i "$clip_dir/%08d.png" -c:v libx264 -pix_fmt yuv420p "$input_video" 2>/dev/null

    # Run inference
    # Note: verify actual CLI flags from repo's argparse before first run.
    # Flags below are from the repo README; if they fail, check:
    #   python inference_upscale_a_video.py --help
    cd "$REPO_DIR"
    python inference_upscale_a_video.py \
        -i "$input_video" \
        -o "$TEMP_DIR/output" \
        -n 150 -g 7 -s 30

    # Extract output frames back to per-clip dir
    output_video=$(find "$TEMP_DIR/output" -name "*.mp4" -newer "$input_video" | head -1)
    if [[ -n "$output_video" ]]; then
        ffmpeg -y -i "$output_video" "$clip_output/%08d.png" 2>/dev/null
        echo "  Saved ${clip_name} frames to $clip_output"
    else
        echo "  Warning: No output video found for $clip_name"
    fi

    # Cleanup temp
    rm -f "$input_video"
    rm -f "$TEMP_DIR/output"/*.mp4
    cd - > /dev/null
done

rm -rf "$TEMP_DIR"
echo "Done. Results at: $OUTPUT"
```

Note: The ffmpeg frame-to-video assembly uses `%08d.png` pattern — this must be verified against the actual DOVE frame naming convention when first run. Adjust the pattern if DOVE uses a different naming scheme.

- [ ] **Step 3: Make scripts executable and commit**

```bash
chmod +x experiments/baselines/upscale_a_video/setup.sh
chmod +x experiments/baselines/upscale_a_video/run_inference.sh
git add experiments/baselines/upscale_a_video/
git commit -m "feat: add Upscale-A-Video setup and inference scripts"
```

---

### Task 6: MGLD-VSR setup and inference scripts

**Files:**
- Create: `experiments/baselines/mgld_vsr/setup.sh`
- Create: `experiments/baselines/mgld_vsr/run_inference.sh`

- [ ] **Step 1: Create setup.sh**

Create `experiments/baselines/mgld_vsr/setup.sh`:
```bash
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
```

- [ ] **Step 2: Create run_inference.sh**

Create `experiments/baselines/mgld_vsr/run_inference.sh`:
```bash
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
```

Note: The `--vqgan_ckpt` filename (`vqgan_ckpt.ckpt`) is a placeholder — verify the actual filename from the HuggingFace download. The output directory structure from MGLD-VSR may not perfectly match the per-clip subdirectory format expected by evaluate.py — this will need verification on first run and the script may need a post-processing step to reorganize frames.

- [ ] **Step 3: Make scripts executable and commit**

```bash
chmod +x experiments/baselines/mgld_vsr/setup.sh
chmod +x experiments/baselines/mgld_vsr/run_inference.sh
git add experiments/baselines/mgld_vsr/
git commit -m "feat: add MGLD-VSR setup and inference scripts"
```

---

### Task 7: Baselines README

**Files:**
- Create: `experiments/baselines/README.md`

- [ ] **Step 1: Create README**

Create `experiments/baselines/README.md`:
```markdown
# VSR Baseline Inference

Reproducible baseline inference for diffusion-based VSR methods on DOVE benchmarks.

## Quick Start

### 1. Download dataset

```bash
bash experiments/baselines/data/download_dove.sh UDM10
```

### 2. Setup models (one-time per model)

```bash
bash experiments/baselines/upscale_a_video/setup.sh
bash experiments/baselines/mgld_vsr/setup.sh
```

### 3. Run inference

```bash
bash experiments/baselines/upscale_a_video/run_inference.sh \
    --input experiments/baselines/data/UDM10/LQ \
    --output experiments/baselines/results/upscale_a_video/UDM10

bash experiments/baselines/mgld_vsr/run_inference.sh \
    --input experiments/baselines/data/UDM10/LQ \
    --output experiments/baselines/results/mgld_vsr/UDM10
```

### 4. Evaluate

```bash
python experiments/baselines/evaluate.py \
    --results experiments/baselines/results/upscale_a_video/UDM10 \
    --gt experiments/baselines/data/UDM10/GT \
    --output experiments/baselines/results/upscale_a_video/UDM10_metrics.json

python experiments/baselines/evaluate.py \
    --results experiments/baselines/results/mgld_vsr/UDM10 \
    --gt experiments/baselines/data/UDM10/GT \
    --output experiments/baselines/results/mgld_vsr/UDM10_metrics.json
```

## Models

| Model | Paper | Env | Notes |
|-------|-------|-----|-------|
| Upscale-A-Video | CVPR 2024 | `uav` | Diffusion + text prompts, expects video input |
| MGLD-VSR | ECCV 2024 | `mgldvsr` | Motion-guided latent diffusion |

## Adding a new model

1. Create `experiments/baselines/<model_name>/setup.sh` and `run_inference.sh`
2. Follow the same `--input <LQ_dir> --output <output_dir>` interface
3. Output frames as per-clip subdirectories with PNG files

## Adding a new dataset

1. Download into `experiments/baselines/data/<dataset>/` with `GT/` and `LQ/` subdirs
2. Run the same inference and evaluate commands pointing to the new paths
```

- [ ] **Step 2: Commit**

```bash
git add experiments/baselines/README.md
git commit -m "docs: add baselines README with end-to-end workflow"
```

---

### Task 8: Verify everything works locally

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All pass. LPIPS tests skip on M1 Mac (no GPU torch). evaluate.py tests pass with `--no-lpips`.

- [ ] **Step 2: Verify directory structure**

```bash
find experiments/ -type f | sort
```

Expected output:
```
experiments/PhD_mentor_reports/Week 1 Feb 23 - Mar 4/...
experiments/baselines/README.md
experiments/baselines/data/.gitignore
experiments/baselines/data/README.md
experiments/baselines/data/download_dove.sh
experiments/baselines/evaluate.py
experiments/baselines/mgld_vsr/run_inference.sh
experiments/baselines/mgld_vsr/setup.sh
experiments/baselines/upscale_a_video/run_inference.sh
experiments/baselines/upscale_a_video/setup.sh
```

- [ ] **Step 3: Verify .gitignore works**

```bash
git status
```

Expected: No untracked data/results/repo directories.

- [ ] **Step 4: Push to remote**

```bash
git push origin main
```
