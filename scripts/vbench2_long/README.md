# VBench 2.0 Long-Video Evaluation

Patched version of [VBench](https://github.com/Vchitect/VBench) for long-video quality evaluation using the `vbench2_beta_long` module.

## Setup

```bash
# 1. Create conda env
conda create -n vbench python=3.10 -y
conda activate vbench

# 2. Install PyTorch (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements_long.txt

# 4. Install VBench source (needed for vbench2_beta_long imports)
export PYTHONPATH=/path/to/this/repo:$PYTHONPATH
```

## Usage

### Evaluate custom videos (SR outputs, generated videos, etc.)

Only the **Quality Score** dimensions are meaningful for custom videos without text prompts:

```bash
python vbench2_beta_long/eval_long.py \
    --videos_path /path/to/mp4/videos/ \
    --output_path /path/to/output/dim_name \
    --full_json_dir vbench2_beta_long/VBench_full_info.json \
    --dimension imaging_quality \
    --mode long_custom_input
```

### Available Quality Score dimensions (7 total)

| Dimension | What it measures |
|-----------|-----------------|
| `imaging_quality` | Image quality (MUSIQ-based) |
| `motion_smoothness` | Motion smoothness across frames (RAFT optical flow) |
| `temporal_flickering` | Frame-to-frame pixel stability |
| `aesthetic_quality` | Aesthetic appeal (LAION aesthetic predictor) |
| `dynamic_degree` | Amount of motion/dynamics |
| `subject_consistency` | Subject appearance consistency across frames (DINOv2) |
| `background_consistency` | Background consistency across frames (DreamSim) |

### Semantic Score dimensions (9 total) — require text prompts

These are designed for text-to-video generation benchmarking and require prompt metadata.
Not applicable for super-resolution or custom video evaluation:
`overall_consistency`, `appearance_style`, `temporal_style`, `human_action`,
`color`, `object_class`, `multiple_objects`, `spatial_relationship`, `scene`.

### Batch evaluation script

```bash
#!/bin/bash
VIDEOS=/path/to/videos
OUTPUT=/path/to/results
GPU=0

for dim in imaging_quality motion_smoothness temporal_flickering \
           aesthetic_quality dynamic_degree subject_consistency \
           background_consistency; do
    echo "=== $dim ==="
    CUDA_VISIBLE_DEVICES=$GPU python vbench2_beta_long/eval_long.py \
        --videos_path "$VIDEOS" \
        --output_path "$OUTPUT/$dim" \
        --full_json_dir vbench2_beta_long/VBench_full_info.json \
        --dimension "$dim" \
        --mode long_custom_input
done
```

### Reading results

```python
import json, glob
for f in sorted(glob.glob("results/*/results_*_eval_results.json")):
    d = json.load(open(f))
    for k, v in d.items():
        score = v[0] if isinstance(v, list) else v
        print(f"{k}: {score:.4f}")
```

## Notes

- Input: MP4 videos only (no MKV). Convert with ffmpeg or cv2 if needed.
- Videos are automatically split into clips for evaluation.
- Model weights are downloaded automatically on first run (~4 GB total: DreamSim, ViCLIP, DINO, UMT, RAFT, CLIP).
  If the server has no internet, download weights locally and place in `~/.cache/`.
- `moviepy==1.0.3` is required (2.x removed `moviepy.editor`).
- `setuptools<81` is required for `pkg_resources` (CLIP dependency).

## Citation

```bibtex
@InProceedings{huang2023vbench,
    title={{VBench}: Comprehensive Benchmark Suite for Video Generative Models},
    author={Huang, Ziqi and He, Yinan and Yu, Jiashuo and Zhang, Fan and Si, Chenyang and Jiang, Yuming and Zhang, Yuanhan and Wu, Tianxing and Jin, Qingyang and Chanpaisit, Nattapol and Wang, Yaohui and Chen, Xinyuan and Wang, Limin and Lin, Dahua and Qiao, Yu and Liu, Ziwei},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}
```
