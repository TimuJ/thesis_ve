# VBench-2.0 Long-Video Evaluation (for Video SR)

Patched [VBench-2.0](https://github.com/Vchitect/VBench/tree/master/VBench-2.0) for evaluating long super-resolution videos. Includes a slow-fast adapter that wraps `Human_Identity` for >1-min videos and three small upstream patches.

For VBench 1.x (the older long-video quality dimensions), see `scripts/vbench1_long/`.

## Setup

```bash
conda create -n vbench python=3.10 -y
conda activate vbench
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # adjust to your CUDA
pip install -r requirements.txt
```

`requirements.txt` is the upstream VBench-2.0 pinned set. Note `mmdet`, `mmengine`, `mmyolo`, `timm`, `decord`, and the upstream-cloned `YOLO-World` and `Instance_detector` repos are required by `Human_Anatomy`. `arcface` and `retinaface` are required by `Human_Identity`.

## Usage

Slow-fast is the default mode for long-video SR evaluation. Both dimensions split the source video into 2-second clips at native fps; the slow branch averages within-clip scores, the fast branch concatenates clip-first-frames and scores cross-clip drift, and the two fuse 50/50. The upstream `custom_input` whole-video mode is also available as a fallback (see "Whole-video fallback" below) but on long videos it collapses (identity drift accumulates) or behaves asymmetrically (anatomy regime shifts on certain content).

Common env setup (server paths — adjust):

```bash
export VBENCH2_CACHE_DIR=/path/to/vbench2_cache
export PYTHONPATH="$PWD:/path/to/YOLO-World:${PYTHONPATH:-}"
```

### Human_Identity — slow-fast adapter

`human_identity_long.py` wraps the patched `Human_Identity`:

- **Slow branch:** split video into 2-sec clips at native fps, run identity per clip, average across clips with a face detected.
- **Fast branch:** concatenate the first frame of each clip into a synthetic "fast" video, run identity on it (catches long-range identity drift across minutes).
- **Fusion:** `w_slow * slow + w_fast * fast`, default 50/50.

```bash
python human_identity_long.py \
    --videos_path /path/to/mp4_videos \
    --output_path /path/to/output \
    [--w_slow 0.5] [--w_fast 0.5] [--clip_duration 2] \
    [--fps_overrides fps_map.json] [--save_clip_detail]
```

`--fps_overrides` takes a JSON map of `{video_basename: fps}` to override the (often wrong) fps tag written by SR pipelines — pass the LQ-source fps for an apples-to-apples comparison. `--save_clip_detail` persists per-clip scores.

Runtime: ~3 hours for 5 ×3-min videos on a single A100, dominated by RetinaFace per-frame.

### Human_Anatomy — slow-fast adapter

`human_anatomy_long.py` runs the ViTDetector ensemble per frame and aggregates into slow-fast, matching the Identity adapter shape:

```bash
python human_anatomy_long.py \
    --videos_path /path/to/mp4_videos \
    --output_path /path/to/output \
    [--w_slow 0.5] [--w_fast 0.5] [--clip_duration 2] \
    [--fps_overrides fps_map.json] [--save_per_frame] [--save_clip_detail]
```

Output JSON shape mirrors `human_identity_long.py` — `per_video[<v>] = {slow, fast, fused, whole_video, ...}` and `overall_{slow, fast, fused, whole_video}`. The `whole_video` field is the upstream's frame-pooled score for sanity-checking against the original `custom_input` mode.

For diagnostic / debugging, the two underlying steps are also exposed separately:

- `diagnose_anatomy_per_frame.py --video V --output OUT.json` — just the per-frame trace (the upstream's `compute_abnormality` computes then discards `frame_results`).
- `aggregate_slow_fast_anatomy.py --per-frame OUT.json --fps F --output SF.json` — post-hoc slow-fast aggregation over an existing per-frame trace.

### Whole-video fallback (upstream `custom_input` mode)

Use this only when slow-fast windowing isn't applicable (very short videos, sanity-checking against upstream numbers).

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --videos_path /path/to/mp4_videos \
    --dimension Human_Anatomy \
    --mode custom_input \
    --output_path /path/to/output
```

`run_vbench2_anatomy.sh` runs MGLD then UAV sequentially in this whole-video mode.

## Layout

```
vbench2_long/
├── README.md                          # this file
├── requirements.txt                   # upstream VBench-2.0 pinned deps
├── human_identity_long.py             # slow-fast adapter for Human_Identity
├── human_anatomy_long.py              # slow-fast adapter for Human_Anatomy (end-to-end)
├── diagnose_anatomy_per_frame.py      # diagnostic — per-frame Anatomy trace only
├── aggregate_slow_fast_anatomy.py     # diagnostic — slow-fast over an existing per-frame trace
├── run_vbench2_anatomy.sh             # launch script we used (server paths — adjust)
├── evaluate.py                        # upstream entry point (mirrored unchanged)
└── vbench2/                           # mirrored upstream sources, source-only
    ├── human_identity.py              # PATCHED (3 changes — diff vs .bak below)
    ├── human_identity.py.bak          # upstream original (kept for diff)
    ├── human_anatomy.py               # used as-is
    ├── third_party/YOLO-World/        # PATCHED config — text encoder path (see below)
    └── ...                        # other VBench-2.0 metric modules
```

The mirror excludes weights (`.pth/.bin/.safetensors`), images, archives, and the LVIS minival annotation file (~110 MB). Re-fetch them from the upstream VBench-2.0 repo (or the `gdown` paths in the upstream README).

## Patches we applied

### 1. `vbench2/human_identity.py` (3 patches)

Diff: `diff -u vbench2/human_identity.py.bak vbench2/human_identity.py`

- **Multi-face frames** — original required exactly 1 face per frame. Patched to pick the largest bounding-box face when multiple faces are detected.
- **Late reference init** — original required a face in frame 0 and broke the loop otherwise. Patched to use the first frame with a detectable face as the reference.
- **ZeroDivisionError guard** — return `-1.0` sentinel when `num == 0` instead of crashing on empty videos.

These three patches together fix `Human_Identity` so it doesn't return artificially zero scores on multi-person/late-face videos.

### 2. `vbench2/third_party/YOLO-World/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py`

Line 20 hard-codes the CLIP-ViT-Base-Patch32 path. We patched it from the upstream `'../pretrained_models/clip-vit-base-patch32-projection'` to a local cache path (the lab server can't reach HuggingFace). Two valid alternatives:

```python
# upstream default (relative to VBench-2.0 root)
text_model_name = '../pretrained_models/clip-vit-base-patch32-projection'

# HuggingFace identifier (works if the machine can reach huggingface.co)
text_model_name = 'openai/clip-vit-base-patch32'

# our patch — local cache (adjust to your machine)
text_model_name = '/data/disk2/timur/cache/clip/clip-vit-base-patch32'
```

If you're behind a firewall that blocks `huggingface.co`, mirror via `hf-mirror.com` first, then point this to the local cache.

## Required env / cache layout

VBench-2.0 reads model weights from `$VBENCH2_CACHE_DIR` (default `~/.cache/vbench2`). The cache must contain:

```
$VBENCH2_CACHE_DIR/
├── YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth
├── anomaly_detector/{human,face,hand}.pth   # 347 MB each — for Human_Anatomy
├── arcface/                                 # for Human_Identity
└── retinaface/                              # for face detection
```

Two pre-existing upstream gotchas to watch for:

- The `anomaly_detector/{human,hand}.pth` files served by upstream's `gdown` URLs were **truncated** (92 MB / 167 MB; they should be 347 MB). Verify with `python -c "import torch; torch.load('human.pth')"` — if it raises `PytorchStreamReader`, re-download.
- `VBENCH2_CACHE_DIR` must be exported before running, otherwise the loader silently falls back to `~/.cache/vbench2/` and may load partial files.

## Results we observed (5 synthetic SR videos, MGLD vs UAV)

| Metric | MGLD | UAV | Notes |
|--------|------|-----|-------|
| `Human_Identity` (whole-video, custom_input) | 0.200 | 0.203 | tied — drift across full video collapses both |
| `Human_Identity` (slow-fast fused) | **0.555** | 0.463 | MGLD wins 4/5 videos (Δ +0.092) |
| `Human_Anatomy` (custom_input) | 0.600 | 0.605 | tie at the mean; MGLD wins 4/5 per-video |

Per-video results live in `results/mgld_synthetic_metrics.md` and `results/vbench2_anatomy/`.

## Citation

```bibtex
@article{huang2025vbench2,
    title={{VBench-2.0}: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness},
    author={Huang, Ziqi and Zhang, Fan and Xu, Xiaojie and He, Yinan and Yu, Jiashuo and Dong, Ziyue and Ma, Qianli and Chanpaisit, Nattapol and Si, Chenyang and Jiang, Yuming and others},
    journal={arXiv preprint arXiv:2503.21755},
    year={2025}
}
```
