# SeedVR2 Standup — Environment Green, Row Deferred (VRAM-bound)

**2026-07-19.** Round-2 contrast model (window-local positions vs FlashVSR's
absolute streaming). Standup completed to the point of working CUDA kernels;
the 4th LR-VCC benchmark row is deferred post-thesis because inference memory
does not fit the shared A100-40G under tenant load. Everything below is
reusable as-is when a freer GPU is available.

## Working environment (server): conda env `seedvr310`

- python 3.10, torch 2.6.0+cu124 + torchvision 0.21.0 (pip, cu124 index).
  NB: the flashvsr env (py3.11) is NOT reusable — all SeedVR2 prebuilt wheels
  are cp310.
- **flash-attn 2.7.4.post1**, wheel
  `flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310...whl` (torch pip
  wheels are cxx11abi=FALSE; the TRUE wheel imports but dies with undefined
  symbols). GitHub release assets crawl at ~25–80 kB/s from the server —
  wheel was bridged from the Mac via the orphan-branch trick.
- **apex: do NOT install the wheel shipped in the SeedVR2-3B weights repo** —
  it is compiled for a different GPU arch (sm90; A100=sm80) and poisons the
  process with `no kernel image is available` on the first fused-kernel
  launch (misattributed to nearby torch ops due to async reporting).
  Installed instead: a 15-line **torch-native shim** at
  `site-packages/apex/normalization/__init__.py` aliasing
  `FusedLayerNorm -> nn.LayerNorm`, `FusedRMSNorm -> nn.RMSNorm`
  (identical math; fusion is a perf variant). If bit-parity with the
  authors' setup ever matters, compile apex from source for sm80.
- deps: omegaconf einops einops-exts tensordict mediapy
  rotary_embedding_torch av opencv-python-headless diffusers transformers
  accelerate safetensors; `setuptools<81` re-pin.
- Setup script: `scripts/server_runners/seedvr2_env_setup.sh` (idempotent).

## Weights + launch pattern

- Weights: `~/weights/seedvr2/SeedVR2-3B/` (14 GB via
  `HF_ENDPOINT=https://hf-mirror.com` snapshot; hf-mirror hosts 3B and 7B).
- Repo `~/repos/SeedVR` needs: `ckpts -> ~/weights/seedvr2/SeedVR2-3B`
  symlink (script expects `./ckpts/seedvr2_ema_3b.pth`, config expects
  `./ckpts/ema_vae.pth`), plus `pos_emb.pt`/`neg_emb.pt` symlinked into the
  repo root (loaded from CWD).
- Launch (single GPU): `cd ~/repos/SeedVR && PYTHONPATH=$PWD
  CUDA_VISIBLE_DEVICES=<g> torchrun --nproc_per_node=1 --master_port=<p>
  projects/inference_seedvr2_3b.py --video_path <dir> --output_dir <dir>
  --res_h 720 --res_w 1280 --sp_size 1`
  — plain `python` fails (unconditional `dist.init_process_group`);
  PYTHONPATH required (repo-local `data`/`common`/`models` packages).
  "Color fix is not avaliable" is a harmless optional-module warning.

## Why the row is deferred

SeedVR2 processes the whole clip in one shot (no streaming): at 720×1280
output, a 33-frame clip peaked ~30 GiB and then tried to allocate a further
7.9 GiB inside `rotary_embedding_torch.get_axial_freqs`; a 17-frame clip
still OOMs with ~7.5 GiB of tenant processes resident (40 GB card, ~32.5 GiB
best-case free). Even if short clips fit, the 22,412-frame benchmark would
need ~1,300 chunked passes — days of shared-GPU time. Needs either a
dedicated 40 GB card (short clips, chunked stitching) or an 80 GB card.

## Thesis framing

Unchanged: SeedVR2 remains the designated round-2 spatial-axis contrast in
Ch6/Ch7 future work. This note + the env script are the handoff.
