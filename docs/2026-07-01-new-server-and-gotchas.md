# New GPU Server + Known Problems / Gotchas (2026-07-01)

> **Purpose:** onboarding for any new work session on the current GPU host —
> including unrelated work like **testing rotary embeddings for VSR models**.
> This captures every problem hit while standing the server up, so the next
> session doesn't rediscover them. Credentials/connection details are in the
> gitignored `docs/private/server-setup.md`; this file is the git-tracked
> technical companion.

## The host in one paragraph

Multi-tenant box on `yc.smartml.cn` (SSH details in `docs/private/server-setup.md`).
**2× A100-PCIE-40 GB** (note: 40 GB, *not* 80 GB), 22 CPU cores, 175 GB RAM,
Ubuntu 22.04, Python 3.10.12 system. Shared with tenants `hyh`, `zrk`, `teme`.
We have our own `timur` user (uid 1003, no sudo) with SSH-key auth. Miniconda
installed at `~/miniconda3`. **GPUs are frequently at 100 % util from other
tenants** — plan for contention (see "GPU sharing" below).

## What's already installed (reuse it, don't rebuild)

| item | location | notes |
|---|---|---|
| miniconda3 | `~/miniconda3` | `source ~/miniconda3/etc/profile.d/conda.sh` to activate |
| `vsr` env | conda | torch **1.13.1+cu117**, metric deps (pyiqa, lpips, opencv, openai-clip) |
| `vbench` env | conda | torch **2.5.1+cu121**, CLIP/identity-adjacent |
| `identity` env | conda | torch **2.5.1+cu121** + ternaus retinaface + arcface, for VBench-2.0 slow-fast identity |
| repos | `~/repos/` | MGLD-VSR (patched), Upscale-A-Video, VBench (incl. VBench-2.0), YOLO-World |
| MGLD checkpoints | `~/repos/MGLD-VSR/checkpoints/mgld/` | unet 6.3 GB + vae 766 MB (SD-2.1 base NOT downloaded — 404 on mirror) |
| our repo | `~/thesis_ve/` | git clone; `results/` + `synthetic_data/` symlinked to `~/results`, `~/synthetic_data` |
| data | `~/results/`, `~/synthetic_data/` | all metric JSONs, MGLD+UAV SR outputs, source videos, refs, masks |

For **rotary-embedding VSR work** you'll likely want the `vsr` env (torch
1.13.1) or a fresh env — check what the target VSR model needs. The MGLD-VSR
repo + checkpoints are already there if you're modifying that architecture.

## THE BIG ONE: Mac ↔ server transfer is ~22 KB/s and drops constantly

Direct `rsync`/`scp`/`cat`-over-ssh from the Mac to this host is **unusable**
for anything over ~1 MB — it measured 22 KB/s and the TCP connection resets
every few minutes. A 3.3 MB file got 8 % across in 5 minutes.

**Server→internet is fine** (GitHub, PyPI, hf-mirror all fast). So:

### The GitHub-branch bridge (use this for any file > a few hundred KB)

```bash
# ON MAC — push file(s) to a throwaway orphan branch (home-network speed):
cd ~/Desktop/Timur/thesis_ve
git stash -u
git checkout --orphan xfer-transfer
git rm -rf --cached . >/dev/null 2>&1; git reset >/dev/null 2>&1
mkdir _xfer && cp /path/to/BIGFILE _xfer/          # split >100 MB first (see below)
git add -f _xfer/* && git commit -q -m "transfer"
git push -u origin xfer-transfer
git checkout -f main; rm -rf _xfer; git stash pop; git branch -D xfer-transfer

# ON SERVER — clone just that branch (retry; server↔GitHub is also a bit flaky):
for t in $(seq 1 10); do
  git clone --depth 1 --branch xfer-transfer --single-branch \
    https://github.com/TimuJ/thesis_ve.git ~/xfer && break; sleep 6; done
```

**GitHub rejects files > 100 MB.** Split first, reassemble on the far side:
```bash
split -b 90m BIGFILE BIGFILE.part      # on Mac before committing
cat BIGFILE.part* > BIGFILE            # on server after clone
```
Delete the throwaway branch afterward (`git push origin --delete xfer-transfer`).

This bridged: source videos (76 MB), MGLD SR outputs (773 MB / 12 chunks),
ArcFace weight (98 MB). It's the single most useful trick for this host.

### Small files / scripts

`scp` of a tiny (< a few KB) script usually succeeds within a few retries. For
launching long jobs, always use `tmux` so a link drop doesn't kill the job.
Wrap the launch so the ssh connection returns immediately (don't hold a
long-lived connection with an inline `sleep`).

## Network reachability from the server (tested 2026-07-01)

| host | reachable? | use |
|---|---|---|
| github.com / codeload / objects.githubusercontent | ✅ 200 | repos, release assets, the bridge |
| pypi / download.pytorch.org | ✅ | pip installs |
| **hf-mirror.com** | ✅ 200 | HuggingFace models: `export HF_ENDPOINT=https://hf-mirror.com` |
| huggingface.co | ❌ 000 | blocked — always use the mirror |
| drive.google.com | ❌ 000 | Google Drive blocked — fetch on Mac + bridge |
| youtube.com | ❌ Errno 101 | blocked — no yt-dlp on the server |

**For HuggingFace model downloads** (e.g. a rotary-embedding VSR checkpoint):
`export HF_ENDPOINT=https://hf-mirror.com` before `huggingface_hub.snapshot_download`
or `from_pretrained`. Note the mirror does not have *every* repo (SD-2.1-base
404'd), and `list_repo_files` may 401 on the mirror even when file downloads
work — just request the specific filename you need.

## open_clip vs HuggingFace

`open_clip.create_model_and_transforms(..., pretrained="openai")` **still
routes through HuggingFace Hub** (for the timm config) and hangs. Use OpenAI's
original `clip` package instead: `pip install git+https://github.com/openai/CLIP.git`,
`clip.load("ViT-B/32")` — downloads from OpenAI's CDN, caches at `~/.cache/clip/`.
Our `scripts/lr_vcc/compute_clip_trajectory.py` already does this.

## conda / pip gotchas

- **`vbench` env needs `setuptools<81`** — Detectron2's `model_zoo` and CLIP's
  `pkg_resources` import break on 81+. Re-pin after any pip install that pulls
  a fresh setuptools: `pip install 'setuptools<81'`. Symptom:
  `ModuleNotFoundError: No module named 'pkg_resources'`.
- **`retinaface` on PyPI is the wrong package** (TensorFlow-based). The one
  with `predict_single` is ternaus/retinaface, which hard-pins `torch==1.9.0`.
  Don't `pip install` it — clone the repo and copy the `retinaface/` package
  dir into site-packages, or `pip install --no-deps git+…` (the git clone in
  pip can fail on the flaky link; the copy-into-site-packages route is most
  reliable).
- **`pip install -r <freeze> --no-deps`** is how the envs were rebuilt from the
  captured freezes (`docs/server_conda_envs_2026-06-15.txt`). `--no-deps` avoids
  re-resolving the whole graph; install torch first with the right CUDA wheel.

## VBench-2.0 identity stack (only relevant if you touch sub-metric I)

- Weights: ArcFace `~/.cache/vbench2/arcface/resnet18_110.pth` (98 MB, from a
  Google-Drive gdown id — bridge it); RetinaFace
  `~/.cache/torch/hub/checkpoints/retinaface_resnet50_2020-07-20-f168fae3c.zip`
  (97 MB GitHub release — curl with retries, watch for truncation).
- **Use our patched `human_identity.py`**, not the fresh clone. Our git-vendored
  `scripts/vbench2_long/vbench2/human_identity.py` is the authoritative patched
  version (multi-face + late-reference + num=0 guard). The upstream clone
  crashes on zero-face videos (`ZeroDivisionError`). Deploy with:
  `cp ~/thesis_ve/scripts/vbench2_long/vbench2/human_identity.py ~/repos/VBench/VBench-2.0/vbench2/human_identity.py`
  (verified via md5 diff: this is the ONLY file that differs from upstream).
- **Concurrent first-load race:** if you run multiple identity processes, they
  race to unzip the same cached RetinaFace weight and corrupt it
  (`PytorchStreamReader ... archive is corrupted`). Warm the cache with ONE
  `model_zoo.load_url(...)` call before spawning workers, and stagger launches
  by ~25 s.

## GPU sharing etiquette

- Both A100s are usually at 100 % util from other tenants, but memory is the
  real constraint — check `nvidia-smi --query-gpu=memory.free --format=csv`
  before launching. Each identity/CLIP process ~2–3 GB; keep total footprint
  modest so you don't OOM `hyh`/`zrk`/`teme`.
- Compute is time-sliced, so wall-clock is much slower than a dedicated A100.
  A job estimated at ~10 min/video on a dedicated card ran ~21 min/video here.
- Parallelising across many small processes helps when the work is partly
  CPU-bound (frame I/O), since there are 22 cores.
- Always `tmux`. `pkill -f <script>` to clean up your own strays;
  `nvidia-smi` to confirm your memory is released.

## Data provenance note for reproducibility

The artefact generator (`scripts/synthetic_artefacts/generate_all.py`) reads
its *base* videos from `results/mgld_synthetic_mp4/` — i.e. the MGLD **SR
outputs**, not the raw LR source. Any new artefact must be generated from those
same base files to stay comparable with the existing 11 families. (This is why
the 773 MB of MGLD SR outputs had to be bridged to the server.)

## Pointers

- Connection + credentials: `docs/private/server-setup.md` (gitignored)
- Env freezes: `docs/server_conda_envs_2026-06-15.txt`
- Restore-from-scratch procedure: `docs/server_restore_guide.md`
- The finalize-flip_invert one-shot: `scripts/finalize_flip_invert.sh`
- Weekly narrative: `reports/Timur_Iakshibaev_2026-06-29_to_2026-07-01.md`

## Addenda (2026-07-02, RoPE-probe / FlashVSR work)

- **setuptools<81 gotcha also hits the `vsr` env** (not just `vbench`):
  `pyiqa`→`clip`→`pkg_resources` import chain. Same fix:
  `pip install 'setuptools<81'` inside `vsr`.
- **Server now has GitHub push auth** (fine-grained PAT, contents:write on
  `TimuJ/thesis_ve` only, expires ~2026-07-09, stored in `~/.git-credentials`).
  Server→Mac bulk transfer = reverse bridge: push orphan branch from server,
  pull on Mac, delete branch. Used for FlashVSR mp4s (314 MB) + metric JSONs.
- **FlashVSR env (`flashvsr`)**: torch 2.6.0+cu124, Block-Sparse-Attention
  compiled with `BLOCK_SPARSE_ATTN_CUDA_ARCHS="80"` (upstream emits compute_120
  → nvcc 12.4 fatal), `import torch` before `block_sparse_attn` (libc10.so),
  `modelscope` required by diffsynth, diffsynth wired via .pth (editable
  install fails on the setuptools gotcha). Repo pinned `pristine-2026-07-02`.
