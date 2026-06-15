# Server Restore Guide — LR-VCC Pipeline on a Fresh GPU Host

> **Audience:** future-Timur (or anyone setting this up on a new lab server after the original `/data/disk2/timur/` host was decommissioned on June 15, 2026).
>
> **Goal:** rebuild the metric pipeline end-to-end such that
> `python -m scripts.lr_vcc.run_lr_vcc ...` produces composite JSONs that
> match the v5 baseline byte-for-byte where the inputs match. This is what
> the thesis chapters depend on.

---

## TL;DR — minimum viable restore (~2 h)

1. Create `vsr` conda env from `docs/server_conda_envs_2026-06-15.txt`.
2. Clone the repo (this one). The Python sources are everything needed for
   composite computation — no patches.
3. Extract `results_jsons_2026-06-15.tar.gz` from the Google Drive backup
   into `<repo>/results/`. That gives you all metric stage JSONs and the v5
   composites without re-running any GPU work.
4. Smoke test: `pytest tests/ -v` → 114 passing.

That alone restores read-only access to every existing result.

The longer steps below are needed only if you want to (a) generate new
artefact clips, (b) re-evaluate new SR-model outputs, or (c) recompute the
identity / CLIP-IQA / D'' stages from raw video.

---

## Backup inventory — what should be on Google Drive

`zju_server_backup_2026-06-15/` should contain:

| item | size | source path on old server |
|---|---|---|
| `synthetic_mkv/` (5 mkv + 5 mp4) | 1.4 GB | `/data/disk2/timur/synthetic_data/synthetic/` |
| `results_jsons_2026-06-15.tar.gz` | 58 MB | `/data/disk2/timur/results/{synthetic_artefacts_eval, lr_vcc, synthetic_artefacts/_{human_masks,references}}` |
| `server_conda_envs_2026-06-15.txt` | small | `pip freeze` for vsr / vbench / uav envs |

The repo git remote already has:
- `scripts/server_runners/*.sh` — 60 runner scripts (commit 5224c29)
- `scripts/{lr_vcc, synthetic_artefacts, vbench2_long, long_range_temporal}/*.py` — all Python source
- `docs/server_conda_envs_2026-06-15.txt` — env spec
- `docs/private/server-setup.md` — original host's setup notes (gitignored locally, available in your `.config` clone)

The deliberately-not-backed-up things:
- 43 GB of artefact MP4s — regenerable with `python scripts/synthetic_artefacts/generate_all.py` (~3 h CPU)
- Raw frames from the SR runs — pruned on the old server already
- Conda env binaries — rebuild from the pip-freeze spec

---

## Prerequisites on the new server

1. **CUDA + matching driver.** The original was CUDA 11.8 / 12.1 (the
   `vsr` env's torch pin was 2.1.0+cu118; vbench used a newer 2.x). Match
   or exceed.
2. **conda or mamba.** Miniconda installation: `bash Miniconda3-latest-Linux-x86_64.sh`.
3. **git, tmux, rsync, ffprobe/ffmpeg.** Standard.
4. **OpenAI CLIP weights download path.** The `clip` PyPI package caches
   `ViT-B-32.pt` (350 MB) to `~/.cache/clip/` on first load. **Run this
   once on a host that *can* reach the OpenAI CDN** (your laptop, for
   example), then scp the cached file to the new server before the first
   D'' job, because the lab network often can't reach HuggingFace Hub but
   sometimes also can't reach the OpenAI CDN cleanly. See "Known gotchas".

---

## Step 1 — repo

```bash
git clone git@github.com:TimuJ/thesis_ve.git
cd thesis_ve
```

Project standards (`CLAUDE.md`): no `pip install -e .` — Python imports use the
`scripts/` package path directly, so always `cd` to the repo root and either
run scripts via `python -m scripts.lr_vcc.run_lr_vcc ...` or set
`PYTHONPATH=$PWD`. The server runners in `scripts/server_runners/` already do
this.

---

## Step 2 — conda envs from frozen specs

`docs/server_conda_envs_2026-06-15.txt` has three `=== <name> ===`-delimited
sections, one per env. Tear out the section for an env, paste into
`requirements-<env>.txt`, then:

```bash
conda create -n vsr python=3.10 -y && conda activate vsr
pip install torch==<see-spec>+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements-vsr.txt
```

Repeat for `vbench` and `uav`. **Three env-specific gotchas** (see below for
details): `vbench` needs `setuptools<81`; the `clip` package's
`pkg_resources` deprecation warning is harmless; `uav` needs the patches in
`docs/private/mgld-vsr-patches.md` if you also want to re-run UAV.

Smallest viable restore: only `vsr` is needed for D / D' / E / composite.
`vbench` is required for the slow-fast identity stage (sub-metric I).
`uav` is for the Upscale-A-Video model itself, not needed for the
metric pipeline.

---

## Step 3 — external repos (only for re-running specific stages)

| repo | needed for | clone target |
|---|---|---|
| VBench-2.0 | sub-metric I (slow-fast identity adapter) | `repos/VBench/VBench-2.0/` |
| YOLO-World | VBench-2.0 dependency | `repos/YOLO-World/` |
| DOVER | DOVER metric (currently unused by v5 composite, but in legacy runners) | `repos/DOVER/` |
| MGLD-VSR | re-running MGLD baseline | `repos/MGLD-VSR/` (patched per `docs/private/mgld-vsr-patches.md`) |
| Upscale-A-Video | re-running UAV baseline | `repos/Upscale-A-Video/` (patched per `docs/private/server-setup.md`) |

Skip everything in this table for a read-only restore. The composite
recomputation only needs the Python sources in this repo.

---

## Step 4 — restore data

```bash
# 1. extract JSONs
mkdir -p results/
tar -xzf ~/Downloads/zju_server_backup_2026-06-15/results_jsons_2026-06-15.tar.gz -C ./

# 2. drop source videos
mkdir -p results/mgld_synthetic_mp4/
cp ~/Downloads/zju_server_backup_2026-06-15/synthetic_mkv/*.mp4 results/mgld_synthetic_mp4/
# (keep the .mkv originals somewhere — they're the canonical sources)

# 3. references and human masks landed via the tarball
ls results/synthetic_artefacts/_references/  # should show 10 png files
ls results/synthetic_artefacts/_human_masks/  # should show 5 npz files
```

---

## Step 5 — smoke test (verifies the read-only restore worked)

```bash
# unit tests
pytest tests/ -v
# expected: 114 passed, 0 failed

# regenerate the v5 verdict matrix from the cached composites
python scripts/lr_vcc/build_verdict_matrix.py \
    --composites_dir results/lr_vcc/composite_artefacts_v5 \
    --out /tmp/matrix_check.md
# expected: 6 existing artefacts + 5/6 flip families (or 6/6 if you got flip_invert)
# matches reports/figures/verdict_matrix_v5.md exactly
diff /tmp/matrix_check.md reports/figures/verdict_matrix_v5.md
```

If both pass: read-only restore is complete. Stop here unless you want to
recompute or extend.

---

## Step 6 (optional) — regenerate artefact MP4s

If you need the 300 artefact clips back (e.g. for new sub-metric experiments
or thesis appendix figures):

```bash
conda activate vsr
cd <repo>
python scripts/synthetic_artefacts/generate_all.py
# CPU, ~3–6 h total. Skip-if-exists logic, safe to interrupt and resume.
```

Output: `results/synthetic_artefacts/<artefact>/<base>_sev<S>.mp4`.

Deterministic given the same reference images + source videos. Should match
the original byte-for-byte for everything except `flip_elastic` (which uses
a seeded RandomState — also deterministic given the same numpy version).

---

## Step 7 (optional) — recompute metric stages on a GPU box

For each artefact / clip set you want to evaluate, the standard pipeline is
(adapt the per-artefact runners from `scripts/server_runners/run_b6_eval.sh`
and `run_b8_flip_eval.sh`):

```
clip-iqa  →  tof_tlp  →  color_histogram  →  color_slope  \
                                            ↓
                                          (CPU)
                                            ↓
            human_identity_long (slow-fast)  →  identity JSONs
            (GPU, vbench env, slowest stage at ~10 s/clip × 83 clips × 5 videos)

D'  ←  color_histogram_anchor (CPU, fast)
D'' ←  compute_clip_trajectory (GPU, ~5 min/video at stride 8)

run_lr_vcc → final composite JSONs
```

The runner scripts in `scripts/server_runners/` encode the right
parameters (uniform tOF weight, α=0.394, β=200/0.5/3.0, etc.) and the
correct conda env per stage.

---

## Known gotchas

### HuggingFace Hub unreachable from lab network

`open_clip.create_model_and_transforms("ViT-B-32", pretrained=...)`
always routes through HF Hub for the timm config download, even when
`pretrained="openai"`. Workaround: use OpenAI's original `clip` package
(`pip install git+https://github.com/openai/CLIP.git`) which downloads from
OpenAI's own CDN. `compute_clip_trajectory.py` already does this. Verify
weights at `~/.cache/clip/ViT-B-32.pt` before first run.

### Detectron2 + setuptools 81

```
ModuleNotFoundError: No module named 'pkg_resources'
```

In the `vbench` env (only). One-shot fix:

```bash
conda activate vbench
pip install 'setuptools<81'
```

### tmux quoting silently breaks `;` separators

```
tmux new-session -d -s name "cmd | tee log; touch /tmp/done"   # BROKEN
```

The `touch` gets parsed as `tee` arg, `.done` flag never created. Always
use `bash -lc 'multi-line block'`:

```
tmux new-session -d -s name "bash -lc 'cmd | tee log; touch /tmp/done'"
```

### Disk pressure on `/data/disk*`

Symptom: SR runs fail with "no space left." Safe-to-prune caches:
- `~/.cache/pip` (rebuilds on next pip use)
- `repos/VBench/VBench-2.0/cache/` (rebuilds on next VBench run)
- `~/.cache/dreamsim/` (rebuilds on model load)
- `results/<artefact>/raw_frames/` if you re-encoded to mp4 (canonical)

Original incident `docs/private/server-incident-2026-04-16.md` has the
detailed audit pattern.

### Per-env weights / cached models

If `~/.cache/clip/ViT-B-32.pt` is missing on the new server, copy it from
the laptop:

```bash
scp ~/.cache/clip/ViT-B-32.pt new-server:~/.cache/clip/
```

Same for any HF Hub cache that was populated on the old server — those
files at `~/.cache/huggingface/hub/` are server-rsyncable in a pinch but
were generally regenerable on the old server before it died.

### GPU sharing etiquette

Original lab was 8× A100 80GB shared with cjk / fyx / xby. GPU 0 was the
intended primary slot but was frequently saturated by `cjk/diffsynth`.
GPU 7 was the backup slot. GPU 1 (fyx/kiwi) was usable when fyx's
process was parked at 0% util — but if fyx resumed mid-run, one process
OOMed. New host may have a totally different policy; check before assuming.

---

## What was *not* backed up and why

| item | reason | recovery cost |
|---|---|---|
| 43 GB of artefact MP4s | regenerable via `generate_all.py` | 3–6 h CPU |
| MGLD-VSR / UAV / DOVER repos | upstream still public, plus patches in `docs/private/` | 1 h clone + patch |
| `repos/*` checkpoints (model weights) | downloadable from original sources | varies, mostly under 10 GB each |
| `miniconda3/` binaries | rebuildable from the pip-freeze spec | 30 min install |
| 22,412 extracted raw frames | already pruned on old server before backup | re-extract with `ffmpeg` |

---

## Verification checklist

When restored, the following should all pass:

- [ ] `pytest tests/ -v` → 114 passed
- [ ] `ls results/synthetic_artefacts_eval/identity/background_drift/` → 2 result_*.json files
- [ ] `ls results/lr_vcc/composite_artefacts_v5/` → 11–12 artefact subdirs, each with 25 JSONs
- [ ] `ls results/synthetic_artefacts/_references/` → 10 png files (5 face + 5 bg)
- [ ] `ls results/synthetic_artefacts/_human_masks/` → 5 .npz files
- [ ] `diff <(python scripts/lr_vcc/build_verdict_matrix.py --composites_dir results/lr_vcc/composite_artefacts_v5 --out /dev/stdout 2>/dev/null) reports/figures/verdict_matrix_v5.md` → empty diff

If all six pass, you have a working LR-VCC v5 read-only pipeline.

---

## Forward references

- `docs/onboarding.md` — broader project orientation (lab/seniors-readable)
- `docs/private/server-setup.md` — original host's setup notes (locally gitignored; live in your `.config` clone)
- `docs/private/mgld-vsr-patches.md` — exact diffs needed on MGLD-VSR
- `docs/superpowers/plans/2026-06-11-benchmark-completion.md` — the experiment plan that produced everything restored here
- `reports/Timur_Iakshibaev_2026-06-05_to_2026-06-18.md` — the work narrative
- `reports/figures/verdict_matrix_v5.md` — what the restored pipeline should reproduce
