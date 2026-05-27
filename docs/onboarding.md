# Project Onboarding

**Last updated:** 2026-05-27
**Audience:** new collaborator, future self, or anyone resuming this project after a gap.

This doc is the single landing page for finding things in the repo, getting an account on the lab server, and running the standard experiment pattern. Sensitive details (server IP, SSH keys, conda env internals) live in `docs/private/server-setup.md`.

---

## 1. Repo layout

```
thesis_ve/
├── proposal/                     # May 31, 2026 proposal (LaTeX)
│   ├── thesis_proposal.tex       # entry point
│   ├── chapters/                 # \input'd sections
│   ├── sections/                 # markdown drafts pre-LaTeX
│   └── figures/                  # plots used in the proposal
│
├── zjuthesis/                    # final thesis (LaTeX, due Sept 30, 2026)
│   └── body/graduate-eng/        # chapter files (introduction, methodology, …)
│
├── scripts/                      # research / experiment code
│   ├── lr_vcc/                   # ★ the metric — composite + 5 sub-metrics
│   │   ├── run_lr_vcc.py         # main CLI entry point
│   │   ├── composite.py          # softmax-log-mean composition
│   │   ├── appearance.py         # sub-metric A (CLIP-IQA wrapper)
│   │   ├── temporal.py           # sub-metric T (multi-k tOF aggregation)
│   │   ├── identity.py           # sub-metric I (Identity slow-fast wrapper)
│   │   ├── color_stability.py    # sub-metric D (color histogram L1)
│   │   ├── color_slope_wrapper.py# sub-metric E (linear regression on Lab means)
│   │   ├── compute_clip_iqa.py   # server-side GPU dumpers …
│   │   ├── compute_color_histogram.py
│   │   └── compute_color_slope.py
│   ├── long_range_temporal/      # tOF/tLP at multi-k (RAFT-based)
│   ├── synthetic_artefacts/      # generators: color_drift, chunk_boundary, flicker, identity_degradation
│   └── vbench2_long/             # VBench-2.0 slow-fast adapters (Identity, Anatomy)
│
├── tests/                        # pytest suite (39 LR-VCC + 11 generator tests)
├── src/                          # legacy infra: PSNR/SSIM, FPS/VRAM tracking
│
├── results/                      # ALL experiment outputs (gitignored)
│   ├── lr_vcc/                   # per-sub-metric JSONs + composite outputs
│   ├── synthetic_artefacts/      # generated test videos
│   ├── synthetic_artefacts_eval/ # metric battery on the artefact videos
│   ├── long_range_temporal/      # multi-k tOF/tLP JSONs
│   ├── vbench2_anatomy/          # legacy VBench outputs
│   ├── mgld_synthetic_mp4/       # MGLD-VSR outputs of the 5 base videos
│   └── uav_synthetic_mp4/        # UAV outputs of the same
│
├── reports/                      # weekly progress reports
├── docs/
│   ├── onboarding.md             # this file
│   ├── thesis-context.md         # situational overview
│   ├── plans/                    # design + implementation plans (most have status banners)
│   ├── notes/                    # observation logs (FPS bug, KZ regime shift, tOF/tLP crossover, LR-VCC validation)
│   ├── meeting-notes/            # records of supervisor / colleague discussions
│   └── private/                  # sensitive: server access, code patches, incident reports (gitignored)
│
└── requirements-{local,gpu}.txt  # deps for M1 vs A100 environments
```

**Single source of truth for paths:** `src/configs/paths.py` (overridable via `VSR_PROJECT_ROOT`, `VSR_DATA_ROOT` env vars).

---

## 2. Where data and results live

### Local (MacBook M1, no GPU)

Used for: code, tests, writing, plotting. Large data lives on the server, not here.

| Path | Contents |
|------|----------|
| `results/mgld_synthetic_mp4/` `results/uav_synthetic_mp4/` | SR outputs of the 5 base videos (mp4, ~30 MB each). Gitignored. |
| `results/synthetic_artefacts/<artefact>/` | Generated test videos (color_drift, chunk_boundary, flicker, identity_degradation × 2 base × 5 severities). Gitignored. |
| `results/lr_vcc/` | Per-sub-metric JSONs + composite outputs (`composite_v3_slope_b200/`, `composite_artefacts_v3_slope_b200/`). Gitignored. |
| `results/synthetic_artefacts_eval/` | Mirror of server-side metric battery on artefact videos. Gitignored. |

### Server (`/data/disk2/timur/`, A100 GPUs)

Used for: all GPU/CPU-heavy experiments (M1 cannot handle it — **hard constraint**).

| Path | Contents |
|------|----------|
| `data/UDM10/`, `data/SPMCS/` | DOVE benchmark datasets (LR + HR pairs) |
| `synthetic_data/synthetic/` | 5 long source MKVs (320×180, 80–208 s) |
| `data/synthetic_frames/` | Pre-extracted frames (22,412 total) |
| `repos/{MGLD-VSR, Upscale-A-Video, DOVE, VBench, detectron2}/` | Cloned + patched baselines |
| `results/mgld_synthetic_mp4/` `results/uav_synthetic_mp4/` | SR outputs of the 5 base videos |
| `results/synthetic_artefacts/<artefact>/` | Generated artefact videos (4 × 2 × 5 = 40 videos) |
| `results/synthetic_artefacts_eval/<metric>/<artefact>/` | Per-metric JSONs (CLIP-IQA, tOF/tLP, Identity, DOVER, E*warp) |
| `results/lr_vcc/{clip_iqa, color_histogram, color_slope, closeup_map_artefacts}/<COND>/` | Per-sub-metric JSON caches |
| `results/lr_vcc/composite_v3_slope_b200/` `composite_artefacts_v3_slope_b200/` | Final LR-VCC composite outputs |
| `scripts/` | Mirror of local `scripts/` (synced manually via rsync) |
| `miniconda3/` | Conda installation (not on PATH — see private doc) |

---

## 3. Connecting to the server

See `docs/private/server-setup.md` for the exact SSH command, key path, and IP. After connecting, the standard activation is:

```bash
eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"
conda activate vsr        # general eval env — used for LR-VCC, tOF/tLP, color sub-metrics
# OR:
conda activate uav        # Upscale-A-Video (torch 2.0.1 + xformers 0.0.22 — do NOT upgrade)
conda activate mgldvsr    # MGLD-VSR (torch 2.0.1 + mmcv 2.1.0 — do NOT upgrade)
conda activate vbench     # VBench-2.0 (Anatomy, Identity)
```

GPUs 3, 5, 7 are usually free; check with `nvidia-smi` first. Pin a GPU with `CUDA_VISIBLE_DEVICES=N` before launching.

---

## 4. Standard experiment workflow

The pattern that's worked all month:

```bash
# 1. Edit + test locally
pytest tests/ -v
git commit -am "<scope>: <change>"

# 2. Push code (only the file(s) you changed) to the server
rsync -av -e "ssh -i ~/.ssh/id_ed25519_timuj" \
  scripts/lr_vcc/<file>.py \
  Timur@223.109.239.43:/data/disk2/timur/scripts/lr_vcc/

# 3. Launch the experiment in tmux (survives disconnects)
ssh Timur@223.109.239.43
tmux new -s expN
eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)" && conda activate vsr
cd /data/disk2/timur && PYTHONPATH=. python -m scripts.lr_vcc.<module> [...args]
# Ctrl+B, D to detach. `tmux attach -t expN` to reattach.

# 4. Pull results back
rsync -av -e "ssh -i ~/.ssh/id_ed25519_timuj" \
  Timur@223.109.239.43:/data/disk2/timur/results/<sub-path>/ \
  results/<sub-path>/
```

The local M1 is for writing/analysis only. **All experiments must run on server** — local can't handle the dataset sizes or model weights.

---

## 5. Running LR-VCC end-to-end

LR-VCC (production settings as of 2026-05-27):

```bash
PYTHONPATH=. python -m scripts.lr_vcc.run_lr_vcc \
  --tof_dir          results/long_range_temporal/<COND> \
  --identity_results results/synthetic_artefacts_eval/identity/<COND> \
  --clip_iqa_dir     results/lr_vcc/clip_iqa/<COND> \
  --color_hist_dir   results/lr_vcc/color_histogram/<COND> \
  --color_slope_dir  results/lr_vcc/color_slope/<COND> \
  --closeup_p50_map  results/lr_vcc/closeup_map_artefacts/<COND>.json \
  --output_path      results/lr_vcc/composite_artefacts_v3_slope_b200/<COND> \
  --temporal_weight  uniform \
  --color_hist_alpha 0.394 \
  --color_slope_beta 200
```

All five sub-metric directories must be populated first (run `compute_clip_iqa.py`, `compute_color_histogram.py`, `compute_color_slope.py`, `eval_tof_tlp.py`, the identity-slow-fast pipeline — each produces one JSON per video). Closeup-p50 map is computed once from the anatomy traces.

The production setting was derived empirically: `--temporal_weight uniform` from Option A, `--color_hist_alpha 0.394` from D recalibration (Option A2), `--color_slope_beta 200` from E tuning (Option B). See `reports/Timur_Iakshibaev_2026-05-22_to_2026-05-27.md` for the iteration history.

---

## 6. Where to start reading

- **Topic + timeline + lessons:** `docs/thesis-context.md`
- **Metric design (what LR-VCC is and why):** `docs/plans/2026-05-21-lr-vcc-design.md`
- **Implementation roadmap (15 tasks):** `docs/plans/2026-05-21-lr-vcc-implementation.md`
- **Key findings:**
  - `docs/notes/2026-05-13-kz-regime-shift-trigger.md` — why Anatomy fails on close-up content
  - `docs/notes/2026-05-14-tof-tlp-long-range-results.md` — k=5–10 temporal crossover
  - `docs/notes/2026-05-21-lr-vcc-validation.md` — Layer 1+2 validation (MGLD wins 5/5)
- **Server access + conda envs:** `docs/private/server-setup.md`
- **Most recent weekly summary:** `reports/Timur_Iakshibaev_2026-05-22_to_2026-05-27.md`
