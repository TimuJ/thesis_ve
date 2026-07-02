# Weekly Progress Report — Timur Iakshibaev

## Period: June 29 – July 3, 2026

## Headline

This short cycle produced the **thesis headline experiment** — the first
real-SR-model ranking under the redesigned LR-VCC v5 metric — despite the
period being dominated by a second infrastructure crisis. The June-15
rescue server (which held the disks from the decommissioned lab GPU box)
went offline, and a brand-new multi-tenant GPU host had to be discovered,
provisioned, and populated from scratch. The Mac↔server link turned out to
be effectively unusable (~22 KB/s, dropping every few minutes), which forced
an unconventional but robust data-transfer method (GitHub as an intermediary
CDN). The upshot: **LR-VCC v5 ranks MGLD-VSR above Upscale-A-Video on all 5
videos, unanimously**, and this result was ultimately computed entirely on
the local machine — meaning the headline no longer depends on any fragile
server at all.

---

## 1. Real-model LR-VCC v5 ranking — the headline result

The central thesis claim has always been that LR-VCC measures *long-range
consistency* in a way per-frame metrics (PSNR/SSIM) cannot. This period
delivered the experiment that demonstrates it on real SR-model outputs.

`reports/figures/realmodel_v5_ranking.md`:

| video | MGLD v3 | UAV v3 | **MGLD v5** | UAV v5 | v5 winner |
|-------|--------:|-------:|------------:|-------:|:---------:|
| hhszUXL1Cu8 | 0.558 | 0.476 | **0.579** | 0.528 | MGLD |
| 7WHI2L_FDNg | 0.529 | 0.458 | **0.652** | 0.609 | MGLD |
| KZ8p6b1zJ9U | 0.571 | 0.528 | **0.732** | 0.712 | MGLD |
| BrRLKMbBTYQ | 0.316 | 0.269 | **0.407** | 0.380 | MGLD |
| mJog8DlRk_4 | 0.409 | 0.374 | **0.575** | 0.533 | MGLD |
| **mean** | 0.476 | 0.421 | **0.589** | 0.552 | MGLD |

**MGLD-VSR wins on all 5 videos** under both v3 and v5 LR-VCC. This is
consistent with the April no-reference evaluation, where MGLD won 8 of 9
quality metrics — so the metric is ranking the methods the way independent
evidence says it should.

v5 sub-metric means (MGLD vs UAV):

| sub-metric | MGLD | UAV | MGLD − UAV |
|------------|-----:|----:|-----------:|
| A (CLIP-IQA) | 0.447 | 0.336 | **+0.111** |
| T (tOF) | 0.934 | 0.942 | −0.008 |
| I (slow-fast identity) | 0.555 | 0.463 | **+0.092** |
| D (histogram stability) | 0.505 | 0.506 | −0.002 |
| E (colour slope) | 0.339 | 0.341 | −0.001 |
| **D' (anchor histogram)** | 0.708 | 0.687 | **+0.021** |
| **D'' (CLIP trajectory)** | 0.893 | 0.884 | +0.009 |

The two new sub-metrics both favour MGLD, reinforcing the appearance and
identity signals rather than contradicting them — evidence that the redesign
didn't destabilise the metric's ranking behaviour on real models. The gap
narrows slightly from v3 (+0.056) to v5 (+0.037) because D'/D'' are near-tied
on these two well-behaved diffusion methods, which is the expected behaviour
(the new sub-metrics were designed to catch *pathological* drift, and neither
MGLD nor UAV drifts pathologically).

### Why this was computable locally

A key realisation de-risked the whole thing: the real-model table needs **no
server**. MGLD and UAV already had clip_iqa + tOF + colour-histogram +
colour-slope + identity from the May Task-8 run (all mirrored locally). Only
the two new sub-metrics were missing, and both run on the Mac —
D' from OpenCV/NumPy, D'' from OpenAI CLIP ViT-B/32 (installed locally,
runs on CPU/MPS). So the headline result is now independent of the fragile
GPU-server situation entirely.

---

## 2. Infrastructure crisis #2 — new GPU server stood up from zero

### The situation

- The June-15 CPU rescue server (holding disk1/disk2/disk3 from the old lab
  box) went offline — connection refused, no recovery.
- A new multi-tenant GPU host was located: **2× A100-PCIE-40 GB, 22 cores,
  175 GB RAM**, Ubuntu 22.04, on `yc.smartml.cn`.
- Note the GPUs are 40 GB, not the 80 GB of the old lab box — batch sizes
  must be halved for anything previously tuned to 80 GB.

### What was provisioned (from nothing)

- Created a dedicated **`timur` user** with SSH-key auth (rather than sharing
  the shared `root` login with the other tenants — cleaner and safer).
- **Miniconda + the `vsr` metrics env** (torch 1.13.1+cu117, 2 GPUs visible)
  rebuilt from the `docs/server_conda_envs_2026-06-15.txt` freeze.
- **The `vbench` env** (torch 2.5.1+cu121) rebuilt for identity/CLIP work.
- **Four upstream repos cloned + MGLD patched** (config paths, video_vae
  symlink, SpyNet try/except).
- **MGLD-VSR checkpoints (7 GB)** downloaded via the HuggingFace China mirror
  (`hf-mirror.com`), since huggingface.co itself is unreachable from the host.
- All metric JSONs + v5 composites synced to the server.

### The transfer problem and its solution

The Mac→server link measured at **~22 KB/s and dropped within minutes** —
completely unusable for the SR-output videos (the smallest, 3.3 MB, only got
8 % across in 5 minutes; rsync, scp, and cat-over-ssh all failed). But both
the Mac and the server can reach GitHub's CDN. So the working method became:

> Push files to a throwaway GitHub branch from the Mac (full home-network
> speed), then `git clone` that branch on the server (GitHub-CDN speed).
> Files over GitHub's 100 MB limit are split into 90 MB chunks with `split`
> and reassembled with `cat` on the far side.

This bridged the source LR videos (76 MB), the MGLD SR outputs (773 MB as 12
chunks), the ArcFace weight (98 MB), and the repo scripts. The method is now
documented in `docs/private/server-setup.md` for reuse. This is a genuinely
useful pattern for any China-network-isolated GPU box.

---

## 3. flip_invert (the 12th matrix cell) — identity stack rebuilt, run in progress

The `flip_invert` identity row — the histogram-disrupting *control* artefact
whose identity-slow-fast stage was killed when the original server died on
June 15 — was pushed all the way to a running state this period. It required
standing up the entire VBench-2.0 identity stack from scratch on the new
server, which surfaced a chain of problems (all solved; see
`docs/2026-07-01-new-server-and-gotchas.md`):

1. **Detectron2 not needed** — the identity path uses only RetinaFace +
   ArcFace; Detectron2 was only for the human masks (already have those).
2. **ArcFace weight** (98 MB, Google-Drive-hosted, Drive unreachable from the
   host) — fetched on the Mac, bridged via GitHub, placed at the exact path
   VBench-2.0 expects (`~/.cache/vbench2/arcface/resnet18_110.pth`).
3. **RetinaFace package** — the PyPI `retinaface` is a *TensorFlow* package;
   the correct one (ternaus, provides `predict_single`) hard-pins
   `torch==1.9.0`. Installed the package source directly into site-packages,
   bypassing pip's dependency resolution.
4. **RetinaFace weight** — a GitHub-release zip; the server download hung
   repeatedly and left a truncated 40 MB partial. Fixed with a fresh
   retry-looping curl in tmux → full 97 MB, verified it loads (456 keys).
5. **VBench-2.0 import chain** — a dedicated `identity` conda env plus an
   auto-installing loop for the transitive deps (gdown, scenedetect, mmengine,
   …) until `import vbench2` was clean.
6. **The patched `human_identity.py`** — the fresh VBench clone is *unpatched*
   and crashed with `ZeroDivisionError` (flip_invert inverts colours →
   RetinaFace finds no faces → `score/num` with num=0). A rigorous md5 diff of
   our git-vendored `vbench2/` package against the fresh clone showed **exactly
   one file differs** (`human_identity.py`, carrying the multi-face +
   late-reference + num-guard patches). Deployed our patched copy over the
   clone — flip_invert identity now uses the *same* code as the other 11
   artefacts.
7. **Parallelisation** — the sequential run was ~9 h due to shared-GPU
   contention (both A100s at 100 % from other tenants). Split the 25 clips
   across 8 parallel workers (4 per GPU). First attempt hit a
   concurrent-cache-extraction race that corrupted the RetinaFace zip; fixed by
   warming the cache once before spawning workers, and staggering any
   relaunches by 25 s. All 8 workers now running clean.

**Status at report time:** all 8 identity workers running; ETA ~1.5–2.5 h.
On completion, `scripts/finalize_flip_invert.sh` merges the 8 batch JSONs,
computes the flip_invert v5 composite (all other stage JSONs already local),
rebuilds the verdict matrix to a complete **12/12**, and commits — one command,
fully staged. This cell is a positive control (predicted PASS via the
histogram-disrupting sub-metrics; identity is expected to read ~N/A because
inverted faces are undetectable), so it confirms rather than changes the story.

---

## 4. State of the thesis experimental record

Everything thesis-bound is safe in at least two places (local + git remote;
the SR videos also on the new server):

| artefact | status |
|----------|--------|
| v5 synthetic verdict matrix (12 artefacts × 5 bases) | complete, 11/12 rows (flip_invert identity pending) |
| v5 real-model ranking (MGLD vs UAV) | **complete, committed** |
| D / D' / D'' three-way comparison | complete |
| All metric-stage JSONs (2627 files) | local + git-tracked figures |
| Source + SR videos | local + new server |
| Conda env specs, restore guide, server notes | committed |

Test suite: **126 passing.**

---

## 5. Next period (July 1 → July 15 blind-review submission)

The experimental freeze was nominally July 1. With the headline real-model
result now in hand, the remaining work is almost entirely writing:

1. **Switch `zjuthesis.tex` to `Period=paper`** and begin the methodology
   chapter rewrite (≈70 % liftable from the proposal; new content: D'/D'',
   the flip ablation, the convergence-rewards-stability diagnosis, the v5
   composite).
2. **Experiments chapter**: the v5 synthetic verdict matrix + the real-model
   ranking table (§1 above) are the two centrepieces.
3. **β/α sensitivity sweep + leave-one-out sub-metric ablation** — both
   recompute-only from cached JSONs, no GPU. Adds rigour reviewers will
   expect. ~1 day.
4. **flip_invert** — finish opportunistically if RetinaFace cooperates;
   otherwise report the 11/12 matrix with the control cell noted as
   "infrastructure-interrupted, predicted PASS."
5. **July 13–14**: internal proofread, `BlindReview=true`, final LaTeX build.
6. **July 15**: submit.

## 6. July 2 — RoPE-extrapolation probe launched; FlashVSR becomes the third benchmark row

The post-thesis research arc (Direction 4 of the long-term plan) started a
day early and produced two results in one sprint.

### 6.1 The probe instrument, built and verified

Working hypothesis: RoPE-based VSR attention fails to *extrapolate* beyond
its trained temporal position range, and this — not just content difficulty —
degrades long-video SR. Spec + 10-task plan committed
(`docs/superpowers/specs/2026-07-02-rope-extrapolation-vsr-design.md`);
Tasks 1–5 executed the same day:

- **FlashVSR** (DiffSynth fork of Wan2.1 DiT, one-step streaming VSR) stood
  up on the SmartML server from zero (new `flashvsr` env, torch 2.6+cu124,
  Block-Sparse-Attention compiled for sm_80 after an upstream gencode bug).
- **Architecture finding:** temporal RoPE positions are a slice into a
  1024-row precomputed table, and the streaming pipelines advance *absolute*
  positions per chunk without ever resetting. Two consequences: long videos
  extrapolate by construction, and **stock FlashVSR cannot process more than
  4089 frames (~2.3 min @ 30 fps) in a single pass, period** — the table
  runs out. (`docs/notes/2026-07-02-flashvsr-rope-site.md`)
- **The injection hook passed its faithfulness gate bit-exactly:** the
  probe swaps the freq table for a wrapper that routes position lookups
  through an override; with a no-op override the output is bit-identical to
  stock (drift 0.0 against a measured 0.0 nondeterminism floor), and a +1
  shift demonstrably changes output (0.295) — the instrument works and is
  provably faithful. The FlashVSR repo itself is untouched (pinned + tagged
  `pristine-2026-07-02`; hooks are runtime-only).

### 6.2 FlashVSR tops the LR-VCC v5 ranking (benchmark by-product)

Because the install is pristine, the same FlashVSR served unmodified as a
third real-SR method. Full-coverage inference on all 5 long synthetic videos,
I/O-matched to MGLD/UAV (1280×720 full content, exact frame counts; the four
5000-frame videos forced into 2 segments each by the ceiling above — an
honest property of the method on long videos). Full 7-stage battery + v5
composite (`reports/figures/realmodel_v5_3method.md`):

| | MGLD | UAV | **FlashVSR** |
|---|-----:|----:|---------:|
| LR-VCC v5 mean | 0.589 | 0.552 | **0.610** |
| videos won | 3 (narrow: ≤0.029) | 0 | 2 (large: +0.085, +0.074) |

Sub-metric reading: FlashVSR's win is driven by **identity** (0.598 vs
0.555/0.463); but it is **worst on D'' CLIP-trajectory** (0.862 vs
0.893/0.884) — the long-range-drift cell. A 2025 streaming model beats the
2023 baselines overall while showing exactly the drift signature the probe
attributes to growing absolute positions: the benchmark and the probe now
point at the same suspect from opposite directions. (Closeup-map choice
bracketed with both existing maps — composite moves ≤0.001, so no
FlashVSR-specific anatomy run was needed.)

### 6.3 Infrastructure notes

- Server→Mac bulk transfer solved with a **reverse GitHub bridge**: a 7-day
  fine-grained PAT (contents:write, this repo only) on the server; orphan
  branch push → local pull → branch delete. Moved 314 MB of FlashVSR outputs
  + all metric JSONs. Token lapses ~Jul 9.
- The setuptools-81 `pkg_resources` gotcha struck a third env (`vsr`, via
  pyiqa→clip); fixed the documented way. All new gotchas recorded as addenda
  in `docs/2026-07-01-new-server-and-gotchas.md`.
- Round-2 contrast model decided: **SeedVR2** (window-local positions)
  replaces SparkVSR — tests whether window attention avoids the drift that
  absolute streaming positions incur.

Next on the probe: Task 6 (shift/stretch sweep driver through the verified
hook), then Phase-1 shift-control curves, and the D''-vs-position causal
check that would explain FlashVSR's weakest benchmark cell.

## Open questions for the meeting

1. Is the unanimous MGLD > UAV v5 ranking (with the honest note that the two
   new sub-metrics are near-tied on these well-behaved methods) a strong
   enough real-model result on its own, or should we add a frame-wise lower
   anchor (RealESRGAN per-frame) to widen the ranking spread? The anchor
   would require a fresh SR run on the new server (feasible now that it's
   provisioned, ~half a day).
2. Given two server losses in three weeks, should the metric pipeline be
   packaged (Docker / a one-command restore) so the next infrastructure
   shock is a non-event? This is the third recovery this term.
3. Confirm the July-1 freeze holds for synthetic experiments, with only the
   real-model anchor (Q1) and flip_invert as permitted exceptions.
4. **Does the 3-method table (FlashVSR added, §6.2) enter the thesis
   experiments chapter, or stay paper-side?** It strengthens the ranking
   story (a modern method wins; LR-VCC still discriminates via D'') but was
   computed after the nominal freeze — needs an explicit freeze-exception
   decision like Q1/Q3.
5. Priority between thesis writing and the RoPE probe for the coming two
   weeks: the probe instrument is ready (Task 5 gate passed), but July 15 is
   the blind-review deadline — proposal: probe runs only as background GPU
   jobs, all foreground time on writing.
