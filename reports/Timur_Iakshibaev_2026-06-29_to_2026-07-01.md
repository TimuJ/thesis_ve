# Weekly Progress Report — Timur Iakshibaev

## Period: June 29 – July 1, 2026

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

## 3. flip_invert (the 12th matrix cell) — attempted, blocked

The one incomplete item is the `flip_invert` identity row — the histogram-
disrupting *control* artefact whose identity-slow-fast stage was killed when
the original server died on June 15.

Progress this period:
- Confirmed identity does **not** need Detectron2 (that was only for the
  human masks, which we already have). The identity path needs only
  RetinaFace + ArcFace.
- ArcFace weight (98 MB, Google-Drive-hosted, and Drive is unreachable from
  the host) was fetched on the Mac and bridged to the server successfully.
- RetinaFace (`pip install retinaface`) did not install cleanly, and its
  weight prefetch (a GitHub release) hung on the flaky server↔GitHub link.

flip_invert remains blocked here. It is the least scientifically important
of the 12 artefacts (a positive control that is predicted to PASS on all
bases), so the synthetic-validation story is essentially complete at 11/12
cells regardless. It can be finished opportunistically once RetinaFace
installs, but it does not gate any thesis claim.

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
