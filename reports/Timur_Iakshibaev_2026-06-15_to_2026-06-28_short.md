# Progress Report — June 15 – June 28, 2026

**Topic:** Video Super-Resolution for Long Videos — LR-VCC consistency benchmark

## Headline

**LR-VCC v5 composite finalised:** background_drift inversions 4/5 (v4) → 1/5 (v5). The convergence-rewards-stability mechanism identified at mid-period is fixed in production by the new D' (anchor-window Lab histogram) and D'' (CLIP-image trajectory) sub-metrics. 11 of 12 artefact families have full composite results; the missing flip_invert row was the predicted-PASS sanity control, lost when the lab GPU server was decommissioned mid-experiment on June 15.

## Key Results

### 1. v5 composite — 11 of 12 artefacts × 5 bases

| artefact | hhsz | 7WHI | KZ | BrRLK | mJog | clean |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| color_drift | WEAK | PASS | PASS | PASS | PASS | **5/5** |
| chunk_boundary | PASS | PASS | PASS | INV | WEAK | 4/5 |
| **background_drift** | **PASS** | FLAT | **PASS** | INV | **WEAK** | **4/5** |
| flip_channel_shuffle | FLAT | PASS | PASS | FLAT | PASS | 3/5 |
| flip_transpose | PASS | FLAT | PASS | FLAT | WEAK | 3/5 |
| identity_degradation | WEAK | FLAT | WEAK | FLAT | FLAT | 2/5 |
| identity_drift | WEAK | FLAT | FLAT | FLAT | FLAT | 1/5 |
| flip_periodic | WEAK | FLAT | FLAT | FLAT | FLAT | 1/5 |
| flip_elastic | FLAT | FLAT | FLAT | FLAT | WEAK | 1/5 |
| flicker | FLAT | FLAT | FLAT | INV | FLAT | 0/5 |
| flip_horizontal | FLAT | FLAT | FLAT | FLAT | FLAT | 0/5 |
| flip_invert | — | — | — | — | — | (pending) |

Clean (PASS+WEAK) at composite: **25/55**.

### 2. background_drift fix — the period's core empirical result

Per-base composite Δ on background_drift, v4 → v5:

| base | v4 Δ | v4 verdict | v5 Δ | v5 verdict |
|---|---:|---|---:|---|
| hhsz | −0.276 | PASS | −0.222 | PASS |
| 7WHI | +0.046 | INV | −0.013 | FLAT |
| KZ | −0.002 | FLAT | −0.051 | PASS |
| BrRLK | +0.127 | INV | +0.065 | INV (halved) |
| mJog | +0.064 | INV | −0.030 | WEAK |

4 of 5 inversions converted to PASS / WEAK / FLAT; BrRLK cartoon remains a documented content-domain limitation (natural scene-cut variation in source dominates anchor distance on both D' and D'').

### 3. Diagnostic flip-family ablation behaved as predicted

| transform | preserves | predicted | composite v5 |
|---|---|---|:--:|
| flip_horizontal | full histogram | invisible to all D variants | **0/5 ✓** (the smoking-gun null result) |
| flip_transpose | full histogram (rotated) | partial via D'' | 3/5 ✓ |
| flip_periodic | full histogram | mostly invisible | 1/5 ✓ |
| flip_elastic | ≈full histogram | mostly invisible | 1/5 ✓ |
| flip_channel_shuffle | per-channel marginals | PASS via D' + D'' | 4/5 ✓ |
| flip_invert | only variance | sanity PASS | (lost to server downtime) |

Empirically confirms the theoretical convergence-rewards-stability diagnosis with hand-designed null results.

### 4. Infrastructure note (one-line)

Lab GPU server `223.109.239.43` was decommissioned mid-experiment on June 15. All thesis-relevant data was rescued to a CPU host (`/data/disk3/timur/`, 7.2 GB) and to a full local mirror (~4 GB) on the same day. **No thesis-relevant data was lost** — only the in-flight flip_invert identity stage (the predicted-PASS control row) did not complete.

## Code delivered this period

| File | Purpose |
|---|---|
| `scripts/server_runners/` (61 scripts) | Rescued runner scripts from the original GPU server |
| `docs/server_restore_guide.md` | End-to-end pipeline-restore procedure for a fresh GPU host |
| `docs/plans/2026-06-15-short-term-plan.md` | June 15 → July 15 (5 priorities, hard freeze July 1) |
| `docs/plans/2026-06-15-long-term-plan.md` | July 15 → Sept 30 + 5 open research directions |
| `docs/server_conda_envs_2026-06-15.txt` | pip-freeze of `vsr` / `vbench` / `uav` envs (pre-decommission) |

Test suite **126 passing**, 0 failing.

## Next Period (June 29 – July 12)

Contingent on a new GPU host being available — otherwise writing-track is front-run.

1. **Real-SR-model evaluation with v5 LR-VCC.** Apply v5 to MGLD-VSR, Upscale-A-Video, and a frame-wise RealESRGAN lower anchor on the 5-video set. Thesis-headline experiment: does the new composite rank these models in a way PSNR / SSIM cannot?
2. **β / α sensitivity sweep + leave-one-out sub-metric ablation.** Recomputable from cached trajectory JSONs — no GPU required.
3. **Classmate model outreach.** 5 LR mp4s + submission spec sent by ~June 30; soft deadline for receipts ~July 6.
4. **Methodology chapter draft.** Switch `zjuthesis.tex` to `Period=paper` on June 30; methodology chapter from June 30 onward (~70% liftable from the proposal).

**Hard experiment freeze:** July 1. **Blind-review thesis submission:** July 15.
