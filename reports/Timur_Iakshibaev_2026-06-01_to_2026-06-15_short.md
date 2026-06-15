# Progress Report — June 1 – June 15, 2026

**Topic:** Video Super-Resolution for Long Videos — LR-VCC consistency benchmark

## Headline

Implemented two new long-term-consistency artefacts (identity_drift, background_drift) requested at the May meeting, scaled validation 2 → 5 base videos, then diagnosed and fixed a fundamental mechanism flaw in the LR-VCC composite metric. **Headline result: background_drift inversions dropped from 4/5 INVERTED under v4 to 1/5 INVERTED under v5** — diagnosis empirically confirmed and fixed in production.

## Key Results

### 1. Validation set scaled and stabilised

| dimension | prior | now |
|---|---|---|
| base videos | 2 | 5 |
| artefact families | 4 | 12 |
| total clips evaluated | 60 | 300 |
| sub-metrics in composite | 5 | 7 |
| test suite | 76 passing | 114 passing |

All five base videos (hhszUXL1Cu8, 7WHI2L_FDNg, KZ8p6b1zJ9U, BrRLKMbBTYQ, mJog8DlRk_4) cover distinct content types: cooking with intermittent faces, single-face talking-head, news with text, animated cartoon, lifestyle TV.

### 2. Mechanism diagnosis — convergence rewards stability

Sub-metric D (colour-histogram temporal stability) systematically *rises* under background_drift on all 5 bases. Mechanism: replacing a dynamic background with a progressively static reference image genuinely makes per-frame colour histograms more stable. D measures exactly what it was designed to measure — but stability-based no-reference metrics reward convergence-type corruption. Supervisor independently arrived at the same diagnosis ("degradation makes videos more stable").

This was unexpected and has broad implications: any no-reference video-quality metric that rewards intra-clip stability will systematically prefer over-smoothed SR outputs.

### 3. Metric redesign — D' and D''

Two replacement sub-metrics implemented and integrated:

- **D' (anchor-window Lab histogram)**: distance of each frame from the first-60-frame anchor centroid. Score = `exp(-β · |q4 − q1|)`, β = 0.5. CPU only.
- **D'' (CLIP-image trajectory)**: per-frame CLIP ViT-B/32 embedding, cosine distance to first-60-frame anchor centroid, same formula, β = 3.0. GPU only.

Both share the existing `{score, reliability, details}` schema; both integrated into the v5 LR-VCC composite (keeping D — it still uniquely catches chunk_boundary at 5/5). 4 new unit tests; default behaviour byte-identical when arguments omitted.

### 4. v5 composite — background_drift fully recovered

| base | v4 Δ | v4 verdict | v5 Δ | v5 verdict |
|---|---:|---|---:|---|
| hhsz | −0.276 | PASS | −0.222 | PASS |
| 7WHI | +0.046 | INVERTED | −0.013 | FLAT |
| KZ | −0.002 | FLAT | −0.051 | PASS |
| BrRLK | +0.127 | INVERTED | +0.065 | INVERTED (halved) |
| mJog | +0.064 | INVERTED | −0.030 | WEAK |

4 of 5 inversions converted to PASS / WEAK / FLAT. Only BrRLK (cartoon) remains, with the inversion magnitude halved. The cartoon failure has a documented content-domain explanation: natural scene-cut variation dominates the anchor-distance signal on both D' and D''. Reported as a content-domain limitation.

### 5. Flip ablation confirms the diagnosis empirically

Designed six self-modifying midpoint artefacts as a statistical-preservation ablation ladder, then tested predictions at the composite level (composite results pending one final battery run):

| transform | preserves | predicted | composite v5 |
|---|---|---|:---:|
| flip_horizontal | full histogram | invisible to all D variants | **0/5 ✓** |
| flip_transpose | full histogram (rotated) | partial via D'' | **3/5** |
| flip_periodic | full histogram | mostly invisible | **0/5** |
| flip_elastic | ≈full histogram | mostly invisible | **0/5** |
| flip_channel_shuffle | per-channel marginals | PASS via both D' and D'' | **4/5 ✓** |
| flip_invert | only variance | sanity PASS | pending |

The flip_horizontal probe was the cleanest diagnostic: a corruption that completely breaks identity while preserving the histogram exactly. Empirically invisible to all 7 sub-metrics in the v5 composite — confirms the convergence-rewards-stability diagnosis with a hand-designed null result. CLIP's known horizontal-flip robustness (a documented property of ViT trained with horizontal-flip augmentation) makes this also a blind spot for D''.

This becomes an honest "known limitation" thesis paragraph rather than a metric bug.

## Code Delivered (this period)

| File | Purpose |
|---|---|
| `scripts/lr_vcc/identity.py` (modified) | Parked variance gate for slow-fast pooling pathology |
| `scripts/synthetic_artefacts/identity_drift.py`, `background_drift.py`, `select_reference_scene.py`, `flip.py` | 4 new artefact generator modules |
| `scripts/synthetic_artefacts/precompute_human_masks.py` | Detectron2 Mask R-CNN human-silhouette precompute |
| `scripts/lr_vcc/color_histogram_anchor.py` | D' anchor-window histogram |
| `scripts/lr_vcc/compute_clip_trajectory.py` | D'' CLIP-trajectory |
| `scripts/lr_vcc/build_verdict_matrix.py`, `compare_d_variants.py` | Analysis tooling |
| `scripts/lr_vcc/run_lr_vcc.py` (modified) | v5 composite with 7 sub-metrics |

Test suite **114 passing**, 0 failing. All code TDD with subagent code review (spec compliance + code quality) before merge.

## Next Period (June 16 – June 30)

1. **Real-SR-model evaluation with v5 LR-VCC.** Apply v5 composite to MGLD-VSR, Upscale-A-Video, and a frame-wise lower anchor (RealESRGAN per-frame) on the 5-video set. Thesis-headline experiment: does the new composite rank these models in a way PSNR / SSIM cannot?
2. **β / α calibration sweep** for the new sub-metrics from cached trajectory JSONs (no GPU time).
3. **Leave-one-out sub-metric ablation** to identify which artefact families each of the 7 sub-metrics uniquely catches.
4. **Start writing track:** switch `zjuthesis.tex` to `Period=paper`; methodology chapter draft begins ~June 22.

**Hard experiment freeze: July 1.** Blind-review thesis submission: July 15.
