# Weekly Progress Report — Timur Iakshibaev

## Period: July 13 – 19, 2026 (IN PROGRESS)

## Headline

The RoPE sensitivity study became a **standalone group deliverable**
(`reports/rope_timespace_sensitivity_matrix.md`) with two substantive
upgrades: corrected resolution-ladder bookkeeping that *sharpens* the story
(the rungs are exactly 1×/1.5×/2× the trained spatial extent, and the
collapse-rung grid divides evenly into attention windows — eliminating the
window-remainder suspect outright), and a **theory section explaining why
Position Interpolation works at all** — with our data as the first
per-axis empirical fingerprint of the interpolation-vs-extrapolation
distinction in a video model. The collapse-decomposition experiments
(sparsity pinning, blur isolation, knee ladder) are running. Thesis
chapters remain the critical path (blind review July 25).

## 1. Sensitivity report — corrections and the PI theory section

- **Grid bookkeeping corrected from the run records:** input padding to
  multiples of 32 quantises the latent grid to multiples of 8, so the
  ladder actually ran at **48² / 72² / 96²** (not 45/72/90 as first
  labelled) with outputs 720²/1152²/**1536²**. Improvements to the story:
  rung 1 sits *exactly at* the trained extent (48); the ladder is a clean
  1×/1.5×/2×; and the collapse grid (96) is divisible by the 8-latent
  window — **the window-remainder hypothesis is ruled out**, leaving
  adaptive sparsity (topk 3.8 → 1.5 → 0.83) as prime suspect and
  upsampled-input blur as secondary.
- **New §5 "Why PI helps at all":** fractional positions are novel values
  but *interpolated within* the trained relative-phase range, where a
  learned smooth attention function is well-behaved; harm comes from
  *extrapolation beyond* it (Chen et al. 2023). Our per-axis data is the
  fingerprint: s=0.75 free everywhere (+0.01 dB) vs extrapolation up to
  −2.6 dB — and the boundary is real: at s=0.5 neighbour distances fall
  below the minimum trained distance (1.0), entering the self-vs-neighbour
  transition region, which is why the zero-shot PI-free zone is narrow
  (~0.75) and aggressive PI needs fine-tuning in the LLM literature.

## 2. Collapse-decomposition experiments (launched this period)

Five arms × 8 YouHQ40 clips, each isolating one suspect for the 1536²
collapse (−11 dB, PI recovers only +0.28 → not positional):

| arm | isolates | prediction if suspect is real |
|---|---|---|
| 1536² stock, content-cropped scoring | scoring-artefact share (pad bands) | small recovery only |
| 1536² with topk pinned to the healthy 1152² value | **adaptive sparsity** | large recovery |
| 1152² grid with ×1.33-upsampled input | input blur at fixed grid | small, bounded cost |
| 1280² (grid 80, topk 1.20) | knee location | healthy |
| 1408² (grid 88, topk 0.99) | knee location | transitional |

**Outcome (all 8 clips): the "collapse" was a scoring-geometry artefact**
— the first pass compared the padded output frame against GT at mismatched
magnification. Content-correct scoring: grid 96 = **24.78 dB**, the best
rung; sparsity pinning Δ−0.03 (cleared); grids 80/88 flat; no blur
penalty. **Strengthened verdict: resolution extrapolation is quality-free
to ≥2× the trained extent.** All reports corrected transparently.

## 3. Thesis track

Blind review **July 25 (hard)**. Chapter work per the July-13–24 schedule in
the biweekly; the RoPE-study chapter's source material is final (sensitivity
report + findings note + verdict figures).

_To be completed through July 19._
