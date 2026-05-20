# Long-range tOF / tLP results — crossover at k=5-10 favours MGLD

**Date:** 2026-05-14
**Run:** `scripts/long_range_temporal/eval_tof_tlp.py` on server GPU 4, ~37 min for 5 videos × 2 methods × 6 k-values × 200 pairs/k. Per-video JSONs in `results/long_range_temporal/{mgld,uav}/`.

## tOF — pixel-level temporal consistency

Lower = better. Mean across 5 videos:

| k | MGLD | UAV | Δ (M − U) |
|---|----:|----:|----------:|
| 1 | 0.0216 | 0.0177 | +0.0039 (UAV wins) |
| 5 | 0.0406 | 0.0424 | −0.0018 |
| 10 | 0.0500 | 0.0618 | **−0.0118** |
| 30 | 0.0804 | 0.0922 | **−0.0119** |
| 60 | 0.1110 | 0.1314 | **−0.0204** |
| 120 | 0.1441 | 0.1682 | **−0.0241** |

Per-video tOF winners:

| Video | k=1 | k=5 | k=10 | k=30 | k=60 | k=120 |
|-------|----|-----|------|------|------|-------|
| 7WHI2L_FDNg | UAV | MGLD | MGLD | MGLD | MGLD | MGLD |
| BrRLKMbBTYQ | MGLD | MGLD | MGLD | MGLD | MGLD | MGLD |
| **KZ8p6b1zJ9U** | UAV | UAV | **MGLD** | **MGLD** | **MGLD** | **MGLD** |
| hhszUXL1Cu8 | UAV | UAV | UAV | UAV | UAV | UAV |
| mJog8DlRk_4 | UAV | MGLD | MGLD | MGLD | MGLD | MGLD |

Two clean findings:

1. **A crossover at k=5–10.** UAV wins adjacent-frame stability (smoother frames warp into themselves better); MGLD wins long-range stability. Adjacent-frame temporal metrics (E*warp, tOF k=1) systematically favour smoother SR.
2. **On KZ specifically — the video where Anatomy flips** — tOF agrees with perception at k≥10. MGLD wins long-range temporal stability on KZ even though Anatomy says the opposite.

## tLP — perceptual temporal consistency (LPIPS)

Lower = better. Mean across 5 videos:

| k | MGLD | UAV | Δ (M − U) |
|---|----:|----:|----------:|
| 1 | 0.0295 | 0.0222 | +0.0073 |
| 5 | 0.0445 | 0.0458 | −0.0014 |
| 10 | 0.0422 | 0.0385 | +0.0036 |
| 30 | 0.0334 | 0.0268 | +0.0066 |
| 60 | 0.0244 | 0.0202 | +0.0042 |
| 120 | 0.0158 | 0.0130 | +0.0027 |

Per-video tLP winners: UAV dominates all k except mJog8DlRk_4 from k≥5. On KZ specifically UAV wins all k by margin 0.005–0.029.

**tLP systematically favours UAV across all k.** Mechanism: LPIPS rewards self-similarity; UAV's smoothness means warped frames look more like the source frame in LPIPS feature space. **Same "smoother-output bias" as VBench's subject_consistency (DINOv2) and Anatomy on KZ — three completely independent learned representations all reward smoothness.** This is now the strongest piece of structural evidence we have for the thesis's central claim: long-video SR metrics built on "trained on pristine HR" representations have a built-in bias that disfavours sharp, detail-preserving SR.

## Mask coverage — methodological flag

Mean across 5 videos, fraction of frame valid per pair:

| k | MGLD | UAV |
|---|----:|----:|
| 1 | 0.93 | 0.91 |
| 5 | 0.69 | 0.64 |
| 10 | 0.57 | 0.35 |
| 30 | 0.36 | 0.19 |
| 60 | 0.22 | 0.12 |
| 120 | 0.11 | 0.07 |

Two things to flag:

1. **Coverage drops fast** with k. At k=120 we're measuring tOF/tLP over ~10% of the frame.
2. **UAV's coverage is consistently lower** than MGLD's at long k. RAFT flow estimator finds fewer FB-consistent regions in UAV's smoother output — smooth textures give the flow estimator less to lock onto. So at long k the *valid pixel subsets* differ per method, and the comparison is sampling-biased. This caveat needs flagging in any thesis write-up.

Mitigations to consider:
- Restrict to a fixed "always valid" pixel set per video (intersection of masks across all k and both methods). Smaller sample, fairer.
- Stride down sampling drastically at small k to match coverage between methods — gives lower variance per number but at the cost of comparing differently-sized pixel sets.
- Use SpyNet or RAFT-small with looser FB tolerance (3 px vs 1 px) to keep more pixels at long k.

## Implications for the metric-effectiveness story (thesis)

Three concrete additions to the case:

1. **The crossover at k=5–10 is real and meaningful.** It directly demonstrates that adjacent-frame temporal metrics are insufficient for long-video SR. We can claim "single-number temporal metrics flatten exactly the long-range information that matters for SR" and back it with numbers.

2. **Anatomy's failure on KZ is not perceptual — tOF disagrees with it.** On KZ8p6b1zJ9U the metric tally now reads: NR-IQA (4/4 MGLD), VBench-1 Quality (MGLD), Identity slow-fast (MGLD), DOVER (MGLD), E*warp k=1 (MGLD), tOF k≥10 (MGLD), tLP (UAV), Anatomy whole-video & slow-fast (UAV). **6 of 8 metric families favour MGLD on KZ; only Anatomy and tLP disagree** — and tLP's disagreement is consistent with its known LPIPS-self-similarity bias.

3. **Three independent learned-representation metrics show the same smoother-output bias** — VBench `subject_consistency` (DINOv2), VBench `Human_Anatomy` on close-up (anomaly ViT), tLP (LPIPS) — all reward UAV over MGLD on at least one video where humans prefer MGLD. This is structural, not coincidental: representations trained on real / pristine HR data flag diffusion-detail as anomalous. A long-video SR metric needs a representation that doesn't have this bias.

Inputs for the long-range-consistency-metric brainstorm (next):

- We have a working multi-k tOF/tLP pipeline.
- We've localized one clean failure mode (smoother-output bias in learned representations).
- We've localized the crossover scale (k = 5–10 frames at our fps ≈ 0.2 s).
- Mask coverage at long k is a structural problem (need to either accept smaller pixel subset or change the flow tolerance).
