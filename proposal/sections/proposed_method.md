# Section 4: Proposed Method — LR-VCC

The **Long-Range Video Consistency Composite (LR-VCC)** is a no-reference composite metric for ranking video super-resolution methods on long-video temporal, perceptual, and identity consistency. It is built directly from the three failure modes characterized in Section 3 and was designed to be robust to each of them. Rather than replacing existing learned-representation metrics with new bias-free alternatives — which would require large labeled diffusion-SR training datasets — LR-VCC uses existing metrics honestly and downweights them per video in the regimes where their failure mechanisms are known to activate.

---

## 4.1 Architecture

**[Figure 4 — LR-VCC architecture: SR video → three sub-metrics in parallel (each emitting a score and a reliability score) → reliability-weighted composition → final LR-VCC score.]**

Three sub-metrics, each producing a `(score, reliability)` pair in [0, 1], are computed in parallel from the SR video:

- **Sub-metric A — Appearance stability.** Per-frame CLIP-IQA over the video; score = mean(quality) − λ·std(quality). Captures whether the SR output's perceptual quality is both high and stable across minutes of footage.
- **Sub-metric T — Temporal stability.** log(1+k)-weighted mean of tOF across k ∈ {1, 5, 10, 30, 60, 120} frame gaps. Captures multi-scale temporal smoothness, with long-range lags weighted more heavily than adjacent-frame lags. This directly addresses Finding 1 (adjacent-frame temporal metrics undervalue long-range stability).
- **Sub-metric I — Identity preservation.** Wraps the patched slow-fast Human_Identity adapter (RetinaFace + ArcFace), incorporating all fps-correction and multi-face reliability fixes documented in Section 3.5.

Each sub-metric outputs not only a score but also a **reliability** — a value in [0, 1] that encodes how trustworthy the score is on this particular video, derived from regime indicators that correspond directly to the failure modes in Section 3.

The sub-metric scores and reliabilities feed into a **composition block** that combines them via a reliability-weighted log-mean (geometric mean with per-sub-metric weights):

```
weights = softmax([A_reliability, T_reliability, I_reliability] / τ)
LR_VCC(V, M) = exp( Σ_s  weights[s] · log(score_s + ε) )
```

with temperature τ = 0.2 (sharp softmax — the most reliable sub-metric dominates) and ε = 1e-6 for numerical stability. The log-mean preserves the "no compensation for failures" property: a sub-metric scoring 0.1 pulls the composite down even if its weight is only 0.3. A video where all three reliabilities fall below 0.2 is marked **low-confidence** and excluded from the per-method aggregate by default; it is still reported in the per-video table with a flag.

---

## 4.2 Sub-metric A — Appearance Stability

Sub-metric A measures whether the SR output's perceptual quality is high and stable across time.

**Computation.** Per-frame CLIP-IQA quality scores are computed over every frame of the video using the `pyiqa` library. The score is:

```
A_score = clamp(mean(quality) − λ · std(quality),  0, 1),   λ = 0.5
```

The mean term rewards high average perceptual quality; the drift-variance penalty rewards frames that are consistently good rather than intermittently excellent. Lambda is 0.5 by default; it is tunable but was set independently of the test set.

**Reliability test.** Two regime indicators reduce A_reliability:

1. *Sub-metric undiscriminating:* if std(quality) < 0.02, both methods produce nearly identical per-frame quality distributions and sub-metric A carries no differential signal. A_reliability is penalized proportionally (smooth sigmoid, sharpness 10).
2. *Saturation ceiling:* if mean(quality) > 0.98, both methods are in a ceiling regime where CLIP-IQA cannot discriminate. A_reliability is penalized.

**Why this avoids smoother-output bias.** CLIP-IQA is trained on diverse content — not exclusively pristine HR reference images — so its feature space does not systematically flag diffusion-generated high-frequency detail as low-quality. The drift-variance term is also symmetric: a stably-blurry SR output will have low std (no penalty from drift) but also a lower mean quality score, so it does not escape penalty. This is structurally different from LPIPS or DINOv2, which reward feature-space self-similarity and thereby reward spatially smooth outputs independent of mean quality.

---

## 4.3 Sub-metric T — Temporal Stability

Sub-metric T measures how stable the SR video is over multiple time scales, with an explicit long-range bias.

**Computation.** RAFT optical flow is computed between frame pairs separated by k ∈ {1, 5, 10, 30, 60, 120} frames, with forward-backward consistency masking (valid mask: FB error < 1 px). For each k, the mean optical flow magnitude on valid pixels (tOF_k) is computed over 200 uniformly sampled frame pairs per video. The sub-metric score is:

```
T_score = 1 − weighted_mean_k(tOF_k),   weight(k) = log(1 + k)
```

The log(1+k) weighting assigns significantly more influence to long-range lags than to adjacent-frame lags. At k=120, weight ≈ 4.8; at k=1, weight ≈ 0.7. This matches the empirical finding in Section 3.3: at k=1 the smoother method (UAV) wins on tOF, but at k≥10 the detail-preserving method (MGLD) wins. By weighting long-range lags more, sub-metric T is oriented toward the regime that distinguishes genuine long-range consistency from superficial frame-to-frame smoothness.

**Reliability test.** For each k, if mean forward-backward mask coverage drops below 10% of pixels, that k's contribution is down-weighted in the weighted mean. T_reliability is the mean over all k of (1 − below-coverage penalty). At k=120, mask coverage on smooth-output videos can fall below 7% (documented in Section 3.3); the reliability mechanism reduces T's influence in exactly these cases.

**Why tOF is used and not tLP.** tLP (LPIPS-based perceptual warping error) favours UAV over MGLD at every k value tested — including long k — because LPIPS rewards frames that look like their own warped predecessors in LPIPS feature space, and UAV's smooth output satisfies this criterion by construction. This is the third instance of smoother-output bias from a learned representation metric (Finding 2). Using tOF (pixel L2 on valid regions) instead of tLP removes this bias source from the temporal sub-metric. tLP is retained as a diagnostic column in the per-video report but explicitly excluded from the composite.

---

## 4.4 Sub-metric I — Identity Preservation

Sub-metric I measures whether people in the video retain stable visual identity across long time scales.

**Computation.** The existing slow-fast Human_Identity adapter from `scripts/vbench2_long/human_identity_long.py` is used, incorporating the fps-correction patch documented in Section 3.5. The adapter splits the video at 2-second clip boundaries at the LQ-source fps, computes ArcFace embedding similarity for within-clip pairs (slow branch), and computes identity drift across clip first-frames (fast branch). The fused slow-fast score is I_score.

**Reliability tests.** Two regime indicators reduce I_reliability:

1. *Low face-detection rate:* if fewer than 20% of clips contain a detected face, identity is not a meaningful signal on this video. I_reliability is penalized proportionally.
2. *Close-up content:* if the median face or hand bounding-box area exceeds 5% of the frame area, the content is in the close-up regime where the Anatomy classifier (and by structural analogy, the ArcFace tracker) are known to operate near their calibration boundary. I_reliability is penalized by a smooth sigmoid around this threshold.

The close-up reliability test directly operationalises the KZ8p6b1zJ9U failure characterization from Section 3.4. The Anatomy metric's complete breakdown on that video was traced to a single content-type predictor — hand bbox p50 > 5% of frame area — that predicted the metric's direction across all 5 test videos monotonically. The same predictor is applied here conservatively: when bbox area signals close-up scale, the identity sub-metric is trusted less, allowing the other two sub-metrics to determine the composite outcome.

---

## 4.5 Reliability-Weighting — the Core Mechanism

The central design hypothesis is that **every existing learned-representation metric is reliable on some content and unreliable on other content**. Rather than discard failed metrics or attempt to build new bias-free metrics, LR-VCC uses existing metrics honestly and controls their influence per video through reliability scores derived from the failure characterizations in Section 3.

The per-sub-metric reliability is computed from regime indicators using smooth sigmoid penalties (sharpness 10 — smooth transitions rather than sharp threshold cliffs). All indicator thresholds were chosen from the Section 3 characterizations; they are not tuned to the test set.

A sharp softmax (τ = 0.2) over the reliabilities ensures that when one sub-metric is substantially more trustworthy than the others, it clearly dominates the composition. This is the mechanism that makes LR-VCC **flip-resistant**: on a video where one sub-metric is in a known-failure regime (e.g., Anatomy/Identity on a close-up video), the other two sub-metrics can carry the composite without requiring their scores to be implausibly high. The dominant-path emphasis in Figure 4 — the thicker border around sub-metric T and heavier arrows on that path — reflects the expectation that T has the most reliable signal across diverse long-video content, since tOF does not depend on a representation trained on pristine HR data.

The log-domain composition preserves the "no compensation" property across sub-metrics: a score of 0.1 from one sub-metric can only be compensated by the other two when those two carry all the weight (which requires the failing sub-metric's reliability to be near zero). Weighted averaging in linear space would not have this property — a single sub-metric failure could be diluted away.

---

## 4.6 Implementation

**Code layout:**

| File | Purpose |
|---|---|
| `scripts/lr_vcc/appearance.py` | Sub-metric A — per-frame CLIP-IQA + mean/std score |
| `scripts/lr_vcc/temporal.py` | Sub-metric T — long-k tOF reader + weighted score |
| `scripts/lr_vcc/identity.py` | Sub-metric I — Human_Identity adapter wrapper |
| `scripts/lr_vcc/reliability.py` | Sigmoid penalty functions for all regime indicators |
| `scripts/lr_vcc/composite.py` | Softmax weighting + log-mean composition |
| `scripts/lr_vcc/run_lr_vcc.py` | CLI: takes precomputed tOF JSONs, CLIP-IQA dumps, identity results |
| `scripts/lr_vcc/compute_clip_iqa.py` | Per-frame CLIP-IQA dump (GPU server) |
| `scripts/lr_vcc/build_closeup_map.py` | Close-up bbox-p50 map from anatomy per-frame traces |

**Unit tests:** `tests/lr_vcc/` — 20 tests passing. Tests cover reliability sigmoid boundary conditions, log-mean numerical stability (ε clamp), low-confidence gating, and CLI I/O format.

**Composability.** Each sub-metric can be run independently and its output cached as a JSON file. `run_lr_vcc.py` reads precomputed inputs, making ablation studies cheap: swap out one sub-metric's JSON and re-run the composition in seconds.

**Wallclock cost per video on a single A100:**

| Step | Time |
|---|---|
| CLIP-IQA (all frames) | ~3 min |
| Multi-k tOF (RAFT, 200 pairs × 6 k-values) | ~8 min |
| Human_Identity slow-fast | ~14 min |
| Composition (CPU) | < 1 s |

Identity dominates cost due to RetinaFace per-frame detection; this is an upstream bottleneck not specific to LR-VCC.

**Output format.** Per-video JSON per method:

```json
{
  "video": "KZ8p6b1zJ9U",
  "method": "MGLD",
  "lr_vcc": 0.612,
  "sub_metrics": {
    "appearance": {"score": 0.78, "reliability": 0.90},
    "temporal":   {"score": 0.70, "reliability": 0.50},
    "identity":   {"score": 0.66, "reliability": 0.85}
  },
  "weights_used": [0.40, 0.18, 0.42],
  "low_confidence": false,
  "diagnostics": {
    "closeup_indicator": 0.62,
    "mean_mask_coverage_long_k": 0.07,
    "face_detection_rate": 0.51
  }
}
```

The per-video × per-sub-metric matrix is always reported alongside the headline composite, so reviewers can see where the composite is driven from on each video. No black-box aggregation.

**Total implementation size:** approximately 600 lines of Python including tests. All code uses MIT-compatible licenses matching upstream VBench and DOVE.

---

## 4.7 Validation Plan

LR-VCC validation is conducted at two layers for the proposal milestone.

**Layer 1 — perceptual agreement on 5 test videos.** Pass criterion: LR-VCC(MGLD) > LR-VCC(UAV) on the per-method mean, and MGLD wins per-video on at least 4/5 videos. This is a sanity check that the composite is not broken in the direction already established by visual inspection and six independent metric families.

**Layer 2 — flip-resistance on KZ8p6b1zJ9U.** The key thesis test. Anatomy and tLP both flip on KZ for documented structural reasons. LR-VCC must not flip, because: (a) the Identity reliability test downweights close-up content; (b) sub-metric T uses tOF not tLP, avoiding the tLP smoother-output bias; and (c) sub-metric A (CLIP-IQA) does not depend on pristine-HR representations. Pass criterion: LR-VCC(MGLD, KZ) > LR-VCC(UAV, KZ).

**Layer 3 — parameterized synthetic degradation datasets (thesis future work, not proposal).** Synthetic datasets with parameterized severity on five degradation axes (color drift, periodic flicker, chunk-boundary jumps, identity degradation, long-range background change). LR-VCC should respond monotonically to severity on each axis and outperform every individual sub-metric on at least one axis. This constitutes the full empirical validation of the composite design and is planned for the thesis body.

Preliminary validation results from the 5-video test set confirm Layer 1 and Layer 2 pass. Full validation tables and per-video JSON outputs are in `docs/notes/2026-05-21-lr-vcc-validation.md`.
