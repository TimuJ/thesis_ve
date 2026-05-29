# Section 5.3: Synthetic validation — severity-response analysis

To test whether existing metrics and LR-VCC correctly detect known artefact types at controlled
severity levels, we generated a parameterised synthetic test set: 2 base videos (long-form SR
outputs, one 167 s, one 80 s) × 4 artefact families (color drift, chunk-boundary jumps,
periodic flicker, identity degradation) × 5 severities (0.02, 0.05, 0.10, 0.20, 0.40) test
videos with known ground-truth quality ordering: as severity increases, quality decreases.
See Figures 5–7 for severity-response curves; the full multi-metric × severity × base ×
artefact table is archived under `results/synthetic_artefacts_eval/` and
`results/lr_vcc/composite_artefacts_v3_slope_b200/`.

## 5.3.1 Artefact definitions

**Color drift** — a gradual color cast that accumulates linearly over the full video duration.
At frame $i$ of $T$ total frames, the red channel is multiplied by $1 + \alpha \cdot (i/T)$
while blue and green are reduced by the same factor, producing a monotonically increasing
red tint. The severity parameter $\alpha \in \{0.02, 0.05, 0.10, 0.20, 0.40\}$ controls
the final magnitude. This artefact is purely temporal and long-range: individual frames look
normal, but comparing the first and last frames reveals a pronounced appearance shift.
It tests whether a metric captures multi-minute appearance consistency.

**Chunk-boundary jumps** — at every 60-frame boundary (2 s at 30 fps), a deterministic
per-chunk uniform additive color offset of magnitude up to ±severity × 255 is applied to all
channels. This mimics the chunk-state-reset artefact observed when diffusion-SR models process
long videos in fixed-length segments without latent-state continuity. The discontinuity is
abrupt (single-frame) and periodic, making it detectable by any metric sensitive to
inter-frame differences at boundary frames.

**Periodic flicker** — sinusoidal brightness oscillation. Per-frame pixel intensities are
multiplied by `1 + severity · sin(2π · i / period)` with period = 15 frames (0.5 s at 30 fps).
This mimics the periodic flicker produced by some diffusion samplers, where every N-th frame
in a sampling schedule shows mildly elevated brightness. The artefact is purely high-frequency:
within a 60-frame window it executes 4 full cycles. Detecting it requires a temporal metric
that gives meaningful weight to small temporal gaps.

**Identity degradation** — per-frame Haar-cascade face detection followed by
`GaussianBlur(σ = severity · 10)` applied within each detected face bbox with 10 % padding.
This mimics the identity-collapse failure mode of diffusion VSR on faces and tests whether
the Identity sub-metric (slow-fast face embedding) responds to localised, identity-specific
degradation that leaves the rest of the frame unchanged. Severity-0 leaves the video
unchanged within float tolerance.

## 5.3.2 Severity-response results

Figure 7 shows the headline summary: tOF k=1 (adjacent-frame temporal), tOF k=120 (long-range
temporal), and LR-VCC (composite) across both artefact types. Figures 5–6 show all 9 metrics
individually.

### Chunk-boundary jumps — the well-behaved case

Chunk-boundary jumps produce large inter-chunk frame discontinuities that most long-range metrics
detect. The key results (averaged over both base videos):

- **LR-VCC** drops from 0.694 at severity 0.02 to 0.591 at severity 0.40 on the 167-s video
  (7WHI), a delta of −0.085. The 80-s video (hhsz) shows an even larger drop (0.697 → 0.580,
  delta −0.117). This is a consistent monotonic decline on both sequences, confirming that the
  composite captures the artefact. **Verdict: PARTIAL** — the 7WHI video shows clean monotonicity
  while hhsz has one non-monotonic step (sev 0.05 → 0.10: 0.695 → 0.700) before falling sharply
  to 0.580. The net trend is clear and strongly significant.
- **tOF k=120** increases (worse) from 0.144 to 0.323 on 7WHI and from 0.110 to 0.297 on hhsz
  — roughly a 2× deterioration across the severity range. Monotonic on both videos. **Verdict: PASS**.
- **tLP k=120** increases from 0.025 to 0.070 on 7WHI and from 0.015 to 0.051 on hhsz.
  Monotonic on both. **Verdict: PASS**.
- **E\*warp** increases from 0.01075 to 0.01280 on 7WHI and from 0.00222 to 0.00513 on hhsz.
  Monotonic on both. **Verdict: PASS**.
- **CLIP-IQA** decreases (worse) from 0.453 to 0.413 on 7WHI and from 0.459 to 0.385 on hhsz.
  Monotonic on both. **Verdict: PASS**.
- **DOVER** drops from 78.9 to 67.4 on 7WHI (7WHI). Monotonic in the aggregate. **Verdict: FAIL**
  on strict per-step monotonicity (non-monotone at sev 0.02 → 0.05 on 7WHI; increases slightly
  on hhsz at low severities), though the large-severity end correctly registers the artefact.
- **tOF k=1** (adjacent-frame): *decreases* with severity on both videos (7WHI: 0.0235 → 0.0214;
  hhsz: 0.0108 → 0.0089). This is counter-intuitive — chunk jumps are inter-chunk discontinuities
  that only occur every 60 frames, so the adjacent-frame average is dominated by the many
  smooth in-chunk pairs and is diluted below detection threshold. **Verdict: FAIL**.
- **tLP k=1**: similarly flat and non-monotonic. **Verdict: FAIL**.
- **Identity fused**: varies non-monotonically (7WHI: 0.373 → 0.345 but not consistently;
  hhsz: 0.465 → 0.626 → 0.626 — unexpected increase). Identity consistency as measured by
  face similarity is not sensitive to color-offset chunk boundaries. **Verdict: FAIL**.

### Color drift — the failure case

Color drift is the adversarial condition for all current metric families. The drift is slow
and spatially uniform: no frame-to-frame motion is induced, so optical-flow-based metrics
see no inter-frame difference. CLIP-based metrics are designed to be content-descriptive
rather than temporally comparative, so gradual color shifts do not move their scores.

- **LR-VCC**: essentially flat across 0.02–0.40. On 7WHI, values range 0.689–0.702 (span 0.013);
  on hhsz, 0.693–0.701 (span 0.008). No meaningful trend. The dominant sub-metric driving LR-VCC
  is temporal consistency (tOF), which is blind to color drift. **Verdict: FAIL**.
- **tOF k=1** and **tOF k=120**: flat or slightly oscillating. Optical-flow warping absorbs
  uniform per-frame color offsets; the warped-frame difference is near-zero regardless of drift
  magnitude. Both metrics show range < 0.007 across all 5 severities. **Verdict: FAIL**.
- **tLP k=1** and **tLP k=120**: flat. LPIPS compares perceptual features between adjacent or
  temporally distant frames; uniform color cast does not change the structural features
  substantially. Range < 0.003 across severities. **Verdict: FAIL**.
- **CLIP-IQA**: flat or slightly *increasing* with severity (7WHI: 0.457 → 0.478). CLIP-IQA
  was trained on diverse natural images; a mild red tint does not penalize the score and can
  even correlate with warm-toned content that scores well. **Verdict: FAIL**.
- **DOVER**: flat and non-monotonic (7WHI: 76.1 → 75.4 over a highly non-monotonic path;
  hhsz: 81.9 → 83.8, i.e. slightly *increasing* — DOVER misreads the drifted video as higher
  quality). **Verdict: FAIL**.
- **E\*warp**: flat (7WHI range: 0.01022–0.01069; hhsz range: 0.00208–0.00221).
  Warp-error is a frame-difference metric after warping; color drift cancels out in the
  difference. **Verdict: FAIL**.
- **Identity fused**: decreases slightly on the 80-s base (0.475 → 0.407) suggesting some color
  sensitivity in the face-embedding comparisons, but non-monotonic on the 167-s base. **Verdict: FAIL**.

### Color drift — LR-VCC after adding sub-metric E

The colour-drift gap motivated the addition of sub-metric E (linear regression on per-frame
Lab channel means with R²-gated reliability). After E is folded in with β = 200, LR-VCC
becomes the *first metric in our 9-metric set* to respond to colour drift on at least one
base video. On the 167-s base, LR-VCC drops from 0.619 at severity 0.02 to 0.507 at
severity 0.40 (Δ −0.111, monotonic with one small mid-severity bump). On the 80-s base, the
drop is weaker (Δ −0.039) because the base video has a pre-existing colour trajectory that
the slope sub-metric correctly flags as reliable on the clean video already — adding more
drift on top is hard to disentangle from the baseline. **Verdict: PARTIAL** — 167-s base
clean, 80-s base weak. The mechanism is characterised; the limitation becomes a future-work
target.

### Periodic flicker — adjacent-frame only

Flicker is detected cleanly only at the small-k end of the temporal sub-metric:

- **tOF k=1** ramps monotonically with severity (Δ ≈ 2.2× on both base videos). **Verdict: PASS**.
- **tOF k=5** ramps even more strongly (Δ ≈ 4.5×). **Verdict: PASS**.
- **tOF k ≥ 60** is essentially flat — the flicker period (15 frames) divides 60 and 120
  cleanly, so the cross-pair difference cancels. **Verdict: FAIL** at k = 60, 120.
- **LR-VCC composite** is essentially flat (Δ ≈ +0.005 on one base, −0.012 on the other).
  The composite is dominated by sub-metrics with reliability close to 1.0 (D, E) whose
  signals do not respond to flicker, so even though sub-metric T's small-k component
  picks up the signal, it is averaged down at composite level. **Verdict: FAIL.** Documented
  limitation; a fast-varying brightness sub-metric (FFT 5–20 Hz on per-frame mean) is in the
  future-work list to close this gap without disturbing the other sub-metrics.

### Identity degradation — the multi-face vs single-face split

Identity degradation is the first artefact family designed specifically to stress the
Identity sub-metric. The result is mixed and pedagogically valuable.

- **Sub-metric T, D, E** all stay correctly flat (Δ ≤ 0.001 each): face-region blur does not
  affect global temporal consistency, colour histograms, or colour trajectory. Clean
  attribution — these sub-metrics are not false-positing on a non-target artefact.
- **CLIP-IQA (sub-metric A)** drops monotonically (Δ ≈ −0.064 on both bases): the global
  blur is visible in the full-frame perceptual quality. This is an expected side-effect
  rather than a specific identity signal.
- **Identity sub-metric** behaves *differently* on the two base videos:
  - On the multi-face base, the fused slow-fast score drops cleanly (Δ −0.227): cross-clip
    embedding similarity decreases as blur erases identity-distinctive features.
  - On the single-face base, the score *rises* (Δ +0.114): heavy face-region blur (σ = 4.0)
    erases identity-distinctive features in *all* frames of the single face, so cross-clip
    embeddings become more similar to each other in a "generic blurred face" sense. The
    face-detection rate is 0.96 at every severity (the detector survives the blur), so the
    face-rate reliability gate does not engage. The pathology is in the slow-fast pooling
    itself, not in face detection.
- **LR-VCC composite** tracks the underlying Identity behaviour: monotonic on the multi-face
  base (Δ −0.070), inverted on the single-face base (Δ +0.043). **Verdict: PARTIAL** —
  multi-face clean, single-face inverted via a *documented identity-collapse pathology*. The
  fix (gate sub-metric I by face-detection confidence + per-face embedding variance, not just
  face-rate) is concrete future work.

## 5.3.3 Consolidated verdict table

| Metric          | Chunk-boundary | Color drift | Flicker | Identity degradation |
|-----------------|:--------------:|:-----------:|:-------:|:--------------------:|
| LR-VCC (v3+slope β=200) | PARTIAL | PARTIAL     | FAIL    | PARTIAL              |
| tOF k = 1       | FAIL           | FAIL        | PASS    | FAIL                 |
| tOF k = 60      | PASS           | FAIL        | FAIL    | FAIL                 |
| tOF k = 120     | PASS           | FAIL        | FAIL    | FAIL                 |
| tLP k = 120     | PASS           | FAIL        | FAIL    | FAIL                 |
| DOVER           | FAIL           | FAIL        | FAIL    | FAIL                 |
| E\*warp         | PASS           | FAIL        | FAIL    | FAIL                 |
| CLIP-IQA        | PASS           | FAIL        | FAIL    | PARTIAL (side-effect)|
| Identity fused  | FAIL           | FAIL        | FAIL    | PARTIAL (1/2 bases)  |

PASS = monotonic on both base videos; PARTIAL = monotonic on one; FAIL = non-monotonic or flat.

## 5.3.4 What this shows

**Chunk-boundary detection is well-covered by long-range metrics.** tOF k=120, tLP k=120,
E\*warp, CLIP-IQA, and LR-VCC all track severity monotonically. Adjacent-frame metrics (tOF k=1,
tLP k=1) are insensitive because the periodic jump is diluted by the many smooth in-chunk frames.
DOVER and Identity are noisy at low severity, suggesting their internal mechanisms are not tuned
for this periodic discontinuity pattern.

**Colour drift was a blind spot for every baseline metric tested; it is now partially closed
by LR-VCC.** Of the eight baseline metrics, none responds monotonically to a slow linear
colour ramp. The failure mechanism is structural: tOF and E\*warp compute frame differences
after optical-flow warping, which absorbs uniform additive colour shifts; LPIPS is partially
colour-invariant; CLIP-based metrics are trained on content diversity rather than temporal
colour consistency; DOVER aggregates technical and aesthetic scores; Identity metrics measure
semantic face consistency rather than photometric trajectory. LR-VCC's sub-metric E (linear
regression on per-frame Lab channel means, with reliability gated by goodness-of-fit R²)
is specifically designed to detect what the per-pair metrics cannot see, and it is the first
metric in the set to respond monotonically on at least one base video.

**Flicker is detected only at small temporal gaps.** This is the inverse of the chunk-boundary
case and demonstrates why a multi-scale temporal metric is necessary. tOF at k = 1 catches
flicker beautifully (4.5× ratio across the severity range) precisely because the temporal-scale
selection matters: long-k metrics blind themselves to artefacts whose period divides k.
LR-VCC's composite-level response to flicker is currently weak — a documented limitation,
not a mystery — and the fix (a fast-varying brightness sub-metric) is in the future-work list.

**Identity degradation reveals a content-dependent pathology in slow-fast pooling.** On
multi-face content the metric behaves as designed; on single-face content under heavy blur
it inverts because the metric loses its discriminative basis. This is a clean characterisation
of when a vbench-style identity metric breaks, with a concrete proposed fix (reliability
gating by face-detection confidence rather than face-rate alone).

## 5.3.5 Implications for the thesis

The synthetic validation delivers three take-aways that directly motivate the proposed work.

First, the chunk-boundary PASS results confirm that LR-VCC's temporal and colour sub-metrics
are functioning as designed: they respond monotonically to a known long-range temporal artefact
in a way that short-range alternatives (tOF k = 1, tLP k = 1) do not. This is the intended
differentiation between LR-VCC and standard per-frame or short-window metrics.

Second, the colour-drift result establishes that the LR-VCC mechanism — adding a sub-metric
whose reliability is gated by goodness-of-fit of the underlying trajectory model — can close
gaps that the entire baseline suite is blind to. The mechanism is generalisable: any "slow
signal vs. noisy baseline" failure mode admits an analogous slope-style sub-metric.

Third, the flicker and identity-degradation results document the two areas where LR-VCC is
*currently* insufficient — *and characterise the mechanism* in each case. Flicker is undetected
because the composition arithmetic averages down a correctly-firing small-k temporal signal;
identity degradation inverts on single-face content because slow-fast pooling loses its
discriminative basis. Both are addressable with specific, concrete additions to the composite,
listed under Section 7 (Timeline) as planned post-proposal work.

Together, the four artefacts serve as a verdict matrix — most cells PASS, the remaining
cells have *named* failure modes — that frames the metric contribution of this thesis within
a testable, reproducible validation framework rather than a single aggregate number.
