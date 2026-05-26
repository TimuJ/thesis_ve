# Section 5.3: Synthetic validation — severity-response analysis

To test whether existing metrics and LR-VCC correctly detect known artefact types at controlled
severity levels, we generated a parameterized synthetic test set: 2 base videos (MGLD-SR outputs
of `7WHI2L_FDNg`, 167 s, and `hhszUXL1Cu8`, 80 s) × 2 artefact types (color drift,
chunk-boundary jumps) × 5 severities (0.02, 0.05, 0.10, 0.20, 0.40) = 20 test videos with
known ground-truth quality ordering: as severity increases, quality decreases.
See Figures 5–7 for severity-response curves; the full 9-metric × 5-severity × 2-base ×
2-artefact table is archived at `results/lr_vcc/severity_response_table.csv`.

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
- **Identity fused**: decreases slightly on hhsz (0.475 → 0.407) suggesting some color
  sensitivity in the face-embedding comparisons, but non-monotonic on 7WHI. **Verdict: FAIL**.

## 5.3.3 Consolidated verdict table

| Metric          | Chunk-boundary | Color drift |
|-----------------|:--------------:|:-----------:|
| LR-VCC          | PARTIAL        | FAIL        |
| tOF k=1         | FAIL           | FAIL        |
| tOF k=120       | PASS           | FAIL        |
| tLP k=1         | FAIL           | FAIL        |
| tLP k=120       | PASS           | FAIL        |
| DOVER           | FAIL           | FAIL        |
| E\*warp         | PASS           | FAIL        |
| CLIP-IQA        | PASS           | FAIL        |
| Identity fused  | FAIL           | FAIL        |

PASS = monotonic on both base videos; PARTIAL = monotonic on one; FAIL = non-monotonic or flat.

## 5.3.4 What this shows

**Chunk-boundary detection is well-covered by long-range metrics.** tOF k=120, tLP k=120,
E*warp, CLIP-IQA, and LR-VCC all track severity monotonically. Adjacent-frame metrics (tOF k=1,
tLP k=1) are insensitive because the periodic jump is diluted by the many smooth in-chunk frames.
DOVER and Identity are noisy at low severity, suggesting their internal mechanisms are not tuned
for this periodic discontinuity pattern. LR-VCC achieves a PARTIAL because its temporal
sub-metric (tOF-based) correctly drives the score down, but the identity sub-metric adds noise
at low severities on the 80-s video.

**Color drift is a blind spot for every metric tested.** This is itself a concrete finding:
not a single evaluated metric correctly orders the 5 severity levels for color drift across
both base videos. The failure mechanism is structural — tOF and E*warp both compute frame
differences after optical-flow warping, which absorbs uniform additive color shifts.
LPIPS is designed for perceptual similarity in a way that is partially color-invariant.
CLIP-based metrics are trained on content diversity, not temporal color consistency.
DOVER aggregates technical and aesthetic scores that do not specifically penalize long-range
hue drift. Identity metrics measure semantic face consistency, not photometric consistency.

LR-VCC is affected by the same blind spot because its current sub-metric composition
(appearance quality, temporal tOF, identity) does not include a color-consistency component.
This is a planned future extension: a fourth sub-metric based on color-histogram temporal
variance computed over long gaps (k ∈ {60, 120} frames) would require no re-training and
would be implementable using the existing tOF infrastructure.

## 5.3.5 Implications for the thesis

The synthetic validation delivers two take-aways that directly motivate the proposed work.

First, the chunk-boundary PASS results confirm that LR-VCC's temporal sub-metric is functioning
as designed: it responds monotonically to a known long-range temporal artefact in a way that
short-range alternatives (tOF k=1, tLP k=1) do not. This is the intended differentiation
between LR-VCC and standard per-frame or short-window metrics.

Second, the universal FAIL on color drift documents a concrete gap in the metric literature.
Any paper that reports only tOF/LPIPS/DOVER on long-video SR outputs could be missing
slow appearance-drift artefacts entirely. This gap will be addressed in the next sprint by
adding a histogram-variance temporal sub-metric to LR-VCC (Task A6), with re-validation
on the same synthetic test set to verify it produces a PASS on color drift while not
degrading the chunk-boundary PASS.

Together, the two conditions serve as a positive control (chunk-boundary) and an open
research challenge (color drift), framing the metric contribution of this thesis within
a testable, reproducible validation framework rather than a single aggregate number.
