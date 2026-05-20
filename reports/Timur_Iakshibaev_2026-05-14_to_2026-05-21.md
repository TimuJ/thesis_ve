# Weekly Progress Report — Timur Iakshibaev

## Period: May 14 – May 21, 2026

## Headline

This week closed out the LR-VCC implementation sprint and produced the two most important results for the May 31 proposal:

1. **Long-range tOF/tLP eval finished — clean crossover finding.** At adjacent frames (k=1) UAV wins on 4/5 videos (smoothness). At k≥10 MGLD wins on 4/5 videos. The standard adjacent-frame temporal metric (`E*warp`, `tOF k=1`) systematically favours smoother SR — long-range stability flips the ordering. This is the strongest single empirical argument that long-video SR needs multi-time-scale metrics.

2. **LR-VCC implemented, validated (Layer 1+2 both pass), and written into the proposal.** The composite metric correctly orders MGLD > UAV on all 5/5 test videos including KZ8p6b1zJ9U (Δ = +0.2317), where three individual metrics flip the ordering. Two of the three Phase 3 proposal writing tasks are done: Preliminary Work (Section 3) and Proposed Method (Section 4) with architecture diagram.

3. **Continuous-aggregation Anatomy implemented.** Halves the KZ8p6b1zJ9U anatomy flip-gap (−0.339 → −0.148) but doesn't eliminate it, confirming the failure has both a threshold-boundary half and a deeper representation half. Committed as opt-in `--continuous` flag.

---

## tOF/tLP Long-Range Eval — Crossover and Smoother-Output Bias

Built `scripts/long_range_temporal/eval_tof_tlp.py` (RAFT forward+backward flow, FB-consistency occlusion mask, masked L2 for tOF and masked LPIPS for tLP). Ran on all 5 synthetic SR videos × 2 methods × 6 `k`-values × 200 pairs per `k`. ~37 min on GPU 4.

### tOF (pixel-level)

| k | Mean MGLD | Mean UAV | Δ (M − U) |
|---|----------:|---------:|----------:|
| 1 | 0.0216 | 0.0177 | **+0.0039 (UAV wins)** |
| 5 | 0.0406 | 0.0424 | −0.0018 |
| 10 | 0.0500 | 0.0618 | −0.0118 |
| 30 | 0.0804 | 0.0922 | −0.0119 |
| 60 | 0.1110 | 0.1314 | −0.0204 |
| 120 | 0.1441 | 0.1682 | **−0.0241 (MGLD wins)** |

Clean crossover at k=5–10. UAV wins adjacent-frame stability (smooth frames warp into themselves better); MGLD wins long-range stability decisively. Per-video at k=120: MGLD wins 4/5.

### tLP (LPIPS-based perceptual)

tLP systematically favours UAV across all k. LPIPS rewards self-similarity → UAV's smoother frames look more like themselves under LPIPS feature distance, even though humans see UAV's output as worse. **Same smoother-output bias as VBench's `subject_consistency` (DINOv2) and `Human_Anatomy` (anomaly ViT) on close-ups** — that's three completely independent learned-representation metrics all exhibiting the same bias. Strong structural evidence for the thesis's central claim.

### Mask coverage collapses at long k

| k | MGLD coverage | UAV coverage |
|---|--------------:|-------------:|
| 1 | 0.93 | 0.91 |
| 10 | 0.57 | 0.35 |
| 60 | 0.22 | 0.12 |
| 120 | 0.11 | 0.07 |

At k=120 only ~10% of the frame is valid. UAV's coverage is consistently lower than MGLD's at long k — RAFT struggles to find FB-consistent flow on UAV's smoother textures. Long-range numbers compare differently-sized pixel subsets per method; flagged in the writeup.

Full doc: `docs/notes/2026-05-14-tof-tlp-long-range-results.md`.

---

## Continuous-Aggregation Anatomy

Added `--continuous` opt-in flag to `aggregate_slow_fast_anatomy.py` and `human_anatomy_long.py`. Default stays upstream's `1 − fraction_above_threshold` for paper reproducibility; `--continuous` switches to `1 − mean(p_abnormal)` averaged per detector category. Implemented in 5 minutes, validated on all 10 cached per-frame traces (no GPU re-run).

Slow-fast comparison across all 5 videos:

| Video | MGLD threshold | MGLD continuous | UAV threshold | UAV continuous | Δ threshold | Δ continuous |
|-------|---------------:|----------------:|--------------:|---------------:|------------:|-------------:|
| 7WHI2L_FDNg | **0.840** | **0.887** | 0.774 | 0.828 | +0.066 | +0.059 |
| BrRLKMbBTYQ | **0.472** | **0.770** | 0.410 | 0.745 | +0.062 | +0.025 |
| **KZ8p6b1zJ9U** | 0.137 | 0.591 | **0.476** | **0.739** | −0.339 | **−0.148 (halved)** |
| hhszUXL1Cu8 | **0.969** | **0.950** | 0.896 | 0.921 | +0.073 | +0.029 |
| mJog8DlRk_4 | **0.622** | **0.832** | 0.531 | 0.803 | +0.091 | +0.029 |
| **Mean** | 0.608 | 0.806 | 0.618 | 0.807 | −0.010 | −0.001 |

**Per-video winners unchanged (MGLD 4/5)** under both schemes. **KZ gap halves** from −0.339 to −0.148 but doesn't disappear. Trade-off: continuous lifts everyone's absolute scores and compresses inter-method gaps on the 4 MGLD-wins videos; recommendation is to report both side-by-side in the thesis.

Full numbers in `docs/notes/2026-05-13-kz-regime-shift-trigger.md`.

---

## KZ8p6b1zJ9U Regime-Shift — Content Trigger Characterised

Bbox-size analysis on cached per-frame anatomy traces (CPU-only):

| Video | Hand bbox p50 (% of frame) | MGLD−UAV anatomy gap |
|-------|---------------------------:|---------------------:|
| **KZ8p6b1zJ9U** | **18%** | **−0.291** |
| BrRLKMbBTYQ | 7% | +0.085 |
| mJog8DlRk_4 | 3% | +0.036 |
| hhszUXL1Cu8 | 1% | +0.047 |
| 7WHI2L_FDNg | 0.5% | +0.097 |

Monotonic across videos. Two natural fixes tested; neither fully rescues KZ:

1. **Close-up frame filtering** (drop frames where face/hand bbox ≥ 5%): KZ gap *widens* from −0.339 → −0.437. The high-fire frames are distributed across the whole video, not concentrated in the close-ups.
2. **Continuous aggregation**: halves KZ gap to −0.148 but doesn't eliminate it.

The failure has two halves: ~50% threshold-near-boundary noise (continuous aggregation fixes), ~50% deeper representation bias (genuine distributional shift of the anomaly classifier at close-up scale — not fixable without classifier retraining). Both halves documented and characterised; the deeper half is flagged as thesis future work.

Full doc: `docs/notes/2026-05-13-kz-regime-shift-trigger.md`.

---

## LR-VCC — Implementation Sprint Complete; Layer 1+2 Validation Passes

This is the main deliverable of the week and the centrepiece of the May 31 proposal. All Phase 1 (implementation) and Phase 2 (validation) tasks are done.

### Design

A no-reference composite metric for ranking long-video SR methods. Three sub-metrics with per-video reliability tests, combined via softmax-weighted log-mean:

- **Sub-metric A — Appearance stability.** Per-frame CLIP-IQA, `score = mean(quality) − 0.5·std(quality)`. Reliability drops if std too small (undiscriminating) or mean saturates at 0.98.
- **Sub-metric T — Temporal stability.** `log(1+k)`-weighted mean of tOF over k ∈ {1, 5, 10, 30, 60, 120}. Reliability drops if RAFT mask coverage too low at long k.
- **Sub-metric I — Identity preservation.** Wraps `human_identity_long.py` fused slow-fast. Reliability drops if face-detection rate < 0.20 or close-up bbox p50 > 5% of frame area.

Composition: `LR_VCC = exp(Σ_s softmax(reliability/0.2)_s · log(score_s + ε))`. Sharp softmax (τ=0.2) lets the most reliable sub-metric dominate; log-mean preserves "no compensation for failures."

Full spec: `docs/plans/2026-05-21-lr-vcc-design.md`.

### Layer 1 + Layer 2 Validation

| Video | MGLD | UAV | Δ (M−U) |
|-------|-----:|----:|--------:|
| 7WHI2L_FDNg | 0.6894 | 0.4979 | +0.1915 |
| BrRLKMbBTYQ | 0.7235 | 0.7037 | +0.0198 |
| **KZ8p6b1zJ9U** | **0.6991** | **0.4674** | **+0.2317** |
| hhszUXL1Cu8 | 0.7334 | 0.6328 | +0.1006 |
| mJog8DlRk_4 | 0.6371 | 0.5255 | +0.1116 |
| **Mean** | **0.6965** | **0.5655** | **+0.131** |

**Layer 1 PASS** — MGLD wins 5/5 with +0.131 aggregate advantage.

**Layer 2 PASS** — on KZ8p6b1zJ9U (the metric-failure case where Anatomy, tLP, and tOF k=1 all flip), LR-VCC gives MGLD a +0.2317 win. Mechanism: the Identity sub-metric's reliability is crushed (close-up bbox 16%, face-rate penalty) and Temporal carries the composite at weight 0.58, where MGLD's long-range tOF is strong. For UAV on the same video, smooth textures collapse RAFT FB-consistency at long k, so Temporal reliability is only 0.554 — Appearance carries UAV's composite at weight 0.64, but UAV's Appearance score is 0.332 < MGLD's 0.481. **The per-method weight asymmetry on the same video is the key validation result — the design hypothesis empirically confirmed.**

Full results: `docs/notes/2026-05-21-lr-vcc-validation.md`.

### Code Delivered

| File | Status |
|------|--------|
| `scripts/lr_vcc/reliability.py` | Done |
| `scripts/lr_vcc/composite.py` | Done |
| `scripts/lr_vcc/temporal.py` | Done |
| `scripts/lr_vcc/identity.py` | Done |
| `scripts/lr_vcc/appearance.py` | Done |
| `scripts/lr_vcc/compute_clip_iqa.py` | Done |
| `scripts/lr_vcc/run_lr_vcc.py` | Done |
| `scripts/lr_vcc/build_closeup_map.py` | Done |
| `tests/lr_vcc/` | 20/20 tests passing |

Total: ~600 lines Python including tests.

---

## Proposal Writing — Sections 3 and 4 Landed

Two of the three remaining Phase 3 proposal writing tasks completed today:

**Task 11 — Preliminary Work (Section 3)** (`proposal/sections/preliminary_work.md`): ~2,200 words covering the test set setup, the seven-metric evaluation summary, the tOF/tLP crossover finding with full per-k table, the KZ8p6b1zJ9U regime shift characterisation, the FPS-mismatch methodological discovery, and a summary of the three motivating findings with the design-response mapping table. Three figures generated: `fig1_kz_vs_hhsz_abnormal_histograms.png`, `fig2_handbbox_vs_anatomy_gap.png`, `fig3_tof_per_k_curves.png`.

**Task 12 — Proposed Method (Section 4)** (`proposal/sections/proposed_method.md`): ~2,000 words (~1,500 prose + code blocks and tables) covering the full LR-VCC architecture (Sections 4.1–4.6): per sub-metric design rationale, reliability-weighting mechanism, implementation file layout, per-video JSON output format, wallclock costs, and two-layer validation plan. Architecture diagram generated: `proposal/figures/fig4_lr_vcc_architecture.png` (377 KB, 300 DPI, boxes-and-arrows matplotlib diagram showing input → three parallel sub-metrics with reliability tests → composition → output, dominant temporal path highlighted).

---

## Commit Log (this week)

```
76764a3 proposal: proposed method (LR-VCC) section + architecture diagram
c294fd4 proposal: preliminary work section + 3 figures
d5f1653 docs: LR-VCC Layer 1+2 validation note — both pass, +0.131 mean, +0.232 on KZ
571f3d0 results: LR-VCC composite on 5 synthetic videos x 2 methods (Layer 1+2)
6eddf10 reports: weekly May 14-21
dcc239a proposal: outline (sections 1-7)
9f7ab7e lr_vcc: close-up bbox-p50 map builder + cached maps for 5 videos x 2 methods
a6811e6 lr_vcc: CLI runner that composes all three sub-metrics
7c60ae9 lr_vcc: sub-metric A (CLIP-IQA per-frame + appearance stability wrapper)
ccb5802 lr_vcc: sub-metric I (Identity slow-fast + face-rate + closeup reliability)
8803b95 lr_vcc: sub-metric T (long-k weighted tOF + coverage reliability)
79ef7d2 lr_vcc: softmax + log-mean composition
2ff68fa lr_vcc: project skeleton + sigmoid reliability helpers
ac03015 docs: LR-VCC 10-day implementation plan
ec760b4 docs: LR-VCC design spec
69a2111 docs: tOF/tLP long-range results — crossover at k=5-10
2c823b1 scripts+docs: --continuous Anatomy tweak + tOF/tLP long-range eval
```

---

## Proposal Status Heading into May 22

| Section | Status |
|---------|--------|
| 1 — Introduction | Outline done, prose TBD |
| 2 — Related Work | Outline done, prose TBD |
| 3 — Preliminary Work | **DONE** (2,200 words + 3 figures) |
| 4 — Proposed Method | **DONE** (2,000 words + architecture diagram) |
| 5 — Validation & Timeline | Outline done, prose TBD |
| 6 — Contributions | Outline done, prose TBD |
| LaTeX assembly | TBD |

Three full writing tasks remain for the last 9 days of the sprint (May 22–31).

---

## Next Steps (May 22 – May 31)

1. **Task 13 — Validation + Timeline section** (~800 words): Layer 1+2 results in proposal format, Layer 3 plan as future work, Gantt-style timeline to July 2026 thesis submission.
2. **Task 14 — Contributions + Related Work sections** (~600 words combined): 3–4 bullet contributions, related work situating LR-VCC vs DOVE, VBench 1.x/2.0, E*warp, NIQE/CLIP-IQA NR-IQA baselines.
3. **Task 15 — LaTeX assembly + internal review**: translate four section `.md` files to LaTeX, build PDF, self-review, send to supervisor before May 31.

Items punted to post-proposal (thesis future work):
- Multi-person Identity v2 implementation (designed only).
- LR-VCC Validation Layer 3 (parameterized synthetic test datasets at controlled artefact severity).
- Long-range temporal metric retraining / classifier replacement to address deeper smoother-output bias.

---

## Talking Points for the May 22 Meeting

1. **The crossover finding** — tOF flips between k=1 (UAV wins) and k≥10 (MGLD wins). Direct evidence that adjacent-frame temporal metrics undervalue long-range stability. Cleanest single result of the week.
2. **Three independent learned representations show the same bias** — DINOv2 (`subject_consistency`), Anatomy ViT (close-up regime), LPIPS (tLP). Structural, not coincidental — "trained on pristine HR" representations all share the bias.
3. **LR-VCC Layer 1+2 both pass** — MGLD wins 5/5 including KZ (Δ +0.2317). Per-method weight asymmetry on the same video is the empirical validation of the reliability-weighting design. This is the proposal's key result.
4. **Two of five proposal sections written** — Preliminary Work (Section 3) and Proposed Method (Section 4) are prose-complete. Three sections (Introduction, Related Work, Validation + Timeline) remain, plus LaTeX assembly, in 9 days.
5. **Continuous-aggregation Anatomy** — halves KZ flip-gap but confirms deeper failure persists; implemented as opt-in for diagnostic use.

## Blocking Questions for the Group

1. **LR-VCC reliability hyperparameters** — drift floor 0.02, mask-coverage floor 0.10, face-rate floor 0.20, close-up bbox threshold 0.05, softmax temperature 0.2. All derived from independent characterisations; none tuned to Layer 1+2 results. Any concerns before publishing them in the proposal?
2. **Validation Layer 3 scope** — 5 parameterized synthetic degradation axes (color drift, periodic flicker, chunk-boundary jumps, identity degradation, long-range BG change). Which 2–3 are highest priority for the full thesis?
3. **Real HR long-video baseline** — currently no-reference only on 5 synthetic videos. Worth sourcing real HR long videos for additional validation, or stay no-reference?
4. **Smoother-output bias — long-term fix** — LR-VCC takes reliability-weighting (cheap, principled, fits proposal scope). Retraining LPIPS / Anatomy ViT on diffusion-SR data would be the principled long-term fix. In-scope for the thesis or not?
5. **Time-scale granularity** — k ∈ {1, 5, 10, 30, 60, 120} for tOF. Is the log(1+k) weighting spectrum the right shape, or should the proposal argue for a single summary statistic?
