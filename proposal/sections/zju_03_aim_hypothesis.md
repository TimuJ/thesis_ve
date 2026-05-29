# Section 3: Research Aim and Hypothesis

## 3.1 Research Aim

To develop and validate a **no-reference long-range video consistency metric for long-form video super-resolution** that:

1. **Catches the long-range failure modes that existing metrics are blind to** — slow color drift, chunk-boundary jumps from tile-based diffusion, periodic flicker from sampler artefacts, identity collapse on recurring faces.
2. **Preserves correct per-video method rankings on real super-resolved outputs** — including in the regime-shift case described in Section 2 where two existing VBench-2.0 metrics flip rankings in disagreement with human visual judgement.
3. **Is reproducible, open-source, and extensible** — single-CLI invocation, JSON sub-metric cache, sub-metric independence so new sub-metrics can be added without disturbing existing ones, full unit-test coverage.

The metric is **LR-VCC** (Long-Range Video Consistency Composite). Its specification, design rationale, and validation are given in Sections 4 and 5.

## 3.2 Research Hypotheses

The research is structured around three falsifiable hypotheses, each tested against the synthetic artefact set and the real SR test set described in Sections 2.4 and 4.

**H1 — Composition.** Reliability-weighted softmax-log-mean composition of independent sub-metrics catches failure modes that no single constituent sub-metric catches.

*Test.* For each artefact family (color_drift, chunk_boundary, flicker, identity_degradation), compute the per-video severity-response of each LR-VCC sub-metric individually and of the LR-VCC composite. Verify that the composite catches at least one artefact family that no single sub-metric catches across both base videos.

*Pass criterion.* Composite catches ≥1 artefact family whose detection requires sub-metric *combination* (e.g., color_drift requires sub-metric E whose reliability gate is informed by R²; chunk_boundary requires both sub-metric T at long-k and sub-metric D for cross-validation).

**H2 — Multi-scale temporal.** Multi-scale temporal aggregation (tOF computed at k ∈ {1, 5, 10, 30, 60, 120} frame gaps) catches both high-frequency and long-range temporal failure modes that single-scale temporal metrics miss.

*Test.* Compare tOF at fixed k = 1 (TecoGAN convention) against the LR-VCC multi-k aggregation under different `--temporal_weight` modes (log, uniform, sqrt) on flicker (high-frequency, period 15 frames) and chunk_boundary (mid-range, chunk size 60 frames) artefacts.

*Pass criterion.* Multi-k aggregation with uniform weighting catches both artefacts at the sub-metric T level (note: the *composite* response to flicker is a separate question, documented in Section 5.2 as a known limitation, because the composition layer weights are determined by reliabilities rather than score magnitudes). Single-k metrics catch only one artefact each.

**H3 — Reliability gating against content-dependent flips.** Per-sub-metric reliability gates (face_rate, closeup_p50, R² floor, entropy floor) reduce content-dependent ranking flips of the type observed in the regime-shift case study (Section 2).

*Test.* On the real SR test videos, including the regime-shift video, verify that the LR-VCC composite produces a consistent MGLD > UAV ranking. Compare against (a) raw Human_Identity slow-fast scores (which flip on the regime-shift video at +0.094 favoring UAV) and (b) raw Human_Anatomy whole-video scores (which flip at +0.291 favoring UAV).

*Pass criterion.* LR-VCC ranks MGLD > UAV on every real SR test video including the regime-shift one, with the per-video gap Δ(LR-VCC) > 0 on every video. Mean Δ across all test videos > +0.04 (well above the noise floor of the constituent CLIP-IQA, tOF, and Identity sub-metric variability).

## 3.3 Success Criteria

The research objectives are met if the following measurable outcomes are achieved over the proposal-to-thesis horizon:

| Criterion | Target | Status (current) |
|---|---|---|
| LR-VCC catches synthetic artefact families monotonically | Majority of (artefact × base-video) conditions | Most conditions PASS; remaining have documented failure modes |
| LR-VCC preserves MGLD wins on real SR test set | All videos including the regime-shift one | All confirmed under v3+slope β=200 |
| Mean Δ(MGLD − UAV) on real SR | ≥ +0.04 | +0.056 confirmed |
| Unit-test coverage | All sub-metrics covered by canonical four-case tests (clean / target artefact / distractor artefact / too-few-frames) | LR-VCC + generator test suites both green |
| Single-CLI reproducibility | All hyperparameters surfaced as CLI flags; JSON cache enables re-derivation without re-scanning videos | `--temporal_weight`, `--color_hist_alpha`, `--color_slope_beta` all implemented |
| Open-source release | Code + synthetic test set + reproducible CLI under MIT licence | Code in `scripts/lr_vcc/` and `scripts/synthetic_artefacts/`; release pending paper submission |

The qualitative criteria are:

- Each LR-VCC sub-metric is **reliability-gated by an interpretable signal**, not a black-box weighting. This is verified by inspecting the reliability gates in the codebase: A's drift/saturation penalty, T's mask-coverage floor, I's face-rate and closeup-p50 thresholds, D's entropy floor, E's R² floor.
- Each LR-VCC iteration (v1 → v3+slope β=200) **closed a specific failure mode characterised on the validation set**, not an unmotivated parameter tweak. The iteration history (Section 5.3) is itself a methodological contribution.
- The three documented LR-VCC failure modes (hhsz color_drift partial, both bases flicker flat, 7WHI identity_degradation inverted) each have an **identified root mechanism**, not "the metric just doesn't work here." Each mechanism becomes a concrete future-work direction (Section 6.3).
