# Section 7: Timeline

## 7.1 Completed Work

The research has been underway since early April 2026. The completed work is summarised below; full per-week detail is in the weekly reports under `reports/`.

| Period | Milestone | Status |
|---|---|---|
| Apr 5 – 15 | Baseline reproduction. One detail-preserving SR baseline matches the DOVE paper UDM10 numbers exactly (PSNR / SSIM / LPIPS / DISTS within 4 decimal places); one text-conditioned smoother baseline reproduces with a documented +1.3 to +1.7 dB PSNR offset attributable to degradation pipeline configuration, not setup error. | Done |
| Apr 16 – 27 | DOVE benchmark alignment; VBench-2.0 long-video slow-fast adapter for Human_Anatomy and Human_Identity; FPS-mismatch bug discovered and patched (the pipelines tagged the wrong fps regardless of source). | Done |
| Apr 28 – May 7 | Per-frame failure-mode diagnostics on the regime-shift video: root cause traced to close-up hand-bbox fraction triggering the Anatomy anomaly classifier high-fire regime. Multi-person Identity metric scoped (designed but deferred). | Done |
| May 8 – 14 | Long-range tOF and tLP at k ∈ {1, 5, 10, 30, 60, 120} infrastructure built; k-crossover finding documented; smoother-output bias on adjacent-frame metrics confirmed. | Done |
| May 15 – 21 | LR-VCC v1 (3 sub-metrics: A, T, I) designed, implemented, validated on the real SR test set (winner ranking preserved including the regime-shift case). Initial proposal sections drafted. | Done |
| May 22 – 28 | LR-VCC sub-metrics D and E added (colour histogram, colour slope); synthetic-artefact test set built (four families × multiple severities × two base videos); production CLI settings derived empirically (`--temporal_weight uniform --color_hist_alpha 0.394 --color_slope_beta 200`); identity_degradation pathology characterised. | Done |
| **May 29 – 31** | **Proposal draft to classmate and supervisor for review; final submission.** | **In progress** |

## 7.2 Upcoming Work

Detailed plans for each of the following items live in `docs/plans/`. Specific durations are estimates pending paper-submission scheduling.

| Period | Milestone | Deliverable |
|---|---|---|
| June (weeks 1–2) | Multi-person Identity sub-metric implementation, per the design in `docs/plans/2026-05-06-multiperson-identity-metric.md`. Per-clip cluster purity (self-consistency) + LQ-reference IoU-matched-pair variant. | Sub-metric I_multi addendum, validated against the multi-face base video in the synthetic test set |
| June (weeks 3–4) | Flicker improvement. Option A: add a high-frequency brightness sub-metric (FFT magnitude in the 5–20 Hz band of per-frame mean). Option B: rebalance `--temporal_weight` with a content-adaptive variant. Choice to be made after option A is sketched. | LR-VCC v4 candidate; new sub-metric closing the flicker gap on the existing test set |
| July (weeks 1–2) | Real-video baseline confirmation: source ≥ 10 real high-resolution clips (no SR, no artefacts) and confirm LR-VCC scores ≥ 0.7 (no false-positive drift detection on natural content). | Sanity-check report; if false positives surface, baseline-correction pass added to LR-VCC |
| Mid-July | **Thesis blind-review submission deadline.** Assemble + proofread blind-review draft including the proposal content plus the new work above. | Thesis blind-review PDF |
| Late July – August | Extended validation: re-run the metric on a broader set of recent SR methods beyond the proposal's two baselines so the empirical claims generalise. Address blind-review feedback in parallel. | Extended-validation chapter; reviewer-response notes |
| September | Thesis final revisions; incorporate reviewer feedback; final figures and tables reflecting the extended validation. | Thesis final draft |
| Late September | **Final thesis submission.** | Thesis submitted |

## 7.3 Risk Management

**Path A (primary).** Thesis receives all A / B evaluations from blind reviewers; defence; MS conferred. This is the planned outcome; the work to date supports this trajectory.

**Path B (backup, per supervisor instruction).** If blind reviews are mixed, the **invention patent** on the reliability-weighted composition mechanism enters substantive examination. Under the ZJU alternate policy, a patent in substantive examination satisfies the graduation requirement without dependence on the blind-review verdict. This is a contingency; the primary effort goes to Path A.

**Key technical risks and mitigations:**

- *Multi-person Identity sub-metric blocks longer than estimated.* Mitigation: cut from thesis core, mention only in future-work. The proposal's hypotheses (H1, H2, H3) do not require multi-person Identity to be tested; they are satisfied by the current single-identity slow-fast adapter.
- *Real-video baseline confirmation surfaces false positives on natural content.* Mitigation: add a baseline-correction pass to the affected sub-metric (most likely E, where natural slow camera-pan can produce a Lab-channel-mean trend); document the correction in an extended-thesis chapter. The synthetic-test-set validation does not require the real-video baseline to be perfect — it requires only that the metric correctly orders severities of injected artefacts, which it already does on most cells.
- *Validation generalisation to additional SR methods.* The proposal's empirical claims rest on two SR baselines. The thesis-track work explicitly extends validation to a broader set of recent SR methods so that the per-video method rankings and the verdict matrix generalise. Mitigation: if any specific failure mode does not reproduce on the expanded set, that absence becomes a documented characterisation in the thesis rather than a contradiction of the proposal.
