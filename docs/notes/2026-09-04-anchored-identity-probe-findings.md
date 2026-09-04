# Anchored identity (I′) — probe findings and verdict

Three probes over the dumped face embeddings, one frame inspection, and one
asset check. Net result: the I′ design survives, the "control failure" was a
battery-asset defect, and the selection-rule complication turned out to be
unnecessary.

## Probe v1/v2 — design variants (offline, same-era)

Twelve variants (anchor window × reference kind × area gating), then a
floor/aggregation grid. Findings:

- **Direction correct on both identity families** with bbox-area floor 0.003 +
  median-per-clip aggregation + W=10 mean reference: response +0.024
  (degradation) and +0.032 (drift) vs the legacy replay's flat/inverted
  behaviour. Monotonicity breaks only at noise scale (+0.002).
- Fraction-above-threshold aggregation inverts on degradation → ruled out.
- **Control appeared to fail:** background_drift responded up to +0.21
  (should be identity-flat), concentrated entirely on BrRLK, where
  face-bearing frames grow ~680 → ~3 760 with severity and survive the area
  floor.

## Probe v3 — selection hypothesis: REFUTED

Hypothesis: the largest-face-per-frame rule starts scoring faces introduced by
the drifting background. Test: candidate dump (`--all_faces`), then per-frame
selection of the candidate *closest to the reference* vs largest.

| rule | 7WHI | BrRLK |
|---|---:|---:|
| largest | −0.004 (flat ✓) | +0.431 |
| best-match | −0.003 (flat ✓) | +0.425 |

Best-match changes nothing. If the subject's face were present and intact,
best-match would find it; similarity still collapses 0.57 → 0.14, so at high
severity no candidate matches the early-video faces.

## The actual mechanism — a battery-asset defect

Frame inspection at severity 0.40: the subject (cartoon fox) is still visible
and intact. The *injected reference background* is real-world broadcast
footage containing people — a name caption and UI elements bleed through the
blend. The corruption genuinely introduces a parade of different real human
faces over time. Early faces ≠ late faces, so an identity-consistency measure
responding is **correct behaviour**; the control premise (identity untouched
by background drift) is false on this base.

Detector sweep over the generator's reference backgrounds:

| asset | detectable faces |
|---|---:|
| ref_bg_for_7WHI | 0 |
| **ref_bg_for_BrRLK** | **7** |
| **ref_bg_for_KZ** | **7** |
| ref_bg_for_hhsz | 0 |
| ref_bg_for_mJog | 0 |

So background_drift on BrRLK and KZ injects identity churn by construction;
on the three clean-reference bases the identity-flatness premise holds, and
on the one probed (7WHI) I′ is properly flat under both selection rules.

## Consequences

1. **I′ proceeds with the simple largest-face rule.** The candidate machinery
   stays in the dump tool (`--all_faces`) for multi-person work later, but
   selection is not the fix for anything observed so far.
2. **Adopted probe settings:** bbox-area floor 0.003 of frame area, median
   per clip, mean over clips, reference = mean of largest-face embeddings
   from the first 10 face-bearing clips.
3. **Control validation for I′ uses clean-reference bases only**; BrRLK and
   KZ background_drift cells are documented as invalid identity controls
   (defect is in the asset, not the metric).
4. **Curation rule for the video-scaling phase:** any reference asset used by
   a corruption generator must be screened face-free with the detector before
   adoption. One command, prevents this class of contamination on the
   enlarged set.
5. Next: wire the anchored statistic into the response table (existing npz
   suffice — no new GPU work for the identity families) and validate through
   the calibration harness on held-out bases.

## Same-host reproducibility footnote

The legacy replay of one identical video differs by ~0.007 between the two
GPUs of this host (0.6948 vs 0.6883) — marginal detections flip even across
GPUs on one machine. All I′-vs-legacy comparisons therefore come from the
same npz files, never from separately-run detector passes.
