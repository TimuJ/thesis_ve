# ADR pair-selection signals — motion / detail / blur probe

**Status:** Complete (probe). Five signal primitives implemented and unit-tested
(`scripts/lr_vcc/adr/quality.py`); validated against the 42 reviewer-labelled pairs
in `data/review/full18_gated.json` using the local GT videos in `data/starter/`.
**Outcome: none wired as a selection gate** — the cheap motion/detail signals do not
reproduce the reviewer's calls; the no-reference blur metric is valid but the
dominant residual complaint is pose, not blur. `selection.py` / `run_adr.py`
untouched. Companion to the Laplacian-blur negative result recorded in
`docs/notes/2026-11-05-adr-consistency-calibration-findings.md`.

## What prompted this

Reviewer triage of all 42 selected cross-gap pairs named four recurring reasons a
pair is unreviewable, by frequency: different pose/angle, motion blur, object too
small, object on the frame edge. Edge and size already have working gates; blur and
pose do not. The proposal (`docs/superpowers/plans/2026-11-06-adr-selection-improvements-proposal.md`,
P3/P4) suggested cheap CPU signals — optical-flow motion and detail/contrast — for
the blur and "small & dark" cases. Before wiring any of them as a gate (the mistake
the Laplacian probe warned against), this probe asks whether they actually track the
reviewer's labels.

## Method

Signals, all pure CPU (cv2/numpy), computed on the GT object crop frozen on the GT
annotation box (selection is GT-only):

- `motion_signal` (P3a): mean dense Farneback flow magnitude between frames i±1
  cropped to the selected box, normalised by box diagonal.
- box-velocity (P3b): box-centre displacement between i−1 and i+1, normalised by box
  diagonal (annotation geometry only, no decode).
- `reblur_blur` (P3c): no-reference blur, Crete, Dolmiere, Ladret & Nicolas (2007) —
  low-pass the crop again and measure the drop in neighbour-to-neighbour variation;
  [0,1], higher = blurrier.
- `rms_contrast`, `shannon_entropy`, `detail_signal` (P4): detail = geometric mean of
  normalised std and normalised histogram entropy; [0,1], higher = more legible.

Each pair takes the worse of its i and j crops (max blur / max motion / min detail).
Labels available: "motion/pose-confounded" = {monkey-19, fox-20} (the only
motion/pose tags encoded); "too small" via the size gate; "small but clear, keep" =
{gecko-12, licenseplate-15}.

## Result

P3 — motion, across three variants (confounded = monkey-19 + fox-20):

| signal | confounded mean | rest mean | rest median | separates? |
|---|---|---|---|---|
| flow-in-box | 0.025 | 0.022 | 0.020 | no |
| box-velocity | 0.107 | 0.083 | 0.057 | weak; monkey absent |
| re-blur | 0.495 | 0.495 | 0.528 | no (in aggregate) |

Per-pair ranks under re-blur (of 42, higher = blurrier) are the real evidence:
**fox-20 ranks #2 / #7 / #12 / #16; monkey-19 ranks #33 / #34 / #35** (among the
sharpest). Re-blur passes its sanity check (Gaussian-blurred crop scores above its
sharp original). The Laplacian, flow, and velocity variants all likewise score
monkey as sharp/slow.

P4 — detail (higher = more legible):

| crop | detail_signal | reviewer intent |
|---|---|---|
| licenseplate-15 | 0.492 | keep (small but clear) |
| gecko-12 | 0.282–0.347 | keep (small but clear) |
| mouse-2 | 0.377 | cut ("small & dark") |
| synthetic dark-flat | <0.05 | — |

`detail_signal` orders the synthetic extremes correctly but **ranks mouse above
gecko**, the opposite of the reviewer's mouse-vs-gecko call.

## Reading / interpretation

- **flow-in-box fails by construction:** the LaSOT box tracks the object, so cropping
  both neighbour frames to that box cancels the object's translation, leaving only
  residual deformation ≈ 0. It measures the wrong thing.
- **re-blur is a *valid* blur detector** — a genuine improvement on the Laplacian,
  which *inverted* on these crops. It correctly flags fox-20 (real blur present) and
  passes the controlled Gaussian test. The aggregate separation washes out
  (0.495 = 0.495) only because the one "confounded" label mixes two mechanisms.
- **monkey is pose, not blur.** Four independent blur/motion estimators (Laplacian,
  flow, velocity, re-blur) all rank monkey-19 as sharp/slow. Its reviewer tag is
  driven by a pose change across the gap, which no blur or box-motion signal can — or
  should — catch.
- **mouse is size, not detail.** Mouse's crop carries adequate contrast and entropy
  (0.377); its unreviewability is its size, which the existing size gate already
  handles. Detail/contrast adds no discrimination for the case it was meant to fix.

The net: the two highest-frequency residual complaints — different pose/angle, and
monkey specifically — are **pose** problems, pointing at P2 (geometric pose-match via
RoMa v2), which is already the approved scoring-axis upgrade and would do double duty
for selection. Re-blur is worth keeping as a soft signal for the cases that are
genuinely blur (fox).

## Honest limitations

- **Thin, confounded ground truth.** Only two clips carry a motion/pose label, over
  42 pairs, from one reviewer; the group means are underpowered, which is why the
  per-pair ranks (not the means) are quoted as evidence. The label itself conflates
  blur and pose, so it cannot cleanly validate a blur-only metric.
- **Resolution / crop confounds.** `reblur_blur` and `detail_signal` are computed on
  bbox crops at native GT resolution (sizes vary widely); absolute thresholds would
  need per-resolution calibration. Crops are plain bbox, not SAM-masked, so
  background texture leaks into both estimates.
- **GT-only.** Signals were validated on GT crops (selection is frozen on GT); SR-side
  behaviour was not assessed here.
- **Negative results retained:** flow-in-box motion, box-velocity motion, and
  detail/contrast gating did not survive this probe and should not be re-attempted as
  cheap selection gates without new evidence.

## What this establishes / next

- Keep the five primitives as tested, reusable soft features; **gate on none** of them
  on current evidence. Selection pipeline unchanged.
- `reblur_blur` supersedes the Laplacian as the blur signal where blur is the real
  issue; fold it in only after P6-style weight-fitting on more labels.
- The real lever for the top complaint is **P2 geometric pose-match (RoMa v2)**, to be
  taken up when the approved RoMa v2 + DINOv3 weights are staged. See the selection
  proposal P2/P6.
