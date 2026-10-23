# Biweekly Report — Timur Iakshibaev

## The period in three headlines

1. **The disappear-reappear evaluation dataset is now chosen and built.** Last
   period's top next-step was a first correspondence result on real footage, which
   depended on settling the dataset backbone. That is resolved: after curating a
   candidate from segmentation masks and then measuring the real binding constraint,
   the source is the **full LaSOT at 30 fps** — **88** sequences in which a sizeable
   subject disappears for several seconds and returns, all extracted and encoded.

2. **Frame rate — not gap length — turned out to be the binding constraint, and it
   reframed the choice.** The mask-curated candidate (LVOS) offered the longest natural
   gaps (up to ~35 s) but only at **6 fps**, for both the frames and the annotations,
   with no higher-rate source and no non-synthetic way to raise it. Six fps is adequate
   for the across-gap identity comparison but makes the adjacent-frame temporal measure
   meaningless, so it was ruled out as super-resolution input. Scanning the **full
   LaSOT (1,400 sequences)** instead produced 88 sizeable-target disappear-reappear
   videos at 30 fps — gaps of *seconds* rather than tens of seconds, but at a usable
   frame rate. For a consistency benchmark that is the correct trade.

3. **The pool is built and a starter set is already in a collaborator's hands.** The
   full 88-video set is extracted, encoded, and packaged with per-frame ground truth;
   a 10-video diverse subset has been handed to a labmate to run the first
   super-resolution + consistency pass — the direct set-up for the quantitative
   correspondence result that was last period's headline next-step.

## Key numbers

- Full LaSOT scan: **1,400** sequences / **70** categories, 0 failures. Qualifying pool
  (sizeable target ≥1 %, disappear-reappear gap ≥2 s, ≥480p): **88** videos across
  **39** categories — **27** with a gap ≥5 s, **9** ≥10 s, **68** at least a minute
  long. Longest gap **23.6 s**; largest target **29.3 %**. All measured from the
  per-frame annotations; only the 30 fps figure is the dataset's stated value.
- Mask-curated alternative (LVOS): **64** sizeable-target sequences, gaps to **34.7 s**
  — but **6 fps**, which decided against it as footage.
- Earlier extension-only scan, for contrast: longest gap **2.5 s**, and gap-rich
  sequences track near-zero-size targets — the full set is what breaks that
  anti-correlation.
- High-resolution alternative (VideoCube): ~30 fps, ~8-minute videos, but **344 GB –
  1.4 TB**, its host unreachable from the working machine, and non-commercial /
  edited-source — shelved.
- Built artefact: **88** encoded clips, **8.5 GB**, with bounding-box and absence
  ground truth per video; a 10-video diverse starter staged for the first run.

## Decisions and framing this period

- **Framing shift — the binding constraint is frame rate, not gap length.** A
  consistency benchmark has two measurement axes: identity/structure *across* the gap
  (frame-rate-independent) and *adjacent-frame* temporal quality (needs a normal frame
  rate). Optimising for the longest possible gap led to footage that cannot support the
  second axis; optimising for frame rate keeps both. This is now the stated basis for
  the dataset choice.
- **The dataset-backbone decision (open last period) is resolved** toward the
  redistributable, standard-frame-rate side: a long-term tracking corpus at 30 fps,
  rather than the high-resolution-but-restricted, movie-sourced alternative.
- **Still open and still gating implementation:** whether the benchmark formally adopts
  the reduced-reference framing. Unchanged from last period; it remains the decision
  that turns the specification into code.
- **A new, smaller choice is teed up:** whether to report on the full 88-video pool or
  lead with a premium subset (the **13** videos that are simultaneously ≥5 % target and
  ≥5 s gap) for the headline disappear-reappear numbers.

## Outcomes this period

- A candidate dataset curated directly from segmentation masks (64 sequences), then
  superseded as footage once the frame-rate constraint was measured.
- The **full** LaSOT scanned from its per-frame absence annotations (no bulk download);
  an 88-video sizeable-target disappear-reappear pool established where only the
  extension had been examined before.
- The high-resolution alternative assessed on access, size, and licence, and shelved
  with the reasoning recorded.
- The 88-video pool **built and encoded** (8.5 GB) with ground truth and a manifest; a
  10-video diverse starter delivered to a collaborator.
- Findings recorded as a committed technical note; local storage managed (~47 GB
  reclaimed) while keeping the curated metadata and re-download paths.

## Next two weeks

1. **The first quantitative correspondence result on real footage** — degrade and
   super-resolve the pool and score whether the output preserves the subject across
   each gap against the ground truth, directly comparable to the external benchmark's
   table. This is the measurement the built dataset now unblocks.
2. **Settle the reduced-reference adoption decision**, which still gates moving the
   specification into implementation.
3. **Fix the pool scope** (full 88 vs the premium subset) and complete the
   collaborator's first super-resolution + consistency pass on the starter set.
4. **Add the collaborator's model row** once his outputs arrive (still pending).

## One-line summary for the meeting

The disappear-reappear evaluation dataset is chosen and built — 88 sizeable-target
sequences at 30 fps from the full LaSOT, after frame rate (not gap length) ruled out
the longer-gap alternative — unblocking the first real correspondence measurement.
