# LaSOT-ext suitability for the SR disappear-reappear consistency benchmark

**Status:** analysis complete. Curated whole-sequence-correspondence subset
defined (36 sequences, manifest: `2026-10-14-lasot-ext-curated-manifest.md`);
figure `reports/figures/lasot_ext_suitability.png`; demo clips in
`data/lasot_ext_demo/` (local, gitignored: badminton gap structure + a
sizeable-target misc-6 example). The correspondence *measurement* is deferred to
the SR phase (below). **Headline: LaSOT-ext is usable for whole-sequence subject
consistency, but NOT for explicit disappear-reappear gaps.**

## What prompted this

The benchmark's correspondence layer needs real footage where a subject can be
tracked (ideally leaves and returns) over a long sequence. LaSOT-ext was chosen
(freely redistributable, ~720p). The question put to the lab — *is the dataset
ready?* — required actually testing whether LaSOT-ext contains sequences that fit
our three needs at once: a disappear-reappear gap, a subject large enough to have
appearance, and resolution high enough for super-resolution.

## Method

- LaSOT-ext is distributed on HuggingFace (`l-lt/LaSOT-ext`) as 15 per-class zips,
  **63.9 GB** total. To avoid downloading all of it, the per-sequence annotation
  files were **range-fetched from the zips' central directories** (`remotezip`):
  `out_of_view.txt`, `full_occlusion.txt`, `groundtruth.txt`, plus the first
  frame for resolution — a few KB per sequence instead of GB.
- Scanned **140 sequences** across all 15 classes. Per sequence computed:
  - **disappear-reappear gap** = longest interior run (≥8 frames) where the
    subject is not visible (`out_of_view` OR `full_occlusion`), subject present
    before and after;
  - **target size** = median ground-truth box area as a fraction of the frame;
  - **resolution** from the first frame.

## Result

**1. Gaps are short.** Across all 140 sequences the single longest
disappear-reappear gap is **75 frames (2.5 s)** (lantern-3); most are ~1 s. The
designed-in "occlusion focus" of LaSOT-ext means *frequent brief* occlusions, not
long disappearances. (Ring Forcing's A-D-R sweeps gaps up to 60 s — LaSOT-ext
does not contain those.)

**2. Gap and target-size are anti-correlated.** The gap-rich classes track small,
fast objects — e.g. badminton's target is the **shuttlecock (~10 px box)** — which
carry no appearance. The sizeable-target classes (animals, people) are
continuously visible and lack gaps. The viable intersection is nearly empty
(figure, left).

**3. The funnel** (of 140 sequences):

| criterion | sequences |
|---|---:|
| ≥720p | 118 |
| ≥720p + target ≥1% of frame (**whole-sequence set**) | **36** |
| ≥720p + target ≥2% | 23 |
| ≥720p + target ≥5% | 16 |
| ≥720p + target ≥1% + disappear-reappear gap (**full spec**) | **3** |

The full-spec set is only `atv-7`, `misc-6`, `rhino-8` — too few to be a
benchmark. The 36-sequence whole-sequence set (target ≥1%, ≥720p; gaps *not*
required) is animal-heavy: rhino 8, raccoon 7, opossum 6, atv 7, misc 3, cosplay
2, wingsuit 2, lantern 1.

## Reading

**LaSOT-ext supports a whole-sequence subject-consistency benchmark, not a
disappear-reappear-gap one.** The 36 curated sequences are ~720p+, ~83 s long,
with a continuously-tracked sizeable subject — enough to ask *does super-
resolution keep the subject consistent over the whole long sequence?* (the
correspondence our metric needs). What LaSOT-ext cannot give naturally is the
minute-scale "gone and returns" case; for that the options are (a) **construct
gaps ourselves** from these 36 sequences (mask/splice the subject out for a
controlled interval), or (b) use **VideoCube/MGIT** (long natural gaps, movie
licence).

## Correspondence measurement — deferred to the SR phase

The quantitative correspondence measurement — *does super-resolution preserve the
subject's appearance across the sequence?* — belongs with the SR pipeline, not
this readiness check. It needs the SR outputs and a pose-robust deep embedder
(DINOv3/ResNet), and runs naturally on the server where the GPU and models already
live. A quick local attempt settled the tooling choice: SIFT keypoint matching is
too strict across the ~80 s span (near-zero stable matches between distant frames),
and a ResNet embedder's weights would not download reliably over the lab network —
both point to running this server-side alongside the SR step. The *dataset* side it
needs is ready: the 36 curated sequences carry per-frame ground-truth boxes that
localize the subject for exactly this comparison, at ≥720p.

## Honest limitations

- **Single-object ground truth.** LaSOT tracks one target per sequence; other
  subjects in frame are unlabelled. Multi-subject correspondence relies on our
  own clustering, without GT.
- **Domain skew.** The 36-set is dominated by animals (21/36); few people. Our
  ArcFace identity sub-metric won't apply there — general matchers (DINOv3/SIFT)
  are needed, as already planned.
- **SIFT is a texture proxy** for the correspondence demo; a deep embedding is the
  production choice.
- **Resolution is 720p** for most (13 of 140 are 1080p); adequate for a
  consistency benchmark, not a high-fidelity one.
- `misc` is a mixed class; `misc-6`'s content was used only as the single largest
  available target.

## What this establishes / next

1. **Answer to "is the dataset ready?":** ready for **whole-sequence
   correspondence** (36 curated sequences, manifest attached); **not** for
   explicit disappear-reappear gaps (only 3) — a decision point, not a blocker.
2. Next: adopt the whole-sequence reframe on the 36 sequences — download them on
   the server, generate LQ inputs, run SR, and score subject consistency
   (HR-referenced and reduced-reference). For the gap version, prototype synthetic
   gap construction from the same 36.
