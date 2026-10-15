# LVOS mask curation — does a sizeable target disappear-reappear in long videos?

**Status:** complete. Curation run over all **560 LVOS v2 train+valid videos**
from per-frame masks. Scan + funnel scripts and the full manifest in
`data/lvos_work/` (`lvos_scan.json`, `lvos_scan.py`, `lvos_funnel.py`,
`curated_manifest.md`; all local, gitignored). Demo clips (mask overlay + status
banner) in `data/lvos_ext_demo/`. This is the dataset readiness check for the
disappear-reappear SR consistency benchmark; it supersedes the metadata-only
estimate in `2026-10-14-dataset-comparison-disappear-reappear.md`.

## What prompted this

The benchmark needs real footage with three properties *at once*: a **sizeable
target** (a subject with appearance), a **natural disappear-reappear gap**, and a
**long video (1 min+)**, at a resolution usable for super-resolution. LaSOT-ext
failed this — the gap-rich classes track tiny objects and the sizeable-target
classes have no gaps (intersection: 3 of 150;
`2026-10-14-lasot-ext-suitability.md`). LVOS v2 was the recommended alternative,
but the recommendation rested on the paper's claims plus metadata. We downloaded
train+valid and measured the three properties directly from the ground-truth
masks.

## Method

- **Downloaded** LVOS v2 `train.zip` (21 GB) and `valid.zip` (11 GB). Extracted
  **only** the `Annotations/` masks (train 682 MB / 160,873 PNGs, valid 214 MB /
  66,197 PNGs) + `meta.json`; the 33 GB of `JPEGImages` were left sealed in the
  zips (disk headroom was ~32 GB). Masks alone carry everything needed — the
  JPEGImages were targeted-extracted per video only for the demo clips, then
  deleted.
- **Metadata is insufficient** (checked first): every object's
  `frame_range.frame_nums / span` ratio is exactly **0.20** (6 fps sampled from
  30 fps), with near-zero within-video spread (max 0.0023) and 45/48 multi-object
  videos sharing identical start+end. So `frame_range` stores only the *presence
  window*, not interior gaps, and has no size field. **Gaps and size are only in
  the masks.**
- **Mask scan** (`lvos_scan.py`, 10-way multiprocess): for each video, one
  `np.bincount` per mask → per-object pixel-count time series. Per object:
  **size** = median mask-area fraction over present frames; **disappear-reappear
  gap** = longest interior run of empty-mask frames flanked by presence, in
  seconds (÷6 fps); plus count of gaps ≥1 s and present-fraction. Resolution from
  the mask dimensions.
- **Funnel** (`lvos_funnel.py`): counts per criterion, and — the decisive test —
  the **same-object intersection** (one object that is both sizeable *and*
  disappears), matching how the LaSOT funnel was computed.

## Result

Funnel over 560 videos (train 420 + valid 140):

| criterion | train | valid | all |
|---|--:|--:|--:|
| 720p+ | 420 | 139 | 559 |
| duration ≥30 s | 417 | 138 | 555 |
| duration ≥60 s | 154 | 70 | 224 |
| sizeable target ≥1% | 359 | 83 | 442 |
| sizeable target ≥5% | 214 | 35 | 249 |
| disappear-reappear gap ≥2 s | 41 | 47 | 88 |
| disappear-reappear gap ≥5 s | 20 | 23 | 43 |
| **same-object: ≥1% & gap ≥2 s & ≥30 s** | 35 | 29 | **64** |
| **same-object: ≥1% & gap ≥2 s & ≥60 s** | 8 | 10 | **18** |

Comparison with LaSOT-ext (the point of the exercise):

| | LaSOT-ext | LVOS train+valid |
|---|--:|--:|
| sizeable × gap × long intersection | **3** / 150 | **64** (≥30 s) / **18** (≥60 s) |
| longest natural gap | 2.5 s | **34.7 s** |
| 720p+ | 118 / 150 | 559 / 560 |

Curated set (an object both ≥1% area and ≥2 s gap): **64 videos**; of those **26**
have a ≥5% target, **34** have a gap ≥5 s, **15** ≥10 s, **5** ≥20 s. Headline
examples: `XfbRshfV` (69 s, 10.9% target, **34.7 s** gap), `mSYwZwke` (46 s, 15.1%,
27.8 s), `mQVrNiUG` (39 s, **33.2%**, 13.7 s, 1080p), `hP6JJw8p` (155 s, 7.0%, 15 s
gap, 14 gaps ≥1 s). Full table: `data/lvos_work/curated_manifest.md`.

**Finding: LVOS clears the bar LaSOT could not.** Sizeable targets genuinely
disappear for *many seconds* (up to 35 s) and return, inside minute-plus 720p
videos — the three requirements co-occur in **64** videos, a usable benchmark pool.

## Reading / interpretation

The mechanism is that LVOS is *built* for long-term reappearance (its design
attribute), so the gap and the tracked subject are the *same* object by
construction — unlike LaSOT, where the gap-bearing target and the sizeable target
are different kinds of object and never coincide. The masks also give per-pixel
size, so "sizeable" is measured, not proxied by a bounding box.

What this supports: LVOS can supply the disappear-reappear portion of the
build-our-own set directly, with ground-truth correspondence across the gap (the
mask identity persists through the absence). What it does not change: this is a
*consistency* benchmark, so the 6 fps / mostly-720p resolution is an accepted
compromise, not a fidelity claim.

## Honest limitations

- **The tension is reduced, not gone.** The biggest-target + longest-gap videos
  cluster at 30–50 s (`mSYwZwke` 46 s, `mQVrNiUG` 39 s, `M2cWIjUt` 30 s); of the
  64, only **18** also reach 60 s, and only a handful are simultaneously ≥5%
  target *and* ≥10 s gap *and* ≥60 s. At the agreed criteria (>30 s, >1%, any
  gap) the pool is comfortable; demanding all "preferred" thresholds at once
  shrinks it.
- **6 fps.** Confirmed from filename stride (every 5th original frame). Fine for
  cross-gap correspondence (our test); weak for adjacent-frame temporal SR.
- **"Gap" = empty mask.** An interior empty-mask run is read as a disappearance.
  This is the LVOS annotation convention, but a full occlusion vs a true exit is
  not distinguished from masks alone; both are valid disappear-reappear for our
  purpose.
- **Size = median over present frames.** A target that is large when near and
  small when far gets one summary number; per-frame area is in the scan if a
  stricter rule is wanted.
- **JPEGImages not yet verified** beyond the three demo videos extracted; frame
  counts match masks 1:1 in the zip central directory, so pairing is expected
  intact, but a full extract was deliberately deferred (disk).
- **Measured vs estimated:** all funnel numbers are *measured* from masks. Only
  the original-fps = 30 (hence 6 fps sampling) is inferred from the 0.20 ratio +
  filename stride.

## What this establishes / next

1. **Dataset readiness: yes for LVOS.** 64-video curated pool (18 at 1 min+) with
   measured sizeable-target disappear-reappear — the real-footage source the
   benchmark needed, where LaSOT-ext was unsuitable.
2. Demo clips rendered (`data/lvos_ext_demo/`, mask overlay + present/absent
   banner, native 6 fps real-time) for `XfbRshfV`, `hP6JJw8p`, `o6HCvWAH`.
3. Next: lock the final per-video criteria with the lab (how hard to push the
   "preferred" thresholds), then extract JPEGImages for the locked shortlist only
   and run the SR + correspondence step server-side.

Related: `2026-10-14-dataset-comparison-disappear-reappear.md`,
`2026-10-14-lasot-ext-suitability.md`.
