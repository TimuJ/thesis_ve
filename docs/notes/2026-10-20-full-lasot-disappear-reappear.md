# Full LaSOT — sizeable-target disappear-reappear videos at 30 fps

**Status:** complete. Scanned **all 1,400 LaSOT sequences** (70 categories, HF
`l-lt/LaSOT`) from their per-frame absence annotations. Scan +
build scripts and raw results in `data/lasot_work/` (`full_scan.py`,
`full_lasot_scan.json`, `build_upload10.py`; local, gitignored). A first **10-video
diverse set** is built and staged in `data/upload_10/` (mp4 + GT + `manifest.csv`).
This supersedes the LaSOT-ext suitability check (`2026-10-14-lasot-ext-suitability.md`),
which scanned only the 15-class extension.

## What prompted this

We need real footage where a **sizeable subject disappears and reappears** inside a
**long video**, at a resolution and **frame rate** usable for super-resolution. LVOS
supplies long natural gaps but only at **6 fps** — unusable as adjacent-frame SR input
(`2026-10-16-lvos-mask-curation.md`). LaSOT is **30 fps**, but we had only ever scanned
the 15-class *extension*, where the longest gap was 2.5 s and always on a ~0 %-area
target. The question: does the **full** 70-class LaSOT contain 30 fps videos that have
both a sizeable target *and* a real disappear-reappear gap?

## Method

- **Source:** `https://huggingface.co/datasets/l-lt/LaSOT` (per-category `.zip`).
- **Scan** (`full_scan.py`, 10-way threaded over categories): per sequence, range-fetch
  only `out_of_view.txt` + `full_occlusion.txt` (comma-delimited 0/1 per frame) — **no
  frames downloaded, disk-safe**. "Absent" = out-of-view **OR** full-occlusion.
  **Disappear-reappear gap** = longest interior run of absent frames flanked by presence,
  in seconds (÷30 fps). **Target size** = median ground-truth bbox area / frame area;
  resolution from the first frame. Target size + resolution fetched only for gap ≥ 1.5 s
  candidates.
- **Qualifying pool:** an object both **≥1 %** median bbox area AND a **≥2 s** gap AND
  resolution **≥480p**.

## Result

1,400 sequences, 70 categories, **0 failures**. Funnel:

| criterion | count |
|---|--:|
| gap ≥2 s | 210 |
| gap ≥2 s & target ≥1 % | 141 |
| **gap ≥2 s & target ≥1 % & ≥480p (the pool)** | **88** |

Within the 88-video pool (**39 distinct categories**):

| band | count | | band | count |
|---|--:|---|---|--:|
| gap ≥3 s | 46 | | target ≥5 % | 45 |
| gap ≥5 s | 27 | | target ≥10 % | 23 |
| gap ≥10 s | 9 | | target ≥20 % | 6 |
| duration ≥60 s | 68 | | duration ≥120 s | 22 |

Longest gap **23.6 s** (`lizard-17`); largest target **29.3 %** (`hippo-12`). About
**12** videos are simultaneously **gap ≥5 s and target ≥5 %** (the premium subset:
`electricfan-8`, `monkey-5`, `hippo-2`, `goldfish-10`, `giraffe-5`, `helmet-13`,
`lion-12`, `electricfan-15`, `giraffe-15`, `fox-10`, `cattle-4`, `goldfish-20`).

Comparison (the point of the exercise):

| | LaSOT-ext | **full LaSOT** | LVOS |
|---|--:|--:|--:|
| sizeable-target disappear-reappear videos | ~6 (gaps 0.3–1.3 s) | **88** (gaps 2–23.6 s) | 64 (gaps 2–34.7 s) |
| frame rate | 30 fps | **30 fps** | 6 fps |
| longest gap on a sizeable target | 1.3 s | **23.6 s** | 34.7 s |

**Finding: the full LaSOT clears the bar.** 88 videos, 30 fps, sizeable targets that
genuinely leave and return for 2–23.6 s, in 1-minute-plus 720p sequences — the
30 fps + real-gap + sizeable + long combination that neither LaSOT-ext (gaps too short)
nor LVOS (6 fps) could give.

### First diverse set delivered (`data/upload_10/`)

10 distinct subjects, all 1280×720 @ 30 fps, raw mp4 (crf 18) + per-video GT
(`groundtruth`/`full_occlusion`/`out_of_view`) + `manifest.csv`, 1.0 GB total:
`lizard-17` (23.6 s gap, 8.1 %), `electricfan-8` (15.5 s, 6.5 %), `monkey-5`
(12.5 s, 13.7 %), `hippo-2` (10.1 s, 16.8 %), `goldfish-10` (9.3 s, 6.4 %, 8 gaps),
`giraffe-5` (7.1 s, 24.1 %), `lion-12` (6.2 s, 9.5 %), `tank-18` (6.4 s, 4.3 %),
`airplane-2` (4.0 s, 6.9 %, 277 s), `spider-1` (2.6 s, 14.4 %).

## Reading / interpretation

The extension failed because its 15 classes track either tiny fast objects (gap-rich,
~0 % area) or sizeable objects with no gaps — the two never coincide. The full 70-class
set contains many categories where a **sizeable** subject naturally goes behind cover or
off-frame and returns (reptiles, primates, large mammals, vehicles, rotating objects),
so gap and appearance belong to the **same** object. Gaps here are **seconds**, not the
**tens of seconds** LVOS reaches — but they occur at **30 fps** with no slideshow
artefact, which for an SR consistency benchmark is the trade we want: **a few-second
30 fps gap is sufficient and preferable to a 35 s gap at 6 fps** (lab decision,
2026-10-20). Target size is a bbox fraction (ground truth is a box, not a mask), so it
is an over-estimate of true object coverage for non-rectangular subjects — fine for
"is there something to super-resolve," not a pixel-accurate area.

## Honest limitations

- **Gap length is modest.** Only 9 of 88 reach ≥10 s and 27 reach ≥5 s; the median
  qualifying gap is ~3–4 s. If the benchmark later needs long gaps specifically, LVOS
  (6 fps) or VideoCube (server-only) remain the only sources — this set does not provide
  them.
- **"Absent" conflates out-of-view and full-occlusion.** The scan unions both flags; a
  true frame-exit and a full occlusion are not distinguished (both are valid
  disappear-reappear for our purpose, but they differ physically).
- **Target size = median GT bbox area**, not segmentation area — over-estimates coverage
  for thin/irregular subjects, and a target that is large when near and small when far
  gets one summary number.
- **Resolution floor 480p; most of the pool is exactly 1280×720** — adequate for a
  *consistency* benchmark, not a high-fidelity (≥1080p) claim.
- **Measured vs inferred:** gaps, target bbox, resolution, frame counts are all measured
  from the annotations/frames. Only fps = 30 is the dataset's stated/standard value.
- **mp4 re-encode.** The delivered clips are H.264 crf 18 (near-lossless, chosen for SR
  input); the originals are JPEG frame sequences if a pristine input is later required.

## What this establishes / next

1. **Dataset readiness: yes, from full LaSOT at 30 fps.** An 88-video pool (68 at ≥60 s,
   45 with ≥5 % targets, 39 categories) of measured sizeable-target disappear-reappear —
   the 30 fps real-footage source the benchmark needed.
2. 10-video diverse set staged (`data/upload_10/`) for the labmate's first experiment run;
   uploads to Google Drive by hand.
3. Next: decide how large to grow the set from the 88 (and whether to bias toward the
   premium gap ≥5 s & target ≥5 % subset); extract the chosen shortlist via the
   contiguous byte-range method (`build_upload10.py` — HF per-member reads are 2.4 s/frame
   and unusable; one bulk Range request per sequence runs at ~6 MB/s).

## Full qualifying pool (88 videos, ranked by gap)

| video | class | gap (s) | target % | dur (s) | #gaps≥1s | WxH |
|---|---|--:|--:|--:|--:|---|
| lizard-17 | lizard | 23.6 | 8.1 | 148 | 2 | 1280x720 |
| helmet-17 | helmet | 18.8 | 1.2 | 267 | 10 | 1280x720 |
| gecko-12 | gecko | 17.6 | 2.4 | 107 | 3 | 1280x720 |
| electricfan-8 | electricfan | 15.5 | 6.5 | 51 | 1 | 1280x720 |
| elephant-17 | elephant | 13.7 | 1.5 | 76 | 1 | 1280x720 |
| licenseplate-15 | licenseplate | 12.5 | 1.6 | 49 | 1 | 1280x720 |
| monkey-5 | monkey | 12.5 | 13.7 | 122 | 1 | 1280x720 |
| monkey-19 | monkey | 10.5 | 2.2 | 73 | 5 | 1280x720 |
| hippo-2 | hippo | 10.1 | 16.8 | 66 | 2 | 1280x720 |
| leopard-9 | leopard | 9.6 | 2.1 | 114 | 1 | 1280x720 |
| goldfish-10 | goldfish | 9.3 | 6.4 | 187 | 8 | 1280x720 |
| electricfan-18 | electricfan | 8.9 | 2.7 | 62 | 0 | 1280x720 |
| lizard-18 | lizard | 7.2 | 4.1 | 40 | 1 | 1280x720 |
| giraffe-5 | giraffe | 7.1 | 24.1 | 44 | 2 | 1280x720 |
| helmet-13 | helmet | 6.4 | 7.7 | 104 | 9 | 1280x720 |
| tank-18 | tank | 6.4 | 4.3 | 139 | 3 | 1280x720 |
| lion-12 | lion | 6.2 | 9.5 | 116 | 0 | 1280x720 |
| bus-20 | bus | 6.1 | 2.1 | 187 | 2 | 1280x720 |
| electricfan-15 | electricfan | 6.0 | 17.4 | 94 | 2 | 1280x720 |
| giraffe-15 | giraffe | 5.9 | 24.7 | 49 | 2 | 1280x720 |
| fox-20 | fox | 5.9 | 2.6 | 139 | 6 | 1280x720 |
| lizard-9 | lizard | 5.8 | 4.5 | 90 | 3 | 1280x720 |
| kangaroo-16 | kangaroo | 5.7 | 3.4 | 52 | 2 | 1280x720 |
| mouse-2 | mouse | 5.5 | 1.0 | 71 | 1 | 1280x720 |
| fox-10 | fox | 5.4 | 10.7 | 56 | 1 | 1280x720 |
| cattle-4 | cattle | 5.3 | 9.1 | 48 | 1 | 1280x720 |
| goldfish-20 | goldfish | 5.0 | 11.9 | 82 | 5 | 1280x720 |
| bus-6 | bus | 4.7 | 6.0 | 217 | 1 | 1280x720 |
| truck-6 | truck | 4.7 | 3.3 | 175 | 2 | 1280x720 |
| elephant-16 | elephant | 4.6 | 1.5 | 89 | 4 | 1280x720 |
| fox-9 | fox | 4.6 | 4.5 | 71 | 3 | 1280x720 |
| monkey-7 | monkey | 4.6 | 6.3 | 68 | 1 | 1280x720 |
| tank-15 | tank | 4.5 | 3.8 | 100 | 2 | 1280x720 |
| pig-18 | pig | 4.4 | 1.5 | 80 | 1 | 1280x720 |
| electricfan-16 | electricfan | 4.2 | 1.3 | 109 | 6 | 1280x720 |
| fox-1 | fox | 4.2 | 13.8 | 42 | 4 | 1280x720 |
| airplane-2 | airplane | 4.0 | 6.9 | 277 | 2 | 1280x720 |
| lion-9 | lion | 4.0 | 11.3 | 57 | 1 | 1280x720 |
| umbrella-13 | umbrella | 3.9 | 1.9 | 250 | 2 | 1280x720 |
| gametarget-11 | gametarget | 3.8 | 9.7 | 65 | 1 | 1280x720 |
| rubicCube-8 | rubicCube | 3.8 | 2.1 | 77 | 1 | 1280x720 |
| kangaroo-13 | kangaroo | 3.5 | 6.1 | 77 | 1 | 1280x720 |
| robot-13 | robot | 3.5 | 9.8 | 241 | 4 | 1280x720 |
| electricfan-1 | electricfan | 3.2 | 23.2 | 55 | 2 | 1280x720 |
| hippo-3 | hippo | 3.0 | 20.6 | 141 | 2 | 1280x720 |
| lizard-10 | lizard | 3.0 | 16.5 | 60 | 2 | 1280x720 |
| frog-7 | frog | 2.9 | 6.4 | 50 | 2 | 1280x720 |
| goldfish-9 | goldfish | 2.9 | 1.0 | 53 | 2 | 1280x720 |
| hippo-12 | hippo | 2.9 | 29.3 | 117 | 1 | 1280x720 |
| frog-2 | frog | 2.8 | 16.2 | 82 | 3 | 1280x720 |
| rabbit-10 | rabbit | 2.8 | 3.2 | 70 | 1 | 1280x720 |
| truck-16 | truck | 2.8 | 1.3 | 241 | 7 | 1280x720 |
| bear-5 | bear | 2.7 | 7.9 | 33 | 1 | 1280x720 |
| frog-8 | frog | 2.7 | 1.6 | 90 | 4 | 1280x720 |
| tiger-7 | tiger | 2.7 | 2.8 | 85 | 1 | 1280x720 |
| book-20 | book | 2.6 | 2.9 | 92 | 2 | 1280x720 |
| flag-8 | flag | 2.6 | 9.2 | 65 | 1 | 1280x720 |
| gorilla-1 | gorilla | 2.6 | 6.0 | 97 | 1 | 1280x720 |
| rabbit-12 | rabbit | 2.6 | 3.7 | 111 | 1 | 1280x720 |
| spider-1 | spider | 2.6 | 14.4 | 130 | 1 | 1280x720 |
| goldfish-14 | goldfish | 2.5 | 3.9 | 138 | 3 | 1280x720 |
| lion-4 | lion | 2.5 | 3.0 | 43 | 2 | 1280x720 |
| monkey-11 | monkey | 2.5 | 8.3 | 80 | 1 | 1280x720 |
| elephant-9 | elephant | 2.4 | 3.9 | 83 | 1 | 1280x720 |
| hippo-8 | hippo | 2.4 | 6.2 | 58 | 2 | 1280x720 |
| zebra-15 | zebra | 2.4 | 4.0 | 181 | 4 | 1280x720 |
| zebra-7 | zebra | 2.4 | 6.5 | 76 | 1 | 1280x720 |
| bottle-14 | bottle | 2.3 | 17.4 | 240 | 1 | 1280x720 |
| gecko-18 | gecko | 2.3 | 4.1 | 60 | 4 | 1280x720 |
| gecko-3 | gecko | 2.3 | 5.1 | 96 | 1 | 1280x720 |
| lion-8 | lion | 2.3 | 15.8 | 50 | 1 | 1280x720 |
| motorcycle-18 | motorcycle | 2.3 | 1.0 | 85 | 1 | 1280x720 |
| monkey-18 | monkey | 2.3 | 15.3 | 70 | 2 | 1280x720 |
| surfboard-1 | surfboard | 2.3 | 1.0 | 333 | 3 | 1280x720 |
| tank-2 | tank | 2.3 | 4.0 | 254 | 4 | 1280x720 |
| gecko-15 | gecko | 2.2 | 9.2 | 93 | 3 | 1280x720 |
| giraffe-7 | giraffe | 2.2 | 23.9 | 57 | 2 | 1280x720 |
| gorilla-8 | gorilla | 2.2 | 3.4 | 61 | 3 | 1280x720 |
| bicycle-9 | bicycle | 2.1 | 1.1 | 100 | 1 | 1280x720 |
| flag-3 | flag | 2.1 | 3.3 | 64 | 0 | 1280x720 |
| electricfan-13 | electricfan | 2.1 | 4.1 | 80 | 1 | 1280x720 |
| guitar-7 | guitar | 2.1 | 1.6 | 110 | 1 | 1280x720 |
| kangaroo-8 | kangaroo | 2.1 | 3.8 | 51 | 2 | 1280x720 |
| leopard-20 | leopard | 2.1 | 11.8 | 92 | 1 | 1280x720 |
| gorilla-2 | gorilla | 2.0 | 8.6 | 124 | 2 | 1280x720 |
| leopard-11 | leopard | 2.0 | 13.5 | 267 | 2 | 1280x720 |
| mouse-19 | mouse | 2.0 | 14.3 | 94 | 1 | 1280x720 |
| tank-13 | tank | 2.0 | 19.2 | 116 | 2 | 1280x720 |

Related: `2026-10-14-lasot-ext-suitability.md`, `2026-10-16-lvos-mask-curation.md`,
`2026-10-16-lvos-curated-manifest.md`.
