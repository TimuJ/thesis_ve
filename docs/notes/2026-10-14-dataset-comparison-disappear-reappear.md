# Dataset comparison — sizeable-target disappear-reappear footage for the SR consistency benchmark

**Status:** research complete. **Recommendation: LVOS v2** (fallback VideoCube/MGIT;
MOSE for dense short-clip examples). LaSOT-ext retained only as a permissive-licence
sanity set. Sizeable-target demo clips (LaSOT + MOSE) in `data/lasot_ext_demo/`.

## What prompted this

The benchmark needs real footage with **(1) a sizeable target** (subject with
appearance), **(2) natural disappear-reappear gaps** (no synthetic), and **(3) long
videos, 1 min+**, at a resolution usable for super-resolution. LaSOT-ext (the first
choice) fails: scanning all 140 sequences, the gap-rich classes track tiny objects
(badminton = ~10 px shuttlecock) and the sizeable-target classes have no gaps — the
three criteria intersect in only 3 sequences. So we researched real alternatives.

## Candidates (verified from papers + sample inspection)

| dataset | 1 min+ | natural gaps | sizeable target | GT | res / fps | licence | China access |
|---|---|---|---|---|---|---|---|
| **LVOS v2** | ✅ 1.14 min avg | ✅ long-term reappearance, gaps ≥100 frames (~17 s) | ⚠️ filter (many <10%) | mask | 720p / **6 fps** | CC-BY-4.0 | Baidu/Kaggle |
| **MOSEv2** | ❌ avg ~3 s (183 videos >1 min) | ✅✅ dense (disappear 61.8% / reappear 50.3%) | ⚠️ mix (incl small) | mask | ~720p–1080p / normal | CC-BY-**NC**-SA | HuggingFace |
| **VideoCube / MGIT** | ✅✅ ~8 min | ✅ long | ✅ | bbox | ~1080p / normal | ❌ movie-sourced | project site |
| LaSOT-ext | ⚠️ ~83 s | ❌ ≤2.5 s; 3 usable seqs | ❌ anti-correlated with gaps | bbox | 720p / 30 fps | ✅ permissive | HuggingFace |

Event-camera sets (EventVOT/FELT) are excluded — not RGB. MOT/DanceTrack excluded —
crowded multi-person, medium targets.

## Reading

**No off-the-shelf dataset satisfies all four needs cleanly; it is a choice of
compromise.** Adding the 1 min+ requirement is what breaks MOSE (its rich
disappear-reappear is in ~3 s clips; only 183 videos exceed a minute).

- **LVOS v2 — recommended.** The only candidate that is *simultaneously* 1 min+,
  real, with **natural long disappearances (~17 s)**, **mask** ground truth, 720p,
  and a research-usable licence (CC-BY-4.0). Two manageable caveats: **6 fps** (fine
  for *correspondence across a gap* — our test — less ideal for adjacent-frame
  temporal SR) and we **filter for sizeable targets** by mask area (same step done on
  LaSOT). Note LVOS re-uses some LaSOT/VOT-LT footage, now with masks + long-term
  curation.
- **VideoCube/MGIT — fallback.** Cleanest on gaps + size + length + fps + resolution,
  **if** the lab accepts the movie licence (no redistribution of a derived set).
- **MOSEv2 — supplement.** Unbeatable gap *density* and masks, but short; usable as its
  183-video long subset, or for short-clip demos (see below).

## Sample inspection (what's in the demos)

- **LaSOT sizeable** (`data/lasot_ext_demo/`): opossum-9 (large opossum, **35%** of
  frame, occluded → reappears), opossum-1 (**10%**), misc-6 (underwater, 5.6%).
- **MOSE valid** (311 videos inspected by first-frame mask size; sizeable subjects up
  to 88%): `MOSE_c4fd77f2` (tigers, ~88%, mutual occlusion), `MOSE_c89239d8` (night
  traffic, 1080p, vehicles enter/leave — clean disappear-reappear), `MOSE_c4ecad66`
  (56%), `MOSE_d4f4cf55` (36%, 720p). Short (~1–5 s) but the gap quality is visible.

## Honest limitations

- LVOS **6 fps** and small-target prevalence are real; sizeable curation + the
  fact that our test is cross-gap correspondence (not adjacent-frame smoothness)
  mitigate but do not erase them.
- MOSE valid ships **first-frame masks only** (semi-supervised VOS), so per-frame
  disappear-reappear can't be auto-verified from valid; the train split (22 GB) has
  full masks.
- Resolution/fps figures for LVOS/MOSE/VideoCube are from papers + the MOSE valid
  sample; confirm exact numbers on full download.
- MOSE/LVOS are **non-commercial research** / CC — fine for a thesis/paper, a
  constraint only if a derived dataset were released.

## What this establishes / next

1. **Recommendation to the lab:** adopt **LVOS v2**; the decision point is
   LVOS (CC, 6 fps) vs **VideoCube** (movie-licence) — a licence call.
2. On a go, **curate LVOS** the way LaSOT was scanned: filter by mask-area
   (sizeable) and confirm long-term-reappearance sequences, producing the curated
   benchmark subset.
3. The correspondence measurement then runs server-side with the SR pipeline.
