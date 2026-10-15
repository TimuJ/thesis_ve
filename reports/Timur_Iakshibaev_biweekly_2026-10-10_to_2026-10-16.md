# Lab-Meeting Report — Timur Iakshibaev (recap since the last meeting)

*Audience: lab members + lab lead. This period was **dataset work** — choosing and
curating the real footage for the disappear-reappear consistency benchmark. The
VBench audit and the VBench-vs-ours comparison were presented before and are not
repeated. This is selection + measured evidence.*

## Where we left off

Last time the evaluation-dataset direction was scoped (long-term tracking footage;
LaSOT-ext as the first choice). This period tested that choice against the real
requirements, searched alternatives, and **curated a working pool from ground-truth
masks** — a clear result.

The requirement, fixed with the three criteria per video: **480p+ (pref 720p+),
target >1% (pref 5%), duration >30 s (pref 1 min+), natural disappear-reappear**.

## 1. LaSOT-ext does not work — gap and sizeable target do not co-occur

Scanned **all 140 LaSOT-ext sequences** (annotations range-fetched from the
HuggingFace zips, no bulk download). Funnel: **720p+ 118 → +target ≥1% 36 → +gap
(full spec) 3**. The gap-rich classes track tiny objects (badminton's target is a
**~10 px shuttlecock**); the sizeable-target classes have no gaps — the two are
**anti-correlated**. And gaps are **short everywhere** (longest across all 150:
**2.5 s**). So LaSOT-ext supports only a *whole-sequence* set, not disappear-reappear.

## 2. LVOS v2 — downloaded, curated from masks, and it works

LVOS v2 is purpose-built for long-term reappearance. We pulled train+valid and
measured the three criteria **directly from the ground-truth masks** (560 videos).

- **Downloaded smart.** 21 GB + 11 GB zips, but we extracted **only the masks**
  (train 682 MB + valid 214 MB) — masks alone give size, gaps, and duration. The
  33 GB of frames stay sealed until we need the final shortlist.
- **Metadata alone is not enough** (worth knowing): every object's presence ratio
  is exactly 0.20 (6 fps from 30 fps) with ~zero spread — the metadata stores only
  a presence *window*, so **gaps and size live only in the masks**. We measured
  them per frame (`np.bincount` per mask → per-object area + longest empty-mask
  run).
- **Result — same-object funnel** (one object that is *both* sizeable *and*
  disappears), the exact test LaSOT failed:

| | LaSOT-ext | **LVOS train+valid** |
|---|--:|--:|
| sizeable × gap × long intersection | 3 / 150 | **64** (≥30 s) / **18** (≥60 s) |
| longest natural gap | 2.5 s | **34.7 s** |
| 720p+ | 118 / 150 | **559 / 560** |

**64 videos** have a target ≥1% that disappears ≥2 s and returns; **26** have a
≥5% target, **34** a gap ≥5 s, **15** ≥10 s. Examples: `XfbRshfV` (69 s, 10.9%
target, **34.7 s** gap), `mQVrNiUG` (39 s, **33.2%**, 13.7 s, 1080p), `hP6JJw8p`
(155 s, 7%, 15 s gap, 14 separate gaps). Caveats: **6 fps** (fine for cross-gap
correspondence, weak for adjacent-frame SR); the very-largest-target + longest-gap
clips cluster at 30–50 s, so only 18 of the 64 also reach 1 min+.

## 3. The other options, briefly

- **MOSE / MOSEv2** — *designed* for disappear-reappear (62% / 50%) with masks, but
  **too short**: 1,507 train videos, longest **500 frames**, none reaches 30 s.
  Excluded for long-video SR; kept as a gap-quality reference.
- **VideoCube / MGIT** — real, ~8 min, long gaps, sizeable, ~1080p — the best pure
  quality, **but movie-sourced** (licence call).

## 4. Decision — build our own set, criteria-driven; LVOS is the gap source

No single off-the-shelf dataset is clean on all axes, so we **combine qualifying
videos across datasets**. Qualified videos against the criteria, by dataset:

- **LVOS: 64 videos** curated and measured (18 at 1 min+) — the natural-gap core.
  Named examples: `XfbRshfV` (10.9% / 34.7 s), `mQVrNiUG` (33.2% / 13.7 s / 1080p),
  `mSYwZwke` (15.1% / 27.8 s), `hP6JJw8p` (7% / 15 s / 2.5 min). Full list:
  `data/lvos_work/curated_manifest.md`.
- **LaSOT-ext: 6 videos** (30 fps complement, sizeable, but gaps only **0.3–1.3 s**):
  `opossum-9` (35%), `opossum-1` (10%), `wingsuit-10` (8.3%), `misc-6` (5.6%),
  `atv-7` (1.5%), `rhino-8` (1.1%) — all 71–83 s, 720p. (A wider 36-sequence 720p
  sizeable-target set exists if gaps are dropped.)
- **VideoCube**: would qualify on every axis (~8 min, ~1080p, long gaps, sizeable)
  — pending a **licence** call, so not yet counted.
- **MOSE**: **0 qualify** — dense gaps but longest video is 500 frames (< 30 s).

Demo clips are ready across datasets — **LVOS** (`XfbRshfV` 35 s gap, `hP6JJw8p`
repeated gaps, `o6HCvWAH` 4.5 min — mask overlay + present/absent banner, real-time
6 fps), **LaSOT** (opossum 35%/10%), **MOSE** (tigers / night-traffic).

## Key numbers

- LaSOT-ext funnel **118 → 36 → 3**; longest natural gap **2.5 s**.
- LVOS funnel: **559/560** 720p+, **224** ≥60 s, **442** target ≥1%, **88** gap
  ≥2 s; **64** full-spec (≥1% & gap ≥2 s & ≥30 s), **18** at ≥60 s; longest gap
  **34.7 s**. All *measured* from masks (only original-fps=30 is inferred).
- MOSE train: 1,507 videos, longest **500 frames** (< 30 s).
- Qualifying pool: **64 (LVOS) + 6 (LaSOT)**; VideoCube pending, MOSE 0 (too short).

## Decisions needed from the lab

1. **How hard to push the "preferred" thresholds** — accept the 64-video LVOS pool
   at >30 s / >1%, or demand 1 min+ and ≥5% (shrinks to a handful)? This sets the
   benchmark size.
2. **VideoCube licence** — use movie-sourced footage for an internal benchmark (no
   redistribution), or rule it out?

## One-line summary

LVOS v2 is curated and works where LaSOT did not — **64** videos (18 at 1 min+)
with a measured sizeable target that disappears up to **35 s** and returns, at
720p; we build our own set from LVOS (gap core) + LaSOT (30 fps complement), with
VideoCube pending a licence call and MOSE excluded for length.
