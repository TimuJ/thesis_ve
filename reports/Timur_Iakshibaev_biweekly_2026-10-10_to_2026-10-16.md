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

## 5. How far the thresholds can be pushed — LVOS eligibility tables

Each cell = number of LVOS videos with a target ≥ (row) % that disappears within
the stated gap range, in a video ≥ (col) s. Measured from masks over all 560
train+valid. A video's gap at size S = the longest disappearance among its objects
that are ≥ S% (so a video drops out when only its always-present large object clears
a high size bar — e.g. `XfbRshfV` counts at ≤10% but not ≥20%).

**Table 1 — gap < 1 s** (sizeable target, *no* real disappearance):

| area≥% \ dur≥ | 30 s | 45 s | 60 s | 90 s | 120 s |
|---|--:|--:|--:|--:|--:|
| ≥1% | 354 | 173 | 123 | 50 | 28 |
| ≥2% | 304 | 131 | 85 | 39 | 24 |
| ≥3% | 272 | 105 | 68 | 29 | 19 |
| ≥5% | 206 | 62 | 35 | 17 | 12 |
| ≥10% | 132 | 22 | 9 | 5 | 3 |
| ≥20% | 66 | 8 | 3 | 2 | 1 |

**Table 2 — 1 s ≤ gap < 10 s** (moderate disappear-reappear):

| area≥% \ dur≥ | 30 s | 45 s | 60 s | 90 s | 120 s |
|---|--:|--:|--:|--:|--:|
| ≥1% | 68 | 29 | 18 | 13 | 9 |
| ≥2% | 57 | 22 | 13 | 10 | 7 |
| ≥3% | 47 | 14 | 6 | 5 | 4 |
| ≥5% | 30 | 9 | 2 | 2 | 2 |
| ≥10% | 18 | 4 | 0 | 0 | 0 |
| ≥20% | 5 | 0 | 0 | 0 | 0 |

**Table 3 — gap ≥ 10 s** (long disappear-reappear — the scarce, valuable case):

| area≥% \ dur≥ | 30 s | 45 s | 60 s | 90 s | 120 s |
|---|--:|--:|--:|--:|--:|
| ≥1% | 15 | 10 | 5 | 1 | 1 |
| ≥2% | 11 | 7 | 4 | 1 | 1 |
| ≥3% | 10 | 6 | 3 | 1 | 1 |
| ≥5% | 9 | 5 | 2 | 1 | 1 |
| ≥10% | 7 | 3 | 1 | 0 | 0 |
| ≥20% | 2 | 1 | 0 | 0 | 0 |

**Reading.** Size and duration are cheap; a *long* gap is the scarce resource.
Table 3 shows the three "preferred" thresholds cannot all be maxed at once — ≥5% &
≥10 s gap & ≥60 s leaves **2** videos, ≥10% & ≥60 s leaves **1**. But relaxing any
one axis recovers a pool: ≥5% & ≥10 s gap at **≥30 s = 9**; ≥3% & ≥10 s at ≥30 s =
**10**; drop to moderate gaps (Tables 2+3, gap ≥1 s) and ≥5% & ≥30 s = **39**, ≥1% &
≥30 s = **83**. Recommendation: set the floor at **≥1% target, ≥30 s, gap ≥1 s**
(~83 videos) and carve a **"premium" long-gap subset** at **≥3–5% & gap ≥10 s & ≥30 s
(9–10 videos)** for the headline disappear-reappear results — rather than demanding
every preferred value simultaneously.

## Key numbers

- LaSOT-ext funnel **118 → 36 → 3**; longest natural gap **2.5 s**.
- LVOS funnel: **559/560** 720p+, **224** ≥60 s, **442** target ≥1%, **88** gap
  ≥2 s; **64** full-spec (≥1% & gap ≥2 s & ≥30 s), **18** at ≥60 s; longest gap
  **34.7 s**. All *measured* from masks (only original-fps=30 is inferred).
- MOSE train: 1,507 videos, longest **500 frames** (< 30 s).
- Qualifying pool: **64 (LVOS) + 6 (LaSOT)**; VideoCube pending, MOSE 0 (too short).

## Decisions needed from the lab

1. **Where to set the thresholds** (see §5 tables). The three preferred values can't
   all be maxed — ≥5% & ≥10 s gap & ≥60 s = 2 videos. Proposal: floor at ≥1% / ≥30 s /
   gap ≥1 s (~83) + a premium ≥3–5% / ≥10 s-gap / ≥30 s subset (9–10) for headline
   results. Accept this split, or set a different point?
2. **VideoCube licence** — use movie-sourced footage for an internal benchmark (no
   redistribution), or rule it out?

## One-line summary

LVOS v2 is curated and works where LaSOT did not — **64** videos (18 at 1 min+)
with a measured sizeable target that disappears up to **35 s** and returns, at
720p; we build our own set from LVOS (gap core) + LaSOT (30 fps complement), with
VideoCube pending a licence call and MOSE excluded for length.
