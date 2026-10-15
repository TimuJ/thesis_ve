# Lab-Meeting Report — Timur Iakshibaev (recap since the last meeting)

*Audience: lab members + lab lead. This period was **dataset work** — choosing the
real footage for the disappear-reappear consistency benchmark. The VBench audit
and the VBench-vs-ours comparison were presented before and are not repeated. This
is selection + evidence, not new metric results.*

## Where we left off

Last time the evaluation-dataset direction was scoped (long-term tracking footage;
LaSOT-ext as the first choice). This period tested that choice against the real
requirements and searched for alternatives — with a clear negative result and a
plan.

## 1. LaSOT-ext does not work — the gap and the sizeable target do not co-occur

Scanned **all 140 LaSOT-ext sequences** for three properties at once — a
disappear-reappear gap, a sizeable target (median ground-truth box area), and
resolution. Annotations were range-fetched from the HuggingFace zips' central
directories, so no bulk download was needed.

- Funnel of qualifying sequences: **720p+ 118 → +target ≥1% 36 → +gap (full spec)
  3**.
- **Mechanism:** the gap-rich classes track tiny objects (badminton's target is a
  **~10 px shuttlecock**), while the sizeable-target classes (animals, people) have
  no gaps. The two requirements are **anti-correlated**, so the intersection is
  nearly empty.
- Separately, natural gaps are **short everywhere** — the longest across all 150
  sequences is **2.5 s**; LaSOT-ext is occlusion-focused, not long-disappearance.

So LaSOT-ext supports only a *whole-sequence* consistency set (36 sequences), not
the explicit disappear-reappear we want.

## 2. Real-dataset search — two hard constraints eliminate the obvious choices

Searched real (non-synthetic) datasets purpose-built for disappear-reappear:

- **MOSE / MOSEv2** — *designed* around it (disappear 62% / reappear 50%), with
  mask ground truth. **But the videos are too short:** 1,507 train videos, the
  **longest only 500 frames** (median 48) — **none reaches 30 s**, and they are
  mostly vertical phone clips. → excluded for long-video SR; kept only as a
  disappear-reappear *quality* reference. (This also explains why the first MOSE
  demo clips looked rushed — few frames, fast content.)
- **LVOS v2** — long (**1.14 min** avg), with genuine **long** reappearances
  (object returns after ≥100 frames ≈ **17 s** gaps), mask GT, 720p, CC licence.
  **But 6 fps** (low for video SR) and many small targets (needs size-filtering);
  and it is **not on HuggingFace** — access is Kaggle/Baidu (Google Drive is
  blocked here), so pulling it needs an account.
- **VideoCube / MGIT** — real video, ~8 min, long gaps, sizeable targets, ~1080p —
  the best pure quality, **but movie-sourced** (no redistribution of a derived
  set).

## 3. Decision — build our own set from the parts that fit, with explicit criteria

No off-the-shelf dataset is clean on all axes, so rather than adopt one wholesale
we **combine qualifying videos across datasets**. Agreed criteria per video:
**480p+ (preferably 720p+), target >1% (preferably 5%), duration >30 s (preferably
1 min+), natural disappear-reappear.**

Current pool against those criteria:
- **LaSOT-ext: 7 videos ready** (30 fps, sizeable targets 1.1–35%, but short gaps).
- **LVOS**: the long-gap source — pending **access** (Kaggle/Baidu account).
- **VideoCube**: best quality — pending a **licence** call.
- **MOSE**: excluded on length.

Sizeable-target example clips are prepared (opossum 35%/10% at natural 30 fps;
MOSE tigers / night-traffic as short disappear-reappear references).

## Key numbers

- LaSOT-ext funnel: **118 → 36 → 3**; longest natural gap **2.5 s**.
- MOSE train: 1,507 videos, longest **500 frames** (< 30 s).
- LVOS v2: **1.14 min** avg, **6 fps**, reappearance gaps ≥ 17 s, 720p, CC.
- VideoCube: ~8 min, ~1080p, movie-licence.
- Qualifying pool so far: **7** (LaSOT) + LVOS / VideoCube pending.

## Decisions needed from the lab

1. **LVOS access** — who has a Kaggle or Baidu account to pull it (the best 1 min+
   natural-gap source)?
2. **VideoCube licence** — acceptable to use movie-sourced footage for an internal
   benchmark (no redistribution), or rule it out?

## One-line summary

No single dataset gives sizeable target + natural disappear-reappear + 1 min+ + a
usable licence; we are building our own by combining qualifying videos — LaSOT
ready (7), LVOS pending access, VideoCube pending a licence call, MOSE excluded for
being too short.
