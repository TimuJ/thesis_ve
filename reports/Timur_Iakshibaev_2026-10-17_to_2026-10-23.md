# Lab-Meeting Report — Timur Iakshibaev (recap since the last meeting)

*Audience: lab members + lab lead. This period **reversed last meeting's dataset
choice and replaced it with a measured result**: the frame-rate problem rules LVOS
out as SR footage, and a scan of the full LaSOT gives a 30 fps disappear-reappear
pool, now **built and staged**. Measured numbers throughout; the LVOS curation and
threshold tables shown last time are not repeated.*

## Where we left off

Last meeting: LVOS v2 curated from masks — 64 sizeable-target disappear-reappear
videos (gaps up to 34.7 s), proposed as the **gap core** of a build-our-own set, with
**6 fps** noted only as a caveat. This period that caveat became the deciding factor.

## 1. The 6 fps reckoning — LVOS is out as SR footage

LVOS ships **only every 5th original frame**, and the confirmation is that this is
true for *both* the masks **and** the JPEG frames: the images lie on the same
every-5th-frame grid as the masks (e.g. frame file `00000846.jpg`, with `846 = 1 +
5·169`). There is **no higher-rate source** in the release (`meta.json` has no fps
field; the repo states the footage was *resampled down* from VOT-LT/LaSOT to 6 fps).

Mechanism of why this disqualifies it: our benchmark has two axes — (a) identity/
structure consistency **across the gap** (a before-vs-after comparison, fps-independent)
and (b) **adjacent-frame** temporal quality (flow alignment, flicker/tOF), which needs
≥24 fps. 6 fps serves (a) but makes (b) meaningless, and the only way to raise it is
frame interpolation — **synthetic** motion that would contaminate the very thing we
measure. **Decision: LVOS dropped as experiment footage** (kept as a reference; data
purged locally, metadata + re-download links retained).

## 2. Full LaSOT — a 30 fps disappear-reappear pool (the new result)

We had only ever scanned the 15-class *extension* (where gap and sizeable target are
anti-correlated; max gap 2.5 s). This period we scanned the **full LaSOT — all 1,400
sequences, 70 categories** — from the per-frame `out_of_view` + `full_occlusion`
flags (range-fetched from HuggingFace, no frames downloaded). 0 failures.

Funnel (gap = longest interior absent-run flanked by presence; target = median GT
bbox area):

| criterion | count |
|---|--:|
| gap ≥2 s | 210 |
| + target ≥1 % | 141 |
| **+ ≥480p (the pool)** | **88** (39 distinct categories) |

Within the 88: **46** gap ≥3 s, **27** ≥5 s, **9** ≥10 s; **45** target ≥5 %, **23**
≥10 %; **68** duration ≥60 s. Longest gap **23.6 s** (`lizard-17`); largest target
**29.3 %** (`hippo-12`).

Why the full set works where the extension failed: the 15 extension classes track
either tiny fast objects (gap-rich, ~0 % area) or sizeable objects that never leave —
the two never coincide. The 70-class set contains many categories where a **sizeable**
subject naturally goes behind cover or off-frame and returns, so gap and appearance
are the **same** object.

| | LaSOT-ext | **full LaSOT** | LVOS |
|---|--:|--:|--:|
| sizeable-target disappear-reappear videos | ~6 | **88** | 64 |
| frame rate | 30 fps | **30 fps** | 6 fps |
| longest gap on a sizeable target | 1.3 s | **23.6 s** | 34.7 s |

**Trade accepted:** gaps here are *seconds* not *tens of seconds*, but at 30 fps with
no slideshow artefact. A few-second 30 fps gap is sufficient and preferable to a 35 s
gap at 6 fps for this benchmark.

## 3. VideoCube — checked, shelved

VideoCube/MGIT matches on paper: **~30 fps** (derived, 7.46 M frames / 69.1 h), ~8 min
avg, 500 videos, explicit Instance-Absent gaps. But it is **344 GB (tiny) – 1.4 TB
(full)**, its only download host refused connection from the lab machine, and the
footage is edited film/TV (shot-cuts), non-commercial. → **server-only, licence-gated;
shelved** unless we later need gaps longer than LaSOT offers.

## 4. Built and staged — the pool is ready to run on

The usable set is extracted and encoded, not just listed:

- **Full 88-video pool** → `data/lasot_pool/`: 88 mp4 (720p/30 fps, crf 18) +
  per-video GT (`groundtruth`/`full_occlusion`/`out_of_view`) + `manifest.csv`,
  **8.5 GB**, 0 failures.
- **10-video diverse starter** → `data/upload_10/` for the first experiment run (10
  distinct subjects: lizard, electricfan, monkey, hippo, goldfish, giraffe, lion,
  tank, airplane, spider).
- **Tooling note (worth knowing):** HF per-member reads are **2.4 s/frame** (throttled
  — unusable for bulk); fetching each sequence's **contiguous member byte-range in one
  Range request** runs at **~6 MB/s**. That is the extraction method; it is what made
  8.5 GB feasible on the laptop.
- **Disk:** freed **~47 GB** (purged MOSE, badminton, and LVOS frame data).

## Key numbers

- Full LaSOT scan: **1,400** seqs / **70** cats, 0 fail. Pool **88** (gap ≥2 s &
  target ≥1 % & ≥480p), **39** categories; **27** gap ≥5 s, **9** ≥10 s; longest gap
  **23.6 s**. All *measured* (only fps = 30 is the dataset's stated value).
- Premium subset (gap ≥5 s **and** target ≥5 %): **13** videos.
- Built set: **88 mp4 / 8.5 GB**, GT + manifest; 10-video starter staged.
- LVOS 6 fps confirmed for frames *and* masks → out as SR footage.

## Next week (plan)

1. **Hand the sets to the labmate and run the first experiment** — SR + the LR-VCC
   (reduced-reference) consistency read on the 10-video starter, then widen to the 88.
2. **Lock the pool size with the lab** — ship all 88, or lead with the premium 13
   (gap ≥5 s & target ≥5 %) for headline disappear-reappear results.
3. **Run SR + correspondence server-side** on the chosen pool (lab GPU), measuring
   consistency across each gap against the LQ input.
4. **Decide VideoCube** — only if we want gaps beyond ~24 s; otherwise keep shelved.

## Decisions needed from the lab

1. **Pool scope** — full 88 (broad, diverse) vs. the premium 13 (gap ≥5 s & target
   ≥5 %) as the headline set. Proposal: run the 10-starter now, ship all 88, report
   headline numbers on the premium subset.
2. **Gap-length ambition** — accept LaSOT's few-second 30 fps gaps as the benchmark's
   operating point, or pursue VideoCube (server-only, licence) for longer gaps later?

## One-line summary

The 6 fps problem ruled LVOS out as SR footage; scanning the **full LaSOT** (1,400
sequences) produced **88** sizeable-target disappear-reappear videos at **30 fps**
(gaps to 23.6 s), now **built and staged** (8.5 GB, GT + manifest) with a 10-video
starter ready for the first experiment run.
