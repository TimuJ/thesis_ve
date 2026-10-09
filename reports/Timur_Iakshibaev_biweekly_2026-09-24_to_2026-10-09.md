# Lab-Meeting Report — Timur Iakshibaev (recap since the last meeting)

*Audience: lab members + lab lead. Written for technical discussion — mechanisms
and numbers are included so the claims can be interrogated.*

## Three headlines

1. **The VBench consistency-dimension audit is closed with a measured reward
   inversion:** VBench's `temporal_flickering` and `motion_smoothness` rate a
   *more* blurred (detail-destroyed) video as *more* consistent — the exact
   failure that matters for super-resolution.
2. **We placed the work against a new, directly competing paper** (Ring Forcing,
   long-video generation) — it independently builds the appear-disappear-reappear
   benchmark we were designing, which both validates the direction and hands us a
   clean differentiator.
3. **The evaluation-dataset direction is locked** (long-term *tracking* data as
   the correspondence backbone) and the construction problems are mapped.

## 1. VBench temporal/motion audit — full five-family result

**Setup.** Our synthetic corruption battery: 5 base videos × 5 severities
(0.02, 0.05, 0.10, 0.20, 0.40 — a 20× range) × corruption family. Each corrupted
video is scored by VBench's long-video extension (`eval_long.py
--mode long_custom_input`), which splits a ~2.8-min video into ~84 two-second
clips (≈1885 clips/family) and we aggregate per-clip scores to a per-(base,
severity) mean. **These are VBench 1.x dimensions**, not 2.0:
`temporal_flickering` = mean adjacent-frame MAE; `motion_smoothness` = AMT-S
frame-interpolation error. A corruption-sensitive dimension should *fall* as
severity rises.

**Result (mean score-spread across the 20× ladder, and direction):**

| family | temporal_flickering | motion_smoothness | behaviour |
|---|---|---|---|
| color_drift (gradual) | 0.0004 | 0.0002 | **blind** (flat) |
| background_drift (gradual) | 0.0016 | 0.0012 | **blind** |
| identity_degradation (face-local blur) | 0.0001 | 0.0001 | **blind** |
| flicker (fast global, period≈0.5 s) | 0.0257 ↓ | 0.0053 ↓ | correct (falls) |
| **global_blur (whole-frame detail loss)** | **+0.0026 ↑** | **+0.0021 ↑** | **inverted (rises)** |

Scores live near a ceiling (~0.98–0.996), so magnitudes are small; the *signs*
are the finding, and they are consistent (global_blur rises on **10/10
base×dimension cells**).

**Mechanisms (for the discussion):**
- *Blind to gradual drift* — `temporal_flickering` only sees adjacent-frame
  differences; a drift slower than ~2 frames is averaged away. Confirmed on
  color/background drift.
- *Correct on flicker* — a 0.5 s-period brightness oscillation does move
  adjacent-frame MAE, so the dimension responds; this is the one thing it was
  built for, and SR models do **not** produce it.
- *Inverted on global blur* — blur removes the high-frequency content that drives
  both adjacent-frame MAE (↓ → higher flicker score) and AMT interpolation error
  (blurred frames interpolate more easily → higher smoothness score). So
  destroying detail *raises* both scores. This is the temporal/motion analogue of
  the identity reductio we found earlier (the degraded LQ input scores highest on
  within-clip self-similarity).

**Why it matters:** global detail loss is precisely the dominant failure of a
weak SR model, so these VBench dimensions would rank the *worse* SR method as
more temporally consistent and smoother. On `color_drift`, our input-anchored
colour sub-metric (D′) responds on 4/5 bases where both VBench dims are flat.
(Note: motion_smoothness needed two env fixes to run — AMT checkpoint fetched via
HF mirror, and `omegaconf`/`einops` installed; recorded for reproducibility.)
Figure: `reports/figures/vbench_reward_audit.png` (2×5).

## 2. Ring Forcing (arXiv 2608.26794) — where we sit

A long-video *generation* paper (Lvmin Zhang / Agrawala group) that builds an
**Appear–Disappear–Reappear (A-D-R) benchmark**: 64 controlled cases, each with
an appear/disappear/reappear prompt and a subject keyword, disappearance gaps
swept over 0/1/5/15/30/60 s. Reappearance consistency is scored by **feature
matching — SIFT (texture), LoFTR (geometry), DINOv3 (semantics)** — plus a
**Qwen3-VL** LLM-judge and **AMT motion_smoothness** (the dimension we just
showed is invertible). Training data is **UltraVideo-Long filtered to
single-shot** clips to isolate temporal consistency.

This is the generation-side twin of our correspondence problem — it confirms the
A-D-R framing is a recognised, benchmarkable question. **Our differentiator:**
generation has no reference, so they must compare the output's reappearance to
the output's *own* first appearance (self-comparison, the weakness our audit
documents). Super-resolution has the low-quality input, so we can run the same
A-D-R test **reduced-reference** — establish who-is-who and what-it-looked-like
from the input, then score whether the SR output preserved it across the gap.
Same benchmark shape, with ground truth.

Consolidated the full case into one note
(`docs/notes/2026-09-18-vbench-vs-lrvcc-comparison.md`): half 1 = what VBench
misses/inverts; half 2 = what the input-referenced metric adds (applicability
differencing collapses a legitimate-pan false positive 6–23×; input-anchored
clustering links an identity across a ~2-min absence; the self-similarity
reductio).

## 3. Evaluation-dataset selection + construction problems

To make the correspondence test *quantitative* we need footage that is naturally
consistent, has real disappear→reappear gaps, is long enough, and carries known
correspondence. Candidates surveyed:

| dataset | resolution | length / gaps | licence | note |
|---|---|---|---|---|
| **LaSOT / LaSOT-ext** | ~720p (YouTube) | avg ~2500 fr; occlusion-focused, short gaps (~40 fr) | **CC (redistributable)** | 1,550 seq; per-frame bbox + absence |
| **VideoCube / MGIT** | ~1080p (est.) | avg **~14,920 fr** (~8 min); rich disappearance | **movie-sourced (restricted)** | 500 seq; longest + most reappearance |
| UltraVideo / UVG | 4K | minutes / seconds | varies | single-shot → **no gaps** |
| EventVOT / FELT | "HD" | long | — | **event-camera, not RGB** — unusable for SR |

There is **no 4K RGB long-term tracking set**; the field tops out ~1080p.

**Construction problems flagged (the ones that will bite):**
1. **Licensing vs resolution is the fork** — VideoCube is higher-res + longer but
   movie-sourced (can't redistribute a derived benchmark); LaSOT-ext is CC but
   720p.
2. **No pristine HR** — frames are already JPEG/re-encoded; fine for a
   *consistency* benchmark (we degrade→SR regardless), weak for fidelity.
3. **Tracking GT = location, single-object** — the bbox localises one target and
   doesn't define its appearance (we use the HR frame); other subjects unlabeled.
4. **Gap duration is content-dependent** — LaSOT gaps are short (~1.3 s);
   long gaps (toward Ring Forcing's 60 s) live in VideoCube/MGIT → must *select*
   by gap length.
5. **Domain/metric mismatch** — tracking targets are often non-face (animals,
   vehicles); our ArcFace identity sub-metric won't apply → use general matchers
   (DINOv3/SIFT/LoFTR), as Ring Forcing does.

## Key numbers (for Q&A)

- VBench inversion: global_blur Δ = **+0.0026 (TF) / +0.0021 (MS)**, **10/10**
  base×dim cells rising; flicker Δ = **−0.0257 (TF)** (the one correct response).
- Reduced-reference applicability: legitimate-pan false positive **↓ 6–23×** under
  input-differencing; real corruption retained.
- Correspondence: identity linked across a **~2-minute** absence from the LQ
  alone (29 clean clusters, one base).
- Candidate datasets: LaSOT-ext **1,550** seq @ ~720p CC; VideoCube **500** seq,
  avg **~14,920** frames @ ~1080p, movie-licensed.

## Infrastructure note

The shared GPU box spent stretches refusing SSH (`kex … Connection reset` — sshd
throttling under load ≈10 with ~60 users). Jobs run detached and completed on
schedule, so nothing was lost, but it delayed result retrieval by days. Flagging
as a shared-machine pace-limiter.

## Next two weeks

1. **Pilot the correspondence backbone:** pull a few long sequences with a clear
   disappear→reappear gap (VideoCube for length / LaSOT-ext for licence),
   degrade → super-resolve, run the reduced-reference correspondence score against
   the tracking identity GT — a first quantitative A-D-R-for-SR point.
2. **Settle the backbone decision** (VideoCube 1080p-but-restricted vs LaSOT-ext
   720p-but-CC) and the reduced-reference **scope decision** (precondition for the
   experiment).
3. Optionally confirm the blur inversion is blur-general by adding a
   downscale–upscale detail-loss family.

## One-line summary

VBench's consistency dimensions provably reward detail loss; we have the
input-referenced fix and the external (Ring Forcing) validation; the next step is
a real disappear-reappear dataset built on long-term tracking footage.
