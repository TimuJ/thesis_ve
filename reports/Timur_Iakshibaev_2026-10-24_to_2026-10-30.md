# Lab-Meeting Report — Timur Iakshibaev (recap since the last meeting)

*Audience: lab members + lab lead. This period is **design + dataset refinement**, not
new SR numbers: the gap threshold is now pinned to the enhancer's **measured context
window**, and **scale variation** became the primary ranking axis. The degradation →
SR → consistency run is set up for next week. Last meeting's LVOS/VideoCube material and
the full-LaSOT scan funnel are not repeated.*

## Where we left off

Last meeting: the full-LaSOT scan gave an **88-video, 30 fps** disappear-reappear pool
(gap ≥ 2 s, target ≥ 1 %, ≥ 480p), built and staged with a 10-video starter. Open
questions were the pool scope and whether a few-second gap is "enough."

## 1. The gap threshold is now set by the enhancer's context window (2 s → 3 s)

The 2 s gap floor from last meeting was arbitrary. This period we derived what it should
be from the model that will actually run on the data — **FlashVSR** (arXiv 2510.12747):

- FlashVSR is a **causal streaming** model with a **sliding-window KV-cache**, not a
  fixed-N-frame model. Its temporal memory is bounded by its **85-frame** full-attention
  baseline and **89-frame** training clips, with only 8-frame lookahead → **context
  window ≈ 85–89 frames ≈ 3.0 s at 30 fps** (the 85/89 are the paper's stated values; the
  ~3 s is those ÷ 30 fps).
- **Mechanism — why this is the right threshold:** a subject can stay consistent across
  the gap two ways: (a) **local temporal propagation** through the KV-cache, or (b)
  **genuine long-range identity memory**. If the gap is *shorter* than the context
  window, route (a) bridges it and we are measuring cache carryover, not memory. Setting
  **gap > context window** removes route (a) and leaves (b) as the only path — which is
  exactly the capability a 3 s-window streaming model structurally lacks. That is the
  effect we want the benchmark to expose.
- **Action:** gap floor raised **2 s → 3 s**; pool re-filtered **88 → 46** videos; the
  headline band is **gap ≥ 5 s (27 videos, ~1.7× the window)** for unambiguous margin.
  The floor is now a *function of the enhancer* — swap in a longer-context model and it
  must rise with it.

## 2. Scale variation is the new primary ranking key

A new axis this period, on the LQ side. Each video is ranked by **how much the subject
changes size** over the clip:

- **Why:** when the subject is large (zoomed-in) at some points and small (zoomed-out) at
  others, the zoomed-out appearance's LQ genuinely *lacks* the fine detail the zoomed-in
  one carries (×4 degradation on a small object leaves almost nothing). So the model
  cannot reuse detail across the scale change by reading the LQ — it has to carry
  identity. This is a **degradation-side stressor that complements the gap**.
- **Metric: `area_change` = `area_max / max(area_min, 1 %)`** — the 95th / 5th percentile
  of GT bbox area (% of frame) over *present* frames, with the denominator floored at
  1 %. The floor is deliberate: below ~1 % of a 720p frame the object is < ~24 px in ×4
  LQ — a second disappearance, not a usable zoomed-out view — so we don't reward going
  smaller. All measured from GT we already hold; it is now a manifest column.
- **Secondary flag `min_at`** — where in the clip the subject is smallest; `0.15–0.85`
  marks the big→small→big **interior dip** (e.g. `bus-20`), which **22 of 46** have. A
  pattern flag, not the sort key.
- Pool range: **50.2** (`leopard-9`, 0.35 → 50 %) down to **1.7** (`electricfan-16`).

## 3. The LQ design — differential degradation (decided, not yet run)

The main design output of the period; it sets the "LQ side" of the benchmark:

- With gap > window (§1), cross-gap consistency can only come from long-range memory —
  **but** if both appearances get the *identical* degradation, a memoryless model can
  still look consistent via "same object + same degradation → similar LQ → similar
  output." That is an input→output coincidence, not memory, and it flatters the model.
- **Differential degradation** (a different degradation realization before vs after the
  gap) removes that shortcut: the LQ inputs genuinely differ, so a memoryless model's
  hallucinated detail **diverges** (measurably inconsistent) while a memory-capable one
  stays locked. It is also realistic — a subject filmed before and after an occlusion
  really does carry different compression / blur / scale.
- Two conditions to run: **(a) same severity, different realization** (isolates
  consistency) and **(b) asymmetric good→bad** (tests whether the model lifts the
  degraded later view using the better earlier one — reference-based restoration). The
  GT anchors both: the subject is identical across the gap, so any divergence is the
  model, not the input. The scale-variation ranking (§2) is where the asymmetry lands
  naturally — the zoomed-out appearance *is* the harder one.

## 4. Starter rebuilt; degradation infra checked (build-up for next week)

- **18-video diverse starter** → `data/starter/` (renamed from the 10-set): one subject
  per class, all gap ≥ 5 s, ranked by `area_change`, and now including the scale-mismatch
  exemplars `bus-20` and `licenseplate-15`. 18 = every class that has a gap ≥ 5 s clip
  (the one-per-class rule caps there, so "top 20 diverse" is 18 in practice).
- **Server checked for the degradation step:** reachable (2× A100-40 GB, both busy —
  10.6 / 4.5 GB free), but the RealBasicVSR degradation deps (`basicsr` / `mmcv`) are
  **absent from every env**, there is **no ffmpeg** on PATH, the pool **isn't staged**,
  and the shared disk is **99 % full (66 GB free)**. Degradation is CPU-only with tiny
  output, so the likely call is to **run it locally** and push only the LQ to the server
  for the GPU SR pass.

## Key numbers

- FlashVSR context window **≈ 85–89 frames ≈ 3.0 s @ 30 fps** (85/89 from the paper, 3 s
  derived). Gap floor **3 s** → pool **46** (was 88 at the 2 s floor); **27** at gap ≥ 5 s.
- Scale variation `area_change`: pool range **50.2 → 1.7**; **22/46** have the interior
  big→small→big dip.
- Starter **18 videos** (`data/starter/`, 1.7 GB), one per class, all gap ≥ 5 s.
- Degradation = RealBasicVSR two-stage **×4** (existing script); server deps missing,
  disk 66 GB free.

## Next week (plan)

1. **Run the LQ degradation** (RealBasicVSR ×4) on the 46-pool / 18-starter — decide
   local vs server, and implement **per-segment degradation** (a different seed/severity
   before vs after the gap boundary, which we already have from the annotations).
2. **First FlashVSR SR pass + consistency read** on the starter: degrade → super-resolve
   → score subject preservation across each gap against GT.
3. **Settle the degradation conditions** — uniform (control) / symmetric-different /
   asymmetric good→bad.

## Decisions needed from the lab

1. **Degradation recipe + conditions** — RealBasicVSR ×4 (matches FlashVSR's training) as
   the base; which conditions to run first. *Proposal: uniform as control + asymmetric
   good→bad as the headline.*
2. **Run location** — local (CPU, sidesteps the server's missing deps and 99 %-full disk)
   vs server. *Proposal: degrade locally, SR on the lab GPU.*
3. **Starter size** — keep 18, or pad to a literal 20 (two second-of-class clips).

## One-line summary

The gap threshold is now **pinned to FlashVSR's measured ~3 s context window** (pool
re-filtered 88 → 46) and **scale variation** became the primary ranking axis — setting up
a differential-degradation benchmark whose first SR + consistency numbers come next week.
