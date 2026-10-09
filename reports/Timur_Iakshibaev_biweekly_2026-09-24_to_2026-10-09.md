# Lab-Meeting Report — Timur Iakshibaev (recap since the last meeting)

*Audience: lab members + lab lead. The VBench reward-inversion audit and the
VBench-vs-ours comparison were presented last time, so they are not repeated
here. This period was groundwork for the next experiment — a close read of the
most relevant competing paper and selection of the evaluation dataset — so this
report is methodology and dataset detail, not new corruption results.*

## Where we left off (context only, not re-presented)

The five-family audit closed: VBench's `temporal_flickering` / `motion_smoothness`
reward global detail loss (blurrier → higher score, 10/10 cells) and are blind to
gradual/local corruption. The question that set this period's work: **how do we
turn the correspondence claim into a real, quantitative benchmark, and on what
data?**

## 1. Ring Forcing — what we extracted beyond the headline

Last time this was a one-line "external corroboration." Having now read the
evaluation section in detail, the actionable specifics:

- **Their A-D-R benchmark** is 64 controlled cases, each a (appear, disappear,
  reappear) prompt plus a **subject keyword**, with disappearance gaps swept over
  **0 / 1 / 5 / 15 / 30 / 60 s** — a graded gap axis, not a single setting.
- **Reappearance is scored by three complementary matchers**: **SIFT** (local
  texture), **LoFTR** (geometric structure), **DINOv3** (high-level semantics) —
  between the first appearance and the reappearance. Plus a **Qwen3-VL**
  multimodal-LLM judge (5-pt: spatiotemporal consistency / physical logic /
  naturalness) and **AMT motion_smoothness** (the dimension our audit shows is
  invertible — worth noting they lean on it).
- **Dataset is filtered to single-shot clips** ("to isolate temporal
  consistency") from UltraVideo-Long.

**What we adopt (concrete):**
1. The **graded gap axis** (0–60 s) becomes our correspondence severity ladder —
   the temporal analogue of our corruption-severity ladder.
2. **DINOv3 + SIFT + LoFTR as correspondence matchers**, which generalise beyond
   faces — fixes a real limitation (our identity sub-metric is ArcFace, face-only)
   and lets correspondence cover object permanence, not just people.
3. **Single-shot filtering** operationalises the "controlled input consistency"
   we agreed on.
4. **Qwen3-VL as a cheap human-anchor proxy** for the correlation study.

**The differentiator (one sentence):** they generate, so they can only compare
the output's reappearance to the output's own first appearance (self-comparison);
we super-resolve, so the low-quality input gives us a *reference* — same A-D-R
test, with ground truth.

## 2. Evaluation-dataset selection — the real decision

Need: naturally consistent footage, genuine disappear→reappear gaps, long enough,
with known correspondence. Surveyed the current landscape and the latest versions:

| dataset | resolution | length / gaps | licence | notes |
|---|---|---|---|---|
| **LaSOT → LaSOT-ext** (latest) | ~720p (YouTube) | avg ~2500 fr; occlusion-focused, **short gaps (~40 fr ≈ 1.3 s)** | **CC — redistributable** | 1,550 seq; per-frame bbox + absence labels |
| **VideoCube / MGIT** | **~1080p (est.)** | avg **~14,920 fr (~8 min)**; disappearance-rich | **movie-sourced — restricted** | 500 seq; longest, most reappearance; action/activity/story annotations |
| UltraVideo / UVG / Inter4K | 4K | minutes / seconds | varies | single-shot → **no disappear-reappear gap** |
| EventVOT / FELT ("HD") | event-cam | long | — | **event cameras, not RGB** → unusable for SR |

Takeaways: there is **no 4K RGB long-term tracking dataset** (field tops out
~1080p); LaSOT's newest form is **LaSOT-ext** (+150 occlusion-heavy videos); and
the single best "HQ + long + reappearance" RGB option is **VideoCube/MGIT**
(≈1080p estimated from ~190 KB/frame vs LaSOT's ~71 KB, movie-quality source).
*Resolution for VideoCube is an estimate — I'll confirm by pulling a sample.*

**Construction problems identified (these will bite):**
1. **Licensing vs resolution is the fork** — VideoCube is higher-res + longer but
   movie-sourced (can't redistribute a derived benchmark); LaSOT-ext is CC but
   720p.
2. **No pristine HR** — frames are already JPEG/re-encoded; acceptable for a
   *consistency* benchmark (we degrade→SR regardless), weak for fidelity.
3. **Tracking GT is location + single-object** — bbox localises one target, does
   not define its appearance (we use the HR frame); other subjects unlabeled.
4. **Gap duration is content-dependent** — long gaps (toward 60 s) live in
   VideoCube/MGIT, not LaSOT (~1.3 s) → must *select* sequences by gap length.
5. **Domain/metric mismatch** — tracking targets are often non-face → the ArcFace
   identity sub-metric won't apply → use the general matchers from §1.

## 3. Open decision + next steps

1. **Backbone decision:** VideoCube (≈1080p, long, movie-licensed) vs LaSOT-ext
   (720p, CC, redistributable). The licence-vs-resolution trade-off is the call to
   make with the lab.
2. **Pilot:** once the backbone is chosen, pull a few sequences with a clear
   disappear→reappear gap, degrade → super-resolve, and run the reduced-reference
   correspondence score (DINOv3/SIFT/LoFTR, gap-binned) against the tracking
   identity GT — a first quantitative A-D-R-for-SR point, directly comparable to
   Ring Forcing's table.
3. **Scope decision** (formally adopt the reduced-reference framing) remains the
   precondition for the experiment.

## One-line summary

This period was build-up, not results: a detailed read of Ring Forcing gave us a
concrete evaluation recipe (graded gaps + general matchers + single-shot
control), and the dataset search converged on a long-term tracking backbone with
one real fork to decide — VideoCube (HQ, restricted) vs LaSOT-ext (CC, 720p).
