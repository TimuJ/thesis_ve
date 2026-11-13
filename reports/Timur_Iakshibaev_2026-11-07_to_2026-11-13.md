# Lab-Meeting Report — Timur Iakshibaev (recap since the last meeting)

*Audience: lab members + lab lead. This period **produced the first cross-gap
consistency numbers**: the evaluation now scores all three Ring-Forcing axes
individually on the FlashVSR predictions, with a verified caveat about combining them,
plus a curated reproducible example pack for the collaborator. Last meeting only noted
the frame-pair selection tool was in an early state; it was then stress-tested this
period, with a negative result worth stating.*

## Where we left off

Last meeting: the FlashVSR super-resolution predictions were in hand and an **early
frame-pair selection tool** was ready. No consistency numbers yet, and the scorer set
was SIFT + DINO only.

## 1. Ring-Forcing consistency reproduced, now with all three axes (geometry added)

The evaluation now scores each pair on the **three axes Ring Forcing uses,
individually** — SIFT (texture), **LoFTR (geometry, new this period)**, DINO v1
(semantic) — matching their appear-disappear-reappear protocol:

- **Mechanism — what each number is.** For a scorer M we measure the object's cross-gap
  *self-similarity* `c = M(crop_before, crop_after)`. Ring Forcing reports this raw
  self-similarity (they generate video, so there is no reference) — our **`c_sr` column
  is exactly their metric**. Because we run on *real* footage **with ground truth**, we
  add a GT-anchored score `s_M = max(0, 1 − |c_sr − c_gt|)`: the target is not
  "identical" but "the SR changes by the same amount the GT does", so it penalizes
  **both** drift (`c_sr < c_gt`) **and** fabricated over-consistency (`c_sr > c_gt`) —
  which the raw `c_sr` alone cannot.
- **First numbers (29 reviewable pairs, FlashVSR).** The SR **drifts on all three axes**
  (`c_sr < c_gt` everywhere): SIFT `s_M` **0.834** (`c_gt` 0.268 → `c_sr` 0.107), LoFTR
  **0.869** (0.386 → 0.259), DINO **0.923** (0.716 → 0.646). Order: texture worst, then
  geometry, then semantics — the model loses fine texture correspondence first and
  preserves coarse semantics best.
- **Why geometry earns its place.** LoFTR catches failures the other two miss on
  textured objects — e.g. `bus-20`, where SIFT (`s_M` 0.889) and DINO (0.908) both read
  "fine" but LoFTR `c_gt 0.875 → c_sr 0.290` (`s_M` 0.415) exposes a real geometry break.
- Semantic axis is **DINO v1**, not DINOv3; RoMa v2 + DINOv3 are the approved upgrade,
  deferred on weights (so these numbers are a lower bound on what the final trio sees).

## 2. A verified composite "free-pass" — why we report the axes individually

Collapsing the three axes into one score has a failure mode we **verified in code and on
real pairs**:

- **Mechanism.** The composite is a reliability-weighted geometric mean, but the same
  per-pair reliability is handed to **every** axis, so the softmax weights come out
  **uniform**. An axis with *no signal* on a crop (SIFT or LoFTR on a low-texture object:
  `c_gt ≈ 0` → `s_M ≈ 1`) therefore votes at full weight and **inflates** the composite.
- **It is already active.** On `leopard-9`, SIFT is pure noise (`c_gt` 0.003) yet scores
  `s_M` 0.997; it pulls the composite to **0.895** versus **0.804** from the real DINO
  signal alone — nearly flipping the pair across the 0.90 preserve/diverge line. Folding
  geometry in makes it worse: **10 pairs** cross the 0.90 band when LoFTR is added,
  because LoFTR free-passes on the low-texture crops where it finds no matches.
- **Decision.** Report the **three axes individually** (immune to this), keep pair
  grouping on the calibrated 2-axis composite, and **defer** a per-axis-reliability fix
  + band recalibration (the principled fix: weight each axis by its *own* signal, not a
  shared per-pair value). The individual-axis table is valid now; only the single
  composite number is affected.

## 3. A curated reproducible example pack, and a selection-tool reality check

- **Example pack (17 pairs).** Matched before/after-gap pairs for the collaborator to
  reproduce: **8 SR-diverges** (failures — gecko, licenseplate, bus, …) + **9
  SR-preserves** (good pairs vetted by eye), each with per-axis `c_gt / c_sr / s_M`, the
  exact frames/boxes, and the scoring method. Positive and negative examples together so
  the metric's behaviour is checkable both ways, not just on failures.
- **Selection tool — negative result worth stating.** The early tool's weak spot is
  picking un-reviewable pairs (wrong pose, motion blur). I tested cheap CPU signals
  against the ~42 hand-labelled pairs: **optical-flow motion fails** (the tracking box
  cancels the object's own translation, leaving ≈ 0 residual), and **detail/contrast does
  not separate** the target cases (the "too small / dark" rejects are already caught by
  the size gate). This mirrors the earlier **Laplacian-sharpness** negative result. A
  proper no-reference blur metric (re-blur) *does* detect blur, but the **dominant
  residual reject reason is pose**, which points at **geometric pose-matching (RoMa v2)**
  — the same model already approved for the geometry axis — as the real selection lever.

## Key numbers

- Consistency over **29 reviewable pairs** (FlashVSR, GT-anchored `s_M`): **SIFT 0.834 /
  LoFTR 0.869 / DINO 0.923**; raw SR self-similarity `c_sr` **0.107 / 0.259 / 0.646** vs
  GT `c_gt` **0.268 / 0.386 / 0.716** — drift on all three axes, texture worst.
- Composite free-pass: `leopard-9` composite **0.895** (dead SIFT axis included) vs
  **0.804** (real signal only); **10 pairs** flip the 0.90 band when LoFTR is folded in.
- Example pack: **17 pairs** = 8 SR-diverges + 9 SR-preserves, three axes per pair.
- Selection probe: motion/detail signals **do not** track the labels; re-blur detects
  blur but the main reject reason is **pose**.

## Next period (plan)

1. **Per-axis-reliability fix + band recalibration** so geometry can safely enter the
   composite (weight each axis by its own signal; re-anchor the 0.90 band on the 3-axis
   scores).
2. **Stage RoMa v2 + DINOv3** — the approved upgrade — for the geometry/semantic axes
   *and* pose-aware pair selection (one model, both jobs).
3. **Second SR model** so the consistency numbers have a comparison point (currently
   FlashVSR only, so the axes describe one model's behaviour, not a ranking).

## Decisions needed from the lab

1. **Composite vs per-axis** — present the three axes individually (current), or invest
   in the per-axis-reliability fix before relying on any single composite? *Proposal:
   individual axes now, fix next period.*
2. **Selection** — accept RoMa v2 for *selection* (not only scoring), or keep hard gates
   plus a blur metric? *Proposal: RoMa v2 — it does double duty.*

## One-line summary

First cross-gap consistency numbers are in — **FlashVSR drifts on all three Ring-Forcing
axes** (texture worst), the single composite has a verified no-signal "free-pass" so we
report the axes individually for now, and a 17-pair reproducible example pack (failures +
approved good) is ready for the collaborator.
