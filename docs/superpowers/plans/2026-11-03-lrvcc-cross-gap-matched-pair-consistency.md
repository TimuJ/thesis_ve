# LR-VCC Cross-Gap Matched-Pair Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically select, per LaSOT disappear-reappear clip, the most-comparable pre-gap/post-gap object pair (frozen on GT) and score each SR model's cross-gap consistency as deviation from the GT pair's own relationship, plus standard fidelity.

**Architecture:** A new `scripts/lr_vcc/adr/` subpackage. All selection and scoring *logic* is pure (numpy, CPU) and consumes precomputed embeddings/metric values — the project's existing "a GPU script dumps arrays/JSON, the sub-metric logic consumes them" pattern (cf. `scripts/lr_vcc/identity.py`). Heavy models (CLIP selector; SIFT/LoFTR/DINOv3 scorers; SAM crops) sit behind injected `FrameEmbedder` / `PairScorer` interfaces, so orchestration is fully unit-tested on an M1 with deterministic fakes, and the real adapters are validated on the server.

**Tech Stack:** Python ≥3.9, numpy, OpenCV (SIFT, CPU), scikit-image (`src/evaluation/metrics.py`); server-only: torch + open_clip (CLIP), DINOv3, LoFTR (kornia), segment-anything. Reuses `scripts/lr_vcc/composite.compose_score` and `scripts/lr_vcc/reliability.{below,above}_threshold_penalty`.

**Spec:** `docs/superpowers/specs/2026-11-02-lrvcc-cross-gap-matched-pair-consistency-design.md`

## Global Constraints

- Run from repo root; imports `from scripts.lr_vcc.adr.X import ...` and `from src.evaluation.metrics import psnr, ssim, lpips_score`. `tests/conftest.py` already puts repo root on `sys.path`.
- **Frame-index alignment ONLY** — pair LQ↔GT↔SR by frame index, never timestamp or clip index.
- **Single canonical GT = the source mp4 decode**, used for both degradation input and scoring reference.
- **One continuous pass** over the uncut clip for both LQ and SR (never per-segment with state reset).
- **SR must match GT resolution exactly** (×4 = 1280×720); a model on a different grid has the GT box rescaled onto it, and a size mismatch is asserted.
- **Selection embedding is separate from the scorers** (CLIP selects; SIFT/LoFTR/DINOv3 score).
- **Configuration-matched selection:** `cost = cos_dist(E_a,E_b) + λ_area·|log(area_a/area_b)| + λ_aspect·|log(aspect_a/aspect_b)|`.
- **GT-anchored reduced-reference consistency:** `s_M = max(0.0, 1.0 − |c_sr − c_gt|)`.
- **Reliability-weighted composition:** reuse `compose_score`; weak anchors get reduced reliability, never a silent drop.
- **Gap floor `MIN_GAP_S = 3.0`** (FlashVSR ~3 s context window); headline 5.0.
- **Weak-anchor policy:** fall back to the next-longest qualifying gap; if none clears the `c_gt` floor, keep the clip at reduced consistency-reliability; never silent-drop or widen the span.
- Builder-tunable values (`λ_area`, `λ_aspect`, `A_MIN`, `C_GT_FLOOR`, `MAX_WINDOW_OFFSET`, crop size, SIFT/LoFTR→[0,1] mapping, single-vs-top-k, headline unit, CLIP-vs-ReID, Ring-Forcing SR-vs-SR column) live in `adr/config.py` as named constants marked PROVISIONAL; do not hard-code them elsewhere.
- Research-outcome framing in all strings/docstrings/filenames; **no thesis mentions**.
- Commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Review Focus

- **No qualifying straddling gap** (no gap ≥ `MIN_GAP_S` with present frames on both sides): `evaluate_clip` returns status `"no_pair"`, does not crash — test in Task 8; gap-level in Task 1.
- **Alignment mismatch / VFR** (SR frame count ≠ GT, or `r_frame_rate ≠ avg_frame_rate`): the clip is quarantined, never scored — test in Task 2; model-side in Task 8.
- **Degenerate / out-of-bounds bbox** (zero `w`/`h`, or box past the frame edge): `bbox_crop` clamps and never returns an empty/garbage crop; zero-area → `area_frac == 0` → excluded by `A_MIN` — test in Task 5 (+ Task 1 for `present_mask`).
- **All candidates below `A_MIN` on one side**: `choose_pair` returns `i=j=None` (no spurious pick) → weak-anchor fallback — test in Task 3 and Task 4.
- **Over-consistent SR** (a blur makes both sides match more than the real footage, `c_sr > c_gt`): `rr_consistency` penalizes it (`s_M < 1`) and the blur is not ranked best — test in Task 6 and Task 8 (the reductio).

---

### Task 1: Config + annotation parsing and present/gap detection

**Files:**
- Create: `scripts/lr_vcc/adr/__init__.py`, `scripts/lr_vcc/adr/config.py`, `scripts/lr_vcc/adr/annotations.py`
- Test: `tests/lr_vcc/adr/__init__.py`, `tests/lr_vcc/adr/test_annotations.py`

**Interfaces:**
- Produces:
  - `config`: `MIN_GAP_S=3.0`, `HEADLINE_GAP_S=5.0`, `FLASHVSR_WINDOW_S=3.0`; PROVISIONAL `LAMBDA_AREA`, `LAMBDA_ASPECT`, `A_MIN` (area fraction), `C_GT_FLOOR`, `MAX_WINDOW_OFFSET` (frames).
  - `Annotations` dataclass: `boxes: np.ndarray (N,4) float (x,y,w,h)`, `out_of_view: np.ndarray (N,) bool`, `full_occlusion: np.ndarray (N,) bool`, `n: int`.
  - `load_annotations(seq_dir: Path) -> Annotations` — `groundtruth.txt` is one `x,y,w,h` per line; `out_of_view.txt` / `full_occlusion.txt` are a single comma-separated line of `0/1` (length `N`).
  - `present_mask(ann: Annotations) -> np.ndarray (N,) bool` = `~out_of_view & ~full_occlusion & (w>0) & (h>0)`.
  - `box_area_frac(box, frame_w, frame_h) -> float`; `box_aspect(box) -> float` (`w/h`, guarded).
  - `Gap` dataclass: `start:int, end:int` (inclusive absent-run indices), `duration_s:float`, `pre_run:(int,int)`, `post_run:(int,int)` (inclusive present-run bounds adjacent to the gap).
  - `find_gaps(present: np.ndarray, fps: float, min_gap_s: float = MIN_GAP_S) -> list[Gap]` — maximal absent runs with `len/fps ≥ min_gap_s` that have ≥1 present frame before and after; sorted by `duration_s` descending.

- [ ] **Step 1: Write the failing test** in `tests/lr_vcc/adr/test_annotations.py`

```python
def test_present_mask_excludes_absent_and_degenerate():
    ann = Annotations(boxes=np.array([[0,0,10,10],[0,0,10,10],[0,0,0,0]], float),
                      out_of_view=np.array([False, True, False]),
                      full_occlusion=np.array([False, False, False]), n=3)
    assert present_mask(ann).tolist() == [True, False, False]  # absent, then zero-area

def test_find_gaps_sorted_desc_and_straddleable_only():
    present = np.array([1,1,0,0,0,0,1,1,1,0,0], bool)  # fps=1 → gap of 4 (straddled), trailing 2 (not)
    gaps = find_gaps(present, fps=1.0, min_gap_s=2.0)
    assert [g.duration_s for g in gaps] == [4.0]        # trailing gap dropped (no post-run)
    assert (gaps[0].start, gaps[0].end) == (2, 5)
    assert gaps[0].pre_run == (0, 1) and gaps[0].post_run == (6, 8)

def test_load_annotations_parses_lines_and_flag_row(tmp_path):
    # groundtruth: N lines "x,y,w,h"; flags: one comma row of 0/1 length N
    ...  # write 3 gt lines + 3-long flag rows, assert shapes and dtypes
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/lr_vcc/adr/test_annotations.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** `config.py` (constants above) and `annotations.py` (`Annotations`, `load_annotations`, `present_mask`, `box_area_frac`, `box_aspect`, `Gap`, `find_gaps`). `find_gaps`: scan `present` for maximal `False` runs, keep those `≥ min_gap_s` with a present frame at `start-1` and `end+1`, record adjacent maximal present runs, sort by duration desc.
- [ ] **Step 4: Run to verify it passes** — same command → PASS.
- [ ] **Step 5: Commit** — `git add scripts/lr_vcc/adr/__init__.py scripts/lr_vcc/adr/config.py scripts/lr_vcc/adr/annotations.py tests/lr_vcc/adr/` ; message `feat(adr): LaSOT annotation parsing, present mask, gap detection`.

---

### Task 2: Alignment guard

**Files:**
- Create: `scripts/lr_vcc/adr/alignment.py`
- Test: `tests/lr_vcc/adr/test_alignment.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `AlignmentResult` dataclass: `ok: bool`, `reasons: list[str]`.
  - `check_alignment(counts: dict[str,int], r_frame_rate: str, avg_frame_rate: str) -> AlignmentResult` — asserts all values in `counts` (keys e.g. `decoded, groundtruth, out_of_view, full_occlusion, manifest`) are equal, and `r_frame_rate == avg_frame_rate` (constant frame rate); each failure adds a human-readable reason.
  - `probe_counts(mp4: Path) -> tuple[int, str, str]` (decoded frame count + `r_frame_rate` + `avg_frame_rate`) — thin IO over ffprobe/cv2; **contract-tested only** (`pytest.importorskip`-style skip if no decoder), not part of the pure unit set.

- [ ] **Step 1: Write the failing test**

```python
def test_check_alignment_ok_when_all_equal_and_cfr():
    r = check_alignment({"decoded":5601,"groundtruth":5601,"out_of_view":5601,
                         "full_occlusion":5601,"manifest":5601}, "30/1", "30/1")
    assert r.ok and r.reasons == []

def test_check_alignment_flags_count_mismatch():
    r = check_alignment({"decoded":5600,"groundtruth":5601,"out_of_view":5601,
                         "full_occlusion":5601,"manifest":5601}, "30/1", "30/1")
    assert not r.ok and any("decoded" in s for s in r.reasons)

def test_check_alignment_flags_vfr():
    r = check_alignment({"decoded":5601,"groundtruth":5601,"out_of_view":5601,
                         "full_occlusion":5601,"manifest":5601}, "30000/1001", "29970/1000")
    assert not r.ok and any("rate" in s.lower() for s in r.reasons)
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** `check_alignment` + `AlignmentResult`, and `probe_counts` (prefer ffprobe `nb_read_frames`/`r_frame_rate`/`avg_frame_rate`; fall back to cv2 `CAP_PROP_FRAME_COUNT`).
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(adr): per-clip frame-index alignment guard`.

---

### Task 3: Window bounding, selection cost, pair choice

**Files:**
- Create: `scripts/lr_vcc/adr/selection.py`
- Test: `tests/lr_vcc/adr/test_selection.py`

**Interfaces:**
- Consumes: `config` (λ, `A_MIN`, `MAX_WINDOW_OFFSET`), `Gap` (Task 1).
- Produces:
  - `bound_windows(present: np.ndarray, gap: Gap, max_offset: int) -> tuple[list[int], list[int]]` — the contiguous present runs adjacent to `gap.start`/`gap.end`, each truncated to the `max_offset` frames nearest the gap edge.
  - `selection_cost(emb_a, emb_b, area_a, area_b, aspect_a, aspect_b, lambda_area, lambda_aspect) -> float` (formula in Global Constraints; `cos_dist = 1 − cosine_similarity`).
  - `PairChoice` dataclass: `i: int|None`, `j: int|None`, `cost: float`, `n_admissible: int`.
  - `choose_pair(pre_idx, post_idx, embeddings: np.ndarray (N,D), areas: np.ndarray, aspects: np.ndarray, a_min, lambda_area, lambda_aspect) -> PairChoice` — argmin `selection_cost` over `(a,b)` with `areas[a] ≥ a_min` and `areas[b] ≥ a_min`; `i=j=None, n_admissible=0` when none admissible.

- [ ] **Step 1: Write the failing test**

```python
def test_cost_zero_for_identical():
    e = np.array([1.0, 0.0]); assert selection_cost(e,e,0.2,0.2,1.0,1.0,1.0,1.0) == 0.0

def test_cost_rises_with_area_and_aspect_mismatch():
    e = np.array([1.0,0.0])
    base = selection_cost(e,e,0.2,0.2,1.0,1.0,1.0,1.0)
    assert selection_cost(e,e,0.4,0.1,1.0,1.0,1.0,1.0) > base
    assert selection_cost(e,e,0.2,0.2,2.0,0.5,1.0,1.0) > base

def test_choose_pair_picks_min_cost_admissible():
    emb = np.array([[1,0],[0,1],[1,0]], float); areas=np.array([.3,.3,.3]); asp=np.ones(3)
    c = choose_pair([0,1],[2], emb, areas, asp, a_min=0.1, lambda_area=1.0, lambda_aspect=1.0)
    assert (c.i, c.j) == (0, 2)              # 0 matches 2 exactly; 1 is orthogonal

def test_choose_pair_none_when_all_below_a_min_one_side():
    emb=np.zeros((3,2)); areas=np.array([.3,.3,.005]); asp=np.ones(3)
    c = choose_pair([0,1],[2], emb, areas, asp, a_min=0.01, lambda_area=1.0, lambda_aspect=1.0)
    assert c.i is None and c.j is None and c.n_admissible == 0

def test_bound_windows_truncates_to_offset():
    present = np.array([1,1,1,1,0,0,0,1,1,1], bool)  # gap 4..6
    from scripts.lr_vcc.adr.annotations import Gap
    pre, post = bound_windows(present, Gap(4,6,3.0,(0,3),(7,9)), max_offset=2)
    assert pre == [2,3] and post == [7,8]
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** `bound_windows`, `selection_cost`, `PairChoice`, `choose_pair`.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(adr): configuration-matched cross-gap pair selection`.

---

### Task 4: Weak-anchor fallback and anchor reliability

**Files:**
- Create: `scripts/lr_vcc/adr/weak_anchor.py`
- Test: `tests/lr_vcc/adr/test_weak_anchor.py`

**Interfaces:**
- Consumes: `scripts.lr_vcc.reliability.below_threshold_penalty`; `config.C_GT_FLOOR`; `Gap` (Task 1), `PairChoice` (Task 3).
- Produces:
  - `anchor_reliability(c_gt: float, floor: float = C_GT_FLOOR, sharpness: float = 10.0) -> float` = `1 − below_threshold_penalty(c_gt, floor, sharpness)`.
  - `FallbackResult` dataclass: `gap: Gap|None`, `pair: PairChoice|None`, `c_gt: float`, `reliability: float`, `status: str` (`"ok"|"weak"|"no_pair"`).
  - `select_with_fallback(gaps_desc: list[Gap], choose_fn: Callable[[Gap], PairChoice], c_gt_fn: Callable[[Gap, PairChoice], float], floor: float = C_GT_FLOOR) -> FallbackResult` — iterate gaps (already duration-desc); first gap whose `choose_fn` yields a pair with `c_gt_fn ≥ floor` → `status "ok"`, `reliability 1.0`; else remember the highest-`c_gt` admissible pair and return it `status "weak"`, `reliability = anchor_reliability(best_c_gt)`; if no gap yields an admissible pair → `status "no_pair"`.

- [ ] **Step 1: Write the failing test**

```python
def test_first_strong_gap_accepted_full_reliability():
    gaps = [G(10.0), G(4.0)]
    r = select_with_fallback(gaps, choose_fn=lambda g: PAIR, c_gt_fn=lambda g,p: 0.9, floor=0.6)
    assert r.status == "ok" and r.reliability == 1.0 and r.gap is gaps[0]

def test_all_weak_keeps_best_with_reduced_reliability():
    gaps = [G(10.0), G(4.0)]
    cgt = {gaps[0]: 0.3, gaps[1]: 0.5}
    r = select_with_fallback(gaps, lambda g: PAIR, lambda g,p: cgt[g], floor=0.6)
    assert r.status == "weak" and r.gap is gaps[1] and 0.0 < r.reliability < 1.0

def test_no_admissible_pair_anywhere():
    r = select_with_fallback([G(10.0)], lambda g: PairChoice(None,None,float("inf"),0),
                             lambda g,p: 0.0, floor=0.6)
    assert r.status == "no_pair" and r.pair is None
```
(`G(d)` and `PAIR` are small local helpers for a `Gap` of duration `d` and a valid `PairChoice`.)

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** `anchor_reliability`, `FallbackResult`, `select_with_fallback`.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(adr): weak-anchor fallback and reliability weighting`.

---

### Task 5: Crop normalization

**Files:**
- Create: `scripts/lr_vcc/adr/crops.py`
- Test: `tests/lr_vcc/adr/test_crops.py`

**Interfaces:**
- Produces:
  - `bbox_crop(frame: np.ndarray, box: tuple) -> np.ndarray|None` — integer-rounded `x,y,w,h`, clamped to frame bounds; returns `None` for a zero-area/empty result (documented).
  - `resize_crop(crop: np.ndarray, short_side: int) -> np.ndarray` (aspect-preserving).
  - `rescale_box(box: tuple, from_hw: tuple[int,int], to_hw: tuple[int,int]) -> tuple` — scale an `(x,y,w,h)` box from the GT grid onto a model whose output grid differs (`scale_x=to_W/from_W`, `scale_y=to_H/from_H`).
  - `sam_mask_on_white(frame, box, sam_predictor) -> np.ndarray` — Ring-Forcing-style mask-on-white; **adapter** (segment-anything + torch), contract-tested with skip-if-unavailable, validated on server; a fast-follow over `bbox_crop`.

- [ ] **Step 1: Write the failing test**

```python
def test_bbox_crop_matches_slice():
    f = np.arange(100*100*3, dtype=np.uint8).reshape(100,100,3)
    np.testing.assert_array_equal(bbox_crop(f,(10,20,30,40)), f[20:60,10:40])

def test_bbox_crop_clamps_out_of_bounds():
    f = np.zeros((50,50,3), np.uint8)
    c = bbox_crop(f,(40,40,30,30)); assert c is not None and c.shape[0]<=10 and c.shape[1]<=10

def test_bbox_crop_none_on_zero_area():
    assert bbox_crop(np.zeros((50,50,3),np.uint8), (10,10,0,5)) is None

def test_resize_crop_short_side():
    c = resize_crop(np.zeros((40,80,3),np.uint8), 20); assert min(c.shape[:2]) == 20

def test_rescale_box_scales_coords():
    assert rescale_box((10,20,30,40), (100,200), (50,100)) == (5,10,15,20)
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** `bbox_crop`, `resize_crop`, `rescale_box`, and `sam_mask_on_white` (lazy-import torch/SAM inside the function).
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(adr): object crop normalization (bbox + SAM-on-white)`.

---

### Task 6: Reduced-reference consistency, SIFT scorer, fidelity

**Files:**
- Create: `scripts/lr_vcc/adr/consistency.py`, `scripts/lr_vcc/adr/scorers.py`
- Test: `tests/lr_vcc/adr/test_consistency.py`, `tests/lr_vcc/adr/test_scorers_sift.py`

**Interfaces:**
- Consumes: `src.evaluation.metrics.{psnr, ssim, lpips_score, HAS_LPIPS}`; `crops.bbox_crop` (Task 5).
- Produces:
  - `rr_consistency(c_gt: float, c_sr: float) -> float` = `max(0.0, 1.0 − abs(c_sr − c_gt))`.
  - `consistency_scores(c_gt: dict[str,float], c_sr: dict[str,float]) -> dict[str,float]` — `rr_consistency` per shared key (`"sift","loftr","dino"`).
  - `fidelity(sr_a, gt_a, sr_b, gt_b, box_a=None, box_b=None) -> dict` — full-frame and (when boxes given) object-crop `psnr`/`ssim` (+ `lpips` when `HAS_LPIPS`), averaged over the two frames.
  - `PairScorer` Protocol: attribute `name: str`, `__call__(crop_a, crop_b) -> float` in `[0,1]`.
  - `SiftTextureScorer` (in `scorers.py`, CPU OpenCV SIFT) implementing `PairScorer`; `[0,1]` mapping = good-match ratio (PROVISIONAL mapping in `config`).

- [ ] **Step 1: Write the failing test**

```python
def test_rr_consistency_equal_is_one():
    assert rr_consistency(0.8, 0.8) == 1.0

def test_rr_consistency_penalizes_over_consistency():   # Review Focus: blur reductio
    assert rr_consistency(0.7, 0.95) < 1.0 and rr_consistency(0.7, 0.95) == rr_consistency(0.7, 0.45)

def test_rr_consistency_clamps_at_zero():
    assert rr_consistency(0.1, 0.95) >= 0.0

def test_fidelity_identical_is_high_and_reports_crop_and_full():
    a = (np.random.default_rng(0).integers(0,255,(64,64,3))).astype(np.uint8)
    d = fidelity(a, a, a, a, box_a=(8,8,16,16), box_b=(8,8,16,16))
    assert d["psnr_full"] > 50 and "psnr_crop" in d

def test_sift_identical_high_unrelated_low():   # in test_scorers_sift.py (cv2, CPU)
    s = SiftTextureScorer()
    img = cv2.imread(...)  # a small textured fixture committed under tests/lr_vcc/adr/fixtures/
    assert s(img, img) > 0.8
    assert s(img, np.zeros_like(img)) < 0.2
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** `rr_consistency`, `consistency_scores`, `fidelity` (reuse `metrics.psnr/ssim/lpips_score`, crop via `bbox_crop`), `PairScorer`, `SiftTextureScorer`. Commit a tiny textured fixture image for the SIFT test.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `feat(adr): GT-anchored RR consistency, SIFT scorer, dual fidelity`.

---

### Task 7: Model adapters — CLIP embedder, DINOv3 & LoFTR scorers, embedding dump

**Files:**
- Create: `scripts/lr_vcc/adr/embed_clip.py`, `scripts/lr_vcc/adr/scorers_gpu.py`, `scripts/lr_vcc/adr/dump_embeddings.py`
- Test: `tests/lr_vcc/adr/test_adapters_contract.py`

**Interfaces:**
- Consumes: `PairScorer` (Task 6), `crops` (Task 5).
- Produces:
  - `FrameEmbedder` Protocol: `embed(crops: list[np.ndarray]) -> np.ndarray (N, D)`.
  - `ClipEmbedder` implementing `FrameEmbedder` (open_clip ViT; CPU-capable, GPU on server) — the **selection** embedding, deliberately distinct from the scorers.
  - `DinoSemanticScorer`, `LoftrGeometryScorer` implementing `PairScorer` (`name` = `"dino"`, `"loftr"`).
  - `dump_embeddings.py` CLI — decode GT frames, crop (bbox or SAM), embed with `ClipEmbedder`, write `embeddings.npy (N,D)` + `index.json` per clip under `~/data/LRVCC/...`. **Frozen on GT** (GT only).

- [ ] **Step 1: Write the contract test** (torch-gated) in `tests/lr_vcc/adr/test_adapters_contract.py`

```python
torch = pytest.importorskip("torch")
def test_clip_embeds_shape_and_identity():
    emb = ClipEmbedder().embed([CROP, CROP])
    assert emb.shape[0] == 2 and np.isfinite(emb).all()
    cos = emb[0] @ emb[1] / (np.linalg.norm(emb[0])*np.linalg.norm(emb[1]))
    assert cos > 0.99
@pytest.mark.parametrize("Scorer", [DinoSemanticScorer, LoftrGeometryScorer])
def test_scorer_range_and_identity(Scorer):
    s = Scorer(); v = s(CROP, CROP); assert 0.0 <= v <= 1.0 and v > 0.7
```

- [ ] **Step 2: Run to verify it fails** (on a torch box) — FAIL (classes missing); on the M1 it **skips** (no torch), which is expected.
- [ ] **Step 3: Implement** the three adapters (lazy-import torch/open_clip/kornia inside `__init__`/first call; device = cuda if available) and the `dump_embeddings.py` CLI. Keep `embed_clip.py`/`scorers_gpu.py` free of top-level torch imports so the `FrameEmbedder`/`PairScorer` symbols can be imported (for typing) without a GPU.
- [ ] **Step 4: Verify on the server** — run `pytest tests/lr_vcc/adr/test_adapters_contract.py -v` inside the `vsr` env on the A100 box; PASS. **Launch as a background server task.**
- [ ] **Step 5: Commit** — `feat(adr): CLIP selector + DINOv3/LoFTR scorer adapters + GT embedding dump`.

---

### Task 8: Orchestration — `evaluate_clip` and the `run_adr` CLI

**Files:**
- Create: `scripts/lr_vcc/adr/run_adr.py`
- Test: `tests/lr_vcc/adr/test_run_adr.py`

**Interfaces:**
- Consumes: Tasks 1–7 + `scripts.lr_vcc.composite.compose_score`.
- Produces:
  - `evaluate_clip(gt_frames, sr_frames, ann: Annotations, fps: float, manifest_frames: int, embedder: FrameEmbedder, scorers: Sequence[PairScorer], cfg=config) -> dict` — runs `check_alignment` (quarantine → `{"status":"misaligned", ...}`); asserts SR matches GT frame count (else `"misaligned"`) and, for a model whose output grid differs in size but not aspect, rescales the chosen GT box onto the SR grid with `rescale_box` before SR crop/fidelity; `find_gaps`; for each gap `bound_windows` → embed **GT** crops → `choose_pair` (via `select_with_fallback`, `c_gt_fn` = a chosen scorer's GT-pair value); then per scorer compute `c_gt=M(GT_i*,GT_j*)`, `c_sr=M(SR_i*,SR_j*)`; `consistency_scores`; `compose_score(scores=s_M values, reliabilities=[fallback.reliability]*k)`; `fidelity`. Returns a dict with `status` (`ok|weak|no_pair|misaligned`), chosen `(i*,j*)`, `gap`, per-metric `c_gt/c_sr/s_M`, composite consistency, reliability, and fidelity (full + crop).
  - `run_adr` CLI — iterate a clip manifest, write one JSON per clip (mirrors `scripts/lr_vcc/run_lr_vcc.py`). **Concrete adapters are imported lazily inside `main()`**, so `evaluate_clip` and its tests stay torch-free.

- [ ] **Step 1: Write the failing test** with deterministic CPU fakes (`FakeEmbedder` → per-crop mean vector; `FakeScorer` → normalized SSIM in `[0,1]`)

```python
def test_sr_equals_gt_is_perfect():
    out = evaluate_clip(GT, GT, ann, fps=30.0, manifest_frames=len(GT),
                        embedder=FakeEmbedder(), scorers=[FakeScorer("dino")])
    assert out["status"] == "ok"
    assert all(abs(v - 1.0) < 1e-6 for v in out["s_M"].values())

def test_blurred_sr_scores_below_gt_as_sr():          # Review Focus: over-consistency reductio
    good = evaluate_clip(GT, GT, ann, 30.0, len(GT), FakeEmbedder(), [FakeScorer("dino")])
    blur = evaluate_clip(GT, [gaussian(f) for f in GT], ann, 30.0, len(GT), FakeEmbedder(), [FakeScorer("dino")])
    assert blur["consistency"] < good["consistency"]

def test_no_qualifying_gap_returns_no_pair():         # Review Focus: no straddling gap
    out = evaluate_clip(GT_NOGAP, GT_NOGAP, ann_nogap, 30.0, len(GT_NOGAP), FakeEmbedder(), [FakeScorer("dino")])
    assert out["status"] == "no_pair"

def test_sr_frame_count_mismatch_is_quarantined():    # Review Focus: alignment
    out = evaluate_clip(GT, GT[:-1], ann, 30.0, len(GT), FakeEmbedder(), [FakeScorer("dino")])
    assert out["status"] == "misaligned"

def test_selection_is_frozen_on_gt():                 # spec: indices chosen on GT, shared across models
    o1 = evaluate_clip(GT, SR_A, ann, 30.0, len(GT), FakeEmbedder(), [FakeScorer("dino")])
    o2 = evaluate_clip(GT, SR_B, ann, 30.0, len(GT), FakeEmbedder(), [FakeScorer("dino")])
    assert (o1["i"], o1["j"]) == (o2["i"], o2["j"])   # SR never moves the pair

def test_half_resolution_sr_is_handled_via_box_rescale():  # spec Q2: off-grid model
    half = [f[::2, ::2] for f in GT]                  # FakeScorer resizes crops to a common short side
    out = evaluate_clip(GT, half, ann, 30.0, len(GT), FakeEmbedder(), [FakeScorer("dino")])
    assert out["status"] in ("ok", "weak")
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** `evaluate_clip` and the `run_adr` CLI (lazy adapter import in `main()`).
- [ ] **Step 4: Run to verify it passes** — `pytest tests/lr_vcc/adr/test_run_adr.py -v` → PASS (CPU, M1).
- [ ] **Step 5: Commit** — `feat(adr): cross-gap consistency orchestration + run_adr CLI`.

---

## Notes for the executor

- Tasks 1–6 and 8 are fully testable on the M1 (pure logic + CPU SIFT + injected fakes). Task 7 (torch adapters) is validated on the server `vsr` env; its contract test skips without torch.
- Do not import torch adapters at module top level anywhere `evaluate_clip` is reachable — keep them lazy so the pure pipeline and its tests run without a GPU.
- The three remaining spec open-decisions (headline unit per-clip vs per-gap; CLIP vs a dedicated ReID selector; whether to also emit Ring Forcing's GT-free SR-vs-SR column) are config/CLI flags — wire them as options, default to the spec recommendations, and surface the choice rather than hard-coding it.
