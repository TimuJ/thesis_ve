"""Task 8: cross-gap consistency orchestration (evaluate_clip).

Deterministic CPU fakes exercise the full pipeline without GPU models:
FakeEmbedder = per-crop mean colour; FakeScorer = SSIM on crops resized to a
common size.
"""
import cv2
import numpy as np

from scripts.lr_vcc.adr.annotations import Annotations
from scripts.lr_vcc.adr.run_adr import _clip_paths, evaluate_clip, evaluate_clip_pairs
from src.evaluation.metrics import ssim

H = W = 64
BOX = (16.0, 16.0, 24.0, 24.0)
FPS = 1.0


def _frame(pattern):
    img = np.full((H, W, 3), 40, np.uint8)
    if pattern == "P1":
        img[16:40, 16:40:2] = 220        # vertical stripes in the box
    else:
        img[16:40:2, 16:40] = 220        # horizontal stripes in the box
    return img


def _make_gt():
    out = []
    for i in range(20):
        if i < 5:
            out.append(_frame("P1"))                      # pre-gap present
        elif i < 10:
            out.append(np.full((H, W, 3), 40, np.uint8))  # gap 5..9
        else:
            out.append(_frame("P2"))                      # post-gap present
    return out


def _ann_with_gap():
    boxes = np.zeros((20, 4))
    oov = np.zeros(20, bool)
    for i in range(20):
        if 5 <= i < 10:
            oov[i] = True
        else:
            boxes[i] = BOX
    return Annotations(boxes=boxes, out_of_view=oov, full_occlusion=np.zeros(20, bool), n=20)


class FakeEmbedder:
    def embed(self, crops):
        return np.array([np.asarray(c, float).reshape(-1, 3).mean(0) for c in crops])


class FakeScorer:
    def __init__(self, name):
        self.name = name

    def __call__(self, a, b):
        a = cv2.resize(a, (32, 32))
        b = cv2.resize(b, (32, 32))
        return float(max(0.0, ssim(a, b)))


class ConstScorer:
    """A scorer with a fixed name and a controllable constant value."""

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __call__(self, a, b):
        return self.value


GT = _make_gt()
ANN = _ann_with_gap()


def _run(sr):
    return evaluate_clip(GT, sr, ANN, FPS, len(GT), FakeEmbedder(), [FakeScorer("dino")],
                         c_gt_floor=0.0, compute_lpips=False)


def test_sr_equals_gt_is_perfect():
    out = _run(GT)
    assert out["status"] == "ok"
    assert all(abs(v - 1.0) < 1e-6 for v in out["s_M"].values())


def test_blurred_sr_scores_below_gt_as_sr():            # over-consistency reductio
    good = _run(GT)
    blur = _run([cv2.GaussianBlur(f, (7, 7), 3) for f in GT])
    assert blur["consistency"] < good["consistency"]


def test_no_qualifying_gap_returns_no_pair():
    gt = [_frame("P1") for _ in range(8)]
    ann = Annotations(boxes=np.tile(BOX, (8, 1)), out_of_view=np.zeros(8, bool),
                      full_occlusion=np.zeros(8, bool), n=8)
    out = evaluate_clip(gt, gt, ann, FPS, 8, FakeEmbedder(), [FakeScorer("dino")],
                        c_gt_floor=0.0, compute_lpips=False)
    assert out["status"] == "no_pair"


def test_sr_frame_count_mismatch_is_quarantined():
    out = _run(GT[:-1])
    assert out["status"] == "misaligned"


def test_selection_is_frozen_on_gt():
    rng = np.random.default_rng(1)
    sr_a = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(20)]
    sr_b = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(20)]
    o1, o2 = _run(sr_a), _run(sr_b)
    assert (o1["i"], o1["j"]) == (o2["i"], o2["j"])


def test_half_resolution_sr_is_handled():
    half = [f[::2, ::2].copy() for f in GT]
    out = _run(half)
    assert out["status"] in ("ok", "weak")


def test_anchor_gate_uses_named_semantic_scorer_not_list_position():
    # SIFT at list position 0 scores low; the gate must use the named semantic
    # metric (dino), so the clip stays "ok" rather than being demoted to "weak".
    scorers = [ConstScorer("sift", 0.1), ConstScorer("dino", 0.9)]
    out = evaluate_clip(GT, GT, ANN, FPS, len(GT), FakeEmbedder(), scorers,
                        c_gt_floor=0.5, compute_lpips=False)
    assert out["status"] == "ok"


def test_resolution_mismatch_is_flagged():
    half = [f[::2, ::2].copy() for f in GT]
    assert _run(half)["resolution_mismatch"] is True
    assert _run(GT)["resolution_mismatch"] is False


def _make_gt_2gap():
    # present 0-4 (P1), gap 5-9 (5f), present 10-14 (P2), gap 15-18 (4f), present 19-23 (P1)
    out = []
    for i in range(24):
        if 5 <= i < 10 or 15 <= i < 19:
            out.append(np.full((H, W, 3), 40, np.uint8))
        elif 10 <= i < 15:
            out.append(_frame("P2"))
        else:
            out.append(_frame("P1"))
    return out


def _ann_2gap():
    boxes = np.zeros((24, 4))
    oov = np.zeros(24, bool)
    for i in range(24):
        if 5 <= i < 10 or 15 <= i < 19:
            oov[i] = True
        else:
            boxes[i] = BOX
    return Annotations(boxes=boxes, out_of_view=oov, full_occlusion=np.zeros(24, bool), n=24)


def test_evaluate_clip_pairs_covers_all_gaps():
    # top-K across the video, round-robin: with 2 gaps and top_k=4 we get 4 pairs
    # and BOTH gaps are represented (coverage-first), same record shape as evaluate_clip.
    gt, ann = _make_gt_2gap(), _ann_2gap()
    out = evaluate_clip_pairs(gt, gt, ann, FPS, len(gt), FakeEmbedder(),
                              [FakeScorer("dino")], c_gt_floor=0.0, compute_lpips=False, top_k=4)
    assert out["status"] == "ok"
    assert len(out["pairs"]) == 4
    assert len({p["gap_s"] for p in out["pairs"]}) == 2  # both gaps covered
    for p in out["pairs"]:
        assert set(p) >= {"i", "j", "c_gt", "c_sr", "s_M", "consistency", "fidelity"}


def test_evaluate_clip_pairs_multi_per_gap():
    # a single gap yields multiple DISTINCT pairs (sliding pre/post offsets)
    out = evaluate_clip_pairs(GT, GT, ANN, FPS, len(GT), FakeEmbedder(),
                              [FakeScorer("dino")], c_gt_floor=0.0, compute_lpips=False, top_k=3)
    adr = [p for p in out["pairs"] if p["pair_type"] == "adr"]
    assert len(adr) == 3
    assert len({(p["i"], p["j"]) for p in adr}) == 3     # distinct pairs
    assert len({p["gap_s"] for p in adr}) == 1           # all from the one gap


def test_evaluate_clip_pairs_pose_gate_rejects_all_is_graceful():
    # a pose gate scoring every pair below threshold admits no gap pair -> no adr
    # pairs forced in (reviewability beats count), and no exception.
    reject_all = lambda a, b: 0.0
    out = evaluate_clip_pairs(GT, GT, ANN, FPS, len(GT), FakeEmbedder(),
                              [FakeScorer("dino")], c_gt_floor=0.0, compute_lpips=False,
                              pose_gate=reject_all)
    assert [p for p in out["pairs"] if p["pair_type"] == "adr"] == []


def test_evaluate_clip_pairs_quarantines_like_evaluate_clip():
    out = evaluate_clip_pairs(GT, GT[:-1], ANN, FPS, len(GT), FakeEmbedder(),
                              [FakeScorer("dino")], c_gt_floor=0.0, compute_lpips=False)
    assert out["status"] == "misaligned" and out["pairs"] == []


def test_evaluate_clip_pairs_headline_matches_single_pair_eval():
    # the best (lowest-cost) pair of the multi-pair path is the same pair the
    # single-pair evaluator picks.
    single = _run(GT)
    multi = evaluate_clip_pairs(GT, GT, ANN, FPS, len(GT), FakeEmbedder(),
                                [FakeScorer("dino")], c_gt_floor=0.0, compute_lpips=False)
    assert multi["pairs"][0]["pair_type"] == "adr"
    assert (multi["pairs"][0]["i"], multi["pairs"][0]["j"]) == (single["i"], single["j"])


BIGBOX = (8.0, 8.0, 48.0, 48.0)      # ~56% of a 64x64 frame
SMALLBOX = (16.0, 16.0, 10.0, 10.0)  # ~2.4%


def _big_frame():
    img = np.full((H, W, 3), 40, np.uint8); img[8:56, 8:56] = 200; return img


def _small_frame():
    img = np.full((H, W, 3), 40, np.uint8); img[16:26, 16:26] = 150; return img


def _make_gt_big_ends():
    out = []
    for i in range(30):
        if i < 5 or i >= 25:
            out.append(_big_frame())                      # big object at start & end
        elif 10 <= i < 15:
            out.append(np.full((H, W, 3), 40, np.uint8))  # gap 10..14
        else:
            out.append(_small_frame())                    # small object around the gap
    return out


def _ann_big_ends():
    boxes = np.zeros((30, 4)); oov = np.zeros(30, bool)
    for i in range(30):
        if 10 <= i < 15:
            oov[i] = True
        elif i < 5 or i >= 25:
            boxes[i] = BIGBOX
        else:
            boxes[i] = SMALLBOX
    return Annotations(boxes=boxes, out_of_view=oov, full_occlusion=np.zeros(30, bool), n=30)


def test_evaluate_clip_pairs_adds_big_long_range_pair_when_gap_object_is_small():
    # The object is big only at the clip's start and end (no disappearance there);
    # the only gap sits where the object is small. Option B must add one big
    # long-range pair on the end frames, at reduced reliability, while the A-D-R
    # pair stays on the small object near the gap. max_offset=2 keeps the gap window
    # off the big ends.
    gt, ann = _make_gt_big_ends(), _ann_big_ends()
    out = evaluate_clip_pairs(gt, gt, ann, FPS, len(gt), FakeEmbedder(), [FakeScorer("dino")],
                              c_gt_floor=0.0, compute_lpips=False, max_offset=2)
    adr = [p for p in out["pairs"] if p["pair_type"] == "adr"]
    lr = [p for p in out["pairs"] if p["pair_type"] == "long_range"]
    assert len(adr) >= 1 and len(lr) == 1
    assert lr[0]["i"] < 5 and lr[0]["j"] >= 25            # uses the big start & end frames
    assert lr[0]["reliability"] < 1.0                     # non-gap pair -> low reliability
    assert adr[0]["i"] in (8, 9) and adr[0]["j"] in (15, 16)  # gap pair stays on the small object


def test_evaluate_clip_pairs_no_gap_big_object_yields_long_range_pair():
    # Object present in EVERY frame (never disappears -> NO qualifying gap), big
    # throughout. Option B must still yield one temporally-separated long-range pair
    # instead of discarding the clip as no_pair (the hippo case: big object, no gap).
    gt = [_big_frame() for _ in range(30)]
    ann = Annotations(boxes=np.tile(BIGBOX, (30, 1)), out_of_view=np.zeros(30, bool),
                      full_occlusion=np.zeros(30, bool), n=30)
    out = evaluate_clip_pairs(gt, gt, ann, FPS, len(gt), FakeEmbedder(), [FakeScorer("dino")],
                              c_gt_floor=0.0, compute_lpips=False)
    adr = [p for p in out["pairs"] if p["pair_type"] == "adr"]
    lr = [p for p in out["pairs"] if p["pair_type"] == "long_range"]
    assert out["status"] == "ok"
    assert len(adr) == 0 and len(lr) == 1                 # no gap -> only the long-range pair
    assert abs(lr[0]["i"] - lr[0]["j"]) >= 3              # temporally separated
    assert lr[0]["reliability"] < 1.0                     # non-gap pair -> low reliability


def test_small_object_pair_gets_low_reliability():
    # A ~2.4%-area object (below BIG_AREA_FRAC=5%) is kept as a low-reliability
    # example even with a perfect anchor.
    gt = ([_small_frame() for _ in range(5)]
          + [np.full((H, W, 3), 40, np.uint8) for _ in range(5)]
          + [_small_frame() for _ in range(5)])
    boxes = np.zeros((15, 4)); oov = np.zeros(15, bool)
    for i in range(15):
        if 5 <= i < 10: oov[i] = True
        else: boxes[i] = SMALLBOX
    ann = Annotations(boxes=boxes, out_of_view=oov, full_occlusion=np.zeros(15, bool), n=15)
    out = evaluate_clip(gt, gt, ann, FPS, 15, FakeEmbedder(), [FakeScorer("dino")],
                        c_gt_floor=0.0, compute_lpips=False)
    assert out["status"] == "ok" and out["reliability"] < 0.6   # small object -> reduced


def _noisy_box_frame(seed):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 40, np.uint8)
    img[16:40, 16:40] = rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)
    return img


def _textured_gap_clip(soft_window):
    """A gap clip whose gap-adjacent (windowed) frames are optionally blurred, while
    the far present frames stay sharp -- so the clip median stays sharp and the
    forced-on-window pair is measurably softer than the clip."""
    frames = []
    for i in range(20):
        if 5 <= i < 10:
            frames.append(np.full((H, W, 3), 40, np.uint8))        # gap 5..9
            continue
        f = _noisy_box_frame(i)
        if soft_window and i in (3, 4, 10, 11):                    # the gap window
            f = cv2.GaussianBlur(f, (9, 9), 4)
        frames.append(f)
    boxes = np.zeros((20, 4)); oov = np.zeros(20, bool)
    for i in range(20):
        if 5 <= i < 10:
            oov[i] = True
        else:
            boxes[i] = BOX
    return frames, Annotations(boxes=boxes, out_of_view=oov,
                               full_occlusion=np.zeros(20, bool), n=20)


def test_reliability_record_includes_quality_factors():
    out = _run(GT)
    assert set(out["reliability_factors"]) >= {"size", "sharp", "edge", "anchor"}


class _GtSourcedRelScorer:
    name = "dino"

    def __call__(self, a, b):
        return 0.9

    def reliability(self, a, b):
        return 1.0 if np.asarray(a).max() > 200 else 0.0   # GT crop is bright; SR here is flat


def test_reliability_is_gt_sourced():
    # SR is flat gray (no bright pixels); GT has bright stripes. The recorded
    # axis_reliability must reflect the GT crop (bright), proving reliability is
    # computed on GT, not SR -- else an over-smoothed SR would hide its own failure.
    gray = [np.full((H, W, 3), 40, np.uint8) for _ in range(20)]
    out = evaluate_clip(GT, gray, ANN, FPS, 20, FakeEmbedder(), [_GtSourcedRelScorer()],
                        c_gt_floor=0.0, compute_lpips=False)
    assert out["axis_reliability"]["dino"] == 1.0


def test_blurry_window_pair_gets_lower_reliability_than_sharp():
    fs, ann_s = _textured_gap_clip(soft_window=False)
    fb, ann_b = _textured_gap_clip(soft_window=True)
    rs = evaluate_clip(fs, fs, ann_s, FPS, 20, FakeEmbedder(), [FakeScorer("dino")],
                       c_gt_floor=0.0, compute_lpips=False, max_offset=2)
    rb = evaluate_clip(fb, fb, ann_b, FPS, 20, FakeEmbedder(), [FakeScorer("dino")],
                       c_gt_floor=0.0, compute_lpips=False, max_offset=2)
    assert rb["reliability_factors"]["sharp"] < 1.0        # blurred crop flagged soft
    assert rb["reliability"] < rs["reliability"] - 1e-9    # and that lowers reliability


def test_clip_paths_resolves_bare_and_extensioned_manifest_names():
    # The real manifest's 'video' column holds bare clip names ("leopard-9");
    # GT/SR decode needs the .mp4 extension, annotations need the bare dir name.
    assert _clip_paths("leopard-9", "gt", "sr", "ann") == (
        "gt/leopard-9.mp4", "sr/leopard-9.mp4", "ann/leopard-9")
    # A manifest that already carries the extension resolves to the same paths.
    assert _clip_paths("leopard-9.mp4", "gt", "sr", "ann") == (
        "gt/leopard-9.mp4", "sr/leopard-9.mp4", "ann/leopard-9")


def test_offframe_box_candidate_is_not_selected_and_does_not_crash():
    # Frame 0 is present but its box is entirely off-frame (nominal area >= A_MIN,
    # yet uncroppable). It must never be selected into the pair nor reach a scorer.
    frames = ([_frame("P1")]
              + [np.full((H, W, 3), 40, np.uint8) for _ in range(5)]
              + [_frame("P2") for _ in range(4)])
    boxes = np.zeros((10, 4))
    boxes[0] = (1000.0, 1000.0, 24.0, 24.0)       # fully off a 64x64 frame
    for i in range(6, 10):
        boxes[i] = BOX
    oov = np.zeros(10, bool)
    oov[1:6] = True
    ann = Annotations(boxes=boxes, out_of_view=oov, full_occlusion=np.zeros(10, bool), n=10)
    out = evaluate_clip(frames, frames, ann, FPS, 10, FakeEmbedder(), [FakeScorer("dino")],
                        c_gt_floor=0.0, compute_lpips=False)
    assert out["status"] == "no_pair"
