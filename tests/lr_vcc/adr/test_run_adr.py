"""Task 8: cross-gap consistency orchestration (evaluate_clip).

Deterministic CPU fakes exercise the full pipeline without GPU models:
FakeEmbedder = per-crop mean colour; FakeScorer = SSIM on crops resized to a
common size.
"""
import cv2
import numpy as np

from scripts.lr_vcc.adr.annotations import Annotations
from scripts.lr_vcc.adr.run_adr import evaluate_clip
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
