"""Task 6: SIFT texture scorer (CPU, OpenCV)."""
import cv2
import numpy as np

from scripts.lr_vcc.adr.scorers import SiftTextureScorer


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((128, 128, 3), 128, np.uint8)
    for _ in range(50):
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        p1 = tuple(int(v) for v in rng.integers(0, 128, 2))
        p2 = tuple(int(v) for v in rng.integers(0, 128, 2))
        cv2.rectangle(img, p1, p2, color, -1)
    return img


def test_sift_identical_high_unrelated_low():
    s = SiftTextureScorer()
    img = _texture()
    assert s(img, img) > 0.8
    assert s(img, np.zeros_like(img)) < 0.2
