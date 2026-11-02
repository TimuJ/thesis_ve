"""Task 5: object crop normalization (bbox crop, resize, box rescale)."""
import numpy as np

from scripts.lr_vcc.adr.crops import bbox_crop, resize_crop, rescale_box


def test_bbox_crop_matches_slice():
    f = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    np.testing.assert_array_equal(bbox_crop(f, (10, 20, 30, 40)), f[20:60, 10:40])


def test_bbox_crop_clamps_out_of_bounds():
    f = np.zeros((50, 50, 3), np.uint8)
    c = bbox_crop(f, (40, 40, 30, 30))
    assert c is not None and c.shape[0] <= 10 and c.shape[1] <= 10


def test_bbox_crop_none_on_zero_area():
    assert bbox_crop(np.zeros((50, 50, 3), np.uint8), (10, 10, 0, 5)) is None


def test_resize_crop_short_side():
    c = resize_crop(np.zeros((40, 80, 3), np.uint8), 20)
    assert min(c.shape[:2]) == 20


def test_rescale_box_scales_coords():
    assert rescale_box((10, 20, 30, 40), (100, 200), (50, 100)) == (5, 10, 15, 20)
