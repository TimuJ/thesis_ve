import cv2
import numpy as np
from scripts.rope_probe.make_long_gt import downsample_x4, build_pair


def test_downsample_x4_shape():
    hr = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    lr = downsample_x4(hr)
    assert lr.shape == (180, 320, 3)
    assert lr.dtype == np.uint8


def test_build_pair_crops_to_multiple_of_4_and_counts(tmp_path):
    hr_dir = tmp_path / "hr"; hr_dir.mkdir()
    lr_dir = tmp_path / "lr"
    gt_dir = tmp_path / "gt"
    # 722x1281 → cropped to 720x1280
    for i in range(3):
        img = np.random.randint(0, 256, (722, 1281, 3), dtype=np.uint8)
        cv2.imwrite(str(hr_dir / f"{i:04d}.png"), img)
    n = build_pair(str(hr_dir), str(lr_dir), str(gt_dir))
    assert n == 3
    gt0 = cv2.imread(str(gt_dir / "0000.png"))
    lr0 = cv2.imread(str(lr_dir / "0000.png"))
    assert gt0.shape == (720, 1280, 3)
    assert lr0.shape == (180, 320, 3)
