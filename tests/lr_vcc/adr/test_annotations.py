"""Task 1: LaSOT annotation parsing, present mask, gap detection."""
import numpy as np

from scripts.lr_vcc.adr.annotations import (
    Annotations, present_mask, find_gaps, load_annotations, box_area_frac, box_aspect,
)


def test_present_mask_excludes_absent_and_degenerate():
    ann = Annotations(
        boxes=np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]], float),
        out_of_view=np.array([False, True, False]),
        full_occlusion=np.array([False, False, False]),
        n=3,
    )
    assert present_mask(ann).tolist() == [True, False, False]  # absent, then zero-area


def test_find_gaps_sorted_desc_and_straddleable_only():
    present = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0], bool)  # fps=1
    gaps = find_gaps(present, fps=1.0, min_gap_s=2.0)
    assert [g.duration_s for g in gaps] == [4.0]            # trailing 2-run dropped (no post-run)
    assert (gaps[0].start, gaps[0].end) == (2, 5)
    assert gaps[0].pre_run == (0, 1) and gaps[0].post_run == (6, 8)


def test_find_gaps_orders_by_duration_desc():
    # two straddleable gaps: short (2 frames, idx 2-3) then long (3 frames, idx 6-8)
    present = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1], bool)
    gaps = find_gaps(present, fps=1.0, min_gap_s=2.0)
    assert [g.duration_s for g in gaps] == [3.0, 2.0]
    assert (gaps[0].start, gaps[0].end) == (6, 8)


def test_load_annotations_parses_lines_and_flag_row(tmp_path):
    seq = tmp_path / "seq"
    seq.mkdir()
    (seq / "groundtruth.txt").write_text("1,2,3,4\n5,6,7,8\n9,10,11,12\n")
    (seq / "out_of_view.txt").write_text("0,1,0")
    (seq / "full_occlusion.txt").write_text("0,0,1")
    ann = load_annotations(seq)
    assert ann.n == 3
    assert ann.boxes.shape == (3, 4)
    assert ann.boxes[1].tolist() == [5, 6, 7, 8]
    assert ann.out_of_view.dtype == bool and ann.out_of_view.tolist() == [False, True, False]
    assert ann.full_occlusion.tolist() == [False, False, True]


def test_box_area_frac_is_fraction_of_frame():
    assert box_area_frac((0, 0, 100, 50), frame_w=1000, frame_h=500) == 0.01  # 5000/500000


def test_box_aspect_width_over_height():
    assert box_aspect((0, 0, 30, 10)) == 3.0
