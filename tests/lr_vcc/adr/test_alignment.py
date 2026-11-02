"""Task 2: per-clip frame-index alignment guard."""
from scripts.lr_vcc.adr.alignment import check_alignment


def _counts(**over):
    c = {"decoded": 5601, "groundtruth": 5601, "out_of_view": 5601,
         "full_occlusion": 5601, "manifest": 5601}
    c.update(over)
    return c


def test_check_alignment_ok_when_all_equal_and_cfr():
    r = check_alignment(_counts(), "30/1", "30/1")
    assert r.ok and r.reasons == []


def test_check_alignment_flags_count_mismatch():
    r = check_alignment(_counts(decoded=5600), "30/1", "30/1")
    assert not r.ok and any("decoded" in s for s in r.reasons)


def test_check_alignment_flags_vfr():
    r = check_alignment(_counts(), "30000/1001", "29970/1000")
    assert not r.ok and any("rate" in s.lower() for s in r.reasons)
