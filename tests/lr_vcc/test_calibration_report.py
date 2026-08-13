from scripts.lr_vcc.calibration import recompose as R
from scripts.lr_vcc.calibration import report as REP
from scripts.lr_vcc.calibration import response_table as RT


def test_v5_conformance_counts_are_39_of_55():
    """Expectation-aware scoring of the unchanged v5 matrix.

    RESPOND 25/40 (background_drift 2, chunk_boundary 4, color_drift 5,
    flicker 1, identity_degradation 2, identity_drift 2, flip_invert 5,
    flip_channel_shuffle 4); SILENT 14/15 (flip_elastic misses on mJog);
    flip_transpose's 5 cells are unconstrained.
    """
    counts = REP.conformance_counts(RT.build_table()["artefacts"],
                                    R.PROD_PARAMS)
    assert counts["respond_conforming"] == 25
    assert counts["respond_total"] == 40
    assert counts["silent_conforming"] == 14
    assert counts["silent_total"] == 15
    assert counts["unconstrained"] == 5
    assert counts["uniform_clean"] == 29  # the old PASS+WEAK rule, for contrast


def test_response_curves_report_has_a_row_per_cell(tmp_path):
    out = tmp_path / "curves.md"
    REP.write_response_curves(RT.build_table()["artefacts"], R.PROD_PARAMS, out)
    text = out.read_text()
    assert text.count("\n|") >= 60
    assert "0p10" in text


def test_expectation_matrix_reports_both_counts(tmp_path):
    out = tmp_path / "matrix.md"
    REP.write_expectation_matrix(RT.build_table()["artefacts"],
                                 R.PROD_PARAMS, out)
    text = out.read_text()
    assert "39/55" in text
    assert "29/60" in text


def test_failure_report_separates_addressable_from_structural(tmp_path):
    out = tmp_path / "fail.md"
    REP.write_failure_attribution(RT.build_table()["artefacts"],
                                  R.PROD_PARAMS, out)
    text = out.read_text()
    assert "calibration-addressable" in text
    assert "structural" in text
