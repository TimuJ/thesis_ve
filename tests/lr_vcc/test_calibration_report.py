import re

import pytest

from scripts.lr_vcc.calibration import fit as F
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
    """Pins the actual counts, not just the presence of the bullet labels.

    calibration-addressable 20 (normalisation 3, gate 4, composition 13)
    versus structural 14 (measurement 9, reward_direction 5); also covers
    the two subsections amendment 3 added: the 15 cells where weight drift
    is invisible to a per-sub-metric-only view, and the single SILENT cell
    (flip_elastic / mJog8DlRk_4) explained via silence_broken_by instead of
    a designed-for sub-metric.
    """
    out = tmp_path / "fail.md"
    REP.write_failure_attribution(RT.build_table()["artefacts"],
                                  R.PROD_PARAMS, out)
    text = out.read_text()
    assert "calibration-addressable" in text
    assert "structural" in text

    addressable = re.search(r"calibration-addressable\*\*[^:]*:\s*(\d+)", text)
    structural = re.search(r"\*\*structural\*\*[^:]*:\s*(\d+)", text)
    assert addressable is not None and int(addressable.group(1)) == 20
    assert structural is not None and int(structural.group(1)) == 14

    drift_cells = re.search(r"In these \*\*(\d+) cells\*\*", text)
    assert drift_cells is not None and int(drift_cells.group(1)) == 15

    silent_row = next(line for line in text.splitlines()
                      if "flip_elastic" in line and "mJog8DlRk_4" in line)
    assert "appearance" in silent_row
    assert "identity" in silent_row
    assert "clip_trajectory" in silent_row


@pytest.fixture(scope="module")
def lobo_result():
    return F.lobo(RT.build_table())


def _per_fold_section(text):
    """The '## Per-fold results' section only, up to the next heading.

    Bounding assertions to this section (rather than the whole document)
    matters for the historical-bug guard below: v5's five-base aggregate
    loss legitimately appears in the Summary section above it.
    """
    start = text.index("## Per-fold results")
    end = text.index("## Final parameters", start)
    return text[start:end]


def test_lobo_report_per_fold_table_has_six_columns(tmp_path, lobo_result):
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, out)
    section = _per_fold_section(out.read_text())
    header = next(line for line in section.splitlines()
                 if line.startswith("| fold"))
    columns = [c.strip() for c in header.strip("|").split("|")]
    assert columns == ["fold (held out)", "train loss", "v6 held-out loss",
                       "v5 loss (same base)", "delta (v5−v6)",
                       "converged"]


def test_lobo_report_has_exactly_five_fold_rows(tmp_path, lobo_result):
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, out)
    section = _per_fold_section(out.read_text())
    data_rows = [line for line in section.splitlines()
                if line.startswith("|") and not line.startswith("|---")
                and not line.startswith("| fold")]
    assert len(data_rows) == 5


def test_lobo_report_names_the_exception_and_the_largest_win(tmp_path, lobo_result):
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, out)
    text = out.read_text()
    assert "KZ8p6b1zJ9U" in text and "exception" in text.lower()
    assert "BrRLKMbBTYQ" in text and "largest win" in text.lower()


def test_lobo_report_delta_sign_is_v5_minus_v6(tmp_path, lobo_result):
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, out)
    text = out.read_text()
    fold = lobo_result["folds"][0]
    expected_delta = fold["v5_test_loss"] - fold["test_loss"]
    assert "{:+.5f}".format(expected_delta) in text


def test_lobo_report_states_the_warm_start_caveat(tmp_path, lobo_result):
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, out)
    text = out.read_text()
    assert "PROD_PARAMS" in text
    assert "not a data leak" in text


def test_lobo_report_fold_table_never_shows_the_five_base_v5_aggregate(
        tmp_path, lobo_result):
    """Guard against the historical bug: a fold's held-out loss compared
    against v5's five-base aggregate instead of v5 scored on that same
    held-out base. v5's five-base aggregate (0.026884, from lobo_result's
    v5_loss / mean_paired_v5) legitimately appears in the Summary section,
    but must never appear inside the per-fold table itself.
    """
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, out)
    section = _per_fold_section(out.read_text())
    assert "0.026884" not in section
    assert "{:.6f}".format(lobo_result["v5_loss"]) not in section
