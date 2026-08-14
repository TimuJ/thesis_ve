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


def test_response_curves_states_its_sign_convention(tmp_path):
    """R = y(0.02) - y(0.40); positive is the correct direction. The other
    two reports print the same quantity with the opposite sign (delta =
    -R) in an unlabelled cell, so each report must say which convention it
    uses.
    """
    out = tmp_path / "curves.md"
    REP.write_response_curves(RT.build_table()["artefacts"], R.PROD_PARAMS, out)
    text = out.read_text()
    assert "R = y(0.02)" in text
    assert "y(0.40)" in text


def test_expectation_matrix_reports_both_counts(tmp_path):
    out = tmp_path / "matrix.md"
    REP.write_expectation_matrix(RT.build_table()["artefacts"],
                                 R.PROD_PARAMS, out)
    text = out.read_text()
    assert "39/55" in text
    assert "29/60" in text


def test_expectation_matrix_states_its_sign_convention(tmp_path):
    out = tmp_path / "matrix.md"
    REP.write_expectation_matrix(RT.build_table()["artefacts"],
                                 R.PROD_PARAMS, out)
    text = out.read_text()
    assert "delta = " in text
    assert "y(0.40)" in text and "y(0.02)" in text


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
def table():
    return RT.build_table()


@pytest.fixture(scope="module")
def lobo_result(table):
    return F.lobo(table)


def _per_fold_section(text):
    """The '## Per-fold results' section only, up to the next heading.

    Bounding assertions to this section (rather than the whole document)
    matters for the historical-bug guard below: v5's five-base aggregate
    loss legitimately appears in the Summary section above it.
    """
    start = text.index("## Per-fold results")
    end = text.index("## Final parameters", start)
    return text[start:end]


def test_lobo_report_per_fold_table_has_six_columns(tmp_path, table, lobo_result):
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    section = _per_fold_section(out.read_text())
    header = next(line for line in section.splitlines()
                 if line.startswith("| fold"))
    columns = [c.strip() for c in header.strip("|").split("|")]
    assert columns == ["fold (held out)", "train loss", "v6 held-out loss",
                       "v5 loss (same base)", "delta (v5−v6)",
                       "converged"]


def test_lobo_report_has_exactly_five_fold_rows(tmp_path, table, lobo_result):
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    section = _per_fold_section(out.read_text())
    data_rows = [line for line in section.splitlines()
                if line.startswith("|") and not line.startswith("|---")
                and not line.startswith("| fold")]
    assert len(data_rows) == 5


def test_lobo_report_names_the_exception_and_the_largest_win(tmp_path, table, lobo_result):
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    text = out.read_text()
    assert "KZ8p6b1zJ9U" in text and "exception" in text.lower()
    assert "BrRLKMbBTYQ" in text and "largest win" in text.lower()


def test_lobo_report_delta_sign_is_v5_minus_v6(tmp_path, table, lobo_result):
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    text = out.read_text()
    fold = lobo_result["folds"][0]
    expected_delta = fold["v5_test_loss"] - fold["test_loss"]
    assert "{:+.5f}".format(expected_delta) in text


def test_lobo_report_states_the_warm_start_caveat(tmp_path, table, lobo_result):
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    text = out.read_text()
    assert "PROD_PARAMS" in text
    assert "not a data leak" in text


def test_lobo_report_fold_table_never_shows_the_five_base_v5_aggregate(
        tmp_path, table, lobo_result):
    """Guard against the historical bug: a fold's held-out loss compared
    against v5's five-base aggregate instead of v5 scored on that same
    held-out base. v5's five-base aggregate (0.026884, from lobo_result's
    v5_loss / mean_paired_v5) legitimately appears in the Summary section,
    but must never appear inside the per-fold table itself.
    """
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    section = _per_fold_section(out.read_text())
    assert "0.026884" not in section
    assert "{:.6f}".format(lobo_result["v5_loss"]) not in section


def test_lobo_report_states_delta_sign_convention_for_held_out_matrix(
        tmp_path, table, lobo_result):
    """response_curves_v5.md's R and this matrix's delta are the same
    quantity with opposite signs; the held-out matrix must say so, not
    just expectation_scored_matrix_v5.md.
    """
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    text = out.read_text()
    section_start = text.index("## Held-out verdict matrix")
    section_end = text.index("## Conformance comparison", section_start)
    section = text[section_start:section_end]
    assert "delta = " in section
    assert "y(0.40)" in section and "y(0.02)" in section


def test_lobo_report_conformance_comparison_table(tmp_path, table, lobo_result):
    """Pins the reviewer's recomputed table (F1): the loss improved but
    the as-designed conformance count did not move at all.
    """
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    text = out.read_text()
    assert "## Conformance comparison: v5 vs v6" in text
    assert "| v5 | 25/40 | 14/15 | 39/55 | 29/60 |" in text
    assert "| v6 held-out | 24/40 | 15/15 | 39/55 | 28/60 |" in text
    assert "| v6 in-sample | 30/40 | 12/15 | 42/55 | 35/60 |" in text


def test_lobo_report_states_conformance_is_unchanged(tmp_path, table, lobo_result):
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    text = out.read_text()
    assert "unchanged at the verdict level" in text
    assert "but conformance did not" in text


def test_lobo_report_notes_inverted_to_flat_wins(tmp_path, table, lobo_result):
    """All four v5-INVERTED cells become FLAT under v6 held-out — the
    genuine verdict-shape win the reviewer identified, distinct from the
    (unchanged) conformance count.
    """
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    text = out.read_text()
    assert "**4 cells**" in text
    assert "INVERTED under v5" in text
    assert "become FLAT under v6 held-out" in text


def test_lobo_report_notes_silent_held_out_reaches_15_of_15(
        tmp_path, table, lobo_result):
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    text = out.read_text()
    assert "SILENT held-out reaches **15/15**" in text


def test_lobo_report_notes_insample_silent_drop_as_overcalibration_evidence(
        tmp_path, table, lobo_result):
    """The in-sample SILENT drop to 12/15 is exactly the over-calibration
    signal the silence penalty exists to catch, and the report must say so.
    """
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    text = out.read_text()
    assert "SILENT in-sample drops to **12/15**" in text
    assert "over-calibration" in text


def test_lobo_report_per_fold_parameter_table_marks_boundaries(
        tmp_path, table, lobo_result):
    """Pins the two folds the reviewer flagged as boundary-heavy: the
    7WHI2L_FDNg fold hits 5/7 response-parameter grid boundaries and
    mJog8DlRk_4 hits 4/7; the other three folds hit none. mask_cov_floor
    hits its grid minimum in every fold; a_sat_ceiling never does.
    """
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    text = out.read_text()
    assert "## Per-fold parameter vectors" in text
    assert "not identified by the data at this sample size" in text
    assert ("boundary hits among the seven response parameters, per fold "
           "(held-out base): 7WHI2L_FDNg 5/7, BrRLKMbBTYQ 0/7, "
           "KZ8p6b1zJ9U 0/7, hhszUXL1Cu8 0/7, mJog8DlRk_4 4/7.") in text

    section_start = text.index("## Per-fold parameter vectors")
    section_end = text.index("## Loss surface", section_start)
    section = text[section_start:section_end]
    mask_row = next(l for l in section.splitlines()
                    if l.startswith("| mask_cov_floor"))
    assert mask_row.count("0*") == 5
    sat_row = next(l for l in section.splitlines()
                  if l.startswith("| a_sat_ceiling"))
    assert "*" not in sat_row


def test_lobo_report_loss_surface_classifies_flat_and_sharp_parameters(
        tmp_path, table, lobo_result):
    """Pins the sensitivity read (F3): the gate thresholds a_sat_ceiling,
    closeup_threshold, face_rate_floor and a_drift_floor are flat — the
    data does not constrain them at n=5 — while tau, beta_t, lambda_a and
    beta_dp show a sharp minimum at the chosen value.
    """
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    text = out.read_text()
    assert "## Loss surface (sensitivity at the chosen point)" in text
    flat_line = next(l for l in text.splitlines() if l.startswith("- **flat**"))
    sharp_line = next(l for l in text.splitlines() if l.startswith("- **sharp**"))
    for p in ("a_sat_ceiling", "closeup_threshold", "face_rate_floor",
             "a_drift_floor"):
        assert p in flat_line, (p, flat_line)
    for p in ("tau", "beta_t", "lambda_a", "beta_dp"):
        assert p in sharp_line, (p, sharp_line)


def test_lobo_report_final_parameters_table_has_all_twelve_searched_params(
        tmp_path, table, lobo_result):
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    text = out.read_text()
    section_start = text.index("## Final parameters")
    section_end = text.index("## Per-fold parameter vectors", section_start)
    section = text[section_start:section_end]
    for key in ("tau", "beta_t", "lambda_a", "alpha", "beta_e", "beta_dp",
               "beta_dpp", "a_drift_floor", "a_sat_ceiling",
               "mask_cov_floor", "face_rate_floor", "closeup_threshold"):
        assert "\n| {} |".format(key) in section, key


def test_lobo_report_calls_out_mask_cov_floor_consequence(
        tmp_path, table, lobo_result):
    """mask_cov_floor moved 0.10 -> 0.0 (v5 -> v6), disabling the tOF
    coverage filter entirely; the report must name the row count this
    affects, computed from the actual rows, not hardcoded (F4).
    """
    out = tmp_path / "lobo.md"
    REP.write_lobo_report(lobo_result, table, out)
    text = out.read_text()
    assert "`mask_cov_floor` 0.1 → 0" in text
    assert "`a_drift_floor` 0.02 → 0" in text
    assert "materially changes sub-metric" in text
    assert "183/315" in text
