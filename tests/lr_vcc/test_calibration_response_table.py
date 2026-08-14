import json
import statistics

from scripts.lr_vcc.calibration import response_table as RT
from scripts.lr_vcc.calibration import expectations as E

REQUIRED = ("unit", "clip", "base", "severity", "a_mean", "a_std", "tof",
            "cov", "identity_fused", "n_clips", "n_clips_with_faces",
            "dispersion", "closeup_p50", "hist_dist", "hist_n_frames",
            "slope_abs", "slope_rel", "anchor_q14", "anchor_rel",
            "clip_q14", "clip_rel")


def test_table_has_full_artefact_matrix():
    table = RT.build_table()
    rows = table["artefacts"]
    assert len(rows) == 300
    seen = {(r["unit"], r["base"], r["severity"]) for r in rows}
    assert len(seen) == 300
    assert {r["base"] for r in rows} == set(E.BASES)
    assert {r["severity"] for r in rows} == set(E.SEVERITIES)


def test_every_row_is_complete():
    table = RT.build_table()
    for r in table["artefacts"] + table["realmodels"]:
        for key in REQUIRED:
            assert key in r, (r["clip"], key)
        for key in ("a_mean", "a_std", "hist_dist", "slope_abs",
                    "anchor_q14", "clip_q14", "identity_fused"):
            assert r[key] is not None, (r["clip"], key)
        assert r["tof"] and r["cov"]


def test_realmodel_rows_cover_three_methods_and_five_videos():
    table = RT.build_table()
    rows = table["realmodels"]
    assert {r["unit"] for r in rows} == {"mgld", "uav", "flashvsr"}
    assert len(rows) == 15
    assert all(r["severity"] is None for r in rows)


def test_appearance_stats_match_source_json():
    table = RT.build_table()
    row = [r for r in table["artefacts"]
           if r["unit"] == "flicker" and r["base"] == "7WHI2L_FDNg"
           and r["severity"] == "0p02"][0]
    src = json.load(open("results/synthetic_artefacts_eval/clip_iqa/flicker/"
                         "7WHI2L_FDNg_sev0p02_clip_iqa.json"))
    assert row["a_mean"] == statistics.mean(src["clip_iqa"])
    assert row["a_std"] == statistics.pstdev(src["clip_iqa"])


def test_save_load_roundtrip_is_exact(tmp_path):
    table = RT.build_table()
    p = tmp_path / "t.json"
    RT.save(table, p)
    assert RT.load(p) == table
