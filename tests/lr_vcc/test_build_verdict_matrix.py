import json

from scripts.lr_vcc.build_verdict_matrix import collect_deltas, verdict


def test_verdict_thresholds():
    assert verdict(-0.10) == "PASS"
    assert verdict(-0.03) == "WEAK"
    assert verdict(-0.01) == "FLAT"
    assert verdict(+0.01) == "FLAT"
    assert verdict(+0.05) == "INVERTED"


def test_collect_deltas(tmp_path):
    art = tmp_path / "background_drift"
    art.mkdir()
    for sev, score in [("0p02", 0.532), ("0p40", 0.256)]:
        (art / f"hhszUXL1Cu8_sev{sev}.json").write_text(
            json.dumps({"video": f"hhszUXL1Cu8_sev{sev}", "lr_vcc": score}))
    deltas = collect_deltas(tmp_path)
    assert abs(deltas[("background_drift", "hhszUXL1Cu8")] - (-0.276)) < 1e-9
