"""Extract the raw statistics every free parameter acts on, once.

Every column here is a *measurement*. Nothing in this module applies a
response function — that is recompose.py's job. Keeping the split sharp is
what lets the fitter run without touching disk.

Usage (repo root):
  python -m scripts.lr_vcc.calibration.response_table
"""
import json
import statistics
from pathlib import Path

from ..identity import clip_score_dispersion
from ..sweep_sensitivity import (
    ARTEFACTS, artefact_units, realmodel_units,
)

REPO = Path(__file__).resolve().parents[3]
TABLE_PATH = REPO / "results" / "lr_vcc" / "calibration" / "response_table.json"


def _split_clip(clip):
    """('7WHI2L_FDNg_sev0p02') -> ('7WHI2L_FDNg', '0p02'); no suffix -> (clip, None)."""
    if "_sev" in clip:
        base, sev = clip.rsplit("_sev", 1)
        return base, sev
    return clip, None


def _row(unit_name, clip, u):
    base, sev = _split_clip(clip)

    qs = json.load(open(Path(u["clip_iqa_dir"]) / (clip + "_clip_iqa.json")))["clip_iqa"]
    tof_payload = json.load(open(Path(u["tof_dir"]) / (clip + "_tof_tlp.json")))
    id_pv = json.load(open(str(u["identity_results"])))["per_video"][clip]
    hist = json.load(open(Path(u["color_hist_dir"]) / (clip + "_color_hist.json")))
    slope = json.load(open(Path(u["color_slope_dir"]) / (clip + "_color_slope.json")))
    anchor = json.load(open(Path(u["color_hist_anchor_dir"]) /
                            (clip + "_color_hist_anchor.json")))
    traj = json.load(open(Path(u["clip_trajectory_dir"]) /
                          (clip + "_clip_trajectory.json")))

    hist_dist = hist.get("mean_hist_dist")
    if hist_dist is None:
        hist_dist = (hist.get("details") or {})["mean_l1_dist"]

    def _q14(payload):
        q = (payload.get("details") or {})["trajectory_mean_per_quarter"]
        return abs(float(q[3]) - float(q[0]))

    return {
        "unit": unit_name,
        "clip": clip,
        "base": base,
        "severity": sev,
        "a_mean": statistics.mean(qs),
        "a_std": statistics.pstdev(qs),
        "tof": tof_payload["tof"],
        "cov": tof_payload["mean_mask_coverage"],
        "identity_fused": float(id_pv["fused"]),
        "n_clips": int(id_pv["n_clips"]),
        "n_clips_with_faces": int(id_pv["n_clips_with_faces"]),
        "dispersion": clip_score_dispersion(id_pv),
        "closeup_p50": u["closeup_map"].get(base if sev else clip),
        "hist_dist": float(hist_dist),
        "hist_n_frames": int(hist.get("n_frames", 0)),
        "slope_abs": float((slope.get("details") or {})["max_abs_slope"]),
        # Asymmetric default is deliberate: mirrors evaluate_one_video's
        # raw_e.get("reliability", 0.0) for sub-metric E exactly (Task 4
        # reproduces it bit-exactly) — do not harmonise with anchor/clip below.
        "slope_rel": float(slope.get("reliability", 0.0)),
        "anchor_q14": _q14(anchor),
        # mirrors evaluate_one_video's raw.get("reliability", 1.0) for D'.
        "anchor_rel": float(anchor.get("reliability", 1.0)),
        "clip_q14": _q14(traj),
        # mirrors evaluate_one_video's raw.get("reliability", 1.0) for D''.
        "clip_rel": float(traj.get("reliability", 1.0)),
    }


def _rows_for_units(units):
    rows = []
    for name, u in units:
        for fa in sorted(Path(u["clip_iqa_dir"]).glob("*_clip_iqa.json")):
            clip = fa.name.replace("_clip_iqa.json", "")
            rows.append(_row(name, clip, u))
    return rows


def build_table():
    """{"artefacts": [...300 rows...], "realmodels": [...15 rows...]}.

    Real-model rows use the gated-canonical inputs (fps-corrected identity,
    closeup gate on every method) — the variant the thesis reports and the one
    the fit's leaderboard guards are evaluated against.
    """
    assert len(ARTEFACTS) == 12
    return {
        "artefacts": _rows_for_units(artefact_units()),
        "realmodels": _rows_for_units(
            realmodel_units(identity_variant="corrected", closeup=True,
                            methods=("mgld", "uav", "flashvsr"))),
    }


def save(table, path=TABLE_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(table, open(path, "w"), indent=2)
    return path


def load(path=TABLE_PATH):
    return json.load(open(path))


if __name__ == "__main__":
    t = build_table()
    p = save(t)
    print("wrote {} ({} artefact rows, {} real-model rows)".format(
        p, len(t["artefacts"]), len(t["realmodels"])))
