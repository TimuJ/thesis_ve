"""Build the severity-response summary for the identity_degradation artefact.

Reads:
    results/lr_vcc/composite_artefacts_v3_slope_b200/identity_degradation/*.json
    (preferred) OR falls back to the raw per-sub-metric JSONs.
Writes:
    stdout: markdown tables per base video showing each sub-metric's score vs
    severity, plus the composite LR-VCC and the softmax weight that sub-metric
    I received.
"""
import json
import math
import sys
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parents[2]
COMP_DIR = HERE / "results" / "lr_vcc" / "composite_artefacts_v3_slope_b200" / "identity_degradation"
RAW_BASE = HERE / "results"
BASES = ["hhszUXL1Cu8", "7WHI2L_FDNg"]
SEVS = ["0p02", "0p05", "0p10", "0p20", "0p40"]
SUB_ORDER = ["appearance", "temporal", "identity", "color_stability", "color_slope"]
SUB_SHORT = {"appearance": "A", "temporal": "T", "identity": "I",
             "color_stability": "D", "color_slope": "E"}


def _maybe_load(p):
    if not Path(p).is_file():
        return None
    return json.load(open(p))


def _raw_sub_scores(base, sev):
    """If composite not built yet, derive sub-metric scores from raw JSONs.

    Returns dict {sub: score_or_None}. Identity comes from the bulk JSON; it
    needs the closeup_p50 map and proper aggregation, so we approximate by
    reading raw .fused if present.
    """
    vid = base + "_sev" + sev
    out = {}

    # Appearance: from clip_iqa raw
    clip_path = RAW_BASE / "synthetic_artefacts_eval" / "clip_iqa" / "identity_degradation" / (vid + "_clip_iqa.json")
    raw = _maybe_load(clip_path)
    if raw is not None:
        # Use appearance_score from the module if reachable; otherwise inline
        try:
            from scripts.lr_vcc.appearance import appearance_score
            a = appearance_score(raw)
            out["appearance"] = a["score"]
        except Exception:
            out["appearance"] = None

    # Temporal: from tof_tlp raw
    tof_path = RAW_BASE / "synthetic_artefacts_eval" / "tof_tlp" / "identity_degradation" / (vid + "_tof_tlp.json")
    raw = _maybe_load(tof_path)
    if raw is not None:
        try:
            from scripts.lr_vcc.temporal import temporal_score
            t = temporal_score(raw, weight_fn="uniform")
            out["temporal"] = t["score"]
        except Exception:
            out["temporal"] = None

    # Color stability (D)
    ch_path = RAW_BASE / "lr_vcc" / "color_histogram" / "identity_degradation" / (vid + "_color_hist.json")
    raw = _maybe_load(ch_path)
    if raw is not None:
        try:
            from scripts.lr_vcc.color_stability import color_stability_score
            d = color_stability_score(raw, alpha=0.394)
            out["color_stability"] = d["score"]
        except Exception:
            out["color_stability"] = None

    # Color slope (E): re-derive with beta=200
    cs_path = RAW_BASE / "lr_vcc" / "color_slope" / "identity_degradation" / (vid + "_color_slope.json")
    raw = _maybe_load(cs_path)
    if raw is not None:
        m = raw.get("details", {}).get("max_abs_slope")
        if m is not None:
            score = max(0.0, min(1.0, math.exp(-200.0 * float(m))))
            out["color_slope"] = score

    # Identity (I): from bulk JSON
    id_dir = RAW_BASE / "synthetic_artefacts_eval" / "identity" / "identity_degradation"
    if id_dir.is_dir():
        bulk = list(id_dir.glob("*.json"))
        if bulk:
            payload = json.load(open(bulk[0]))
            pv = payload.get("per_video", {}).get(vid)
            if pv is not None:
                out["identity_fused"] = pv.get("fused")
                out["identity_face_rate"] = (pv.get("n_clips_with_faces", 0)
                                             / max(1, pv.get("n_clips", 1)))
    return out


def fmt(x, w=6, p=3):
    if x is None:
        return " " * w
    return ("{:" + str(w) + "." + str(p) + "f}").format(x)


def load_one(base, sev):
    f = COMP_DIR / (base + "_sev" + sev + ".json")
    if not f.is_file():
        return None
    return json.load(open(f))


def main():
    use_composite = COMP_DIR.is_dir() and any(COMP_DIR.glob("*.json"))
    if not use_composite:
        print("# Partial summary (composite not built yet); using raw per-metric JSONs\n")
        for base in BASES:
            print("\n## Base = " + base + "\n")
            header = "| sev  | A     | T     | I_fused | D     | E     | face_rate |"
            sep = "|------|-------|-------|---------|-------|-------|-----------|"
            print(header)
            print(sep)
            for sev in SEVS:
                s = _raw_sub_scores(base, sev)
                a = s.get("appearance"); t = s.get("temporal"); i = s.get("identity_fused")
                d = s.get("color_stability"); e = s.get("color_slope"); fr = s.get("identity_face_rate")
                print("| " + sev + " | " + fmt(a) + " | " + fmt(t) + " | "
                      + fmt(i) + " | " + fmt(d) + " | " + fmt(e) + " | " + fmt(fr) + " |")
        return

    # Per-base table.
    for base in BASES:
        print("\n## Base = " + base + "\n")
        header = "| sev  | " + " | ".join(SUB_SHORT[s] for s in SUB_ORDER) + " | LR-VCC | w(I) |"
        sep = "|------|" + "|".join(["-------"] * (len(SUB_ORDER) + 2)) + "|"
        print(header)
        print(sep)
        for sev in SEVS:
            r = load_one(base, sev)
            if r is None:
                print("| " + sev + " | (missing) |")
                continue
            sub = r["sub_metrics"]
            scores = [sub[k]["score"] if k in sub else None for k in SUB_ORDER]
            # weights are in same order: appearance, temporal, identity, [D, E]
            weights = r["weights_used"]
            # find which slot is identity (sub-metric I)
            present = [s for s in SUB_ORDER if s in sub]
            i_slot = present.index("identity") if "identity" in present else None
            w_i = weights[i_slot] if i_slot is not None else None
            cells = [fmt(s) for s in scores]
            print("| " + sev + " | " + " | ".join(cells)
                  + " | " + fmt(r["lr_vcc"]) + " | " + fmt(w_i) + " |")

    # Cross-base summary: monotonic decrease check on sub-metric I.
    print("\n## Monotonicity check (sub-metric I)\n")
    for base in BASES:
        i_scores = []
        for sev in SEVS:
            r = load_one(base, sev)
            if r is None:
                i_scores.append(None)
                continue
            i_scores.append(r["sub_metrics"]["identity"]["score"])
        non_decr_violations = 0
        for a, b in zip(i_scores, i_scores[1:]):
            if a is None or b is None:
                continue
            if b > a + 1e-3:
                non_decr_violations += 1
        good_chain = all(s is not None for s in i_scores)
        flat_str = "monotone-non-increasing" if non_decr_violations == 0 and good_chain else (
            str(non_decr_violations) + " violations")
        print("- " + base + ": I scores = " + ", ".join(fmt(s, 5, 3) for s in i_scores)
              + " -> " + flat_str)


if __name__ == "__main__":
    main()
