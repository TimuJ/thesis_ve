"""Build the severity-response summary for the identity_degradation artefact.

Reads:
    results/lr_vcc/composite_artefacts_v3_slope_b200/identity_degradation/*.json
Writes:
    stdout: markdown tables per base video showing each sub-metric's score vs
    severity, plus the composite LR-VCC and the softmax weight that sub-metric
    I received.
"""
import json
import sys
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parents[2]
COMP_DIR = HERE / "results" / "lr_vcc" / "composite_artefacts_v3_slope_b200" / "identity_degradation"
BASES = ["hhszUXL1Cu8", "7WHI2L_FDNg"]
SEVS = ["0p02", "0p05", "0p10", "0p20", "0p40"]
SUB_ORDER = ["appearance", "temporal", "identity", "color_stability", "color_slope"]
SUB_SHORT = {"appearance": "A", "temporal": "T", "identity": "I",
             "color_stability": "D", "color_slope": "E"}


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
    if not COMP_DIR.is_dir():
        sys.exit("[error] missing composite dir: " + str(COMP_DIR))

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
