"""Build the artefact × base verdict matrix from per-video composite JSONs.

Usage:
    python scripts/lr_vcc/build_verdict_matrix.py \
        --composites_dir results/lr_vcc/composite_artefacts_v4 \
        --out reports/figures/verdict_matrix.md
"""
import argparse
import json
import re
from pathlib import Path

_SEV_RE = re.compile(r"^(?P<base>.+)_sev(?P<sev>\d+p\d+)$")
_LO, _HI = "0p02", "0p40"


def verdict(delta: float) -> str:
    if delta <= -0.05:
        return "PASS"
    if delta <= -0.02:
        return "WEAK"
    if delta < +0.02:
        return "FLAT"
    return "INVERTED"


def collect_deltas(composites_dir) -> dict:
    """{(artefact, base): lr_vcc(sev 0.40) - lr_vcc(sev 0.02)}"""
    scores = {}
    for artefact_dir in Path(composites_dir).iterdir():
        if not artefact_dir.is_dir():
            continue
        for f in artefact_dir.glob("*.json"):
            m = _SEV_RE.match(f.stem)
            if not m:
                continue
            d = json.load(open(f))
            scores[(artefact_dir.name, m["base"], m["sev"])] = float(d["lr_vcc"])
    deltas = {}
    for (art, base, sev), v in scores.items():
        if sev == _HI and (art, base, _LO) in scores:
            deltas[(art, base)] = v - scores[(art, base, _LO)]
    return deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--composites_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    deltas = collect_deltas(args.composites_dir)
    artefacts = sorted({a for a, _ in deltas})
    bases = sorted({b for _, b in deltas})

    lines = ["| artefact | " + " | ".join(bases) + " |",
             "|---|" + "---|" * len(bases)]
    for art in artefacts:
        cells = []
        for base in bases:
            d = deltas.get((art, base))
            cells.append("—" if d is None else f"{d:+.3f} {verdict(d)}")
        lines.append(f"| {art} | " + " | ".join(cells) + " |")

    n_pass = sum(1 for d in deltas.values() if verdict(d) in ("PASS", "WEAK"))
    lines.append(f"\nclean (PASS+WEAK): {n_pass}/{len(deltas)} conditions")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
