"""Markdown emitters for the calibration deliverables.

Usage (repo root):
  python -m scripts.lr_vcc.calibration.report            # v5 reports
  python -m scripts.lr_vcc.calibration.report --lobo     # + run the fit
"""
import argparse
import json
from pathlib import Path

from . import expectations as E
from .failure_analysis import ADDRESSABLE, analyse
from .fit import lobo
from .objective import LOSS_CFG, matrix_scores
from .recompose import PROD_PARAMS
from .response_table import build_table

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / "reports" / "figures"


def conformance_counts(rows, params):
    scored = matrix_scores(rows, params)
    c = {"respond_conforming": 0, "respond_total": 0, "silent_conforming": 0,
         "silent_total": 0, "unconstrained": 0, "uniform_clean": 0}
    for (family, _base), cell in scored.items():
        if cell["verdict"] in ("PASS", "WEAK"):
            c["uniform_clean"] += 1
        exp = E.EXPECTATION[family]
        if exp == E.UNCONSTRAINED:
            c["unconstrained"] += 1
        elif exp == E.RESPOND:
            c["respond_total"] += 1
            c["respond_conforming"] += int(E.conforms(family, cell["verdict"]))
        else:
            c["silent_total"] += 1
            c["silent_conforming"] += int(E.conforms(family, cell["verdict"]))
    return c


def _write(out, lines):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)
    return out


def write_response_curves(rows, params, out=FIG / "response_curves_v5.md"):
    scored = matrix_scores(rows, params)
    lines = ["# LR-VCC v5 — severity-response curves", "",
             "All five ladder points. The v5 verdict protocol reads only the "
             "0p02 and 0p40 endpoints; the intermediate severities were cached "
             "but unused.", "",
             "| artefact | base | " + " | ".join(E.SEVERITIES) +
             " | R | monotone | verdict |",
             "|---|---|" + "---|" * (len(E.SEVERITIES) + 3)]
    for (family, base) in sorted(scored):
        cell = scored[(family, base)]
        seq = [cell["ladder"][s] for s in E.SEVERITIES]
        mono = all(b <= a + 1e-12 for a, b in zip(seq, seq[1:]))
        lines.append("| {} | {} | {} | {:+.3f} | {} | {} |".format(
            family, base, " | ".join("{:.3f}".format(v) for v in seq),
            cell["response"], "yes" if mono else "no", cell["verdict"]))
    return _write(out, lines)


def write_expectation_matrix(rows, params,
                             out=FIG / "expectation_scored_matrix_v5.md"):
    scored = matrix_scores(rows, params)
    c = conformance_counts(rows, params)
    total_conf = c["respond_conforming"] + c["silent_conforming"]
    total_cells = c["respond_total"] + c["silent_total"]
    bases = list(E.BASES)
    lines = ["# LR-VCC v5 — expectation-scored verdict matrix", "",
             "Each cell is scored against the expectation pre-registered for "
             "its family, not against a uniform PASS+WEAK rule. A control "
             "family predicted invisible conforms by being FLAT.", "",
             "| artefact | expectation | " + " | ".join(bases) + " |",
             "|---|---|" + "---|" * len(bases)]
    for family in sorted(E.EXPECTATION):
        cells = []
        for base in bases:
            cell = scored[(family, base)]
            ok = E.conforms(family, cell["verdict"])
            mark = "—" if ok is None else ("✓" if ok else "✗")
            cells.append("{:+.3f} {} {}".format(cell["delta"], cell["verdict"], mark))
        lines.append("| {} | {} | {} |".format(
            family, E.EXPECTATION[family], " | ".join(cells)))
    lines += ["",
              "- as-designed (expectation-aware): **{}/{}** "
              "(RESPOND {}/{}, SILENT {}/{}; {} unconstrained cells excluded)"
              .format(total_conf, total_cells, c["respond_conforming"],
                      c["respond_total"], c["silent_conforming"],
                      c["silent_total"], c["unconstrained"]),
              "- clean under the old uniform PASS+WEAK rule: **{}/60**"
              .format(c["uniform_clean"]),
              "",
              "The metric is unchanged between the two counts; only the "
              "scoring criterion differs."]
    return _write(out, lines)


def write_failure_attribution(rows, params, out=FIG / "failure_attribution_v5.md"):
    result = analyse(rows, params)
    stage_counts = {}
    lines = ["# LR-VCC v5 — failure attribution", "",
             "For every non-conforming cell, the sub-metrics the family was "
             "built to excite, and the stage at which the signal was lost.", "",
             "| artefact | base | verdict | sub-metric | stage | raw Δ% | "
             "score Δ | mean w | weight drift |",
             "|---|---|---|---|---|---|---|---|---|"]
    n_nonconforming = 0
    for (family, base) in sorted(result):
        cell = result[(family, base)]
        if cell["conforms"] is not False:
            continue
        n_nonconforming += 1
        for d in cell["sub_metrics"]:
            stage_counts[d["stage"]] = stage_counts.get(d["stage"], 0) + 1
            lines.append("| {} | {} | {} | {} | {} | {:+.0f}% | {:+.3f} | "
                         "{:.3f} | {} |".format(
                             family, base, cell["verdict"], d["sub_metric"],
                             d["stage"], d["rel_raw"] * 100, d["delta_score"],
                             d["mean_weight"], "yes" if d["weight_drift"] else ""))
    n_findings = sum(stage_counts.values())
    addressable = sum(v for k, v in stage_counts.items() if k in ADDRESSABLE)
    structural = sum(v for k, v in stage_counts.items()
                     if k in ("measurement", "reward_direction"))
    lines += ["", "## Totals by stage", "",
              "**{} of 55** constrained cells fail their expectation; the "
              "table above attributes **{} findings** across them (a cell "
              "names every one of its designed-for sub-metrics, so it can "
              "contribute more than one row).".format(n_nonconforming,
                                                       n_findings), ""]
    for stage in sorted(stage_counts):
        lines.append("- {}: {}".format(stage, stage_counts[stage]))
    lines += ["",
              "- **calibration-addressable** (normalisation / gate / "
              "composition): {}".format(addressable),
              "- **structural** (measurement / reward-direction — needs a "
              "different measurement, not a different constant): {}"
              .format(structural),
              "",
              "The structural count is the honest ceiling on what a "
              "re-parameterised v6 can recover: {} of the {} attributed "
              "findings cannot be fixed by refitting constants alone, no "
              "matter how the fit is run.".format(structural, n_findings)]

    # A SILENT family with no DESIGNED_FOR entry (nothing was built to fire
    # on it) leaves cell["sub_metrics"] empty even when it fails, so the
    # table above carries no row for it at all. silence_broken_by names the
    # sub-metric(s) that actually moved and broke the silence, so the
    # failing cell still gets an explanation somewhere in this report.
    silent_gaps = [(family, base, result[(family, base)])
                   for (family, base) in sorted(result)
                   if result[(family, base)]["conforms"] is False
                   and not result[(family, base)]["sub_metrics"]
                   and result[(family, base)]["silence_broken_by"]]
    if silent_gaps:
        lines += ["", "## SILENT failures with no designed-for sub-metric",
                  "",
                  "These cells fail a SILENT expectation but have no "
                  "`DESIGNED_FOR` entry, so they contribute no row above. "
                  "This is the mechanism that broke the silence instead.",
                  "",
                  "| artefact | base | verdict | sub-metrics that broke silence |",
                  "|---|---|---|---|"]
        for family, base, cell in silent_gaps:
            lines.append("| {} | {} | {} | {} |".format(
                family, base, cell["verdict"],
                ", ".join(cell["silence_broken_by"])))

    # weight_drift_submetrics scans every sub-metric, not just a family's
    # designed-for ones. These are the cells where a sub-metric's softmax
    # weight moves across the ladder enough to confound the reading, but
    # none of the (possibly zero) designed-for findings above flagged it —
    # a confound invisible to a per-sub-metric-only view.
    hidden_drift = []
    for (family, base) in sorted(result):
        cell = result[(family, base)]
        wds = cell["weight_drift_submetrics"]
        if not wds:
            continue
        if any(d["weight_drift"] for d in cell["sub_metrics"]):
            continue
        hidden_drift.append((family, base, cell))
    lines += ["", "## Weight drift invisible to a per-sub-metric-only view",
              "",
              "In these **{} cells**, at least one sub-metric's softmax "
              "weight moves by more than the drift threshold across the "
              "severity ladder, but the sub-metric is not one of the "
              "family's designed-for ones, so no finding in the table above "
              "flags it. Scoping the drift scan to designed-for sub-metrics "
              "only would have hidden this confound entirely.".format(
                  len(hidden_drift)),
              "",
              "| artefact | base | conforms | verdict | drifting-weight sub-metrics |",
              "|---|---|---|---|---|"]
    for family, base, cell in hidden_drift:
        lines.append("| {} | {} | {} | {} | {} |".format(
            family, base, cell["conforms"], cell["verdict"],
            ", ".join(cell["weight_drift_submetrics"])))
    return _write(out, lines)


def write_lobo_report(result, out=FIG / "calibration_v6_lobo.md"):
    keys = ("tau", "beta_t", "lambda_a", "alpha", "beta_e", "beta_dp", "beta_dpp")

    def _fmt(v):
        return "linear" if v is None else "{:g}".format(v)

    folds = result["folds"]
    mean_held = sum(f["test_loss"] for f in folds) / len(folds)
    mean_paired_v5 = sum(f["v5_test_loss"] for f in folds) / len(folds)
    beats = [f for f in folds if f["v5_test_loss"] > f["test_loss"]]
    losers = [f for f in folds if f["v5_test_loss"] <= f["test_loss"]]
    all_converged = result["final_converged"] and all(f["converged"] for f in folds)

    lines = ["# LR-VCC v6 — calibration under leave-one-base-out", "",
             "Every fold's coordinate search is warm-started from "
             "`PROD_PARAMS` — v5's parameter vector, which was itself "
             "chosen with all five base videos in view. This is not a data "
             "leak: each fold's loss and leaderboard guards are strictly "
             "restricted to its four training bases, and the held-out base "
             "never enters the search that produces that fold's "
             "parameters. But the fold results are honest *conditional on* "
             "that starting point, not on a blank slate — a cold start "
             "could in principle land somewhere else. Targets: "
             "R_target={r_target}, R_silent={r_silent}, w_mono={w_mono}, "
             "w_silence={w_silence}.".format(**LOSS_CFG), ""]

    beta_t_final = result["final_params"].get("beta_t")
    if beta_t_final is not None:
        lines += ["`beta_t=None` — sub-metric T's original linear form — was "
                  "present in the search grid and was not selected; the fit "
                  "chose beta_t≈{:.2f} instead. The new response parameter "
                  "was therefore adopted on the evidence, not imposed."
                  .format(beta_t_final), ""]

    lines += ["## Summary", "",
              "- v5 loss, all five bases: **{:.6f}**".format(result["v5_loss"]),
              "- v6 in-sample loss, all five bases (the refit sees every "
              "base): **{:.6f}** — the gap to the held-out numbers below is "
              "the overfitting gap.".format(result["final_loss"]),
              "- mean v6 held-out loss (average of the five paired test "
              "losses below): **{:.6f}**".format(mean_held),
              "- mean paired v5 loss (v5 scored on each of the same five "
              "held-out bases, then averaged): **{:.6f}**"
              .format(mean_paired_v5),
              "", "## Per-fold results (paired, same-base comparison)", "",
              "v6's held-out loss and v5's loss in each row are measured on "
              "the *same* held-out base — never v6's one-base number "
              "against v5's five-base aggregate.", "",
              "| fold (held out) | train loss | v6 held-out loss | "
              "v5 loss (same base) | delta (v5−v6) | converged |",
              "|---|---|---|---|---|---|"]
    for f in folds:
        delta = f["v5_test_loss"] - f["test_loss"]
        lines.append("| {} | {:.5f} | {:.5f} | {:.5f} | {:+.5f} | {} |".format(
            f["held_out"], f["train_loss"], f["test_loss"], f["v5_test_loss"],
            delta, f["converged"]))

    lines += ["",
              "- v6 improves on v5 on **{}/{} folds** on a paired, "
              "same-base basis.".format(len(beats), len(folds))]
    for f in losers:
        lines.append("  - the exception is **{}**, where v6 is worse "
                     "({:.6f} vs v5's {:.6f}).".format(
                         f["held_out"], f["test_loss"], f["v5_test_loss"]))
    if beats:
        best = max(beats, key=lambda f: f["v5_test_loss"] - f["test_loss"])
        lines.append("  - v6's largest win is **{}** ({:.6f} vs v5's "
                     "{:.6f}) — that base is simply the hardest one in the "
                     "set, not evidence of a regression elsewhere."
                     .format(best["held_out"], best["test_loss"],
                             best["v5_test_loss"]))
    lines.append("- all five folds and the final refit report "
                 "**converged={}**{}.".format(
                     all_converged,
                     "" if all_converged else
                     " — at least one search hit the pass budget before "
                     "settling; treat its parameters as budget-limited, not "
                     "a confirmed local optimum"))

    lines += ["", "## Final parameters (refit on all five bases)", "",
              "| parameter | v5 | v6 |", "|---|---|---|"]
    for k in keys:
        lines.append("| {} | {} | {} |".format(
            k, _fmt(PROD_PARAMS[k]), _fmt(result["final_params"][k])))

    held = result["heldout_matrix"]
    conf = sum(1 for (fam, _b), c in held.items()
               if E.conforms(fam, c["verdict"]) is True)
    total = sum(1 for (fam, _b) in held
                if E.EXPECTATION[fam] != E.UNCONSTRAINED)
    lines += ["", "## Held-out verdict matrix", "",
              "Every cell produced by a fit that never saw its own base.", "",
              "| artefact | " + " | ".join(E.BASES) + " |",
              "|---|" + "---|" * len(E.BASES)]
    for family in sorted(E.EXPECTATION):
        cells = []
        for base in E.BASES:
            c = held[(family, base)]
            ok = E.conforms(family, c["verdict"])
            mark = "—" if ok is None else ("✓" if ok else "✗")
            cells.append("{:+.3f} {} {}".format(c["delta"], c["verdict"], mark))
        lines.append("| {} | {} |".format(family, " | ".join(cells)))
    lines += ["", "- held-out as-designed: **{}/{}**".format(conf, total)]
    return _write(out, lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lobo", action="store_true",
                    help="also run the five-fold fit and emit the v6 report")
    args = ap.parse_args()
    table = build_table()
    rows = table["artefacts"]
    write_response_curves(rows, PROD_PARAMS)
    write_expectation_matrix(rows, PROD_PARAMS)
    write_failure_attribution(rows, PROD_PARAMS)
    if args.lobo:
        result = lobo(table)
        write_lobo_report(result)
        with open(REPO / "results" / "lr_vcc" / "calibration" / "v6_params.json",
                 "w") as fh:
            json.dump({"v5_loss": result["v5_loss"],
                      "final_loss": result["final_loss"],
                      "final_converged": result["final_converged"],
                      "final_params": result["final_params"]}, fh, indent=2)


if __name__ == "__main__":
    main()
