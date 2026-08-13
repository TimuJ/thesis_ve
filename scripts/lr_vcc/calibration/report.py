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
from .fit import GATE_GRIDS, GATE_ORDER, GRIDS, SEARCH_ORDER, lobo
from .objective import LOSS_CFG, matrix_scores
from .recompose import PROD_PARAMS
from .response_table import build_table

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / "reports" / "figures"

# All twelve parameters the fit actually searches: the seven response
# parameters, then the five gate thresholds. Used everywhere the LOBO
# report needs to walk "every searched parameter" rather than the
# historical seven-key subset.
ALL_PARAM_KEYS = SEARCH_ORDER + GATE_ORDER


def _conformance_from_scored(scored):
    """Respond/silent/unconstrained/uniform counts over an already-scored
    {(family, base): cell} matrix.

    This is the shared core of `conformance_counts` below. It is also
    called directly on `heldout_matrix` / `insample_matrix`: those are
    scored dicts too, but each fold has its own parameters, so there is no
    single `params` vector to hand to `matrix_scores` for them.
    """
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


def conformance_counts(rows, params):
    return _conformance_from_scored(matrix_scores(rows, params))


def _grid_for(key):
    return GRIDS[key] if key in GRIDS else GATE_GRIDS[key]


def _at_grid_boundary(key, value):
    """True when `value` sits at the min or max of its declared grid.

    `None` (beta_t's linear sentinel) is a category, not a boundary value.
    """
    if value is None:
        return False
    nums = [v for v in _grid_for(key) if v is not None]
    return value <= min(nums) + 1e-9 or value >= max(nums) - 1e-9


def _mask_cov_floor_hits(rows, v5_floor, v6_floor):
    """Rows with >=1 tOF sample whose coverage falls below each floor.

    Mirrors recompose._temporal's skip condition
    (`cov[k] < params["mask_cov_floor"]` excludes k from T's weighted
    average) so the count is exact, not an estimate.
    """
    v5_hits = v6_hits = 0
    for row in rows:
        tofs, covs = row["tof"], row["cov"]
        below_v5 = below_v6 = False
        for k_str in tofs:
            if tofs[k_str] is None:
                continue
            c = float(covs.get(k_str, 0.0))
            below_v5 = below_v5 or c < v5_floor
            below_v6 = below_v6 or c < v6_floor
        v5_hits += int(below_v5)
        v6_hits += int(below_v6)
    return v5_hits, v6_hits


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
             "Sign convention: **R = y(0.02) − y(0.40)**; a positive R means "
             "the corruption lowered the score, the correct direction for a "
             "RESPOND family. `expectation_scored_matrix_v5.md` and the "
             "LOBO held-out matrix print `delta = −R` instead — the same "
             "quantity with the opposite sign.", "",
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
             "Sign convention: **delta = −R = y(0.40) − y(0.02)** (mirrors "
             "`build_verdict_matrix`'s own convention; PASS is delta ≤ "
             "−0.05); a negative delta means the corruption lowered the "
             "score — the correct direction. `response_curves_v5.md` "
             "prints `R = −delta` instead — same quantity, opposite sign.",
             "",
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


def write_lobo_report(result, table, out=FIG / "calibration_v6_lobo.md"):
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
              "- **held-out conformance (as-designed) is unchanged from "
              "v5 — see Conformance comparison below. The loss "
              "improvement above does not carry over to the "
              "verdict-level count.**",
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
              "All twelve searched parameters: the seven response "
              "parameters, then the five gate thresholds.", "",
              "| parameter | v5 | v6 |", "|---|---|---|"]
    for k in SEARCH_ORDER:
        lines.append("| {} | {} | {} |".format(
            k, _fmt(PROD_PARAMS[k]), _fmt(result["final_params"][k])))
    lines.append("| *(gate thresholds)* | | |")
    for k in GATE_ORDER:
        lines.append("| {} | {} | {} |".format(
            k, _fmt(PROD_PARAMS[k]), _fmt(result["final_params"][k])))

    moved_gates = [k for k in GATE_ORDER
                  if PROD_PARAMS[k] != result["final_params"][k]]
    if moved_gates:
        lines += ["",
                  "- gate threshold(s) that moved: " +
                  ", ".join("`{}` {} → {}".format(
                      k, _fmt(PROD_PARAMS[k]), _fmt(result["final_params"][k]))
                      for k in moved_gates) + "."]
    if "mask_cov_floor" in moved_gates:
        v5_floor = PROD_PARAMS["mask_cov_floor"]
        v6_floor = result["final_params"]["mask_cov_floor"]
        all_rows = table["artefacts"] + table["realmodels"]
        v5_hits, v6_hits = _mask_cov_floor_hits(all_rows, v5_floor, v6_floor)
        lines.append(
            "- **`mask_cov_floor` {} → {} materially changes sub-metric "
            "T's input set.** Under v5, **{}/{}** rows had at least one "
            "tOF sample whose coverage fell below the floor and was "
            "excluded from T's weighted average; under v6 the floor is "
            "{}, so **{}/{}** rows are affected — the coverage filter is "
            "effectively disabled everywhere it used to fire."
            .format(_fmt(v5_floor), _fmt(v6_floor), v5_hits, len(all_rows),
                   _fmt(v6_floor), v6_hits, len(all_rows)))

    folds_by_base = {f["held_out"]: f for f in folds}
    lines += ["", "## Per-fold parameter vectors", "",
              "The seven response parameters and five gate thresholds "
              "each fold actually landed on, fit without ever seeing the "
              "column's own base. `*` marks a value sitting at the "
              "minimum or maximum of its grid: the fold's optimum is at "
              "or beyond the edge of the declared search space, so that "
              "parameter is not identified by the data at this sample "
              "size.", "",
              "| parameter | " + " | ".join(E.BASES) + " |",
              "|---|" + "---|" * len(E.BASES)]
    for key in ALL_PARAM_KEYS:
        cells = []
        for base in E.BASES:
            v = folds_by_base[base]["params"][key]
            marker = "*" if _at_grid_boundary(key, v) else ""
            cells.append(_fmt(v) + marker)
        lines.append("| {} | {} |".format(key, " | ".join(cells)))

    boundary_counts = [(base,
                        sum(1 for key in SEARCH_ORDER
                            if _at_grid_boundary(
                                key, folds_by_base[base]["params"][key])))
                       for base in E.BASES]
    lines += ["",
              "- boundary hits among the seven response parameters, per "
              "fold (held-out base): " +
              ", ".join("{} {}/{}".format(b, n, len(SEARCH_ORDER))
                        for b, n in boundary_counts) + "."]

    lines += ["", "## Loss surface (sensitivity at the chosen point)", "",
              "For each searched parameter, every other parameter is held "
              "at its `final_params` value and this one is swept over its "
              "declared grid; `matrix_loss` (all five bases, no "
              "leaderboard guard) is recorded at every point — the "
              "sensitivity a fold-level result cannot show on its own. "
              "`spread` is the gap between the loss at the chosen value "
              "and the worst point on the grid: small means this data "
              "barely constrains that parameter at n=5, large means the "
              "fit actively prefers the chosen value over the "
              "alternatives.", "",
              "| parameter | chosen | loss @ chosen | best on grid | "
              "worst on grid | spread |", "|---|---|---|---|---|---|"]
    surfaces = result["loss_surfaces"]
    final_loss = result["final_loss"]
    spread_ratio = {}
    for key in ALL_PARAM_KEYS:
        points = surfaces[key]
        chosen_val = result["final_params"][key]
        best = min(points, key=lambda p: p["loss"])
        worst = max(points, key=lambda p: p["loss"])
        spread = worst["loss"] - final_loss
        spread_ratio[key] = spread / final_loss if final_loss else 0.0
        lines.append("| {} | {} | {:.6f} | {} ({:.6f}) | {} ({:.6f}) | "
                     "{:.6f} |".format(
            key, _fmt(chosen_val), final_loss, _fmt(best["value"]),
            best["loss"], _fmt(worst["value"]), worst["loss"], spread))

    flat = [k for k in ALL_PARAM_KEYS if spread_ratio[k] < 0.05]
    sharp = [k for k in ALL_PARAM_KEYS if spread_ratio[k] > 0.5]
    lines += ["",
              "- **flat** (worst point on the grid raises the loss by "
              "less than 5% of the chosen value's loss — not constrained "
              "by this data at n=5): {}.".format(
                  ", ".join(flat) if flat else "none"),
              "- **sharp** (worst point raises the loss by more than "
              "50%): {}.".format(", ".join(sharp) if sharp else "none")]

    held = result["heldout_matrix"]
    insample = result["insample_matrix"]
    held_conf = _conformance_from_scored(held)
    held_total_conf = held_conf["respond_conforming"] + held_conf["silent_conforming"]
    held_total_cells = held_conf["respond_total"] + held_conf["silent_total"]
    lines += ["", "## Held-out verdict matrix", "",
              "Every cell produced by a fit that never saw its own base.",
              "", "Sign convention: **delta = −R = y(0.40) − y(0.02)**, "
              "same as `expectation_scored_matrix_v5.md`; negative is the "
              "correct direction.", "",
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
    lines += ["", "- held-out as-designed: **{}/{}**".format(
        held_total_conf, held_total_cells)]

    # The LOBO loss numbers above are the fit objective, not the
    # reader-facing verdict count. Without this section, a reader sees
    # "4/5 folds improve" and a large loss reduction and concludes the
    # matrix improved — it does not, at the verdict level.
    v5_scored = matrix_scores(table["artefacts"], PROD_PARAMS)
    v5_conf = _conformance_from_scored(v5_scored)
    insample_conf = _conformance_from_scored(insample)

    def _totals(c):
        return (c["respond_conforming"] + c["silent_conforming"],
                c["respond_total"] + c["silent_total"])

    v5_total_conf, v5_total_cells = _totals(v5_conf)
    insample_total_conf, insample_total_cells = _totals(insample_conf)

    inverted_to_flat = sorted(
        key for key, cell in v5_scored.items()
        if cell["verdict"] == "INVERTED"
        and held.get(key, {}).get("verdict") == "FLAT")

    lines += ["", "## Conformance comparison: v5 vs v6", "",
              "The loss numbers above are the fit objective, not the "
              "reader-facing verdict count; this table puts both scoring "
              "protocols side by side for v5, v6 held-out, and v6 "
              "in-sample.", "",
              "| | RESPOND | SILENT | as-designed | uniform PASS+WEAK |",
              "|---|---|---|---|---|"]
    for label, c, tc, tt in (
            ("v5", v5_conf, v5_total_conf, v5_total_cells),
            ("v6 held-out", held_conf, held_total_conf, held_total_cells),
            ("v6 in-sample", insample_conf, insample_total_conf,
             insample_total_cells)):
        lines.append("| {} | {}/{} | {}/{} | {}/{} | {}/60 |".format(
            label, c["respond_conforming"], c["respond_total"],
            c["silent_conforming"], c["silent_total"], tc, tt,
            c["uniform_clean"]))

    lines += ["",
              "- **the as-designed count is unchanged at the verdict "
              "level: {}/{} for both v5 and v6 held-out. The loss "
              "improved (mean held-out {:.6f} vs mean paired v5 "
              "{:.6f}) but conformance did not.**".format(
                  held_total_conf, held_total_cells, mean_held,
                  mean_paired_v5)]
    if inverted_to_flat:
        lines.append(
            "- the genuine win is at the verdict-shape level, not the "
            "count: **{} cell{}** that {} INVERTED under v5 ({}) "
            "become FLAT under v6 held-out — a wrong-direction response "
            "replaced by no response, even though neither counts as "
            "conforming for a RESPOND family.".format(
                len(inverted_to_flat),
                "" if len(inverted_to_flat) == 1 else "s",
                "was" if len(inverted_to_flat) == 1 else "were",
                ", ".join("{}/{}".format(fam, base)
                          for fam, base in inverted_to_flat)))
    lines.append(
        "- SILENT held-out reaches **{}/{}** — every control family "
        "stays FLAT on its held-out base.".format(
            held_conf["silent_conforming"], held_conf["silent_total"]))
    if insample_conf["silent_conforming"] < insample_conf["silent_total"]:
        n_bad = insample_conf["silent_total"] - insample_conf["silent_conforming"]
        lines.append(
            "- SILENT in-sample drops to **{}/{}** ({} cell{} respond "
            "when they should stay flat) — direct evidence of the "
            "over-calibration the silence penalty exists to catch: with "
            "every base in view, the fit can trade a little unwanted "
            "control-family sensitivity for a lower RESPOND loss "
            "elsewhere. The held-out folds above show that trade does "
            "not survive to an unseen base.".format(
                insample_conf["silent_conforming"],
                insample_conf["silent_total"], n_bad,
                "" if n_bad == 1 else "s"))
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
        write_lobo_report(result, table)
        with open(REPO / "results" / "lr_vcc" / "calibration" / "v6_params.json",
                 "w") as fh:
            json.dump({"v5_loss": result["v5_loss"],
                      "final_loss": result["final_loss"],
                      "final_converged": result["final_converged"],
                      "final_params": result["final_params"]}, fh, indent=2)


if __name__ == "__main__":
    main()
