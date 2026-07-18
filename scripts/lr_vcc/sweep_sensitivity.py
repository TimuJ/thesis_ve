"""Hyperparameter sensitivity sweep + leave-one-out ablation from cached JSONs.

Recomposes LR-VCC v5 (no video re-scanning) over two surfaces:
  - real-model ranking: mgld / uav / flashvsr x 5 videos
  - synthetic verdict matrix: 12 artefact families x 5 bases x 5 severities

Modes:
  gate   — reproduce the stored production composites (hard prerequisite)
  sweep  — grid over (dprime_beta, dprime2_beta, tau) + one-at-a-time
           (color_hist_alpha, color_slope_beta); headline-stability tables
  loo    — drop each of the 7 sub-metrics at production settings
  all    — gate, then sweep, then loo

Usage (repo root):
  python -m scripts.lr_vcc.sweep_sensitivity --mode all
"""
import argparse
import glob
import itertools
import json
import os
import re
from pathlib import Path
from statistics import mean

from .run_lr_vcc import evaluate_one_video
from .build_verdict_matrix import verdict

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
LRV = RES / "lr_vcc"
EVAL = RES / "synthetic_artefacts_eval"

PROD = dict(color_hist_alpha=0.394, color_slope_beta=200.0,
            dprime_beta=0.5, dprime2_beta=3.0,
            temporal_weight="uniform", tau=0.2)

ARTEFACTS = [
    "background_drift", "chunk_boundary", "color_drift", "flicker",
    "flip_channel_shuffle", "flip_elastic", "flip_horizontal", "flip_invert",
    "flip_periodic", "flip_transpose", "identity_degradation", "identity_drift",
]

_SEV_RE = re.compile(r"^(?P<base>.+)_sev(?P<sev>\dp\d+)$")


def _newest(pattern):
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(pattern)
    return files[-1]


def _artefact_identity(art):
    d = EVAL / "identity" / art
    for name in ("_merged_v5.json", "_merged_v4.json"):
        if (d / name).is_file():
            return str(d / name)
    for pattern in ("results_merged*.json", "results_*_eval_results.json"):
        try:
            return _newest(d / pattern)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(d)


def realmodel_units(identity_variant="production", closeup="production"):
    """[(unit_name, kwargs_common)] — one unit per method.

    Production provenance (verified bit-exact against stored composites):
    the July-2 mgld/uav v5 runs applied NO closeup map (diagnostics record
    closeup_bbox_p50=None) and used the PRE-fps-override identity files
    (uav KZ fused 0.75055 = fps_fixed; the corrected fps_overrides gives
    0.62916); flashvsr used mgld's closeup map (mapmgld variant).

    identity_variant: "production" (fps_fixed) | "corrected" (fps_overrides)
    closeup: "production" (only flashvsr gated) | True (all gated) | False
    """
    units = []
    for m in ("mgld", "uav", "flashvsr"):
        if m == "flashvsr":
            identity = _newest(EVAL / "identity" / "flashvsr" /
                               "results_*_eval_results.json")
            gated = closeup in ("production", True)
            closeup_map = (json.load(open(LRV / "closeup_map" / "mgld.json"))
                           if gated else {})
            slope_dir = LRV / "color_slope" / "flashvsr"
        else:
            idir = ("identity_fps_fixed" if identity_variant == "production"
                    else "identity_fps_overrides")
            identity = _newest(RES / "vbench2_anatomy" / idir / m /
                               "results_*_eval_results.json")
            closeup_map = (json.load(open(LRV / "closeup_map" / (m + ".json")))
                           if closeup is True else {})
            slope_dir = LRV / "color_slope" / (m + "_synthetic_mp4")
        units.append((m, dict(
            clip_iqa_dir=LRV / "clip_iqa" / m,
            tof_dir=RES / "long_range_temporal" / m,
            identity_results=identity,
            closeup_map=closeup_map,
            color_hist_dir=LRV / "color_histogram" / m,
            color_slope_dir=slope_dir,
            color_hist_anchor_dir=LRV / "color_hist_anchor_realmodels" / m,
            clip_trajectory_dir=LRV / "clip_trajectory_realmodels" / m,
        )))
    return units


def artefact_units():
    units = []
    for art in ARTEFACTS:
        units.append((art, dict(
            clip_iqa_dir=EVAL / "clip_iqa" / art,
            tof_dir=EVAL / "tof_tlp" / art,
            identity_results=_artefact_identity(art),
            closeup_map={},
            color_hist_dir=LRV / "color_histogram" / art,
            color_slope_dir=LRV / "color_slope" / art,
            color_hist_anchor_dir=LRV / "color_hist_anchor" / art,
            clip_trajectory_dir=LRV / "clip_trajectory" / art,
        )))
    return units


def compose_unit(unit, cfg, drop=None):
    """{clip_id: lr_vcc} for every clip_iqa JSON in the unit."""
    name, u = unit
    out = {}
    for fa in sorted(Path(u["clip_iqa_dir"]).glob("*_clip_iqa.json")):
        base = fa.name.replace("_clip_iqa.json", "")
        ft = Path(u["tof_dir"]) / (base + "_tof_tlp.json")
        if not ft.is_file():
            continue
        r = evaluate_one_video(
            video_id=base,
            clip_iqa_path=str(fa),
            tof_path=str(ft),
            identity_results_path=str(u["identity_results"]),
            closeup_bbox_p50=u["closeup_map"].get(base),
            color_hist_path=str(Path(u["color_hist_dir"]) / (base + "_color_hist.json")),
            color_hist_alpha=cfg["color_hist_alpha"],
            color_slope_path=str(Path(u["color_slope_dir"]) / (base + "_color_slope.json")),
            color_slope_beta=cfg["color_slope_beta"],
            color_hist_anchor_path=str(Path(u["color_hist_anchor_dir"]) / (base + "_color_hist_anchor.json")),
            dprime_beta=cfg["dprime_beta"],
            clip_trajectory_path=str(Path(u["clip_trajectory_dir"]) / (base + "_clip_trajectory.json")),
            dprime2_beta=cfg["dprime2_beta"],
            temperature=cfg["tau"],
            temporal_weight=cfg["temporal_weight"],
            drop=drop,
        )
        out[base] = (r["lr_vcc"], r["low_confidence"])
    return out


def real_summary(cfg, drop=None):
    per_method = {}
    for unit in realmodel_units():
        vals = compose_unit(unit, cfg, drop=drop)
        per_method[unit[0]] = {k: v[0] for k, v in vals.items()
                               if not v[1]}  # high-confidence only, as production
    means = {m: mean(v.values()) for m, v in per_method.items()}
    order = ">".join(sorted(means, key=means.get, reverse=True))
    videos = sorted(per_method["mgld"])
    mgld_beats_uav = sum(per_method["mgld"][v] > per_method["uav"][v]
                         for v in videos)
    return {"means": {m: round(x, 4) for m, x in means.items()},
            "order": order,
            "mgld_beats_uav_pervideo": f"{mgld_beats_uav}/{len(videos)}",
            "per_video": per_method}


def artefact_deltas(cfg, drop=None):
    deltas = {}
    for unit in artefact_units():
        vals = compose_unit(unit, cfg, drop=drop)
        by = {}
        for clip, (score, _lc) in vals.items():
            m = _SEV_RE.match(clip)
            if m:
                by[(m["base"], m["sev"])] = score
        for (b, s), v in by.items():
            if s == "0p40" and (b, "0p02") in by:
                deltas[(unit[0], b)] = v - by[(b, "0p02")]
    return deltas


def matrix_summary(deltas, ref_verdicts=None):
    verdicts = {k: verdict(d) for k, d in deltas.items()}
    clean = sum(1 for v in verdicts.values() if v in ("PASS", "WEAK"))
    out = {"clean": f"{clean}/{len(verdicts)}"}
    if ref_verdicts is not None:
        changed = {f"{a}/{b}": f"{ref_verdicts[(a, b)]}->{v}"
                   for (a, b), v in verdicts.items()
                   if ref_verdicts.get((a, b)) != v}
        out["n_changed_vs_prod"] = len(changed)
        out["changed"] = changed
    return out, verdicts


def gate():
    """Reproduce stored production composites; report max abs diff."""
    ok = True
    # real models
    for m, stored_dir in (("mgld", "mgld"), ("uav", "uav"),
                          ("flashvsr", "flashvsr_mapmgld")):
        unit = [u for u in realmodel_units() if u[0] == m][0]
        vals = compose_unit(unit, PROD)
        worst = 0.0
        for f in (LRV / "composite_v5_realmodels" / stored_dir).glob("*.json"):
            if f.name == "_aggregate.json":
                continue
            stored = json.load(open(f))
            got = vals.get(stored["video"])
            if got is None:
                print(f"GATE MISS {m}/{stored['video']}: not recomposed"); ok = False
                continue
            worst = max(worst, abs(got[0] - stored["lr_vcc"]))
        status = "OK " if worst < 1e-9 else ("ok~" if worst < 5e-4 else "FAIL")
        if status == "FAIL":
            ok = False
        print(f"GATE real  {m:9s} max|diff| = {worst:.2e}  {status}")
    # artefacts. The six pre-June-28 families were composed with the (since
    # parked) dispersion gate ON — current code cannot and should not
    # byte-reproduce them; annotated KNOWN, quantified by `provenance` mode.
    GATE_ERA = {"background_drift", "chunk_boundary", "color_drift", "flicker",
                "identity_degradation", "identity_drift"}
    for unit in artefact_units():
        art = unit[0]
        vals = compose_unit(unit, PROD)
        worst = 0.0
        n = 0
        for f in (LRV / "composite_artefacts_v5" / art).glob("*.json"):
            if f.name == "_aggregate.json":
                continue
            stored = json.load(open(f))
            got = vals.get(stored["video"])
            if got is None:
                continue
            worst = max(worst, abs(got[0] - stored["lr_vcc"])); n += 1
        if worst < 1e-9:
            status = "OK "
        elif art in GATE_ERA:
            status = "KNOWN (dispersion-gate era)"
        else:
            status = "FAIL"
            ok = False
        print(f"GATE artef {art:22s} n={n:2d} max|diff| = {worst:.2e}  {status}")
    return ok


def run_sweep(out_json, out_md):
    grid = [dict(PROD, dprime_beta=a, dprime2_beta=b, tau=t)
            for a, b, t in itertools.product(
                [0.25, 0.5, 1.0, 2.0], [1.0, 2.0, 3.0, 5.0], [0.1, 0.2, 0.5])]
    extras = ([dict(PROD, color_hist_alpha=a) for a in (0.2, 0.8)] +
              [dict(PROD, color_slope_beta=b) for b in (100.0, 300.0)])
    _, prod_verdicts = matrix_summary(artefact_deltas(PROD))
    rows = []
    for cfg in grid + extras:
        real = real_summary(cfg)
        mat, _ = matrix_summary(artefact_deltas(cfg), prod_verdicts)
        rows.append({"cfg": {k: cfg[k] for k in
                             ("dprime_beta", "dprime2_beta", "tau",
                              "color_hist_alpha", "color_slope_beta")},
                     "real": {k: real[k] for k in
                              ("means", "order", "mgld_beats_uav_pervideo")},
                     "matrix": mat})
        print(f"cfg {rows[-1]['cfg']} -> order={real['order']} "
              f"clean={mat['clean']} changed={mat['n_changed_vs_prod']}")
    json.dump(rows, open(out_json, "w"), indent=2)

    lines = ["# LR-VCC v5 — hyperparameter sensitivity sweep",
             "",
             f"Production: dprime_beta=0.5, dprime2_beta=3.0, tau=0.2, "
             f"alpha=0.394, slope_beta=200. {len(rows)} configs recomposed "
             "from cached sub-metric JSONs (no video re-scanning).",
             "",
             "| b_D' | b_D'' | tau | alpha | b_E | 3-method order | MGLD>UAV "
             "per-video | matrix clean | cells changed vs prod |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["cfg"]
        lines.append(
            f"| {c['dprime_beta']} | {c['dprime2_beta']} | {c['tau']} "
            f"| {c['color_hist_alpha']} | {c['color_slope_beta']:g} "
            f"| {r['real']['order']} | {r['real']['mgld_beats_uav_pervideo']} "
            f"| {r['matrix']['clean']} | {r['matrix']['n_changed_vs_prod']} |")
    n_order = sum(1 for r in rows if r["real"]["order"] != "flashvsr>mgld>uav")
    n_pv = sum(1 for r in rows
               if r["real"]["mgld_beats_uav_pervideo"] != "5/5")
    lines += ["",
              f"Headline stability: 3-method mean order flashvsr>mgld>uav holds "
              f"in {len(rows)-n_order}/{len(rows)} configs; MGLD>UAV on every "
              f"video holds in {len(rows)-n_pv}/{len(rows)} configs."]
    Path(out_md).write_text("\n".join(lines) + "\n")
    print("wrote", out_md)


def run_loo(out_json, out_md):
    subs = ["appearance", "temporal", "identity", "color_stability",
            "color_slope", "color_hist_anchor", "clip_trajectory"]
    _, prod_verdicts = matrix_summary(artefact_deltas(PROD))
    prod_real = real_summary(PROD)
    rows = []
    for s in [None] + subs:
        drop = [s] if s else None
        real = real_summary(PROD, drop=drop)
        mat, _ = matrix_summary(artefact_deltas(PROD, drop=drop), prod_verdicts)
        rows.append({"dropped": s or "(none)",
                     "real": {k: real[k] for k in
                              ("means", "order", "mgld_beats_uav_pervideo")},
                     "matrix": mat})
        print(f"drop={s or '(none)':18s} order={real['order']} "
              f"clean={mat['clean']} changed={mat.get('n_changed_vs_prod')}")
    json.dump({"production_real": {k: prod_real[k] for k in
                                   ("means", "order")},
               "rows": rows}, open(out_json, "w"), indent=2)

    lines = ["# LR-VCC v5 — leave-one-out sub-metric ablation",
             "",
             "Each row recomposes the production configuration with one "
             "sub-metric removed (cached JSONs, no re-scanning).",
             "",
             "| dropped | 3-method order | MGLD>UAV per-video | matrix clean "
             "| cells changed vs prod | changed cells |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        ch = r["matrix"].get("changed", {})
        ch_str = "; ".join(f"{k}: {v}" for k, v in sorted(ch.items())[:6])
        if len(ch) > 6:
            ch_str += f" (+{len(ch)-6} more)"
        lines.append(
            f"| {r['dropped']} | {r['real']['order']} "
            f"| {r['real']['mgld_beats_uav_pervideo']} "
            f"| {r['matrix']['clean']} "
            f"| {r['matrix'].get('n_changed_vs_prod', 0)} | {ch_str or '—'} |")
    Path(out_md).write_text("\n".join(lines) + "\n")
    print("wrote", out_md)


def run_provenance(out_md):
    """Quantify stored-vs-uniform matrix and real-model input variants."""
    from .build_verdict_matrix import collect_deltas
    stored = {k: verdict(d) for k, d in
              collect_deltas(LRV / "composite_artefacts_v5").items()}
    uniform_deltas = artefact_deltas(PROD)
    uniform = {k: verdict(d) for k, d in uniform_deltas.items()}
    changed = {f"{a}/{b}": f"{stored[(a, b)]}->{v}"
               for (a, b), v in uniform.items()
               if stored.get((a, b)) != v}
    clean_stored = sum(1 for v in stored.values() if v in ("PASS", "WEAK"))
    clean_uniform = sum(1 for v in uniform.values() if v in ("PASS", "WEAK"))

    variants = [
        ("published (replica)", dict(identity_variant="production",
                                     closeup="production")),
        ("corrected identity (fps overrides)",
         dict(identity_variant="corrected", closeup="production")),
        ("corrected identity + closeup gate on all",
         dict(identity_variant="corrected", closeup=True)),
    ]
    lines = ["# LR-VCC v5 — provenance check (recomposition, current code)",
             "",
             "## Synthetic matrix: published composites vs uniform recompose",
             "",
             "The published 12/12 matrix mixes composition eras: six families "
             "composed with the (since parked) identity dispersion gate ON, "
             "six with it OFF. Uniform current-code recompose (gate off "
             "everywhere):",
             "",
             f"- clean (PASS+WEAK): published {clean_stored}/60 -> uniform "
             f"{clean_uniform}/60",
             f"- cells changing verdict class: {len(changed)}"]
    for k, v in sorted(changed.items()):
        lines.append(f"  - {k}: {v}")
    lines += ["", "## Real-model table under input variants", ""]
    lines += ["| variant | MGLD | UAV | FlashVSR | order | MGLD>UAV/video |",
              "|---|---|---|---|---|---|"]
    for name, kw in variants:
        units = realmodel_units(**kw)
        per = {}
        for unit in units:
            vals = compose_unit(unit, PROD)
            per[unit[0]] = {k: v[0] for k, v in vals.items() if not v[1]}
        means = {m: mean(v.values()) for m, v in per.items()}
        order = ">".join(sorted(means, key=means.get, reverse=True))
        pv = sum(per["mgld"][v] > per["uav"][v] for v in sorted(per["mgld"]))
        lines.append(f"| {name} | {means['mgld']:.4f} | {means['uav']:.4f} "
                     f"| {means['flashvsr']:.4f} | {order} "
                     f"| {pv}/{len(per['mgld'])} |")
    Path(out_md).write_text("\n".join(lines) + "\n")
    print("wrote", out_md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gate", "provenance", "sweep", "loo",
                                       "all"],
                    default="all")
    ap.add_argument("--out_dir", default=str(LRV / "sweeps"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode in ("gate", "all"):
        if not gate() and args.mode == "all":
            raise SystemExit("gate FAILED — fix input paths before sweeping")
    if args.mode in ("provenance", "all"):
        run_provenance(REPO / "reports" / "figures" /
                       "lr_vcc_provenance_check.md")
    if args.mode in ("sweep", "all"):
        run_sweep(out / "sensitivity_sweep.json",
                  REPO / "reports" / "figures" / "sensitivity_sweep_v5.md")
    if args.mode in ("loo", "all"):
        run_loo(out / "loo_ablation.json",
                REPO / "reports" / "figures" / "loo_ablation_v5.md")


if __name__ == "__main__":
    main()
