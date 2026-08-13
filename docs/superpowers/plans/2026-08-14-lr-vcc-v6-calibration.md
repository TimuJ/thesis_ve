# LR-VCC v6 Sensitivity Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit LR-VCC's sub-metric response parameters to the cached severity battery under leave-one-base-out cross-validation, producing a provisional v6 alongside a per-cell failure analysis of the frozen v5 reference.

**Architecture:** A one-time extraction turns the cached sub-metric JSONs into a flat table of raw statistics. A pure-arithmetic recomposer maps (row, parameter vector) → composite, pinned bit-exact against the existing `evaluate_one_video` at production parameters. A severity-response loss scores whole matrices; a deterministic coordinate search minimises it inside LOBO folds. Reporting emits four markdown deliverables. Nothing re-scans video; nothing touches the GPU server.

**Tech Stack:** Python 3.9, stdlib only (`json`, `math`, `statistics`, `pathlib`, `argparse`), pytest. No numpy needed — the arithmetic is scalar. Existing modules reused: `scripts/lr_vcc/run_lr_vcc.py`, `scripts/lr_vcc/reliability.py`, `scripts/lr_vcc/sweep_sensitivity.py`, `scripts/lr_vcc/build_verdict_matrix.py`.

**Spec:** `docs/superpowers/specs/2026-08-14-metric-v6-calibration-design.md`

## Global Constraints

- Run everything from the repo root; imports use the `scripts.lr_vcc.*` package path (there is no `setup.py`).
- Python 3.9 — no `match`, no `X | Y` type unions in annotations evaluated at runtime.
- **v5 is frozen.** No file under `results/lr_vcc/composite_v5_realmodels/`, `results/lr_vcc/composite_artefacts_v5/`, or `results/lr_vcc/composite_artefacts_v5_uniform/` may be modified. New outputs go to `results/lr_vcc/calibration/`.
- No modifications to `run_lr_vcc.py`, `appearance.py`, `temporal.py`, `identity.py`, `color_stability.py`, `composite.py`, or `reliability.py`. The calibration package reads them and reimplements composition in pure form; the bit-exactness test is what keeps the two in agreement.
- Canonical sub-metric order, used everywhere weights are emitted: `appearance, temporal, identity, color_stability, color_slope, color_hist_anchor, clip_trajectory`.
- Severity ladder: `0p02, 0p05, 0p10, 0p20, 0p40` → `0.02, 0.05, 0.10, 0.20, 0.40`.
- Bases: `7WHI2L_FDNg, BrRLKMbBTYQ, KZ8p6b1zJ9U, hhszUXL1Cu8, mJog8DlRk_4`.
- Verdict thresholds stay as `build_verdict_matrix.verdict`: PASS ≤ −0.05, WEAK ≤ −0.02, FLAT < +0.02, else INVERTED, all on `Δ = score(0.40) − score(0.02)`.
- The loss works in `R = −Δ` so that larger is better. Never mix the two conventions in one function.
- The identity dispersion gate stays parked (`dispersion_threshold=None`).
- Commit after every task.

---

### Task 1: Freeze v5 as a regression test

Pins the reference before any new code can perturb it. `sweep_sensitivity.gate()` already performs this check as a print-only script; this promotes it to pytest.

**Files:**
- Create: `tests/test_lr_vcc_v5_frozen.py`

**Interfaces:**
- Consumes: `scripts.lr_vcc.sweep_sensitivity.realmodel_units`, `artefact_units`, `compose_unit`, `PROD`
- Produces: nothing importable; a guard other tasks rely on

- [ ] **Step 1: Write the failing test**

```python
"""v5 is frozen. These composites must never change."""
import json
from pathlib import Path

import pytest

from scripts.lr_vcc.sweep_sensitivity import (
    PROD, artefact_units, compose_unit, realmodel_units,
)

REPO = Path(__file__).resolve().parents[1]
LRV = REPO / "results" / "lr_vcc"

# The six flip families were composed with the current code path (dispersion
# gate off) and reproduce bit-exact. The other six predate the parking of that
# gate and are deliberately excluded — see sweep_sensitivity.gate().
GATE_ERA_FREE = [
    "flip_channel_shuffle", "flip_elastic", "flip_horizontal",
    "flip_invert", "flip_periodic", "flip_transpose",
]


@pytest.mark.parametrize("method,stored_dir", [
    ("mgld", "mgld"), ("uav", "uav"), ("flashvsr", "flashvsr_mapmgld"),
])
def test_v5_realmodel_composites_frozen(method, stored_dir):
    unit = [u for u in realmodel_units() if u[0] == method][0]
    got = compose_unit(unit, PROD)
    n = 0
    for f in sorted((LRV / "composite_v5_realmodels" / stored_dir).glob("*.json")):
        if f.name == "_aggregate.json":
            continue
        stored = json.load(open(f))
        assert stored["video"] in got, stored["video"]
        assert abs(got[stored["video"]][0] - stored["lr_vcc"]) < 1e-12
        n += 1
    assert n == 5


@pytest.mark.parametrize("artefact", GATE_ERA_FREE)
def test_v5_artefact_composites_frozen(artefact):
    unit = [u for u in artefact_units() if u[0] == artefact][0]
    got = compose_unit(unit, PROD)
    n = 0
    for f in sorted((LRV / "composite_artefacts_v5" / artefact).glob("*.json")):
        if f.name == "_aggregate.json":
            continue
        stored = json.load(open(f))
        assert abs(got[stored["video"]][0] - stored["lr_vcc"]) < 1e-12
        n += 1
    assert n == 25
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_lr_vcc_v5_frozen.py -v`
Expected: 9 PASS. If any fail, stop — the cached inputs have drifted and the whole plan rests on them.

- [ ] **Step 3: Commit**

```bash
git add tests/test_lr_vcc_v5_frozen.py
git commit -m "test: pin v5 composites as a frozen regression guard"
```

---

### Task 2: Pre-registered expectations module

**Files:**
- Create: `scripts/lr_vcc/calibration/__init__.py` (empty)
- Create: `scripts/lr_vcc/calibration/expectations.py`
- Test: `tests/test_lr_vcc_expectations.py`

**Interfaces:**
- Produces: `RESPOND`, `SILENT`, `UNCONSTRAINED` (str constants); `EXPECTATION: dict[str, str]`; `DESIGNED_FOR: dict[str, tuple]`; `SEVERITIES: tuple`; `SEVERITY_VALUES: dict[str, float]`; `BASES: tuple`; `SUB_METRICS: tuple`; `conforms(family, verdict) -> bool`

- [ ] **Step 1: Write the failing test**

```python
from scripts.lr_vcc.calibration import expectations as E
from scripts.lr_vcc.sweep_sensitivity import ARTEFACTS


def test_every_family_has_an_expectation():
    assert set(E.EXPECTATION) == set(ARTEFACTS)


def test_every_responding_family_declares_designed_for_submetrics():
    for fam, exp in E.EXPECTATION.items():
        if exp == E.RESPOND:
            assert fam in E.DESIGNED_FOR, fam
            assert set(E.DESIGNED_FOR[fam]) <= set(E.SUB_METRICS), fam


def test_partition_is_eight_three_one():
    counts = {k: sum(1 for v in E.EXPECTATION.values() if v == k)
              for k in (E.RESPOND, E.SILENT, E.UNCONSTRAINED)}
    assert counts == {E.RESPOND: 8, E.SILENT: 3, E.UNCONSTRAINED: 1}


def test_conforms_rules():
    # RESPOND wants a downward move; SILENT wants no move.
    assert E.conforms("flicker", "PASS")
    assert E.conforms("flicker", "WEAK")
    assert not E.conforms("flicker", "FLAT")
    assert not E.conforms("flicker", "INVERTED")
    assert E.conforms("flip_horizontal", "FLAT")
    assert not E.conforms("flip_horizontal", "WEAK")
    assert not E.conforms("flip_horizontal", "INVERTED")
    # UNCONSTRAINED never counts either way.
    assert E.conforms("flip_transpose", "FLAT") is None
    assert E.conforms("flip_transpose", "PASS") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lr_vcc_expectations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.lr_vcc.calibration'`

- [ ] **Step 3: Create the package and module**

```bash
mkdir -p scripts/lr_vcc/calibration && touch scripts/lr_vcc/calibration/__init__.py
```

`scripts/lr_vcc/calibration/expectations.py`:

```python
"""Pre-registered expectations for the LR-VCC severity battery.

Declared BEFORE any calibration is fitted. The six designed-for families and
the flip control family carry predictions stated in the thesis experiments
chapter: flip_invert (histogram-destroying) is caught everywhere,
flip_horizontal / periodic / elastic are invisible, flip_channel_shuffle is
caught partially.

flip_transpose is UNCONSTRAINED on purpose: it preserves the histogram but
destroys geometry, so the pre-registration is genuinely ambiguous. Assigning it
an expectation after its results were seen would be reading the answer off the
data, so it is excluded from the fit objective and reported only.
"""

RESPOND = "RESPOND"
SILENT = "SILENT"
UNCONSTRAINED = "UNCONSTRAINED"

EXPECTATION = {
    # designed-for long-range families
    "color_drift": RESPOND,
    "background_drift": RESPOND,
    "chunk_boundary": RESPOND,
    "flicker": RESPOND,
    "identity_degradation": RESPOND,
    "identity_drift": RESPOND,
    # flip controls with a positive prediction
    "flip_invert": RESPOND,
    "flip_channel_shuffle": RESPOND,
    # flip controls predicted invisible
    "flip_horizontal": SILENT,
    "flip_periodic": SILENT,
    "flip_elastic": SILENT,
    # ambiguous pre-registration
    "flip_transpose": UNCONSTRAINED,
}

SUB_METRICS = ("appearance", "temporal", "identity", "color_stability",
               "color_slope", "color_hist_anchor", "clip_trajectory")

# Which sub-metrics each family was constructed to excite. Used by failure
# attribution to decide which sub-metric "should have fired" in a given cell.
DESIGNED_FOR = {
    "color_drift": ("color_stability", "color_slope", "color_hist_anchor"),
    "background_drift": ("color_hist_anchor", "clip_trajectory", "appearance"),
    "chunk_boundary": ("temporal", "color_stability"),
    "flicker": ("temporal", "appearance"),
    "identity_degradation": ("identity", "appearance"),
    "identity_drift": ("identity", "clip_trajectory"),
    "flip_invert": ("color_stability", "color_hist_anchor", "clip_trajectory",
                    "appearance"),
    "flip_channel_shuffle": ("color_hist_anchor", "clip_trajectory",
                             "appearance"),
}

SEVERITIES = ("0p02", "0p05", "0p10", "0p20", "0p40")
SEVERITY_VALUES = {"0p02": 0.02, "0p05": 0.05, "0p10": 0.10,
                   "0p20": 0.20, "0p40": 0.40}

BASES = ("7WHI2L_FDNg", "BrRLKMbBTYQ", "KZ8p6b1zJ9U", "hhszUXL1Cu8",
         "mJog8DlRk_4")


def conforms(family, verdict):
    """True/False for constrained families, None for UNCONSTRAINED ones.

    RESPOND conforms on PASS or WEAK (a downward response of at least 0.02).
    SILENT conforms only on FLAT.
    """
    exp = EXPECTATION[family]
    if exp == UNCONSTRAINED:
        return None
    if exp == RESPOND:
        return verdict in ("PASS", "WEAK")
    return verdict == "FLAT"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lr_vcc_expectations.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/lr_vcc/calibration/ tests/test_lr_vcc_expectations.py
git commit -m "lr_vcc: pre-registered battery expectations for calibration"
```

---

### Task 3: Response-table extraction

Reads every cached sub-metric JSON once and writes a flat table of the raw statistics that free parameters act on. All 12 families × 25 clips × 6 inputs were verified present, so missing inputs are a hard error.

**Files:**
- Create: `scripts/lr_vcc/calibration/response_table.py`
- Test: `tests/test_lr_vcc_response_table.py`

**Interfaces:**
- Consumes: `sweep_sensitivity.artefact_units`, `realmodel_units`, `ARTEFACTS`
- Produces: `build_table() -> dict` with keys `"artefacts"` and `"realmodels"`, each a list of row dicts; `save(table, path)`; `load(path) -> dict`; `TABLE_PATH`
- Row schema (every key present; `severity`, `closeup_p50` and `dispersion` may be `None`): `unit, clip, base, severity, a_mean, a_std, tof, cov, identity_fused, n_clips, n_clips_with_faces, dispersion, closeup_p50, hist_dist, hist_n_frames, slope_abs, slope_rel, anchor_q14, anchor_rel, clip_q14, clip_rel`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lr_vcc_response_table.py -v`
Expected: FAIL with `ImportError: cannot import name 'response_table'`

- [ ] **Step 3: Write the implementation**

`scripts/lr_vcc/calibration/response_table.py`:

```python
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
        "identity_fused": float(id_pv.get("fused", 0.0)),
        "n_clips": int(id_pv.get("n_clips", 0)),
        "n_clips_with_faces": int(id_pv.get("n_clips_with_faces", 0)),
        "dispersion": clip_score_dispersion(id_pv),
        "closeup_p50": u["closeup_map"].get(base if sev else clip),
        "hist_dist": float(hist_dist),
        "hist_n_frames": int(hist.get("n_frames", 0)),
        "slope_abs": float((slope.get("details") or {})["max_abs_slope"]),
        "slope_rel": float(slope.get("reliability", 0.0)),
        "anchor_q14": _q14(anchor),
        "anchor_rel": float(anchor.get("reliability", 1.0)),
        "clip_q14": _q14(traj),
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lr_vcc_response_table.py -v`
Expected: 5 PASS

- [ ] **Step 5: Generate and inspect the table**

Run: `python -m scripts.lr_vcc.calibration.response_table`
Expected: `wrote .../response_table.json (300 artefact rows, 15 real-model rows)`

- [ ] **Step 6: Commit**

```bash
git add scripts/lr_vcc/calibration/response_table.py tests/test_lr_vcc_response_table.py
git commit -m "lr_vcc: response-table extraction from cached sub-metric JSONs"
```

Note on generated outputs: `.gitignore` line 48 ignores `results/lr_vcc/*`, but `results/lr_vcc/sweeps/*.json` is tracked (force-added, so the ignore rule no longer applies to it). Follow that precedent — the small fit outputs get force-added in Task 8 so the reported numbers are reviewable:

```bash
git add -f results/lr_vcc/calibration/lobo_result.json results/lr_vcc/calibration/v6_params.json
```

`response_table.json` stays out of git: it is ~0.5 MB and regenerates deterministically in seconds from the cached JSONs.

---

### Task 4: Pure recomposer with β_T

The heart of the plan. Must reproduce `evaluate_one_video` bit-exactly at production parameters, which requires keeping T's linear form reachable: `beta_t=None` → `1 − weighted_mean_tof` (v5), a float → `exp(−β_T · weighted_mean_tof)` (the new lever). This refines the spec's continuity note — β_T = 1 only *approximates* v5, so `None` stays in the grid to keep v5 exactly reachable and to let the fit decline the new parameter.

**Files:**
- Create: `scripts/lr_vcc/calibration/recompose.py`
- Test: `tests/test_lr_vcc_recompose.py`

**Interfaces:**
- Consumes: `response_table` rows, `..reliability.below_threshold_penalty`, `..reliability.above_threshold_penalty`
- Produces: `PROD_PARAMS: dict`; `sub_metric_values(row, p) -> list[tuple[str, float, float]]` (name, score, reliability, in canonical order); `composite(row, p) -> dict` with keys `lr_vcc`, `scores`, `reliabilities`, `weights`, `low_confidence`

- [ ] **Step 1: Write the failing test**

```python
import math

import pytest

from scripts.lr_vcc.calibration import recompose as R
from scripts.lr_vcc.calibration import response_table as RT
from scripts.lr_vcc.calibration import expectations as E
from scripts.lr_vcc.sweep_sensitivity import (
    PROD, artefact_units, compose_unit, realmodel_units,
)


@pytest.fixture(scope="module")
def table():
    return RT.build_table()


def test_bit_exact_vs_evaluate_one_video_on_artefacts(table):
    by_unit = {}
    for r in table["artefacts"]:
        by_unit.setdefault(r["unit"], []).append(r)
    checked = 0
    for unit in artefact_units():
        ref = compose_unit(unit, PROD, full=True)
        for row in by_unit[unit[0]]:
            got = R.composite(row, R.PROD_PARAMS)["lr_vcc"]
            assert abs(got - ref[row["clip"]]["lr_vcc"]) < 1e-12, row["clip"]
            checked += 1
    assert checked == 300


def test_bit_exact_vs_evaluate_one_video_on_realmodels(table):
    units = realmodel_units(identity_variant="corrected", closeup=True,
                            methods=("mgld", "uav", "flashvsr"))
    by_unit = {}
    for r in table["realmodels"]:
        by_unit.setdefault(r["unit"], []).append(r)
    checked = 0
    for unit in units:
        ref = compose_unit(unit, PROD, full=True)
        for row in by_unit[unit[0]]:
            got = R.composite(row, R.PROD_PARAMS)["lr_vcc"]
            assert abs(got - ref[row["clip"]]["lr_vcc"]) < 1e-12, row["clip"]
            checked += 1
    assert checked == 15


def test_submetric_order_is_canonical(table):
    names = [n for n, _, _ in R.sub_metric_values(table["artefacts"][0],
                                                  R.PROD_PARAMS)]
    assert tuple(names) == E.SUB_METRICS


def test_beta_t_is_monotone_decreasing_in_tof(table):
    row = dict(table["artefacts"][0])
    p = dict(R.PROD_PARAMS, beta_t=10.0)
    prev = None
    for scale in (1.0, 1.5, 2.0, 3.0):
        r = dict(row, tof={k: (None if v is None else v * scale)
                           for k, v in row["tof"].items()})
        t = [s for n, s, _ in R.sub_metric_values(r, p) if n == "temporal"][0]
        if prev is not None:
            assert t < prev
        prev = t


def test_beta_t_one_approximates_the_linear_v5_form(table):
    """exp(-x) ~= 1 - x over the observed tOF range [0.04, 0.17]."""
    for row in table["artefacts"][:40]:
        lin = [s for n, s, _ in
               R.sub_metric_values(row, dict(R.PROD_PARAMS, beta_t=None))
               if n == "temporal"][0]
        exp1 = [s for n, s, _ in
                R.sub_metric_values(row, dict(R.PROD_PARAMS, beta_t=1.0))
                if n == "temporal"][0]
        assert abs(lin - exp1) < 0.02


def test_low_confidence_flag(table):
    row = table["artefacts"][0]
    out = R.composite(row, R.PROD_PARAMS)
    assert out["low_confidence"] == all(r < 0.2 for r in out["reliabilities"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lr_vcc_recompose.py -v`
Expected: FAIL with `ImportError: cannot import name 'recompose'`

- [ ] **Step 3: Write the implementation**

`scripts/lr_vcc/calibration/recompose.py`:

```python
"""(row, parameter vector) -> LR-VCC composite, as pure arithmetic.

No disk access, no JSON, no video. This is what makes a five-fold search over
a real parameter grid affordable: a full 315-row matrix recomposes in under a
millisecond, against 2.7 s for the JSON-reading path in sweep_sensitivity.

The bit-exactness test against run_lr_vcc.evaluate_one_video is what keeps
this module honest. Any change here that breaks it is a bug here, not there.
"""
import math

from ..reliability import above_threshold_penalty, below_threshold_penalty
from .expectations import SUB_METRICS

# v5 production settings, expressed in this module's parameter vocabulary.
# beta_t=None selects T's original linear form and is what makes bit-exact
# reproduction of v5 possible; a float switches T to the exponential response.
PROD_PARAMS = {
    "lambda_a": 0.5,
    "beta_t": None,
    "alpha": 0.394,
    "beta_e": 200.0,
    "beta_dp": 0.5,
    "beta_dpp": 3.0,
    "tau": 0.2,
    "a_drift_floor": 0.02,
    "a_sat_ceiling": 0.98,
    "mask_cov_floor": 0.10,
    "face_rate_floor": 0.20,
    "closeup_threshold": 0.05,
    "temporal_weight": "uniform",
    "hist_min_frames": 240,
    "low_confidence_floor": 0.2,
    "eps": 1e-6,
}

_WEIGHT_FNS = {
    "log": lambda k: math.log(1 + k),
    "uniform": lambda k: 1.0,
    "sqrt": lambda k: math.sqrt(k),
}


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _appearance(row, p):
    score = _clamp01(row["a_mean"] - p["lambda_a"] * row["a_std"])
    drift_pen = below_threshold_penalty(row["a_std"], p["a_drift_floor"])
    sat_pen = above_threshold_penalty(row["a_mean"], p["a_sat_ceiling"])
    return score, max(0.0, 1.0 - max(drift_pen, sat_pen))


def _temporal(row, p):
    weight_func = _WEIGHT_FNS.get(p["temporal_weight"], _WEIGHT_FNS["log"])
    tofs, covs = row["tof"], row["cov"]
    weighted_sum = weight_total = 0.0
    for k_str in tofs:
        if tofs[k_str] is None:
            continue
        if float(covs.get(k_str, 0.0)) < p["mask_cov_floor"]:
            continue
        w = weight_func(int(k_str))
        weighted_sum += w * float(tofs[k_str])
        weight_total += w
    if weight_total == 0:
        score = 0.0
    else:
        wm = weighted_sum / weight_total
        score = _clamp01(1.0 - wm if p["beta_t"] is None
                         else math.exp(-p["beta_t"] * wm))
    rel_terms = [1.0 - below_threshold_penalty(float(covs.get(k, 0.0)),
                                               p["mask_cov_floor"])
                 for k in tofs]
    reliability = sum(rel_terms) / len(rel_terms) if rel_terms else 0.0
    return score, reliability


def _identity(row, p):
    n_clips = row["n_clips"]
    face_rate = row["n_clips_with_faces"] / n_clips if n_clips > 0 else 0.0
    face_pen = below_threshold_penalty(face_rate, p["face_rate_floor"])
    if row["closeup_p50"] is None:
        closeup_pen = 0.0
    else:
        closeup_pen = above_threshold_penalty(float(row["closeup_p50"]),
                                              p["closeup_threshold"])
    # The dispersion gate stays parked: its penalty is always 0.0.
    return _clamp01(row["identity_fused"]), (1.0 - face_pen) * (1.0 - closeup_pen)


def _exp_sub(raw, beta, reliability):
    return _clamp01(math.exp(-beta * raw)), reliability


def sub_metric_values(row, p):
    """[(name, score, reliability)] in canonical SUB_METRICS order."""
    a_s, a_r = _appearance(row, p)
    t_s, t_r = _temporal(row, p)
    i_s, i_r = _identity(row, p)
    d_s, d_r = _exp_sub(
        row["hist_dist"], p["alpha"],
        _clamp01(1.0 - below_threshold_penalty(row["hist_n_frames"],
                                               p["hist_min_frames"],
                                               sharpness=0.02)))
    e_s, e_r = _exp_sub(row["slope_abs"], p["beta_e"], row["slope_rel"])
    dp_s, dp_r = _exp_sub(row["anchor_q14"], p["beta_dp"], row["anchor_rel"])
    dpp_s, dpp_r = _exp_sub(row["clip_q14"], p["beta_dpp"], row["clip_rel"])
    values = [a_s, t_s, i_s, d_s, e_s, dp_s, dpp_s]
    rels = [a_r, t_r, i_r, d_r, e_r, dp_r, dpp_r]
    return list(zip(SUB_METRICS, values, rels))


def composite(row, p):
    triples = sub_metric_values(row, p)
    scores = [s for _, s, _ in triples]
    rels = [r for _, _, r in triples]
    z = [r / p["tau"] for r in rels]
    z_max = max(z)
    exps = [math.exp(zi - z_max) for zi in z]
    total = sum(exps)
    weights = [e / total for e in exps]
    log_sum = sum(w * math.log(s + p["eps"]) for w, s in zip(weights, scores))
    return {
        "lr_vcc": math.exp(log_sum),
        "scores": scores,
        "reliabilities": rels,
        "weights": weights,
        "low_confidence": all(r < p["low_confidence_floor"] for r in rels),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lr_vcc_recompose.py -v`
Expected: 6 PASS. The two bit-exactness tests are the gate for the whole plan — if they fail, do not proceed to Task 5.

- [ ] **Step 5: Commit**

```bash
git add scripts/lr_vcc/calibration/recompose.py tests/test_lr_vcc_recompose.py
git commit -m "lr_vcc: pure recomposer, bit-exact vs v5, with beta_T response for T"
```

---

### Task 5: Severity-response objective

**Files:**
- Create: `scripts/lr_vcc/calibration/objective.py`
- Test: `tests/test_lr_vcc_objective.py`

**Interfaces:**
- Consumes: `recompose.composite`, `expectations.*`, `..build_verdict_matrix.verdict`
- Produces: `LOSS_CFG: dict`; `response(ladder) -> float`; `monotonicity_violation(ladder) -> float`; `cell_loss(family, ladder, cfg) -> float`; `matrix_scores(rows, params, bases=None) -> dict[(family, base), dict]`; `matrix_loss(rows, params, cfg, bases=None) -> float`; `guards_ok(realmodel_rows, params, bases=None) -> bool`

`ladder` is `{severity_str: lr_vcc}` for all five severities.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from scripts.lr_vcc.calibration import objective as O
from scripts.lr_vcc.calibration import recompose as R
from scripts.lr_vcc.calibration import response_table as RT

FALLING = {"0p02": 0.90, "0p05": 0.85, "0p10": 0.80, "0p20": 0.75, "0p40": 0.70}
FLAT = {s: 0.80 for s in FALLING}
BUMPY = {"0p02": 0.90, "0p05": 0.95, "0p10": 0.80, "0p20": 0.85, "0p40": 0.70}


def test_response_is_negative_delta():
    assert O.response(FALLING) == pytest.approx(0.20)
    assert O.response(FLAT) == pytest.approx(0.0)


def test_monotonicity_violation_counts_upward_steps_only():
    assert O.monotonicity_violation(FALLING) == pytest.approx(0.0)
    assert O.monotonicity_violation(BUMPY) == pytest.approx(0.10)


def test_respond_cell_is_penalised_for_under_responding():
    cfg = O.LOSS_CFG
    assert O.cell_loss("flicker", FLAT, cfg) > 0
    assert O.cell_loss("flicker", FALLING, cfg) == pytest.approx(0.0)


def test_silent_cell_is_penalised_for_responding():
    cfg = O.LOSS_CFG
    assert O.cell_loss("flip_horizontal", FLAT, cfg) == pytest.approx(0.0)
    assert O.cell_loss("flip_horizontal", FALLING, cfg) > 0


def test_silence_penalty_is_asymmetric():
    """Equal-magnitude misses cost more on a control than on a target."""
    cfg = O.LOSS_CFG
    over = {"0p02": 0.90, "0p05": 0.88, "0p10": 0.86, "0p20": 0.84, "0p40": 0.78}
    assert O.cell_loss("flip_horizontal", over, cfg) > O.cell_loss("flicker", over, cfg)


def test_unconstrained_family_contributes_nothing():
    assert O.cell_loss("flip_transpose", FALLING, O.LOSS_CFG) == 0.0
    assert O.cell_loss("flip_transpose", FLAT, O.LOSS_CFG) == 0.0


def test_matrix_scores_covers_sixty_cells():
    rows = RT.build_table()["artefacts"]
    scored = O.matrix_scores(rows, R.PROD_PARAMS)
    assert len(scored) == 60
    cell = scored[("flicker", "7WHI2L_FDNg")]
    assert cell["verdict"] == "FLAT"
    assert cell["delta"] == pytest.approx(-0.001, abs=5e-3)


def test_matrix_loss_respects_base_subset():
    rows = RT.build_table()["artefacts"]
    four = [b for b in ("7WHI2L_FDNg", "BrRLKMbBTYQ", "KZ8p6b1zJ9U",
                        "hhszUXL1Cu8")]
    assert len(O.matrix_scores(rows, R.PROD_PARAMS, bases=four)) == 48


def test_production_parameters_pass_the_guards():
    table = RT.build_table()
    assert O.guards_ok(table["realmodels"], R.PROD_PARAMS) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lr_vcc_objective.py -v`
Expected: FAIL with `ImportError: cannot import name 'objective'`

- [ ] **Step 3: Write the implementation**

`scripts/lr_vcc/calibration/objective.py`:

```python
"""Severity-response loss over the artefact matrix.

Sign convention: build_verdict_matrix works in delta = y(0.40) - y(0.02), so
PASS is delta <= -0.05. This module works in R = -delta so that larger is
better. The two never mix inside one function.

The loss reads all five ladder points. The v5 verdict protocol reads only the
two endpoints; severities 0.05 / 0.10 / 0.20 are cached and were previously
unused.
"""
from statistics import mean

from ..build_verdict_matrix import verdict
from . import expectations as E
from .recompose import composite

LOSS_CFG = {
    "r_target": 0.10,   # wanted response for a RESPOND cell; PASS is 0.05
    "r_silent": 0.02,   # the FLAT band; a SILENT cell must stay inside it
    "w_mono": 1.0,
    "w_silence": 3.0,   # asymmetric: over-calibration is the guarded failure
}

# Real-model guards. v6 may not buy matrix cells with the leaderboard.
GUARD_ORDER = ("flashvsr", "mgld", "uav")


def response(ladder):
    """R = y(0.02) - y(0.40). Positive means the corruption lowered the score."""
    return ladder["0p02"] - ladder["0p40"]


def monotonicity_violation(ladder):
    """Total upward movement along the ladder — zero for a clean response."""
    seq = [ladder[s] for s in E.SEVERITIES]
    return sum(max(0.0, b - a) for a, b in zip(seq, seq[1:]))


def cell_loss(family, ladder, cfg=LOSS_CFG):
    exp = E.EXPECTATION[family]
    if exp == E.UNCONSTRAINED:
        return 0.0
    r = response(ladder)
    if exp == E.RESPOND:
        shortfall = max(0.0, cfg["r_target"] - r)
        return shortfall ** 2 + cfg["w_mono"] * monotonicity_violation(ladder)
    excess = max(0.0, abs(r) - cfg["r_silent"])
    return cfg["w_silence"] * excess ** 2


def matrix_scores(rows, params, bases=None):
    """{(family, base): {"ladder", "delta", "verdict", "response"}}."""
    ladders = {}
    for row in rows:
        if bases is not None and row["base"] not in bases:
            continue
        key = (row["unit"], row["base"])
        ladders.setdefault(key, {})[row["severity"]] = \
            composite(row, params)["lr_vcc"]
    out = {}
    for key, ladder in ladders.items():
        if set(ladder) != set(E.SEVERITIES):
            raise ValueError("incomplete ladder for {}".format(key))
        r = response(ladder)
        out[key] = {"ladder": ladder, "response": r, "delta": -r,
                    "verdict": verdict(-r)}
    return out


def matrix_loss(rows, params, cfg=LOSS_CFG, bases=None):
    scored = matrix_scores(rows, params, bases=bases)
    respond, silent = [], []
    for (family, _base), cell in scored.items():
        exp = E.EXPECTATION[family]
        if exp == E.UNCONSTRAINED:
            continue
        loss = cell_loss(family, cell["ladder"], cfg)
        (respond if exp == E.RESPOND else silent).append(loss)
    return (mean(respond) if respond else 0.0) + (mean(silent) if silent else 0.0)


def guards_ok(realmodel_rows, params, bases=None):
    """Canonical order flashvsr > mgld > uav, and MGLD > UAV on every video.

    During LOBO fitting, pass the fold's TRAINING bases so held-out videos do
    not leak into the fit through the guard.
    """
    per_method = {}
    for row in realmodel_rows:
        if bases is not None and row["base"] not in bases:
            continue
        out = composite(row, params)
        if out["low_confidence"]:
            continue
        per_method.setdefault(row["unit"], {})[row["base"]] = out["lr_vcc"]
    if set(per_method) != set(GUARD_ORDER):
        return False
    means = {m: mean(v.values()) for m, v in per_method.items()}
    if sorted(means, key=means.get, reverse=True) != list(GUARD_ORDER):
        return False
    videos = sorted(per_method["mgld"])
    return all(per_method["mgld"][v] > per_method["uav"][v] for v in videos)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lr_vcc_objective.py -v`
Expected: 9 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/lr_vcc/calibration/objective.py tests/test_lr_vcc_objective.py
git commit -m "lr_vcc: severity-response loss with control-silence penalty and leaderboard guards"
```

---

### Task 6: Coordinate search and LOBO folds

**Files:**
- Create: `scripts/lr_vcc/calibration/fit.py`
- Test: `tests/test_lr_vcc_fit.py`

**Interfaces:**
- Consumes: `objective.matrix_loss`, `objective.guards_ok`, `recompose.PROD_PARAMS`
- Produces: `GRIDS: dict`; `GATE_GRIDS: dict`; `logspace(lo, hi, n) -> list`; `coordinate_search(art_rows, real_rows, bases, cfg, start, passes) -> tuple[dict, float]`; `lobo(table, cfg) -> dict`; `FIT_DIR`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from scripts.lr_vcc.calibration import fit as F
from scripts.lr_vcc.calibration import objective as O
from scripts.lr_vcc.calibration import recompose as R
from scripts.lr_vcc.calibration import response_table as RT
from scripts.lr_vcc.calibration import expectations as E


def test_logspace_endpoints_and_length():
    xs = F.logspace(0.1, 10.0, 5)
    assert len(xs) == 5
    assert xs[0] == pytest.approx(0.1)
    assert xs[-1] == pytest.approx(10.0)


def test_v5_is_reachable_from_every_grid():
    """The fit must be able to decline each new lever."""
    assert None in F.GRIDS["beta_t"]
    for key in ("alpha", "beta_e", "beta_dp", "beta_dpp", "tau", "lambda_a"):
        prod = R.PROD_PARAMS[key]
        assert min(F.GRIDS[key]) <= prod <= max(F.GRIDS[key]), key


def test_coordinate_search_does_not_increase_loss():
    table = RT.build_table()
    bases = E.BASES[:4]
    start_loss = O.matrix_loss(table["artefacts"], R.PROD_PARAMS,
                               O.LOSS_CFG, bases=bases)
    params, loss = F.coordinate_search(table["artefacts"], table["realmodels"],
                                       bases, O.LOSS_CFG, R.PROD_PARAMS,
                                       passes=1)
    assert loss <= start_loss + 1e-12
    assert O.guards_ok(table["realmodels"], params, bases=bases) is True


def test_coordinate_search_is_deterministic():
    table = RT.build_table()
    bases = E.BASES[:4]
    a, la = F.coordinate_search(table["artefacts"], table["realmodels"],
                                bases, O.LOSS_CFG, R.PROD_PARAMS, passes=1)
    b, lb = F.coordinate_search(table["artefacts"], table["realmodels"],
                                bases, O.LOSS_CFG, R.PROD_PARAMS, passes=1)
    assert a == b and la == lb


def test_lobo_folds_are_disjoint():
    """The central methodological claim: a fold never trains on its own base."""
    table = RT.build_table()
    result = F.lobo(table, O.LOSS_CFG)
    assert len(result["folds"]) == 5
    for fold in result["folds"]:
        assert fold["held_out"] not in fold["train_bases"]
        assert len(fold["train_bases"]) == 4
        assert set(fold["train_bases"]) | {fold["held_out"]} == set(E.BASES)


def test_lobo_heldout_matrix_has_one_column_per_fold():
    table = RT.build_table()
    result = F.lobo(table, O.LOSS_CFG)
    cells = result["heldout_matrix"]
    assert len(cells) == 60
    for (_family, base), cell in cells.items():
        assert cell["fitted_without"] == base
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lr_vcc_fit.py -v`
Expected: FAIL with `ImportError: cannot import name 'fit'`

- [ ] **Step 3: Write the implementation**

`scripts/lr_vcc/calibration/fit.py`:

```python
"""Deterministic coordinate search inside leave-one-base-out folds.

Determinism matters more than optimality at this scale: the same table and the
same grids must always produce the same parameters, or the reported numbers
are not reproducible. No randomness, no early stopping on wall-clock.

Usage (repo root):
  python -m scripts.lr_vcc.calibration.fit
"""
import json
from pathlib import Path

from . import expectations as E
from .objective import LOSS_CFG, guards_ok, matrix_loss, matrix_scores
from .recompose import PROD_PARAMS
from .response_table import TABLE_PATH, build_table

FIT_DIR = TABLE_PATH.parent


def logspace(lo, hi, n):
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


# Response parameters, searched first. v5's value lies inside every grid, and
# beta_t=None keeps v5's linear T reachable, so the fit can decline any lever.
GRIDS = {
    "lambda_a": [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
    "beta_t": [None] + logspace(1.0, 50.0, 11),
    "alpha": logspace(0.05, 3.0, 11),
    "beta_e": logspace(20.0, 2000.0, 11),
    "beta_dp": logspace(0.1, 5.0, 11),
    "beta_dpp": logspace(0.5, 30.0, 11),
    "tau": logspace(0.05, 5.0, 11),
}

# Gate thresholds, searched in a second pass with response parameters fixed.
GATE_GRIDS = {
    "a_drift_floor": [0.0, 0.01, 0.02, 0.05, 0.10],
    "a_sat_ceiling": [0.90, 0.95, 0.98, 1.0],
    "mask_cov_floor": [0.0, 0.05, 0.10, 0.20],
    "face_rate_floor": [0.0, 0.10, 0.20, 0.40],
    "closeup_threshold": [0.02, 0.05, 0.10, 1.0],
}

SEARCH_ORDER = ("tau", "beta_t", "lambda_a", "alpha", "beta_e", "beta_dp",
                "beta_dpp")
GATE_ORDER = ("mask_cov_floor", "a_drift_floor", "a_sat_ceiling",
              "face_rate_floor", "closeup_threshold")


def _evaluate(art_rows, real_rows, bases, cfg, params):
    """Loss, or None when the leaderboard guards reject this vector."""
    if not guards_ok(real_rows, params, bases=bases):
        return None
    return matrix_loss(art_rows, params, cfg, bases=bases)


def coordinate_search(art_rows, real_rows, bases, cfg=LOSS_CFG,
                      start=PROD_PARAMS, passes=3):
    """Minimise the loss one parameter at a time. Returns (params, loss)."""
    params = dict(start)
    best = _evaluate(art_rows, real_rows, bases, cfg, params)
    if best is None:
        raise ValueError("starting parameters violate the leaderboard guards")
    for _ in range(passes):
        improved = False
        for key in SEARCH_ORDER + GATE_ORDER:
            grid = GRIDS.get(key) or GATE_GRIDS[key]
            for value in grid:
                if value == params[key]:
                    continue
                trial = dict(params, **{key: value})
                loss = _evaluate(art_rows, real_rows, bases, cfg, trial)
                if loss is not None and loss < best - 1e-12:
                    params, best, improved = trial, loss, True
        if not improved:
            break
    return params, best


def lobo(table, cfg=LOSS_CFG, passes=3):
    """Five folds. Each fold's parameters never saw its held-out base."""
    art, real = table["artefacts"], table["realmodels"]
    folds, heldout = [], {}
    for held in E.BASES:
        train = tuple(b for b in E.BASES if b != held)
        params, train_loss = coordinate_search(art, real, train, cfg,
                                               PROD_PARAMS, passes)
        test_loss = matrix_loss(art, params, cfg, bases=(held,))
        for key, cell in matrix_scores(art, params, bases=(held,)).items():
            heldout[key] = dict(cell, fitted_without=held)
        folds.append({"held_out": held, "train_bases": list(train),
                      "params": params, "train_loss": train_loss,
                      "test_loss": test_loss})
    final_params, final_loss = coordinate_search(art, real, E.BASES, cfg,
                                                 PROD_PARAMS, passes)
    insample = {k: dict(v) for k, v in
                matrix_scores(art, final_params).items()}
    return {"folds": folds, "heldout_matrix": heldout,
            "insample_matrix": insample, "final_params": final_params,
            "final_loss": final_loss,
            "v5_loss": matrix_loss(art, PROD_PARAMS, cfg)}


def _jsonable(matrix):
    return {"{}|{}".format(f, b): v for (f, b), v in matrix.items()}


if __name__ == "__main__":
    table = build_table()
    result = lobo(table)
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    out = dict(result,
               heldout_matrix=_jsonable(result["heldout_matrix"]),
               insample_matrix=_jsonable(result["insample_matrix"]))
    json.dump(out, open(FIT_DIR / "lobo_result.json", "w"), indent=2)
    print("v5 loss           {:.6f}".format(result["v5_loss"]))
    print("v6 in-sample loss {:.6f}".format(result["final_loss"]))
    for f in result["folds"]:
        print("fold {:14s} train {:.6f}  held-out {:.6f}".format(
            f["held_out"], f["train_loss"], f["test_loss"]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lr_vcc_fit.py -v`
Expected: 6 PASS. The LOBO tests run a full five-fold search; allow a couple of minutes.

- [ ] **Step 5: Run the fit**

Run: `python -m scripts.lr_vcc.calibration.fit`
Expected: v5 loss, v6 in-sample loss, and five per-fold lines. Record the numbers — if held-out loss is not below v5 loss on a majority of folds, that is the "harness built, v6 deferred" outcome the spec anticipates, and Task 8's report must say so plainly.

- [ ] **Step 6: Commit**

```bash
git add scripts/lr_vcc/calibration/fit.py tests/test_lr_vcc_fit.py
git commit -m "lr_vcc: coordinate search with leave-one-base-out folds"
```

---

### Task 7: Per-cell failure attribution

**Files:**
- Create: `scripts/lr_vcc/calibration/failure_analysis.py`
- Test: `tests/test_lr_vcc_failure_analysis.py`

**Interfaces:**
- Consumes: `recompose.sub_metric_values`, `recompose.composite`, `expectations.DESIGNED_FOR`, `objective.matrix_scores`
- Produces: `STAGES: tuple`; `ADDRESSABLE: tuple`; `attribute(rows_by_severity, family, params, conforming) -> list[dict]` where each dict has `sub_metric, stage, rel_raw, delta_score, mean_weight, contribution, weight_drift`; `analyse(rows, params) -> dict[(family, base), dict]` with per-cell keys `verdict, delta, conforms, sub_metrics`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from scripts.lr_vcc.calibration import failure_analysis as FA
from scripts.lr_vcc.calibration import recompose as R
from scripts.lr_vcc.calibration import response_table as RT


@pytest.fixture(scope="module")
def rows():
    return RT.build_table()["artefacts"]


def test_stage_vocabulary_is_closed():
    assert FA.STAGES == ("measurement", "reward_direction", "normalisation",
                         "gate", "composition", "ok")
    assert set(FA.ADDRESSABLE) == {"normalisation", "gate", "composition"}


def test_identity_degradation_on_7WHI_is_reward_direction(rows):
    """I rises 0.375 -> 0.489 as identity degrades; the cell is INVERTED."""
    result = FA.analyse(rows, R.PROD_PARAMS)
    cell = result[("identity_degradation", "7WHI2L_FDNg")]
    stages = {d["sub_metric"]: d["stage"] for d in cell["sub_metrics"]}
    assert stages["identity"] == "reward_direction"
    assert cell["conforms"] is False


def test_flicker_on_7WHI_is_a_composition_failure(rows):
    """A and T both respond; D, E and D' outweigh them in the wrong direction."""
    result = FA.analyse(rows, R.PROD_PARAMS)
    cell = result[("flicker", "7WHI2L_FDNg")]
    stages = {d["sub_metric"]: d["stage"] for d in cell["sub_metrics"]}
    assert stages["temporal"] == "composition"
    assert cell["conforms"] is False


def test_conforming_cells_are_marked_ok(rows):
    result = FA.analyse(rows, R.PROD_PARAMS)
    cell = result[("flip_invert", "KZ8p6b1zJ9U")]
    assert cell["conforms"] is True
    assert all(d["stage"] == "ok" for d in cell["sub_metrics"])


def test_unconstrained_cells_report_none(rows):
    result = FA.analyse(rows, R.PROD_PARAMS)
    assert result[("flip_transpose", "KZ8p6b1zJ9U")]["conforms"] is None


def test_every_cell_is_analysed(rows):
    assert len(FA.analyse(rows, R.PROD_PARAMS)) == 60


def test_weight_drift_flag_fires_on_background_drift_brrlk(rows):
    """I's weight moves 0.017 -> 0.176 across this ladder."""
    result = FA.analyse(rows, R.PROD_PARAMS)
    cell = result[("background_drift", "BrRLKMbBTYQ")]
    drift = {d["sub_metric"]: d["weight_drift"] for d in cell["sub_metrics"]}
    assert drift["identity"] is True


def test_assigned_stages_stay_inside_the_vocabulary(rows):
    """Coverage check: no cell gets a stage outside STAGES, and the two
    mechanisms the probes demonstrated both occur somewhere in the matrix."""
    result = FA.analyse(rows, R.PROD_PARAMS)
    seen = {d["stage"] for cell in result.values() for d in cell["sub_metrics"]}
    assert seen <= set(FA.STAGES)
    assert "reward_direction" in seen
    assert "composition" in seen
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lr_vcc_failure_analysis.py -v`
Expected: FAIL with `ImportError: cannot import name 'failure_analysis'`

- [ ] **Step 3: Write the implementation**

`scripts/lr_vcc/calibration/failure_analysis.py`:

```python
"""Attribute every non-conforming matrix cell to the stage that lost the signal.

Five stages, matched in order. Only three of them are reachable by
re-parameterisation; `measurement` and `reward_direction` need a different
measurement, not a different constant, so the count of cells in those two
classes is the honest ceiling on what a fitted v6 can deliver.
"""
import math

from . import expectations as E
from .objective import matrix_scores
from .recompose import composite, sub_metric_values

STAGES = ("measurement", "reward_direction", "normalisation", "gate",
          "composition", "ok")
ADDRESSABLE = ("normalisation", "gate", "composition")

_RAW_FIELD = {
    "appearance": "a_mean",
    "identity": "identity_fused",
    "color_stability": "hist_dist",
    "color_slope": "slope_abs",
    "color_hist_anchor": "anchor_q14",
    "clip_trajectory": "clip_q14",
}

_EPS = 1e-6
_RAW_STATIC = 0.05
_RAW_STRONG = 0.20
_SCORE_DEAD = 0.02
_WRONG_WAY = 0.01
_WEIGHT_DEAD = 0.05
_WEIGHT_DRIFT = 0.05


def _raw_value(row, sub_metric):
    if sub_metric == "temporal":
        vals = [v for v in row["tof"].values() if v is not None]
        return sum(vals) / len(vals) if vals else 0.0
    return float(row[_RAW_FIELD[sub_metric]])


def _traces(rows_by_severity, params):
    """{sub_metric: {"raw": [...], "score": [...], "weight": [...]}} over the ladder."""
    out = {name: {"raw": [], "score": [], "weight": []} for name in E.SUB_METRICS}
    for sev in E.SEVERITIES:
        row = rows_by_severity[sev]
        comp = composite(row, params)
        for idx, (name, score, _rel) in enumerate(sub_metric_values(row, params)):
            out[name]["raw"].append(_raw_value(row, name))
            out[name]["score"].append(score)
            out[name]["weight"].append(comp["weights"][idx])
    return out


def attribute(rows_by_severity, family, params, conforming):
    traces = _traces(rows_by_severity, params)
    designed = E.DESIGNED_FOR.get(family, ())
    contributions = {
        name: (sum(t["weight"]) / len(t["weight"])) *
              (math.log(t["score"][-1] + _EPS) - math.log(t["score"][0] + _EPS))
        for name, t in traces.items()
    }
    findings = []
    for name in designed:
        t = traces[name]
        raw0, raw1 = t["raw"][0], t["raw"][-1]
        rel_raw = abs(raw1 - raw0) / (abs(raw0) + _EPS)
        delta_score = t["score"][-1] - t["score"][0]
        mean_w = sum(t["weight"]) / len(t["weight"])
        drift = (max(t["weight"]) - min(t["weight"])) > _WEIGHT_DRIFT

        if conforming:
            stage = "ok"
        elif rel_raw < _RAW_STATIC:
            stage = "measurement"
        elif delta_score > _WRONG_WAY:
            stage = "reward_direction"
        elif rel_raw >= _RAW_STRONG and abs(delta_score) < _SCORE_DEAD:
            stage = "normalisation"
        elif delta_score <= -_SCORE_DEAD and mean_w < _WEIGHT_DEAD:
            stage = "gate"
        else:
            opposing = sum(c for n, c in contributions.items()
                           if n != name and c > 0)
            stage = ("composition" if opposing >= abs(contributions[name])
                     else "normalisation")

        findings.append({
            "sub_metric": name, "stage": stage,
            "rel_raw": rel_raw, "delta_score": delta_score,
            "mean_weight": mean_w, "contribution": contributions[name],
            "weight_drift": drift,
        })
    return findings


def analyse(rows, params):
    """{(family, base): {"verdict", "delta", "conforms", "sub_metrics"}}."""
    by_cell = {}
    for row in rows:
        by_cell.setdefault((row["unit"], row["base"]), {})[row["severity"]] = row
    scored = matrix_scores(rows, params)
    out = {}
    for key, ladder_rows in by_cell.items():
        family, _base = key
        cell = scored[key]
        conforming = E.conforms(family, cell["verdict"])
        out[key] = {
            "verdict": cell["verdict"], "delta": cell["delta"],
            "conforms": conforming,
            "sub_metrics": attribute(ladder_rows, family, params,
                                     conforming is True),
        }
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lr_vcc_failure_analysis.py -v`
Expected: 8 PASS. If `test_flicker_on_7WHI_is_a_composition_failure` fails, print the cell's `sub_metrics` entries and check which branch T landed in before adjusting any threshold — the thresholds are documented constants, not free knobs.

- [ ] **Step 5: Commit**

```bash
git add scripts/lr_vcc/calibration/failure_analysis.py tests/test_lr_vcc_failure_analysis.py
git commit -m "lr_vcc: per-cell failure attribution across five stages"
```

---

### Task 8: Report emitters

Produces the four deliverables. Expected values are stated so the implementer can verify the output rather than eyeball it.

**Files:**
- Create: `scripts/lr_vcc/calibration/report.py`
- Test: `tests/test_lr_vcc_calibration_report.py`

**Interfaces:**
- Consumes: everything above
- Produces: `write_response_curves(rows, params, out) -> Path`; `write_expectation_matrix(rows, params, out) -> Path`; `write_failure_attribution(rows, params, out) -> Path`; `write_lobo_report(result, out) -> Path`; `conformance_counts(rows, params) -> dict`

- [ ] **Step 1: Write the failing test**

```python
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


def test_expectation_matrix_reports_both_counts(tmp_path):
    out = tmp_path / "matrix.md"
    REP.write_expectation_matrix(RT.build_table()["artefacts"],
                                 R.PROD_PARAMS, out)
    text = out.read_text()
    assert "39/55" in text
    assert "29/60" in text


def test_failure_report_separates_addressable_from_structural(tmp_path):
    out = tmp_path / "fail.md"
    REP.write_failure_attribution(RT.build_table()["artefacts"],
                                  R.PROD_PARAMS, out)
    text = out.read_text()
    assert "calibration-addressable" in text
    assert "structural" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lr_vcc_calibration_report.py -v`
Expected: FAIL with `ImportError: cannot import name 'report'`

- [ ] **Step 3: Write the implementation**

`scripts/lr_vcc/calibration/report.py`:

```python
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


def write_expectation_matrix(rows, params, out=FIG / "expectation_scored_matrix_v5.md"):
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
    for (family, base) in sorted(result):
        cell = result[(family, base)]
        if cell["conforms"] is not False:
            continue
        for d in cell["sub_metrics"]:
            stage_counts[d["stage"]] = stage_counts.get(d["stage"], 0) + 1
            lines.append("| {} | {} | {} | {} | {} | {:+.0f}% | {:+.3f} | "
                         "{:.3f} | {} |".format(
                             family, base, cell["verdict"], d["sub_metric"],
                             d["stage"], d["rel_raw"] * 100, d["delta_score"],
                             d["mean_weight"], "yes" if d["weight_drift"] else ""))
    addressable = sum(v for k, v in stage_counts.items() if k in ADDRESSABLE)
    structural = sum(v for k, v in stage_counts.items()
                     if k in ("measurement", "reward_direction"))
    lines += ["", "## Totals by stage", ""]
    for stage in sorted(stage_counts):
        lines.append("- {}: {}".format(stage, stage_counts[stage]))
    lines += ["",
              "- **calibration-addressable** (normalisation / gate / "
              "composition): {}".format(addressable),
              "- **structural** (measurement / reward-direction — needs a "
              "different measurement, not a different constant): {}"
              .format(structural),
              "",
              "The structural count is the ceiling on what a re-parameterised "
              "v6 can recover."]
    return _write(out, lines)


def write_lobo_report(result, out=FIG / "calibration_v6_lobo.md"):
    keys = ("tau", "beta_t", "lambda_a", "alpha", "beta_e", "beta_dp", "beta_dpp")

    def _fmt(v):
        return "linear" if v is None else "{:g}".format(v)

    lines = ["# LR-VCC v6 — calibration under leave-one-base-out", "",
             "Provisional: fitted on five base videos. Held-out columns are "
             "the honest estimate; the in-sample row shows the overfitting "
             "gap. Targets: R_target={r_target}, R_silent={r_silent}, "
             "w_mono={w_mono}, w_silence={w_silence}.".format(**LOSS_CFG), "",
             "| fold (held out) | train loss | held-out loss | " +
             " | ".join(keys) + " |",
             "|---|---|---|" + "---|" * len(keys)]
    for f in result["folds"]:
        lines.append("| {} | {:.5f} | {:.5f} | {} |".format(
            f["held_out"], f["train_loss"], f["test_loss"],
            " | ".join(_fmt(f["params"][k]) for k in keys)))
    lines += ["",
              "- v5 loss (all bases): **{:.5f}**".format(result["v5_loss"]),
              "- v6 in-sample loss (all bases): **{:.5f}**".format(result["final_loss"]),
              "- mean held-out loss: **{:.5f}**".format(
                  sum(f["test_loss"] for f in result["folds"]) / len(result["folds"])),
              "",
              "## Final parameters (refit on all five bases)", "",
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
        json.dump({"v5_loss": result["v5_loss"],
                   "final_loss": result["final_loss"],
                   "final_params": result["final_params"]},
                  open(REPO / "results" / "lr_vcc" / "calibration" /
                       "v6_params.json", "w"), indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lr_vcc_calibration_report.py -v`
Expected: 4 PASS. `test_v5_conformance_counts_are_39_of_55` is the important one — it pins the reframing claim to the actual data. If the numbers differ, the expectation table or the v5 matrix has changed; investigate before adjusting the test.

- [ ] **Step 5: Generate all four deliverables**

Run: `python -m scripts.lr_vcc.calibration.report --lobo`
Expected: four files written under `reports/figures/`, plus `results/lr_vcc/calibration/v6_params.json`.

- [ ] **Step 6: Run the whole suite**

Run: `pytest tests/ -v`
Expected: all pass, including `tests/test_lr_vcc_v5_frozen.py` — v5 must still reproduce after every change.

- [ ] **Step 7: Commit**

```bash
git add scripts/lr_vcc/calibration/report.py tests/test_lr_vcc_calibration_report.py reports/figures/
git add -f results/lr_vcc/calibration/lobo_result.json results/lr_vcc/calibration/v6_params.json
git commit -m "lr_vcc: calibration reports — response curves, expectation scoring, failure attribution, v6 LOBO"
```

---

## Verification

After Task 8, confirm against the spec:

1. `pytest tests/ -v` — every test passes, v5 frozen test included.
2. `reports/figures/expectation_scored_matrix_v5.md` reports both 39/55 and 29/60.
3. `reports/figures/failure_attribution_v5.md` gives a calibration-addressable versus structural split.
4. `reports/figures/calibration_v6_lobo.md` shows five folds, per-fold parameters, the in-sample gap, and a held-out matrix.
5. `git diff --stat HEAD~8` touches no file under `results/lr_vcc/composite_v5_*`.

Report the held-out result honestly. If mean held-out loss does not beat v5, the outcome is "harness built, v6 deferred to the enlarged base set" — the spec accepts that as a valid result.

## Deferred to a later tranche

Surfaced by this work, out of scope here: a scene-cut-aware anchor for D′/D″ (the BrRLK inversions), a mirror-sensitive sub-metric (flip_horizontal's designed-in blind spot), un-parking the identity dispersion gate, and replacing `R_target` with Phase B's measured human severity curve. Each needs either server time or data this tranche does not have.
