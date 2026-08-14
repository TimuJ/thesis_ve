"""v5 is frozen. These composites must never change."""
import json
from pathlib import Path

import pytest

from scripts.lr_vcc.sweep_sensitivity import (
    PROD, artefact_units, compose_unit, realmodel_units,
)

REPO = Path(__file__).resolve().parents[2]
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
