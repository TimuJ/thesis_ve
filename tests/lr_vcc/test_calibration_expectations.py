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
