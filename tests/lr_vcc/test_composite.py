from scripts.lr_vcc.composite import compose_score


def test_all_equal_reliabilities_geometric_mean():
    # equal reliability -> uniform weights -> geometric mean
    scores = [0.6, 0.8, 0.4]
    rels = [0.9, 0.9, 0.9]
    out = compose_score(scores, rels, temperature=0.2)
    geom = (0.6 * 0.8 * 0.4) ** (1 / 3)
    assert abs(out["score"] - geom) < 1e-3
    for w in out["weights"]:
        assert abs(w - 1 / 3) < 1e-3
    assert not out["low_confidence"]


def test_one_reliable_dominates():
    # one sub-metric is much more reliable -> its score dominates the composite
    scores = [0.1, 0.9, 0.5]
    rels = [0.05, 0.95, 0.05]
    out = compose_score(scores, rels, temperature=0.2)
    # weight on the 2nd should be >0.9
    assert out["weights"][1] > 0.9
    # composite should be close to 0.9 (the dominant score), not the geometric mean
    assert out["score"] > 0.7


def test_all_unreliable_marks_low_confidence():
    out = compose_score([0.5, 0.5, 0.5], [0.1, 0.1, 0.1], temperature=0.2,
                        low_confidence_floor=0.2)
    assert out["low_confidence"]


def test_some_reliable_not_low_confidence():
    out = compose_score([0.5, 0.5, 0.5], [0.1, 0.5, 0.1], temperature=0.2,
                        low_confidence_floor=0.2)
    assert not out["low_confidence"]
