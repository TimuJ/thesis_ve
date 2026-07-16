from scripts.rope_probe.run_probe import cond_id, expand_grid
from scripts.rope_probe.position_override import is_noop


def test_grid_starts_with_noop_and_dedupes():
    grid = expand_grid([0, 100], [1.0, 2.0])
    assert is_noop(grid[0])                      # baseline first
    ids = {(o.shift, o.stretch) for o in grid}
    assert ids == {(0, 1.0), (0, 2.0), (100, 1.0), (100, 2.0)}
    assert len(grid) == len(ids)                 # no duplicates


def test_cond_id_is_filename_safe_and_unique():
    grid = expand_grid([0, 8], [1.0, 1.5])
    ids = [cond_id(o) for o in grid]
    assert len(set(ids)) == len(ids)
    assert all("/" not in i and " " not in i for i in ids)
    assert cond_id(grid[0]) == "shift0_stretch1.0"


def test_continuous_grid_keeps_integer_baseline_and_suffixes_ids():
    grid = expand_grid([0], [1.0, 0.5], continuous=True)
    assert is_noop(grid[0])                       # baseline stays integer no-op
    assert not grid[0].continuous
    rest = grid[1:]
    assert all(o.continuous for o in rest)
    assert {cond_id(o) for o in rest} == {"shift0_stretch1.0c", "shift0_stretch0.5c"}


def test_parse_rung_tokens():
    from scripts.rope_probe.run_resolution_ladder import parse_rung
    assert parse_rung("270") == (270, 270)
    assert parse_rung("270u") == (270, 202)   # crop 202, upscale x4/3
    assert parse_rung("360") == (360, 360)
