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
