"""RV-10: calibrator multiple-testing count must reflect the FULL grid, not just valid configs."""
from okxb.research.calibrator import Grid


def test_grid_size_is_full_product():
    g = Grid()
    assert g.size() == (len(g.S) * len(g.C) * len(g.N) * len(g.H) * len(g.tp_rr) * len(g.modes))
    assert g.size() == 1215        # default 5*3*3*3*3*3 — the honest DSR/PBO n_trials


def test_grid_size_tracks_dimensions():
    g = Grid(S=[60, 70], C=[0.3], N=[2], H=[60], tp_rr=[1.6], modes=["barrier"])
    assert g.size() == 2
