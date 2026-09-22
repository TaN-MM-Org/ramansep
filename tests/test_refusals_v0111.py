"""v0.11.1 regression tests: configuration errors are refused, not
turned into silent results.

* `fit_map` used to catch every per-pixel ValueError, so a bad fit
  configuration (a mistyped baseline name, reversed or overlapping
  windows, a window with too few points) masked EVERY pixel and
  returned an all-NaN result with no error. It now refuses the
  configuration once, before fitting, exactly as `fit_two_modes` does.
* `bayesian_map_inversion` did not check the rank of K. A rank-1 K
  makes its sparse system singular (the constant field along the
  unidentifiable direction costs nothing), so it returned NaN maps
  at lam = 0 and arbitrary finite maps at lam > 0. It now refuses,
  with the same message as `MultiModeModel`.
"""
import numpy as np
import pytest

import ramansep as rs
from ramansep.fitting import lorentzian

W1, W2 = (395.0, 415.0), (440.0, 465.0)
REF1, REF2 = 404.7, 452.0


def _cube():
    x = np.linspace(380.0, 480.0, 401)
    y = (lorentzian(x, REF1, 3.0, 800.0, 50.0)
         + lorentzian(x, REF2, 4.0, 500.0, 0.0))
    return x, np.broadcast_to(y, (2, 2, x.size)).copy()


def test_fit_map_refuses_bad_baseline_name():
    x, cube = _cube()
    with pytest.raises(ValueError, match="baseline"):
        rs.fit_map(x, cube, W1, W2, REF1, REF2, baseline="Linear")


def test_fit_map_refuses_bad_windows():
    x, cube = _cube()
    with pytest.raises(ValueError, match="overlap"):
        rs.fit_map(x, cube, (395.0, 450.0), W2, REF1, REF2)
    with pytest.raises(ValueError, match="lo < hi"):
        rs.fit_map(x, cube, (415.0, 395.0), W2, REF1, REF2)
    with pytest.raises(ValueError, match="fewer than"):
        rs.fit_map(x, cube, (400.0, 400.5), W2, REF1, REF2)
    bad_axis = x.copy()
    bad_axis[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rs.fit_map(bad_axis, cube, W1, W2, REF1, REF2)


def test_fit_map_valid_configuration_unchanged():
    x, cube = _cube()
    res = rs.fit_map(x, cube, W1, W2, REF1, REF2)
    assert res.n_masked == 0 and res.ok.all()
    d1, d2, s1, s2, _, _ = rs.fit_two_modes(x, cube[1, 0], W1, W2,
                                            REF1, REF2)
    assert res.dw1[1, 0] == d1 and res.dw2[1, 0] == d2
    assert res.sigma1[1, 0] == s1 and res.sigma2[1, 0] == s2


def test_bayesian_refuses_rank_deficient_K():
    K = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])     # rank 1
    shifts = np.random.default_rng(0).normal(size=(3, 4, 4))
    for lam in (0.0, 1.0):
        with pytest.raises(ValueError, match="rank < 2"):
            rs.bayesian_map_inversion(K, shifts, [0.1, 0.1, 0.1], lam)
