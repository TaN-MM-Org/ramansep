"""Planning anchors: the planned covariance IS the calibration's
covariance (two code paths, machine precision); the orthogonal unit
design predicts the exact identity covariance before any data exist;
repeating a design r times scales the covariance by exactly 1/r; the
greedy reference design reproduces its own rule, beats every random
subset tried, and picks the off-line point a collinear sweep needs;
collinear candidate sets are refused by the same rank argument as the
calibration itself."""
import numpy as np
import pytest

from ramansep import (calibrate_lever_arms, design_references,
                      plan_calibration, repeats_for_sigma)

K_TRUE = np.array([[-5.1, -2.2], [-20.9, 0.0]])
EPS = np.array([0.0, 0.3, 0.6, 0.0, 0.2, 0.5])
RHO = np.array([0.0, 0.0, 0.1, 0.8, 0.5, 0.9])
SIG = 0.15


def test_plan_equals_calibration_covariance_exactly():
    shifts = np.column_stack([EPS, RHO]) @ K_TRUE.T
    res = calibrate_lever_arms(EPS, RHO, shifts, sigmas=SIG)
    plan = plan_calibration(EPS, RHO, sigmas=SIG)
    assert plan["identifiable"]
    for k in range(2):
        assert np.allclose(plan["cov"], res.cov[k], rtol=0, atol=1e-15)
        assert np.allclose(plan["K_sigma"], res.K_sigma[k], rtol=0,
                           atol=1e-12)
    assert np.allclose(res.K, K_TRUE, rtol=0, atol=1e-10)


def test_orthogonal_unit_design_predicted_identity():
    """The calibration suite's exact anchor -- unit references (1,0)
    and (0,1) with unit sigmas give exactly the identity covariance --
    must now be predictable before any spectrum is taken."""
    plan = plan_calibration([1.0, 0.0], [0.0, 1.0], sigmas=1.0)
    assert np.allclose(plan["cov"], np.eye(2), rtol=0, atol=1e-12)
    assert np.allclose(plan["K_sigma"], 1.0, rtol=0, atol=1e-12)


def test_repeats_scale_covariance_exactly():
    """r identical copies of every reference measurement multiply
    X^T W X by exactly r, so the covariance shrinks by exactly 1/r --
    asserted by tiling the design and comparing matrices."""
    base = plan_calibration(EPS, RHO, sigmas=SIG)
    r = 7
    tiled = plan_calibration(np.tile(EPS, r), np.tile(RHO, r),
                             sigmas=SIG)
    assert np.allclose(tiled["cov"], base["cov"] / r, rtol=1e-12,
                       atol=0)
    target = 0.6 * float(np.max(base["K_sigma"]))
    r_need, plan = repeats_for_sigma(target, EPS, RHO, sigmas=SIG)
    assert float(np.max(plan["K_sigma"])) <= target
    if r_need > 1:
        prev = plan_calibration(np.tile(EPS, r_need - 1),
                                np.tile(RHO, r_need - 1), sigmas=SIG)
        assert float(np.max(prev["K_sigma"])) > target


def test_design_picks_off_line_point_and_beats_random():
    """Candidates: a strain-only sweep plus one gated point. Any
    identifiable subset must contain the gated point, so the greedy
    design necessarily picks it; and its determinant never loses to a
    random same-size subset."""
    eps = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.2])
    rho = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.7])
    out = design_references(eps, rho, 3, sigmas=SIG)
    assert 5 in out["indices"]
    assert out["K_sigma"] is not None

    def logdet_of(idx):
        plan = plan_calibration(eps[list(idx)], rho[list(idx)],
                                sigmas=SIG)
        if not plan["identifiable"]:
            return -np.inf
        s, d = np.linalg.slogdet(np.linalg.inv(plan["cov"]))
        return d

    best = logdet_of(out["indices"])
    rng = np.random.default_rng(2)
    for _ in range(30):
        assert best >= logdet_of(rng.choice(6, 3, replace=False)) - 1e-9


def test_collinear_refusals_match_calibration():
    eps = np.array([0.1, 0.2, 0.4])
    rho = 2.0 * eps                       # exactly collinear
    plan = plan_calibration(eps, rho, sigmas=SIG)
    assert not plan["identifiable"]
    with pytest.raises(ValueError, match="collinear"):
        calibrate_lever_arms(eps, rho, np.zeros((3, 2)), sigmas=SIG)
    with pytest.raises(ValueError, match="collinear"):
        design_references(eps, rho, 2, sigmas=SIG)
    with pytest.raises(ValueError, match="collinear"):
        repeats_for_sigma(0.1, eps, rho, sigmas=SIG)


def test_input_refusals():
    with pytest.raises(ValueError, match="positive"):
        plan_calibration(EPS, RHO, sigmas=-1.0)
    with pytest.raises(ValueError, match="finite"):
        plan_calibration([np.nan, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="n_pick"):
        design_references(EPS, RHO, 1)
    with pytest.raises(ValueError, match="positive"):
        repeats_for_sigma(0.0, EPS, RHO)
