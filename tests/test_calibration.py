"""Anchors for the lever-arm calibration: exact noise-free recovery,
agreement of two independent solver paths, the closed-form covariance
of the orthogonal unit design, Monte-Carlo compatibility of reported
sigmas, identifiability refusals, and the calibrate -> invert round
trip through the package's own separation model."""
import numpy as np
import pytest

import ramansep as rs

K_TRUE = np.array([[-5.1, -2.2],
                   [-20.9, 0.0]])


def _design(n=12, seed=0):
    rng = np.random.default_rng(seed)
    eps = rng.uniform(-0.5, 0.5, n)
    rho = rng.uniform(0.0, 2.0, n)
    return eps, rho


def test_noise_free_recovery_is_exact():
    eps, rho = _design()
    shifts = np.column_stack([eps, rho]) @ K_TRUE.T
    res = rs.calibrate_lever_arms(eps, rho, shifts,
                                  mode_names=["A'1", "2LA(M)"])
    assert np.allclose(res.K, K_TRUE, rtol=0, atol=1e-12)
    # exact data -> zero residual -> zero estimated uncertainty
    assert np.allclose(res.K_sigma, 0.0, atol=1e-10)


def test_normal_equations_agree_with_independent_qr_solve():
    rng = np.random.default_rng(3)
    eps, rho = _design(20, seed=3)
    sig = rng.uniform(0.05, 0.3, (20, 2))
    shifts = (np.column_stack([eps, rho]) @ K_TRUE.T
              + rng.normal(0.0, sig))
    res = rs.calibrate_lever_arms(eps, rho, shifts, sigmas=sig)
    X = np.column_stack([eps, rho])
    for k in range(2):
        w = 1.0 / sig[:, k]
        beta, *_ = np.linalg.lstsq(X * w[:, None], shifts[:, k] * w,
                                   rcond=None)
        assert np.allclose(res.K[k], beta, rtol=0, atol=1e-10)


def test_orthogonal_unit_design_has_identity_covariance():
    # references (strain, density) = (1, 0) and (0, 1) with unit
    # sigmas: X^T W X = I exactly, so Cov = I and every arm's sigma
    # is exactly 1
    eps = np.array([1.0, 0.0])
    rho = np.array([0.0, 1.0])
    shifts = np.column_stack([eps, rho]) @ K_TRUE.T
    res = rs.calibrate_lever_arms(eps, rho, shifts, sigmas=1.0)
    assert np.allclose(res.cov, np.broadcast_to(np.eye(2), (2, 2, 2)),
                       rtol=0, atol=1e-12)
    assert np.allclose(res.K_sigma, 1.0, rtol=0, atol=1e-12)
    assert np.allclose(res.K, K_TRUE, rtol=0, atol=1e-12)
    assert res.chi2_dof == 0


def test_monte_carlo_scatter_matches_reported_sigma():
    eps, rho = _design(15, seed=1)
    sig = 0.2
    rng = np.random.default_rng(7)
    fits = []
    for _ in range(400):
        shifts = (np.column_stack([eps, rho]) @ K_TRUE.T
                  + rng.normal(0.0, sig, (15, 2)))
        r = rs.calibrate_lever_arms(eps, rho, shifts, sigmas=sig)
        fits.append(r.K)
        reported = r.K_sigma            # design-set, identical each run
    emp = np.std(np.asarray(fits), axis=0)
    assert np.allclose(emp, reported, rtol=0.15)
    # and the chi2/dof of the last run is O(1), not O(10)
    assert np.all(r.chi2 / r.chi2_dof < 3.0)


def test_degenerate_and_malformed_references_raise():
    eps = np.linspace(0.1, 0.5, 6)
    shifts = np.column_stack([eps, eps])
    with pytest.raises(ValueError):        # strain-only sweep
        rs.calibrate_lever_arms(eps, np.zeros(6), shifts)
    with pytest.raises(ValueError):        # collinear ray
        rs.calibrate_lever_arms(eps, 2.0 * eps, shifts)
    with pytest.raises(ValueError):        # n = 1
        rs.calibrate_lever_arms([0.1], [1.0], [[1.0, 2.0]], sigmas=0.1)
    with pytest.raises(ValueError):        # n = 2 without sigmas
        rs.calibrate_lever_arms([0.1, 0.0], [0.0, 1.0],
                                [[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):        # non-positive sigma
        rs.calibrate_lever_arms(eps, np.ones(6), shifts, sigmas=0.0)
    bad = shifts.copy()
    bad[2, 1] = np.nan
    with pytest.raises(ValueError):        # non-finite shift
        rs.calibrate_lever_arms(eps, np.ones(6), bad)


def test_calibrate_then_invert_round_trip():
    """Calibrate on synthetic references, package the result, and run
    the package's own two-mode inversion on maps generated with the
    TRUE matrix: the recovered maps equal the inputs."""
    eps, rho = _design(10, seed=4)
    shifts = np.column_stack([eps, rho]) @ K_TRUE.T
    res = rs.calibrate_lever_arms(eps, rho, shifts, sigmas=0.1,
                                  mode_names=["A'1", "2LA(M)"])
    coeff = res.coefficients(
        reference="synthetic calibration for the test suite")
    model = rs.SeparationModel(coeff)
    rng = np.random.default_rng(9)
    strain = rng.uniform(-0.3, 0.3, (5, 5))
    density = rng.uniform(0.0, 1.5, (5, 5))
    dw1, dw2 = model.forward(strain, density)
    out = model.invert(dw1, dw2)
    assert np.allclose(out.strain, strain, rtol=0, atol=1e-10)
    assert np.allclose(out.density, density, rtol=0, atol=1e-10)
    # a three-mode calibration refuses to masquerade as a two-mode set
    shifts3 = np.column_stack([shifts, shifts[:, :1]])
    res3 = rs.calibrate_lever_arms(eps, rho, shifts3, sigmas=0.1)
    with pytest.raises(ValueError):
        res3.coefficients(reference="x")
