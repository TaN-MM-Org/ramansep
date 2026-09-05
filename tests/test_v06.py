"""v0.6 anchors: the Voigt fitter against its exact lineshape limits
and analytic Jacobian, and the joint Bayesian map inversion against the
provable behavior of a linear-Gaussian estimator -- the per-pixel GLS
at lam = 0, exact recovery of a constant noiseless truth at any lam,
the pooled GLS in the lam -> infinity limit, and Loewner-order variance
shrinkage."""
import numpy as np
import pytest

from ramansep import (MultiModeModel, VoigtFit, bayesian_map_inversion,
                      fit_voigt, voigt)
from ramansep.bayesian import _grid_laplacian
from ramansep.fitting import _voigt_model_and_jacobian

K = np.array([[-2.3, 1.1], [-0.9, -1.7], [0.4, 2.2]])
SIG = np.array([0.03, 0.05, 0.02])


def _noisy_maps(seed=7, H=5, W=6):
    rng = np.random.default_rng(seed)
    ts = rng.normal(0.0, 0.01, (H, W))
    tn = rng.normal(0.0, 0.05, (H, W))
    shifts = np.einsum("mj,jhw->mhw", K, np.stack([ts, tn])) \
        + rng.normal(size=(3, H, W)) * SIG[:, None, None]
    return ts, tn, shifts


# ----------------------------- Voigt ------------------------------


def test_voigt_gamma_zero_is_the_gaussian_exactly():
    """Re w(x) = exp(-x^2) for real x, so gamma = 0 must give the
    Gaussian pointwise to machine precision -- an identity of the
    Faddeeva function, not an approximation."""
    x = np.linspace(-5.0, 5.0, 201)
    c, s = 0.3, 0.8
    v = voigt(x, c, s, 0.0, 2.0, 0.1)
    g = 2.0 * np.exp(-((x - c) ** 2) / (2.0 * s * s)) + 0.1
    assert np.abs(v - g).max() < 1e-14


def test_voigt_sigma_to_zero_converges_to_the_lorentzian():
    """The deviation from the half-width-gamma Lorentzian must fall
    linearly with sigma (first-order convolution smearing)."""
    x = np.linspace(-5.0, 5.0, 201)
    c, g = 0.3, 0.5
    lor = 2.0 * g * g / ((x - c) ** 2 + g * g)
    d3 = np.abs(voigt(x, c, 1e-3, g, 2.0) - lor).max()
    d4 = np.abs(voigt(x, c, 1e-4, g, 2.0) - lor).max()
    assert d3 < 1e-5
    assert d4 < 0.11 * d3          # one decade in sigma, one in error


def test_voigt_peak_height_normalization_is_exact():
    """amplitude is the peak height above offset by construction:
    the value at the center is amplitude + offset exactly."""
    v = voigt(np.array([412.0]), 412.0, 1.3, 0.7, 2.0, 0.1)
    assert v[0] == pytest.approx(2.1, abs=1e-15)


def test_voigt_analytic_jacobian_matches_finite_differences():
    x = np.linspace(-5.0, 5.0, 201)
    p = np.array([0.2, 0.7, 0.4, 1.5, 0.05])
    _, J = _voigt_model_and_jacobian(x, p)
    h = 1e-7
    for i in range(5):
        pp, pm = p.copy(), p.copy()
        pp[i] += h
        pm[i] -= h
        fd = (_voigt_model_and_jacobian(x, pp)[0]
              - _voigt_model_and_jacobian(x, pm)[0]) / (2.0 * h)
        assert np.abs(J[:, i] - fd).max() < 1e-6


def test_fit_voigt_recovers_a_noiseless_line_from_default_start():
    x = np.linspace(500.0, 540.0, 401)
    true = (520.0, 1.2, 0.9, 30.0, 5.0)
    f = fit_voigt(x, voigt(x, *true))
    assert isinstance(f, VoigtFit)
    est = np.array([f.center, f.sigma, f.gamma, f.amplitude, f.offset])
    assert np.abs(est - np.array(true)).max() < 1e-8
    assert f.converged
    assert f.gaussian_fwhm == pytest.approx(
        2.0 * np.sqrt(2.0 * np.log(2.0)) * f.sigma, rel=1e-14)
    assert f.lorentzian_fwhm == pytest.approx(2.0 * f.gamma, rel=1e-14)


def test_voigt_refusals():
    x = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError):
        voigt(x, 0.5, 0.0, 0.1, 1.0)          # sigma must be positive
    with pytest.raises(ValueError):
        voigt(x, 0.5, 0.1, -0.1, 1.0)         # gamma must be >= 0
    with pytest.raises(ValueError):
        fit_voigt(x[:5], np.ones(5))          # too few points
    y = np.ones(11)
    y[3] = np.nan
    with pytest.raises(ValueError):
        fit_voigt(x, y)                        # non-finite data


# --------------------------- Bayesian -----------------------------


def test_grid_laplacian_quadratic_form_is_the_edge_sum():
    rng = np.random.default_rng(1)
    h, w = 4, 5
    L = _grid_laplacian(h, w).toarray()
    x = rng.normal(size=(h, w))
    q = x.ravel() @ L @ x.ravel()
    edges = ((x[1:, :] - x[:-1, :]) ** 2).sum() \
        + ((x[:, 1:] - x[:, :-1]) ** 2).sum()
    assert abs(q - edges) < 1e-12
    assert np.abs(L - L.T).max() == 0.0
    assert np.abs(L.sum(axis=1)).max() == 0.0   # constants cost nothing


def test_lam_zero_reproduces_the_per_pixel_gls_maps_and_sigmas():
    """lam = 0 removes the prior, so the joint solve must agree with
    MultiModeModel.invert -- maps to solver round-off, sigmas from the
    same (K^T W K)^{-1}."""
    _, _, shifts = _noisy_maps()
    r_gls = MultiModeModel(K).invert(list(shifts), sigmas=list(SIG))
    r0 = bayesian_map_inversion(K, shifts, SIG, lam_strain=0.0,
                                posterior_sigma=True)
    assert np.abs(r0.strain - r_gls.strain).max() < 1e-12
    assert np.abs(r0.density - r_gls.density).max() < 1e-12
    assert np.abs(r0.strain_sigma - r_gls.strain_sigma).max() < 1e-13
    assert np.abs(r0.density_sigma - r_gls.density_sigma).max() < 1e-13


def test_constant_noiseless_truth_is_exact_at_every_lam():
    """The prior is a seminorm vanishing on constants, so a constant
    truth costs nothing and must be recovered exactly at any lam."""
    cs, cn = 0.004, -0.12
    shifts = (K @ np.array([cs, cn]))[:, None, None] * np.ones((1, 5, 6))
    for lam in (0.0, 1.0, 250.0):
        r = bayesian_map_inversion(K, shifts, SIG, lam_strain=lam)
        assert np.abs(r.strain - cs).max() < 1e-12
        assert np.abs(r.density - cn).max() < 1e-12


def test_lam_infinity_is_the_pooled_precision_weighted_gls():
    """Independent computation: the infinite-smoothness limit is the
    spatially constant GLS on the mode-averaged shifts."""
    _, _, shifts = _noisy_maps()
    r = bayesian_map_inversion(K, shifts, SIG, lam_strain=1e9)
    W = np.diag(1.0 / SIG ** 2)
    sbar = shifts.reshape(3, -1).mean(axis=1)
    pooled = np.linalg.solve(K.T @ W @ K, K.T @ W @ sbar)
    assert max(np.ptp(r.strain), np.ptp(r.density)) < 1e-5
    assert np.abs(r.strain - pooled[0]).max() < 1e-5
    assert np.abs(r.density - pooled[1]).max() < 1e-5


def test_posterior_sigmas_shrink_monotonically_with_lam():
    """Adding a PSD precision term shrinks the covariance in the
    Loewner order -- every per-pixel sigma is non-increasing in lam."""
    _, _, shifts = _noisy_maps()
    prev = None
    for lam in (0.0, 0.5, 2.0, 10.0):
        r = bayesian_map_inversion(K, shifts, SIG, lam_strain=lam,
                                   posterior_sigma=True)
        if prev is not None:
            assert np.all(r.strain_sigma <= prev[0] + 1e-14)
            assert np.all(r.density_sigma <= prev[1] + 1e-14)
        prev = (r.strain_sigma, r.density_sigma)


def test_separate_field_smoothness_weights():
    """lam_density = 0 with lam_strain > 0 must still shrink the strain
    sigmas below their unsmoothed values (fields couple through K)."""
    _, _, shifts = _noisy_maps()
    r00 = bayesian_map_inversion(K, shifts, SIG, lam_strain=0.0,
                                 posterior_sigma=True)
    rmx = bayesian_map_inversion(K, shifts, SIG, lam_strain=5.0,
                                 lam_density=0.0, posterior_sigma=True)
    assert np.all(rmx.strain_sigma < r00.strain_sigma)


def test_bayesian_refusals():
    _, _, shifts = _noisy_maps()
    with pytest.raises(ValueError):
        bayesian_map_inversion(K, shifts, SIG, lam_strain=-1.0)
    with pytest.raises(ValueError):
        bayesian_map_inversion(K, shifts, np.array([0.0, 0.05, 0.02]),
                               lam_strain=1.0)
    with pytest.raises(ValueError):
        bayesian_map_inversion(K, shifts[:, 0], SIG, lam_strain=1.0)
    with pytest.raises(ValueError):   # dense posterior refused, not approximated
        bayesian_map_inversion(K, np.zeros((3, 80, 80)), SIG, 1.0,
                               posterior_sigma=True)
