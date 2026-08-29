"""Peak-fitting tests: analytic Jacobian, exact recovery, honest errors,
and the full spectrum -> shifts -> inversion round trip."""
import numpy as np
import pytest

from ramansep import SeparationModel, synthetic_demo
from ramansep.fitting import (fit_lorentzian, fit_two_modes, lorentzian,
                              _model_and_jacobian)


def test_jacobian_matches_finite_differences():
    x = np.linspace(370.0, 390.0, 121)
    p = np.array([380.5, 4.2, 900.0, 55.0])
    _, J = _model_and_jacobian(x, p)
    eps = 1e-6
    for k in range(4):
        dp = np.zeros(4)
        dp[k] = eps * max(abs(p[k]), 1.0)
        y_plus, _ = _model_and_jacobian(x, p + dp)
        y_minus, _ = _model_and_jacobian(x, p - dp)
        num = (y_plus - y_minus) / (2.0 * dp[k])
        assert np.allclose(J[:, k], num, rtol=1e-5, atol=1e-7)


def test_noiseless_exact_recovery_from_auto_guess():
    x = np.linspace(395.0, 415.0, 201)
    truth = dict(center=404.73, fwhm=3.6, amplitude=1250.0, offset=80.0)
    y = lorentzian(x, **truth)
    fit = fit_lorentzian(x, y)
    assert fit.converged
    assert abs(fit.center - truth["center"]) < 1e-8
    assert abs(fit.fwhm - truth["fwhm"]) < 1e-8
    assert abs(fit.amplitude - truth["amplitude"]) < 1e-6
    assert abs(fit.offset - truth["offset"]) < 1e-6
    assert fit.residual_rms < 1e-8
    # noiseless data: reported uncertainties collapse toward zero
    assert fit.center_sigma < 1e-8


def test_noisy_center_error_compatible_with_reported_sigma():
    rng = np.random.default_rng(11)
    x = np.linspace(395.0, 415.0, 201)
    y0 = lorentzian(x, 404.73, 3.6, 1250.0, 80.0)
    errors, sigmas = [], []
    for _ in range(40):
        fit = fit_lorentzian(x, y0 + rng.normal(0.0, 12.0, x.size))
        errors.append(fit.center - 404.73)
        sigmas.append(fit.center_sigma)
    errors = np.array(errors)
    sigmas = np.array(sigmas)
    assert np.all(sigmas > 0)
    # the reported sigma predicts the observed scatter (chi^2-like check)
    z = errors / sigmas
    assert 0.5 < np.std(z) < 2.0
    assert abs(np.mean(z)) < 0.6


def test_two_mode_round_trip_through_inversion():
    coeffs = synthetic_demo()
    model = SeparationModel(coeffs)
    ref1, ref2 = 385.0, 455.0
    strain_true, density_true = 0.42, -1.3
    dw1, dw2 = model.forward(strain_true, density_true)
    x = np.linspace(360.0, 480.0, 1201)
    y = (lorentzian(x, ref1 + dw1, 4.0, 800.0, 40.0)
         + lorentzian(x, ref2 + dw2, 6.0, 500.0, 0.0))
    got = fit_two_modes(x, y, (370.0, 400.0), (440.0, 470.0), ref1, ref2)
    dw1_fit, dw2_fit, s1, s2, fit1, fit2 = got
    # the far tail of the other Lorentzian shifts a center only weakly
    assert abs(dw1_fit - dw1) < 5e-3
    assert abs(dw2_fit - dw2) < 5e-3
    res = model.invert(dw1_fit, dw2_fit, s1, s2)
    assert abs(float(res.strain) - strain_true) < 5e-3
    assert abs(float(res.density) - density_true) < 5e-3


def test_input_validation():
    x = np.linspace(0.0, 10.0, 50)
    y = lorentzian(x, 5.0, 1.0, 10.0)
    with pytest.raises(ValueError):
        fit_lorentzian(x[:4], y[:4])
    with pytest.raises(ValueError):
        fit_lorentzian(x, np.r_[y[:-1], np.nan])
    with pytest.raises(ValueError):
        fit_lorentzian(x, y, p0=[5.0, -1.0, 10.0, 0.0])
    with pytest.raises(ValueError):
        fit_two_modes(x, y, (2.0, 8.0), (6.0, 9.5), 5.0, 7.0)
    with pytest.raises(ValueError):
        fit_two_modes(x, y, (8.0, 2.0), (6.0, 9.5), 5.0, 7.0)
