"""Calibration-propagation anchors: with zero calibration covariance
the result equals the plain inversion exactly; the derivative
identity dx/dK_mj = -K^-1 E_mj x is checked against finite
differences of the actual re-solve (two code paths); seeded Monte
Carlo over sampled lever arms matches the reported calibration part
of the error bars; the two noise contributions add in quadrature by
construction and are reported separately; and a multimode
calibration is refused by the two-mode tool."""
import dataclasses

import numpy as np
import pytest

from ramansep import (CalibrationResult, calibrate_lever_arms,
                      separation_with_calibration, synthetic_demo)
from ramansep.core import SeparationModel

K_TRUE = np.array([[-5.1, -2.2], [-20.9, 0.0]])
EPS_REF = np.array([0.0, 0.4, 0.8, 0.0, 0.3])
RHO_REF = np.array([0.0, 0.0, 0.2, 0.9, 0.6])


def _cal(sig=0.05):
    shifts = np.column_stack([EPS_REF, RHO_REF]) @ K_TRUE.T
    return calibrate_lever_arms(EPS_REF, RHO_REF, shifts, sigmas=sig)


def test_zero_calibration_cov_reduces_exactly():
    cal = _cal()
    cal0 = dataclasses.replace(cal, cov=np.zeros((2, 2, 2)))
    dw1 = np.array([-2.0, -3.1])
    dw2 = np.array([-8.0, -12.5])
    out = separation_with_calibration(cal0, dw1, dw2, sigma1=0.1,
                                      sigma2=0.1)
    coeffs = synthetic_demo()
    model = SeparationModel(dataclasses.replace(
        coeffs, k1_strain=cal.K[0, 0], k1_density=cal.K[0, 1],
        k2_strain=cal.K[1, 0], k2_density=cal.K[1, 1]))
    base = model.invert(dw1, dw2, sigma1=0.1, sigma2=0.1)
    assert np.allclose(out["strain"], base.strain, atol=1e-12)
    assert np.allclose(out["strain_sigma"], base.strain_sigma,
                       atol=1e-12)
    assert np.allclose(out["density_sigma"], base.density_sigma,
                       atol=1e-12)
    assert np.allclose(out["strain_sigma_calibration"], 0.0)


def test_derivative_identity_vs_finite_difference():
    K = K_TRUE
    Kinv = np.linalg.inv(K)
    y = np.array([-2.7, -9.4])
    x = Kinv @ y
    h = 1e-7
    for m in range(2):
        for j in range(2):
            Kp = K.copy()
            Kp[m, j] += h
            fd = (np.linalg.inv(Kp) @ y - x) / h
            exact = -Kinv[:, m] * x[j]
            assert np.allclose(fd, exact, rtol=1e-5, atol=1e-10)


def test_monte_carlo_matches_calibration_part():
    """Sample lever arms from the calibration covariance, re-invert
    noiseless shifts: the scatter of the maps must match the
    reported calibration contribution."""
    cal = _cal(sig=0.2)
    truth = np.array([0.45, 0.6])                # (strain, density)
    y = K_TRUE @ truth
    out = separation_with_calibration(cal, y[0], y[1])
    rng = np.random.default_rng(3)
    draws = []
    for _ in range(400):
        Ks = np.array([rng.multivariate_normal(cal.K[m], cal.cov[m])
                       for m in range(2)])
        draws.append(np.linalg.inv(Ks) @ y)
    emp = np.std(np.array(draws), axis=0, ddof=1)
    assert np.isclose(emp[0], float(out["strain_sigma_calibration"]),
                      rtol=0.2)
    assert np.isclose(emp[1],
                      float(out["density_sigma_calibration"]),
                      rtol=0.2)
    # without shift sigmas the total IS the calibration part
    assert np.allclose(out["strain_sigma"],
                       out["strain_sigma_calibration"])


def test_quadrature_split_reported():
    cal = _cal()
    out = separation_with_calibration(cal, -2.0, -8.0, sigma1=0.15,
                                      sigma2=0.15)
    tot = float(out["strain_sigma"])
    a = float(out["strain_sigma_shifts"])
    b = float(out["strain_sigma_calibration"])
    assert abs(tot - np.hypot(a, b)) < 1e-12
    assert a > 0 and b > 0


def test_multimode_refused():
    shifts3 = np.column_stack([
        K_TRUE[0, 0] * EPS_REF + K_TRUE[0, 1] * RHO_REF,
        K_TRUE[1, 0] * EPS_REF + K_TRUE[1, 1] * RHO_REF,
        -3.0 * EPS_REF - 1.0 * RHO_REF])
    cal3 = calibrate_lever_arms(EPS_REF, RHO_REF, shifts3, sigmas=0.1)
    with pytest.raises(ValueError, match="two-mode"):
        separation_with_calibration(cal3, -1.0, -2.0)
