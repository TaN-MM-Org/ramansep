"""v0.9 thermal anchors: the three-cause GLS against an independent
per-pixel lstsq path and against the two-cause solver at known
temperature; the anti-Stokes Bose-factor thermometer as an exact
closed-form round trip, its constant checked against SciPy's CODATA
table, and its error propagation against a finite difference."""
import numpy as np
import pytest

from ramansep import (HC_OVER_KB_CM_K, MultiModeModel, ThreeCauseModel,
                      anti_stokes_ratio, calibrate_anti_stokes,
                      temperature_from_anti_stokes)

# A well-conditioned synthetic (4, 3) lever-arm matrix. Labeled
# non-physical, exactly like synthetic_demo(): values chosen only so
# that no column is close to a combination of the others.
K4 = np.array([[-2.1, -0.8, -0.011],
               [-1.0, -2.3, -0.016],
               [-3.2, -0.4, -0.007],
               [-0.6, -1.1, -0.021]])


def test_three_cause_exact_recovery_and_independent_path():
    rng = np.random.default_rng(7)
    model = ThreeCauseModel(K4)
    strain = rng.normal(0.0, 0.3, size=(5, 4))
    dens = rng.normal(0.0, 1.0, size=(5, 4))
    temp = rng.normal(20.0, 30.0, size=(5, 4))
    shifts = model.forward(strain, dens, temp)
    res = model.invert(shifts, sigmas=[0.05, 0.08, 0.05, 0.1])
    # noise-free round trip: machine precision, chi2 exactly ~ 0
    assert np.allclose(res.strain, strain, rtol=0, atol=1e-9)
    assert np.allclose(res.density, dens, rtol=0, atol=1e-9)
    assert np.allclose(res.temperature, temp, rtol=0, atol=1e-8)
    assert res.dof == 1 and np.all(res.chi2_map < 1e-16)
    # independent path: per-pixel whitened lstsq (no shared solver)
    sig = np.array([0.05, 0.08, 0.05, 0.1])
    S = np.stack([np.asarray(s) for s in shifts]).reshape(4, -1)
    for p in [0, 7, 19]:
        xs, *_ = np.linalg.lstsq(K4 / sig[:, None], S[:, p] / sig,
                                 rcond=None)
        assert abs(xs[0] - res.strain.ravel()[p]) < 1e-9
        assert abs(xs[1] - res.density.ravel()[p]) < 1e-9
        assert abs(xs[2] - res.temperature.ravel()[p]) < 1e-8
    # covariance against the direct textbook inverse, per pixel
    W = np.diag(1.0 / sig ** 2)
    cov_direct = np.linalg.inv(K4.T @ W @ K4)
    assert np.allclose(res.covariance.reshape(-1, 3, 3)[3], cov_direct,
                       rtol=1e-12, atol=0)
    assert abs(res.strain_sigma.ravel()[0]
               - np.sqrt(cov_direct[0, 0])) < 1e-12


def test_three_cause_reduces_to_two_cause_at_known_temperature():
    """Subtracting a KNOWN temperature's thermal shift and running the
    existing two-cause solver must equal the truth the three-cause
    forward model was built from -- the exact consistency between the
    old and new estimators."""
    rng = np.random.default_rng(11)
    model3 = ThreeCauseModel(K4)
    strain = rng.normal(0.0, 0.3, size=7)
    dens = rng.normal(0.0, 1.0, size=7)
    T = 45.0                                     # known, uniform
    shifts = model3.forward(strain, dens, np.full(7, T))
    corrected = [s - K4[k, 2] * T for k, s in enumerate(shifts)]
    model2 = MultiModeModel(K4[:, :2])
    res2 = model2.invert(corrected)
    assert np.allclose(res2.strain, strain, rtol=0, atol=1e-10)
    assert np.allclose(res2.density, dens, rtol=0, atol=1e-10)


def test_three_cause_refusals():
    with pytest.raises(ValueError):
        ThreeCauseModel(K4[:2])                  # m < 3
    K_dep = K4.copy()
    K_dep[:, 2] = 2.0 * K_dep[:, 0] - K_dep[:, 1]   # rank 2
    with pytest.raises(ValueError):
        ThreeCauseModel(K_dep)
    model = ThreeCauseModel(K4)
    with pytest.raises(ValueError):
        model.invert([np.zeros(3)] * 3)          # wrong mode count
    with pytest.raises(ValueError):
        model.invert([np.zeros(3)] * 4, sigmas=[0.1, -0.1, 0.1, 0.1])


def test_second_radiation_constant_against_scipy():
    """Two independent sources: our value from the exact SI defining
    constants vs SciPy's CODATA table (m K -> cm K)."""
    from scipy.constants import physical_constants
    c2_m_K, unit, unc = physical_constants["second radiation constant"]
    assert unit == "m K" and unc == 0.0          # exact in the 2019 SI
    # older SciPy tables store the exact value truncated to fewer
    # digits, so agree to the table's own precision, not beyond it
    assert abs(HC_OVER_KB_CM_K - 100.0 * c2_m_K) < 1e-9 * HC_OVER_KB_CM_K


def test_anti_stokes_round_trip_and_calibration():
    w, T_true, C = 385.0, 431.7, 0.83
    r = anti_stokes_ratio(T_true, w, C)
    assert 0.0 < r < C                           # Bose factor < 1
    T_back = temperature_from_anti_stokes(r, w, C)
    assert abs(T_back - T_true) < 1e-9 * T_true  # exact closed forms
    # calibration at a known temperature recovers C exactly
    C_cal = calibrate_anti_stokes(r, w, T_true)
    assert abs(C_cal - C) < 1e-12 * C
    # monotonicity: hotter sample, larger ratio, and vice versa
    assert anti_stokes_ratio(T_true + 50, w, C) > r
    assert anti_stokes_ratio(T_true - 50, w, C) < r
    # the ratio at 300 K for a ~400 cm^-1 phonon is well below 1:
    # hc omega / kB T ~ 1.85, exp(-1.85) ~ 0.157 (sanity, wide margin)
    assert 0.05 < anti_stokes_ratio(300.0, 385.0, 1.0) < 0.3


def test_anti_stokes_error_propagation_and_refusals():
    w, C = 385.0, 0.83
    r = anti_stokes_ratio(350.0, w, C)
    out = temperature_from_anti_stokes(r, w, C, ratio_sigma=0.01 * r)
    # analytic sigma against a central finite difference
    h = 1e-7 * r
    T_p = temperature_from_anti_stokes(r + h, w, C)
    T_m = temperature_from_anti_stokes(r - h, w, C)
    dTdr = (T_p - T_m) / (2 * h)
    assert abs(out["temperature_sigma"] - abs(dTdr) * 0.01 * r) \
        < 1e-5 * out["temperature_sigma"]
    with pytest.raises(ValueError):
        temperature_from_anti_stokes(C, w, C)        # ratio >= C
    with pytest.raises(ValueError):
        temperature_from_anti_stokes(1.5 * C, w, C)
    with pytest.raises(ValueError):
        temperature_from_anti_stokes(-0.1, w, C)
    with pytest.raises(ValueError):
        temperature_from_anti_stokes(r, -385.0, C)
    with pytest.raises(ValueError):
        anti_stokes_ratio(-10.0, w, C)
