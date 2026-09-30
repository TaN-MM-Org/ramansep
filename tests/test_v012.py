"""v0.12 anchors.

1. Joint two-peak fit: analytic Jacobian against finite differences;
   exact recovery of two overlapping noiseless Lorentzians (the
   separate-window fit is measurably biased on the same spectrum);
   seeded Monte Carlo scatter and centre correlation against the
   reported linearised values.
2. fit_map quality checks: a mixed cube of peak-free and genuine
   pixels (the truth is known by construction), reason codes, and
   unchanged results for good pixels.
3. bayesian_map_inversion with per-pixel sigmas and masked pixels:
   lam = 0 against MultiModeModel with the same per-pixel sigmas; the
   discrete-harmonic property of a filled pixel; a constant truth
   through the gaps; an independent dense least-squares solve.
4. multimode_with_calibration: exact reduction to the two-mode tool;
   the derivative formula against finite differences of the re-solve;
   seeded Monte Carlo over the calibration covariance.
5. Input validation and NaN-pixel handling fixed in v0.12.
"""
import dataclasses
import warnings

import numpy as np
import pytest

import ramansep as rs
from ramansep.bayesian import _grid_laplacian
from ramansep.fitting import _two_lorentzians_model_and_jacobian, lorentzian

# ------------------------------------------------------------------
# 1. joint two-peak fit

X = np.linspace(360.0, 430.0, 561)
W1, W2 = (375.0, 394.0), (395.0, 415.0)


def _pair(c1=385.0, c2=404.0, g1=5.0, g2=6.0, a1=600.0, a2=1000.0, b=20.0):
    return lorentzian(X, c1, g1, a1, b) + lorentzian(X, c2, g2, a2, 0.0)


@pytest.mark.parametrize("lin", [False, True])
def test_two_lorentzian_jacobian_matches_finite_differences(lin):
    p = np.array([385.2, 4.7, 610.0, 403.6, 6.3, 980.0, 18.0]
                 + ([2.5] if lin else []))
    x0 = 395.0
    _, J = _two_lorentzians_model_and_jacobian(X, p, x0, lin)
    for i in range(p.size):
        h = 1e-6 * max(1.0, abs(p[i]))
        pp, pm = p.copy(), p.copy()
        pp[i] += h
        pm[i] -= h
        fd = (_two_lorentzians_model_and_jacobian(X, pp, x0, lin)[0]
              - _two_lorentzians_model_and_jacobian(X, pm, x0, lin)[0]) \
            / (2.0 * h)
        scale = np.abs(J[:, i]).max()
        assert np.abs(J[:, i] - fd).max() < 1e-6 * scale + 1e-9


def test_joint_fit_removes_the_tail_bias_exactly():
    """Noiseless overlapping pair: the separate-window fit is biased by
    the neighbouring tail; the joint fit recovers both centres."""
    y = _pair()
    d1, d2, *_ = rs.fit_two_modes(X, y, W1, W2, 385.0, 404.0)
    assert abs(d1) > 0.05 and abs(d2) > 0.02          # the old bias
    j1, j2, s1, s2, f1, f2 = rs.fit_two_modes(X, y, W1, W2, 385.0, 404.0,
                                              joint=True)
    assert abs(j1) < 1e-8 and abs(j2) < 1e-8
    assert abs(f1.fwhm - 5.0) < 1e-8 and abs(f2.fwhm - 6.0) < 1e-8
    assert abs(f1.amplitude - 600.0) < 1e-6
    assert abs(f2.amplitude - 1000.0) < 1e-6
    assert abs(f1.offset - 20.0) < 1e-6 and f1.offset == f2.offset
    assert f1.converged and f2.converged
    assert f1.center_correlation == f2.center_correlation
    # a sloped background with the linear baseline
    y_lin = y + 3.0 * (X - 400.0)
    j1, j2, _, _, f1, _ = rs.fit_two_modes(X, y_lin, W1, W2, 385.0, 404.0,
                                           joint=True, baseline="linear")
    assert abs(j1) < 1e-8 and abs(j2) < 1e-8
    assert abs(f1.slope - 3.0) < 1e-8
    # the windows may also be given in the other order
    k2, k1, *_ = rs.fit_two_modes(X, y, W2, W1, 404.0, 385.0, joint=True)
    assert abs(k1) < 1e-8 and abs(k2) < 1e-8


def test_joint_fit_errors_match_monte_carlo():
    """600 seeded noisy spectra of a strongly overlapping pair: the
    centre scatter matches the reported sigma, the centre correlation
    matches the reported one, and the mean error is compatible with
    zero."""
    x = X
    y0 = lorentzian(x, 385.0, 6.0, 600.0, 20.0) \
        + lorentzian(x, 397.0, 7.0, 1000.0, 0.0)
    w1, w2 = (375.0, 390.9), (391.1, 410.0)
    rng = np.random.default_rng(11)
    err, sig, rho = [], [], []
    for _ in range(600):
        y = y0 + rng.normal(0.0, 10.0, x.size)
        d1, d2, s1, s2, f1, _ = rs.fit_two_modes(x, y, w1, w2, 385.0,
                                                 397.0, joint=True)
        err.append((d1, d2))
        sig.append((s1, s2))
        rho.append(f1.center_correlation)
    err, sig = np.array(err), np.array(sig)
    ratio = err.std(axis=0, ddof=1) / sig.mean(axis=0)
    assert np.all(np.abs(ratio - 1.0) < 0.12)
    assert np.all(np.abs(err.mean(axis=0))
                  < 4.0 * sig.mean(axis=0) / np.sqrt(600))
    emp_rho = np.corrcoef(err.T)[0, 1]
    assert abs(emp_rho - np.mean(rho)) < 0.12
    assert np.mean(rho) > 0.05                # a real, reported coupling


def test_fit_map_joint_equals_fit_two_modes_joint():
    rng = np.random.default_rng(5)
    cube = np.stack([[_pair(c1=385.0 + 0.3 * i, c2=404.0 - 0.2 * j)
                      + rng.normal(0.0, 5.0, X.size)
                      for j in range(3)] for i in range(2)])
    res = rs.fit_map(X, cube, W1, W2, 385.0, 404.0, joint=True)
    assert res.ok.all()
    for i in range(2):
        for j in range(3):
            d1, d2, s1, s2, _, _ = rs.fit_two_modes(
                X, cube[i, j], W1, W2, 385.0, 404.0, joint=True)
            assert (res.dw1[i, j], res.dw2[i, j]) == (d1, d2)
            assert (res.sigma1[i, j], res.sigma2[i, j]) == (s1, s2)


# ------------------------------------------------------------------
# 2. fit_map quality checks

XM = np.linspace(380.0, 480.0, 401)
MW1, MW2 = (395.0, 415.0), (440.0, 465.0)


def _mixed_cube(seed=1, n_noise=30, n_peak=10):
    """Row 0: peak-free pixels (flat background plus noise). Row 1:
    genuine, weak two-peak pixels. The truth is known by construction."""
    rng = np.random.default_rng(seed)
    n = max(n_noise, n_peak)
    cube = np.full((2, n, XM.size), np.nan)
    for j in range(n_noise):
        cube[0, j] = 50.0 + rng.normal(0.0, 3.0, XM.size)
    for j in range(n_peak):
        cube[1, j] = (lorentzian(XM, 404.7, 3.0, 40.0, 50.0)
                      + lorentzian(XM, 452.0, 4.0, 25.0, 0.0)
                      + rng.normal(0.0, 3.0, XM.size))
    return cube[:, :n_noise], cube[1:, :n_peak]


def test_quality_checks_separate_peak_free_from_genuine_pixels():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        noise, peaks = _mixed_cube()
        noise = noise[:1]
        plain = rs.fit_map(XM, noise, MW1, MW2, 404.7, 452.0)
        gated = rs.fit_map(XM, noise, MW1, MW2, 404.7, 452.0, min_snr=3.0)
        good_plain = rs.fit_map(XM, peaks, MW1, MW2, 404.7, 452.0)
        good_gated = rs.fit_map(XM, peaks, MW1, MW2, 404.7, 452.0,
                                min_snr=3.0, fwhm_range=(1.0, 15.0))
    # without min_snr some peak-free pixels still pass (the problem) ...
    assert plain.ok.sum() >= 1
    # ... but never with a negative fitted height (new default check)
    assert np.any(plain.reason == 5)
    # with min_snr = 3 none of the 30 peak-free pixels is kept
    assert gated.ok.sum() == 0
    assert set(np.unique(gated.reason)) <= {3, 4, 5, 6}
    # every genuine pixel is kept, with identical numbers
    assert good_plain.ok.all() and good_gated.ok.all()
    for name in ("dw1", "dw2", "sigma1", "sigma2"):
        assert np.array_equal(getattr(good_plain, name),
                              getattr(good_gated, name))
    assert np.all(good_gated.reason == 0)


def test_mask_reason_codes():
    y_ok = (lorentzian(XM, 404.7, 3.0, 800.0, 50.0)
            + lorentzian(XM, 452.0, 4.0, 500.0, 0.0))
    y_dip = 100.0 - lorentzian(XM, 404.0, 3.0, 40.0, 0.0) \
        + lorentzian(XM, 452.0, 4.0, 500.0, 0.0)
    y_nan = y_ok.copy()
    y_nan[7] = np.nan
    cube = np.stack([[y_ok, y_dip, y_nan]])
    res = rs.fit_map(XM, cube, MW1, MW2, 404.7, 452.0)
    assert res.reason.tolist() == [[0, 5, 1]]
    assert res.ok.tolist() == [[True, False, False]]
    assert rs.MASK_REASONS[5].startswith("fitted peak height")
    # a width range that excludes the true 3 and 4 cm^-1 widths
    res = rs.fit_map(XM, cube[:, :1], MW1, MW2, 404.7, 452.0,
                     fwhm_range=(5.0, 20.0))
    assert res.reason.tolist() == [[7]]
    with pytest.raises(ValueError, match="min_snr"):
        rs.fit_map(XM, cube, MW1, MW2, 404.7, 452.0, min_snr=0.0)
    with pytest.raises(ValueError, match="fwhm_range"):
        rs.fit_map(XM, cube, MW1, MW2, 404.7, 452.0, fwhm_range=(4.0, 2.0))


# ------------------------------------------------------------------
# 3. bayesian_map_inversion: per-pixel sigmas and masked pixels

KB = np.array([[-2.3, 1.1], [-0.9, -1.7], [0.4, 2.2]])
SIGB = np.array([0.03, 0.05, 0.02])


def _maps(seed=7, H=5, W=6):
    rng = np.random.default_rng(seed)
    ts = rng.normal(0.0, 0.01, (H, W))
    tn = rng.normal(0.0, 0.05, (H, W))
    sig = SIGB[:, None, None] * rng.uniform(0.5, 2.0, (3, H, W))
    shifts = np.einsum("mj,jhw->mhw", KB, np.stack([ts, tn])) \
        + rng.normal(size=(3, H, W)) * sig
    return shifts, sig


def _dense_reference(K, shifts, sig, lam_s, lam_n):
    """Independent solve of the same objective as one stacked dense
    least-squares problem: rows sqrt(w) (K x_p - s_p) for every
    measured entry, sqrt(lam) (x_i - x_j) for every lattice edge."""
    m, H, W = shifts.shape
    n = H * W
    rows, rhs = [], []
    for k in range(m):
        for p in range(n):
            i, j = divmod(p, W)
            if not (np.isfinite(shifts[k, i, j]) and np.isfinite(sig[k, i, j])):
                continue
            r = np.zeros(2 * n)
            sw = 1.0 / sig[k, i, j]
            r[p] = sw * K[k, 0]
            r[n + p] = sw * K[k, 1]
            rows.append(r)
            rhs.append(sw * shifts[k, i, j])
    edges = [(i * W + j, i * W + j + 1) for i in range(H) for j in range(W - 1)]
    edges += [(i * W + j, (i + 1) * W + j) for i in range(H - 1) for j in range(W)]
    for off, lam in ((0, lam_s), (n, lam_n)):
        for a, b in edges:
            r = np.zeros(2 * n)
            r[off + a] = np.sqrt(lam)
            r[off + b] = -np.sqrt(lam)
            rows.append(r)
            rhs.append(0.0)
    sol = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    return sol[:n].reshape(H, W), sol[n:].reshape(H, W)


def test_per_pixel_sigmas_at_lam_zero_equal_multimode():
    shifts, sig = _maps()
    ref = rs.MultiModeModel(KB).invert(list(shifts), sigmas=list(sig))
    r0 = rs.bayesian_map_inversion(KB, shifts, sig, 0.0, posterior_sigma=True)
    assert np.abs(r0.strain - ref.strain).max() < 1e-12
    assert np.abs(r0.density - ref.density).max() < 1e-12
    assert np.abs(r0.strain_sigma - ref.strain_sigma).max() < 1e-13
    assert np.abs(r0.density_sigma - ref.density_sigma).max() < 1e-13


def test_constant_per_pixel_sigmas_equal_the_scalar_path():
    shifts, _ = _maps()
    full = np.broadcast_to(SIGB[:, None, None], shifts.shape)
    a = rs.bayesian_map_inversion(KB, shifts, SIGB, 3.0, 7.0,
                                  posterior_sigma=True)
    b = rs.bayesian_map_inversion(KB, shifts, full, 3.0, 7.0,
                                  posterior_sigma=True)
    assert np.abs(a.strain - b.strain).max() < 1e-12
    assert np.abs(a.density - b.density).max() < 1e-12
    assert np.abs(a.strain_sigma - b.strain_sigma).max() < 1e-13


def test_per_pixel_and_masked_agree_with_dense_least_squares():
    shifts, sig = _maps()
    shifts[:, 2, 3] = np.nan                  # a pixel with no data
    shifts[1, 0, 0] = np.nan                  # one mode missing at a corner
    sig[2, 4, 5] = np.nan                     # a missing sigma
    r = rs.bayesian_map_inversion(KB, shifts, sig, 2.0, 5.0,
                                  fill_masked=True)
    ds, dn = _dense_reference(KB, shifts, sig, 2.0, 5.0)
    assert np.abs(r.strain - ds).max() < 1e-10
    assert np.abs(r.density - dn).max() < 1e-10


def test_filled_pixel_is_the_average_of_its_neighbours():
    shifts, sig = _maps()
    shifts[:, 2, 3] = np.nan                  # interior pixel, 4 neighbours
    shifts[:, 0, 5] = np.nan                  # corner pixel, 2 neighbours
    r = rs.bayesian_map_inversion(KB, shifts, sig, 4.0, 9.0,
                                  fill_masked=True, posterior_sigma=True)
    for f in (r.strain, r.density):
        nb = (f[1, 3] + f[3, 3] + f[2, 2] + f[2, 4]) / 4.0
        assert abs(f[2, 3] - nb) < 1e-12
        assert abs(f[0, 5] - (f[0, 4] + f[1, 5]) / 2.0) < 1e-12
    # the interpolated pixel is less certain than its measured neighbours
    assert r.strain_sigma[2, 3] > max(r.strain_sigma[1, 3], r.strain_sigma[2, 2])


def test_constant_truth_is_recovered_through_the_gaps():
    cs, cn = 0.004, -0.12
    shifts = (KB @ np.array([cs, cn]))[:, None, None] * np.ones((1, 5, 6))
    shifts[:, 1:3, 1:4] = np.nan              # a 2 x 3 hole
    for lam in (0.5, 250.0):
        r = rs.bayesian_map_inversion(KB, shifts, SIGB, lam, fill_masked=True)
        assert np.abs(r.strain - cs).max() < 1e-12
        assert np.abs(r.density - cn).max() < 1e-12


def test_masked_refusals():
    shifts, sig = _maps()
    shifts[:, 1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        rs.bayesian_map_inversion(KB, shifts, sig, 1.0)
    with pytest.raises(ValueError, match="lam_density > 0"):
        rs.bayesian_map_inversion(KB, shifts, sig, 1.0, 0.0, fill_masked=True)
    with pytest.raises(ValueError, match="lam_strain > 0"):
        rs.bayesian_map_inversion(KB, shifts, sig, 0.0, fill_masked=True)
    only_one_mode = shifts.copy()
    only_one_mode[1:] = np.nan                # a single mode everywhere
    with pytest.raises(ValueError, match="pooled information"):
        rs.bayesian_map_inversion(KB, only_one_mode, sig, 1.0,
                                  fill_masked=True)
    with pytest.raises(ValueError, match="per-pixel sigmas"):
        rs.bayesian_map_inversion(KB, shifts, sig[:, :2], 1.0)
    bad = sig.copy()
    bad[0, 0, 0] = -0.1
    with pytest.raises(ValueError, match="positive"):
        rs.bayesian_map_inversion(KB, np.nan_to_num(shifts), bad, 1.0)


# ------------------------------------------------------------------
# 4. multimode_with_calibration

K3 = np.array([[-5.1, -2.2], [-20.9, 0.0], [-8.0, -1.1]])
EPS = np.array([0.0, 0.4, 0.8, 0.0, 0.3, 0.6])
RHO = np.array([0.0, 0.0, 0.2, 0.9, 0.6, 0.3])


def _cal3(sig=0.1, seed=4):
    rng = np.random.default_rng(seed)
    y = np.column_stack([EPS, RHO]) @ K3.T + rng.normal(0.0, sig, (6, 3))
    return rs.calibrate_lever_arms(EPS, RHO, y, sigmas=sig)


def test_two_mode_calibration_reduces_to_separation_with_calibration():
    rng = np.random.default_rng(2)
    y = np.column_stack([EPS, RHO]) @ K3[:2].T + rng.normal(0, 0.1, (6, 2))
    cal = rs.calibrate_lever_arms(EPS, RHO, y, sigmas=0.1)
    dw1 = np.array([-2.0, -3.1, 0.4])
    dw2 = np.array([-8.0, -12.5, 1.0])
    a = rs.separation_with_calibration(cal, dw1, dw2, 0.1, 0.15)
    b = rs.multimode_with_calibration(cal, [dw1, dw2], [0.1, 0.15])
    for key in ("strain", "density", "strain_sigma", "density_sigma",
                "strain_sigma_calibration", "density_sigma_calibration",
                "strain_sigma_shifts", "density_sigma_shifts"):
        assert np.allclose(a[key], b[key], rtol=1e-10, atol=1e-13), key
    c = rs.separation_with_calibration(cal, dw1, dw2)
    d = rs.multimode_with_calibration(cal, [dw1, dw2])
    assert np.allclose(c["strain_sigma"], d["strain_sigma"], rtol=1e-10)
    assert d["strain_sigma_shifts"] is None


def test_derivative_formula_matches_finite_differences():
    """dx/dK_mj of the weighted least-squares estimate with a non-zero
    residual, against a central difference of the MultiModeModel
    re-solve (a different code path)."""
    cal = _cal3()
    K = cal.K
    sig = np.array([0.15, 0.10, 0.12])
    y = K @ np.array([0.3, 0.5]) + np.array([0.2, -0.1, 0.25])   # r != 0
    w = sig ** -2.0
    x = rs.MultiModeModel(K).invert(list(y), list(sig))
    x = np.array([float(x.strain), float(x.density)])
    A = K.T @ np.diag(w) @ K
    r = y - K @ x
    assert np.abs(r).max() > 0.05
    h = 1e-6
    for m in range(3):
        for j in range(2):
            e = np.zeros(2)
            e[j] = 1.0
            exact = w[m] * np.linalg.solve(A, e * r[m] - K[m] * x[j])
            Kp, Km = K.copy(), K.copy()
            Kp[m, j] += h
            Km[m, j] -= h
            xp = rs.MultiModeModel(Kp).invert(list(y), list(sig))
            xm = rs.MultiModeModel(Km).invert(list(y), list(sig))
            fd = (np.array([float(xp.strain), float(xp.density)])
                  - np.array([float(xm.strain), float(xm.density)])) / (2 * h)
            assert np.allclose(fd, exact, rtol=1e-6, atol=1e-10)


def test_monte_carlo_over_the_calibration_covariance():
    cal = _cal3(sig=0.2)
    sig = [0.15, 0.10, 0.12]
    y = K3 @ np.array([0.45, 0.6]) + np.array([0.1, -0.05, 0.12])
    out = rs.multimode_with_calibration(cal, list(y), sig)
    rng = np.random.default_rng(3)
    draws = []
    for _ in range(400):
        Ks = np.array([rng.multivariate_normal(cal.K[m], cal.cov[m])
                       for m in range(3)])
        r = rs.MultiModeModel(Ks).invert(list(y), sig)
        draws.append((float(r.strain), float(r.density)))
    emp = np.std(np.array(draws), axis=0, ddof=1)
    assert np.isclose(emp[0], float(out["strain_sigma_calibration"]), rtol=0.2)
    assert np.isclose(emp[1], float(out["density_sigma_calibration"]), rtol=0.2)
    # the two parts add in quadrature, and zero covariance gives the plain GLS
    assert np.isclose(float(out["strain_sigma"]) ** 2,
                      float(out["strain_sigma_shifts"]) ** 2
                      + float(out["strain_sigma_calibration"]) ** 2,
                      rtol=1e-12)
    cal0 = dataclasses.replace(cal, cov=np.zeros((3, 2, 2)))
    z = rs.multimode_with_calibration(cal0, list(y), sig)
    plain = rs.MultiModeModel(cal.K).invert(list(y), sig)
    assert float(z["strain_sigma"]) == pytest.approx(float(plain.strain_sigma),
                                                     rel=1e-12)
    assert float(z["strain_sigma_calibration"]) == 0.0


def test_multimode_with_calibration_masks_and_refusals():
    cal = _cal3()
    y = [np.array([-2.0, np.nan]), np.array([-8.0, -9.0]),
         np.array([-3.4, -3.9])]
    out = rs.multimode_with_calibration(cal, y, [0.1, 0.1, 0.1])
    assert np.isfinite(out["strain_sigma"][0])
    assert np.isnan(out["strain"][1]) and np.isnan(out["strain_sigma"][1])
    assert out["dof"] == 1
    bad = dataclasses.replace(cal, cov=np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="cov"):
        rs.multimode_with_calibration(bad, y, [0.1, 0.1, 0.1])


# ------------------------------------------------------------------
# 5. validation and NaN pixels


def test_three_cause_model_keeps_nan_pixels_local():
    K = np.array([[-2.1, -0.8, -0.011], [-1.0, -2.3, -0.016],
                  [-3.2, -0.4, -0.007], [-0.6, -1.1, -0.021]])
    model = rs.ThreeCauseModel(K)
    rng = np.random.default_rng(0)
    truth = [rng.normal(0, 0.1, (3, 4)), rng.normal(0, 0.5, (3, 4)),
             rng.normal(0, 20.0, (3, 4))]
    shifts = model.forward(*truth)
    sig = [np.full((3, 4), 0.05) for _ in range(4)]
    clean = model.invert(shifts, sig)
    shifts[1] = shifts[1].copy()
    sig[2] = sig[2].copy()
    shifts[1][0, 0] = np.nan                  # masked shift
    sig[2][2, 3] = np.nan                     # masked sigma (as fit_map gives)
    res = model.invert(shifts, sig)           # refused before v0.12
    bad = np.zeros((3, 4), bool)
    bad[0, 0] = bad[2, 3] = True
    for name in ("strain", "density", "temperature", "temperature_sigma",
                 "p_value"):
        got, ref = getattr(res, name), getattr(clean, name)
        assert np.all(np.isnan(got[bad]))
        assert np.array_equal(got[~bad], ref[~bad])


def test_new_refusals():
    with pytest.raises(ValueError, match="finite"):
        rs.SeparationModel(rs.ModeCoefficients("a", "b", np.nan, 1.0, 2.0,
                                               3.0, reference="test"))
    with pytest.raises(ValueError, match="finite"):
        rs.MultiModeModel([[np.inf, 1.0], [2.0, 3.0], [1.0, 1.0]])
    model = rs.SeparationModel(rs.mos2_a1_2la())
    with pytest.raises(ValueError, match="negative"):
        model.invert(0.1, 0.2, -0.1, 0.1)
    # NaN sigmas of masked pixels still pass through as NaN
    r = model.invert(np.array([0.1, np.nan]), np.array([0.2, np.nan]),
                     np.array([0.1, np.nan]), np.array([0.1, np.nan]))
    assert np.isfinite(r.strain_sigma[0]) and np.isnan(r.strain_sigma[1])
    for bad_sig in ([-0.1, 0.1, 0.1], [0.0, 0.1, 0.1],
                    [np.nan, 0.1, 0.1], [np.inf, 0.1, 0.1]):
        with pytest.raises(ValueError, match="finite and positive"):
            rs.compare_mode_sets(K3, bad_sig)
    for bad_val in (np.nan, np.inf):
        Kb = K3.copy()
        Kb[0, 0] = bad_val
        with pytest.raises(ValueError, match="finite"):
            rs.compare_mode_sets(Kb, [0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match=r"shape \(m, 2\)"):
        rs.compare_mode_sets([1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
