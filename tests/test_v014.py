"""v0.14 anchors.

Fixed (each test fails on 0.13.1):
* the identifiability tests of `calibrate_lever_arms`, `plan_calibration`
  and `design_references`, and the pooled-information test of
  `bayesian_map_inversion(..., fill_masked=True)`, depended on the
  units: with the density in cm^-2 (strain in percent) a well-spread
  design was refused as "collinear" / "singular";
* `compare_mode_sets` returned an empty list for a subset size below 2
  or above the number of modes.

Performance: `fit_map` fits all pixels together; its outputs agree with
those of the old per-pixel loop over `fit_two_modes`, kept here as the
reference (exactly on the machine of 0.14; within stated tolerances,
derived at TOL_SHIFT, since v0.15, because another CPU may round the
stacked and the single fits differently in the last digit).

Added: correlated shift errors in the two-mode inversion
(`shift_correlation`, `MapFitResult.shift_correlation`), and region
means and contrasts with a split error budget (`region_mean`,
`region_contrast`). Each is checked against an independent calculation
and seeded Monte Carlo.

Tutorials: both scripts under examples/ run and print the numbers
pasted in the README.
"""
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import ramansep as rs
from ramansep import mapfit as _mapfit

K_MOS2 = np.array(rs.mos2_a1_2la().matrix())
CM2 = 1e13                     # 1e13 cm^-2 per unit of the MoS2 density


# ----------------------------------------------------------------------
# Fixed: unit-dependent identifiability tests


def _design():
    strain = np.array([0.0, 0.2, 0.4, 0.0, 0.2, 0.4])      # percent
    density = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0])     # 1e13 cm^-2
    return strain, density


def test_calibration_does_not_depend_on_the_density_unit():
    strain, density = _design()
    rng = np.random.default_rng(1)
    shifts = np.column_stack([strain, density]) @ K_MOS2.T
    shifts = shifts + rng.normal(0.0, 0.1, shifts.shape)
    a = rs.calibrate_lever_arms(strain, density, shifts, sigmas=0.1)
    # the same data with the density in cm^-2: 0.13.1 refused it as
    # "collinear"; the arms must be the same numbers, rescaled
    b = rs.calibrate_lever_arms(strain, density * CM2, shifts, sigmas=0.1)
    np.testing.assert_allclose(b.K[:, 0], a.K[:, 0], rtol=1e-10)
    np.testing.assert_allclose(b.K[:, 1] * CM2, a.K[:, 1], rtol=1e-10)
    np.testing.assert_allclose(b.K_sigma[:, 1] * CM2, a.K_sigma[:, 1],
                               rtol=1e-10)
    np.testing.assert_allclose(b.chi2, a.chi2, rtol=1e-8)
    # with the free offset, and without sigmas
    c = rs.calibrate_lever_arms(strain, density, shifts + 400.0,
                                fit_offset=True)
    d = rs.calibrate_lever_arms(strain, density * CM2, shifts + 400.0,
                                fit_offset=True)
    np.testing.assert_allclose(d.K[:, 1] * CM2, c.K[:, 1], rtol=1e-9)
    np.testing.assert_allclose(d.offset, c.offset, rtol=1e-12)
    # a design that IS collinear is still refused in any unit
    with pytest.raises(ValueError, match="collinear"):
        rs.calibrate_lever_arms(strain, 2.0 * strain * CM2, shifts,
                                sigmas=0.1)
    with pytest.raises(ValueError, match="straight line"):
        rs.calibrate_lever_arms(strain, (0.3 + 2.0 * strain) * CM2,
                                shifts, sigmas=0.1, fit_offset=True)


def test_planning_does_not_depend_on_the_density_unit():
    strain, density = _design()
    a = rs.plan_calibration(strain, density, 0.1)
    b = rs.plan_calibration(strain, density * CM2, 0.1)
    assert a["identifiable"] and b["identifiable"]   # b: False in 0.13.1
    np.testing.assert_allclose(b["K_sigma"] * [1.0, CM2], a["K_sigma"],
                               rtol=1e-10)
    bo = rs.plan_calibration(strain, density * CM2, 0.1, fit_offset=True)
    ao = rs.plan_calibration(strain, density, 0.1, fit_offset=True)
    np.testing.assert_allclose(bo["K_sigma"] * [1.0, CM2], ao["K_sigma"],
                               rtol=1e-10)
    cs = np.array([0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.3])
    cd = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0])
    pa = rs.design_references(cs, cd, 4, sigmas=0.1)
    pb = rs.design_references(cs, cd * CM2, 4, sigmas=0.1)   # refused before
    assert pa["indices"] == pb["indices"]
    r_a, _ = rs.repeats_for_sigma(0.05, strain, density, 0.1)
    assert r_a >= 1
    assert not rs.plan_calibration(strain, 3.0 * strain * CM2,
                                   0.1)["identifiable"]


def test_bayesian_fill_masked_does_not_depend_on_the_density_unit():
    K = np.array([[-5.1, -2.2], [-20.9, 0.0], [-3.0, -1.0]])
    Ks = K / [1.0, CM2]
    rng = np.random.default_rng(0)
    H = W = 6
    truth = np.stack([0.2 + 0.01 * rng.standard_normal((H, W)),
                      0.5 + 0.01 * rng.standard_normal((H, W))])
    shifts = np.einsum("mk,khw->mhw", K, truth)
    shifts[:, 2, 3] = np.nan
    sig = np.array([0.1, 0.2, 0.15])
    a = rs.bayesian_map_inversion(K, shifts, sig, 1.0, 1.0,
                                  fill_masked=True)
    # density in cm^-2: the same problem with lam_density scaled by
    # 1/CM2^2 ("pooled information matrix is singular" in 0.13.1)
    b = rs.bayesian_map_inversion(Ks, shifts, sig, 1.0, 1.0 / CM2 ** 2,
                                  fill_masked=True)
    # agreement to the precision a sparse solve with columns 1e13 apart
    # keeps (about 1e-7 here), far below any error bar
    np.testing.assert_allclose(b.strain, a.strain, rtol=1e-5)
    np.testing.assert_allclose(b.density / CM2, a.density, rtol=1e-5)
    # a K that really cannot fix both fields from the data left is
    # still refused (only mode 2 measured: strain only)
    only2 = shifts.copy()
    only2[[0, 2]] = np.nan
    with pytest.raises(ValueError, match="pooled information"):
        rs.bayesian_map_inversion(Ks, only2, sig, 1.0, 1.0,
                                  fill_masked=True)


def test_compare_mode_sets_refuses_a_subset_size_it_cannot_rank():
    K = np.array([[-5.1, -2.2], [-20.9, 0.0], [-8.0, -1.1]])
    sig = [0.1, 0.1, 0.12]
    for bad in (0, 1, 4, 2.5, True):
        with pytest.raises(ValueError, match="subset_size"):
            rs.compare_mode_sets(K, sig, subset_size=bad)   # [] in 0.13.1
    assert len(rs.compare_mode_sets(K, sig, subset_size=2)) == 3
    assert len(rs.compare_mode_sets(K, sig, subset_size=3.0)) == 1


# ----------------------------------------------------------------------
# Performance: fit_map agrees with the old per-pixel loop


def _reference_fit_map(x, cube, w1, w2, ref1, ref2, baseline, joint,
                       min_snr=None, fwhm_range=None):
    """The per-pixel loop of fit_map in 0.13.1, verbatim in substance."""
    H, W = cube.shape[:2]
    out = {k: np.full((H, W), np.nan) for k in ("dw1", "dw2", "s1", "s2")}
    ok = np.zeros((H, W), dtype=bool)
    reason = np.zeros((H, W), dtype=int)
    for i in range(H):
        for j in range(W):
            y = cube[i, j]
            if not np.all(np.isfinite(y)):
                reason[i, j] = 1
                continue
            try:
                d1, d2, e1, e2, f1, f2 = rs.fit_two_modes(
                    x, y, w1, w2, ref1, ref2, baseline=baseline,
                    joint=joint)
            except (ValueError, np.linalg.LinAlgError):
                reason[i, j] = 2
                continue
            if not (f1.converged and f2.converged):
                reason[i, j] = 3
                continue
            if not (w1[0] <= f1.center <= w1[1]
                    and w2[0] <= f2.center <= w2[1]):
                reason[i, j] = 4
                continue
            if not (f1.amplitude > 0.0 and f2.amplitude > 0.0):
                reason[i, j] = 5
                continue
            if min_snr is not None and not (
                    f1.amplitude >= min_snr * f1.amplitude_sigma
                    and f2.amplitude >= min_snr * f2.amplitude_sigma):
                reason[i, j] = 6
                continue
            if fwhm_range is not None and not (
                    fwhm_range[0] <= f1.fwhm <= fwhm_range[1]
                    and fwhm_range[0] <= f2.fwhm <= fwhm_range[1]):
                reason[i, j] = 7
                continue
            out["dw1"][i, j], out["dw2"][i, j] = d1, d2
            out["s1"][i, j], out["s2"][i, j] = e1, e2
            ok[i, j] = True
    return out, ok, reason


def _noisy_map(seed=11, H=6, W=8, with_peaks=False):
    rng = np.random.default_rng(seed)
    x = np.linspace(360.0, 430.0, 141)
    c1 = 385 + 0.5 * rng.standard_normal((H, W))
    c2 = 404 + 0.5 * rng.standard_normal((H, W))
    peak = np.where(rng.random((H, W)) < 0.25, 0.0, 1.0)   # no-peak pixels
    cube = (peak[..., None] * (rs.lorentzian(x, c1[..., None], 6.0, 60.0)
                               + rs.lorentzian(x, c2[..., None], 7.0, 80.0))
            + 10.0 + 0.05 * (x - 395.0) + rng.normal(0, 3.0, (H, W, x.size)))
    cube[1, 2, 7] = np.nan
    if with_peaks:                     # (new in v0.15) which pixels carry peaks
        return x, cube, peak == 1.0
    return x, cube


# Tolerances for comparing the stacked fits of `fit_map` with the
# single-spectrum fits (since v0.15; before, these tests demanded exact
# equality). The arithmetic is the same, but the stacked and the single
# matrix products are different NumPy/BLAS calls, which another CPU may
# round differently in the last digit. A converged fit stops once a step
# lowers the cost C by less than a relative 1e-12. Near the minimum C*,
# C - C* = dp^T J^T J dp, and for each parameter
# (dp_i / sigma_i)^2 <= (C - C*) / s^2 with s^2 = C* / (n - p), so a stop
# one step earlier or later moves a parameter by at most about
# sqrt(1e-12 (n - p)) = 1e-6 sqrt(n - p) of its error bar: below 4e-5
# here (n <= 401). Centers (shifts) are therefore compared to 1e-4 of
# their error bar, error bars to a relative 1e-5 and the correlation to
# 1e-5. Mask decisions are integers and are compared exactly, on pixels
# that carry peaks, whose decisions are far from every threshold (checked
# below). Pixels WITHOUT a peak are not compared: their fits to pure noise
# can run away (joint fits reach centers of 1e20 to 1e76 cm^-1) along a
# chaotic path, and changing the counts of the map by 1 ulp (a relative
# 2.2e-16) already flips the reason code of 1 or 2 of the 18 peak-free
# pixels of this map between 3 and 4 (measured in v0.15), so no
# implementation-independent result exists for them.
TOL_SHIFT = 1e-4                       # times the error bar
TOL_SIGMA = 1e-5                       # relative
TOL_RHO = 1e-5                         # absolute


def assert_same_fits(dw, sigma, ref_dw, ref_sigma, ok):
    """Shift maps dw and their error bars sigma (tuples of maps) agree
    with the reference ones within the tolerances above at the pixels
    `ok`."""
    for d, e, rd, re in zip(dw, sigma, ref_dw, ref_sigma):
        assert np.all(np.abs(d[ok] - rd[ok]) <= TOL_SHIFT * re[ok])
        np.testing.assert_allclose(e[ok], re[ok], rtol=TOL_SIGMA, atol=0)


@pytest.mark.parametrize("baseline", ["constant", "linear"])
@pytest.mark.parametrize("joint", [False, True])
def test_fit_map_agrees_with_the_per_pixel_loop(baseline, joint):
    """The stacked fit_map against the 0.13.1 per-pixel loop over
    fit_two_modes, kept here as the reference (exact equality before
    v0.15; within the stated tolerances since, see TOL_SHIFT)."""
    x, cube, peaks = _noisy_map(with_peaks=True)
    w1, w2 = (370.0, 394.5), (395.0, 420.0)
    kw = dict(min_snr=2.0, fwhm_range=(2.0, 14.0))
    ref, ok, reason = _reference_fit_map(x, cube, w1, w2, 385.0, 404.0,
                                         baseline, joint, **kw)
    res = rs.fit_map(x, cube, w1, w2, 385.0, 404.0, baseline, joint=joint,
                     **kw)
    cmp = peaks.copy()
    cmp[1, 2] = True                      # the broken pixel: reason 1
    # the decisions at the pixels with peaks are far from the thresholds
    for i, j in np.argwhere(peaks & ok):
        _, _, _, _, f1, f2 = rs.fit_two_modes(x, cube[i, j], w1, w2, 385.0,
                                              404.0, baseline=baseline,
                                              joint=joint)
        for f, w in ((f1, w1), (f2, w2)):
            assert f.n_iter < 100 and f.converged
            assert f.amplitude > 4.0 * f.amplitude_sigma
            assert 3.0 < f.fwhm < 13.0 and w[0] + 3 < f.center < w[1] - 3
    assert np.array_equal(res.ok[cmp], ok[cmp])
    assert np.array_equal(res.reason[cmp], reason[cmp])
    good = cmp & ok
    assert good.sum() >= 30
    assert_same_fits((res.dw1, res.dw2), (res.sigma1, res.sigma2),
                     (ref["dw1"], ref["dw2"]), (ref["s1"], ref["s2"]), good)
    for f in (res.dw1, res.dw2, res.sigma1, res.sigma2):
        assert np.all(np.isnan(f[~res.ok])) and np.all(np.isfinite(f[res.ok]))
    assert res.n_masked == int((~res.ok).sum())
    assert len(set(reason.ravel().tolist())) >= 3    # several mask reasons
    # the reported correlation: 0 for separate fits, the joint fit's own
    # value with joint=True, NaN where masked
    if joint:
        i, j = np.argwhere(good)[0]
        f1 = rs.fit_two_modes(x, cube[i, j], w1, w2, 385.0, 404.0,
                              baseline=baseline, joint=True)[4]
        assert abs(res.shift_correlation[i, j]
                   - f1.center_correlation) <= TOL_RHO
    else:
        assert np.all(res.shift_correlation[res.ok] == 0.0)
    assert np.all(np.isnan(res.shift_correlation[~res.ok]))


def test_fit_map_results_do_not_depend_on_the_block_size(monkeypatch):
    x, cube, peaks = _noisy_map(seed=3, H=5, W=7, with_peaks=True)
    a = rs.fit_map(x, cube, (370, 394.5), (395, 420), 385, 404, joint=True)
    monkeypatch.setattr(_mapfit, "_BLOCK", 4)
    b = rs.fit_map(x, cube, (370, 394.5), (395, 420), 385, 404, joint=True)
    # pixels with peaks: same decisions, numbers within the tolerances
    # (pixels without a peak: see TOL_SHIFT)
    assert np.array_equal(a.reason[peaks], b.reason[peaks])
    good = peaks & a.ok
    assert good.sum() >= 20
    assert_same_fits((b.dw1, b.dw2), (b.sigma1, b.sigma2),
                     (a.dw1, a.dw2), (a.sigma1, a.sigma2), good)
    assert np.all(np.abs(a.shift_correlation[good]
                         - b.shift_correlation[good]) <= TOL_RHO)


def test_fit_map_refuses_like_fit_lorentzian_on_a_repeated_axis_value():
    # fit_lorentzian refuses a zero starting width (an axis whose first
    # two points coincide, carrying the only counts above half height);
    # fit_map masks that pixel with reason 2, as the per-pixel loop did
    x, cube = _noisy_map(seed=5, H=2, W=3)
    x = x.copy()
    x[1] = x[0]
    cube[0, 0] = 0.0
    cube[0, 0, :2] = 100.0
    w1, w2 = (359.0, 394.5), (395.0, 420.0)
    _, ok, reason = _reference_fit_map(x, cube, w1, w2, 385.0, 404.0,
                                       "constant", False)
    res = rs.fit_map(x, cube, w1, w2, 385.0, 404.0)
    assert reason[0, 0] == 2 and np.array_equal(res.reason, reason)


# ----------------------------------------------------------------------
# Added: correlated shift errors in the two-mode inversion


def test_shift_correlation_is_the_gum_covariance():
    model = rs.SeparationModel(rs.mos2_eprime_a1())
    rng = np.random.default_rng(4)
    dw1, dw2 = rng.normal(0, 1, (2, 3, 4))
    s1, s2 = rng.uniform(0.05, 0.3, (2, 3, 4))
    rho = rng.uniform(-0.9, 0.9, (3, 4))
    res = model.invert(dw1, dw2, s1, s2, shift_correlation=rho)
    # independent: K^-1 S K^-T with the full 2 x 2 shift covariance
    S = np.empty((3, 4, 2, 2))
    S[..., 0, 0], S[..., 1, 1] = s1 ** 2, s2 ** 2
    S[..., 0, 1] = S[..., 1, 0] = rho * s1 * s2
    Ki = np.linalg.inv(np.array(rs.mos2_eprime_a1().matrix()))
    C = np.einsum("ij,...jk,lk->...il", Ki, S, Ki)
    np.testing.assert_allclose(res.strain_sigma, np.sqrt(C[..., 0, 0]),
                               rtol=1e-12)
    np.testing.assert_allclose(res.density_sigma, np.sqrt(C[..., 1, 1]),
                               rtol=1e-12)
    np.testing.assert_allclose(
        res.correlation,
        C[..., 0, 1] / np.sqrt(C[..., 0, 0] * C[..., 1, 1]), rtol=1e-10)
    # zero correlation is the old result bit for bit
    a = model.invert(dw1, dw2, s1, s2)
    b = model.invert(dw1, dw2, s1, s2, shift_correlation=0.0)
    for f in ("strain", "density", "strain_sigma", "density_sigma",
              "correlation"):
        assert np.array_equal(getattr(a, f), getattr(b, f))
    # NaN marks a masked pixel; refusals
    r2 = rho.copy()
    r2[0, 0] = np.nan
    out = model.invert(dw1, dw2, s1, s2, shift_correlation=r2)
    assert np.isnan(out.strain_sigma[0, 0])
    assert np.isfinite(out.strain_sigma[1:]).all()
    for bad in (1.5, -1.01, np.inf):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            model.invert(dw1, dw2, s1, s2, shift_correlation=bad)
    with pytest.raises(ValueError, match="needs both"):
        model.invert(dw1, dw2, s1, shift_correlation=0.3)
    with pytest.raises(ValueError, match="broadcast"):
        model.invert(dw1, dw2, s1, s2, shift_correlation=np.zeros(5))


def test_shift_correlation_matches_repeated_joint_fits():
    """Monte Carlo: 400 noisy repeats of two overlapping peaks, fitted
    jointly. The spread of the inverted density matches the error bar
    WITH the fitted correlation; ignoring it overstates the error bar
    by more than 20 percent here."""
    model = rs.SeparationModel(rs.mos2_eprime_a1())
    dw = model.forward(0.10, 0.20)
    x = np.linspace(350.0, 440.0, 181)
    clean = (rs.lorentzian(x, 384.6 + dw[0], 14.0, 400.0)
             + rs.lorentzian(x, 402.7 + dw[1], 16.0, 500.0) + 30.0)
    rng = np.random.default_rng(17)
    cube = clean + rng.normal(0.0, 10.0, (20, 20, x.size))
    f = rs.fit_map(x, cube, (360, 393.5), (394, 430), 384.6, 402.7,
                   joint=True)
    ok = f.ok
    assert ok.sum() >= 395
    rho = f.shift_correlation[ok]
    assert 0.3 < np.median(rho) < 0.5
    # the fitted correlation is the actual correlation of the shifts
    emp = np.corrcoef(f.dw1[ok], f.dw2[ok])[0, 1]
    assert abs(emp - np.median(rho)) < 0.12        # ~2.5 MC sigma
    a = model.invert(f.dw1[ok], f.dw2[ok], f.sigma1[ok], f.sigma2[ok])
    b = model.invert(f.dw1[ok], f.dw2[ok], f.sigma1[ok], f.sigma2[ok],
                     shift_correlation=rho)
    spread = np.std(a.density, ddof=1)
    assert abs(np.median(b.density_sigma) / spread - 1.0) < 0.12
    assert np.median(a.density_sigma) / spread > 1.15
    # separation_with_calibration passes the correlation through
    cs, cd = np.array([0.0, 0.2, 0.0, 0.2]), np.array([0.0, 0.0, 0.5, 0.5])
    K = np.array(rs.mos2_eprime_a1().matrix())
    cal = rs.calibrate_lever_arms(cs, cd, np.column_stack([cs, cd]) @ K.T,
                                  sigmas=0.05)
    s = rs.separation_with_calibration(cal, f.dw1[ok], f.dw2[ok],
                                       f.sigma1[ok], f.sigma2[ok],
                                       shift_correlation=rho)
    m = rs.SeparationModel(cal.coefficients("test"))
    want = m.invert(f.dw1[ok], f.dw2[ok], f.sigma1[ok], f.sigma2[ok],
                    shift_correlation=rho)
    np.testing.assert_array_equal(s["density_sigma_shifts"],
                                  want.density_sigma)


# ----------------------------------------------------------------------
# Added: error bars for noise that varies along the spectrum


def test_varying_noise_error_bars_match_shot_noise_scatter():
    """Seeded shot-noise (Poisson) spectra of one peak 400 counts high on
    a 30-count baseline: the actual scatter of the fitted center is 1.7
    times the default ("constant") error bar, and matches the sandwich
    ("varying") error bar. The fitted centers themselves are the same
    with either setting."""
    x = np.linspace(380.0, 430.0, 201)
    clean = rs.lorentzian(x, 404.0, 6.0, 400.0, 30.0)
    rng = np.random.default_rng(3)
    c, s_const, s_vary = [], [], []
    for y in rng.poisson(clean, (600, x.size)).astype(float):
        a = rs.fit_lorentzian(x, y)
        b = rs.fit_lorentzian(x, y, noise="varying")
        assert a.center == b.center and a.fwhm == b.fwhm
        c.append(a.center)
        s_const.append(a.center_sigma)
        s_vary.append(b.center_sigma)
    spread = np.std(c, ddof=1)
    r_const = spread / np.sqrt(np.mean(np.square(s_const)))
    r_vary = spread / np.sqrt(np.mean(np.square(s_vary)))
    assert 1.5 < r_const < 1.9            # the documented underestimate
    assert abs(r_vary - 1.0) < 0.1        # 600 spectra: ~3 percent MC error


def test_varying_noise_on_maps_and_its_independent_formula():
    x = np.linspace(370.0, 430.0, 241)
    clean = (rs.lorentzian(x, 385.0, 5.0, 400.0)
             + rs.lorentzian(x, 404.0, 6.0, 400.0) + 30.0)
    rng = np.random.default_rng(21)
    cube = rng.poisson(clean, (20, 30, x.size)).astype(float)
    w1, w2 = (372.0, 394.5), (395.0, 425.0)
    a = rs.fit_map(x, cube, w1, w2, 385.0, 404.0, joint=True)
    b = rs.fit_map(x, cube, w1, w2, 385.0, 404.0, joint=True,
                   noise="varying")
    assert np.array_equal(a.dw1, b.dw1) and np.array_equal(a.ok, b.ok)
    ok = b.ok
    assert ok.sum() == 600
    for dw, sa, sb in ((b.dw1, a.sigma1, b.sigma1), (b.dw2, a.sigma2,
                                                      b.sigma2)):
        spread = np.std(dw[ok], ddof=1)
        assert spread / np.sqrt(np.mean(sa[ok] ** 2)) > 1.25
        assert abs(spread / np.sqrt(np.mean(sb[ok] ** 2)) - 1.0) < 0.1
    # noise of the same size everywhere: both settings match the scatter
    g = clean + rng.normal(0.0, 12.0, (20, 30, x.size))
    for nz in ("constant", "varying"):
        f = rs.fit_map(x, g, w1, w2, 385.0, 404.0, joint=True, noise=nz)
        for dw, sg in ((f.dw1, f.sigma1), (f.dw2, f.sigma2)):
            ratio = np.std(dw[f.ok], ddof=1) / np.sqrt(np.mean(sg[f.ok] ** 2))
            assert abs(ratio - 1.0) < 0.1
    # the single-spectrum path, at every pixel checked (exactly equal
    # before v0.15; within the tolerances derived at TOL_SHIFT since)
    for i, j in ((0, 0), (7, 11), (19, 29)):
        d = rs.fit_two_modes(x, cube[i, j], w1, w2, 385.0, 404.0,
                             joint=True, noise="varying")
        assert abs(d[0] - b.dw1[i, j]) <= TOL_SHIFT * d[2]
        assert abs(d[1] - b.dw2[i, j]) <= TOL_SHIFT * d[3]
        assert d[2] == pytest.approx(b.sigma1[i, j], rel=TOL_SIGMA, abs=0)
        assert d[3] == pytest.approx(b.sigma2[i, j], rel=TOL_SIGMA, abs=0)
        assert abs(d[4].center_correlation
                   - b.shift_correlation[i, j]) <= TOL_RHO
    # the sandwich from an independent finite-difference Jacobian
    y = cube[3, 4][(x >= w2[0]) & (x <= w2[1])]
    xs = x[(x >= w2[0]) & (x <= w2[1])]
    f = rs.fit_lorentzian(xs, y, baseline="linear", noise="varying")
    p = np.array([f.center, f.fwhm, f.amplitude, f.offset, f.slope])
    x0 = xs.mean()

    def model(q):
        return rs.lorentzian(xs, q[0], q[1], q[2], q[3]) + q[4] * (xs - x0)
    J = np.empty((xs.size, 5))
    for k in range(5):
        h = 1e-6 * max(1.0, abs(p[k]))
        e = np.zeros(5)
        e[k] = h
        J[:, k] = (model(p + e) - model(p - e)) / (2 * h)
    r = y - model(p)
    B = np.linalg.inv(J.T @ J)
    cov = xs.size / (xs.size - 5) * B @ (J.T * r ** 2) @ J @ B
    np.testing.assert_allclose(
        [f.center_sigma, f.fwhm_sigma, f.amplitude_sigma, f.offset_sigma,
         f.slope_sigma], np.sqrt(np.diag(cov)), rtol=1e-5)
    for call in (lambda: rs.fit_lorentzian(xs, y, noise="poisson"),
                 lambda: rs.fit_two_modes(x, cube[0, 0], w1, w2, 385.0,
                                          404.0, noise="shot"),
                 lambda: rs.fit_map(x, cube, w1, w2, 385.0, 404.0,
                                    noise=None)):
        with pytest.raises(ValueError, match="noise"):
            call()


# ----------------------------------------------------------------------
# Added: region means and contrasts


def _region_data(seed=0, H=8, W=10):
    rng = np.random.default_rng(seed)
    model = rs.SeparationModel(rs.mos2_a1_2la())
    strain = np.full((H, W), 0.1)
    density = np.full((H, W), 0.03)
    edge = np.zeros((H, W), dtype=bool)
    edge[:, :2] = True
    density[edge] = 0.25
    dw1, dw2 = model.forward(strain, density)
    s1 = rng.uniform(0.05, 0.15, (H, W))
    s2 = rng.uniform(0.1, 0.3, (H, W))
    dw1 = dw1 + s1 * rng.standard_normal((H, W))
    dw2 = dw2 + s2 * rng.standard_normal((H, W))
    return model, dw1, dw2, s1, s2, edge


def test_region_mean_matches_the_per_pixel_inversion():
    model, dw1, dw2, s1, s2, edge = _region_data()
    rho = np.full(dw1.shape, -0.3)
    dw1[3, 5] = np.nan                                  # a masked pixel
    inner = ~edge
    r = rs.region_mean(rs.mos2_a1_2la(), dw1, dw2, s1, s2, inner,
                       shift_correlation=rho, ref_sigma=(0.2, 0.3))
    inv = model.invert(dw1, dw2, s1, s2, shift_correlation=rho)
    use = inner & np.isfinite(dw1)
    N = int(use.sum())
    assert r.n_pixels == N and r.n_excluded == 1
    # value: the mean of the per-pixel maps (an independent path)
    assert r.strain == pytest.approx(np.mean(inv.strain[use]), abs=1e-14)
    assert r.density == pytest.approx(np.mean(inv.density[use]), abs=1e-14)
    # shift part: the sum of the per-pixel covariances over N^2
    vs = np.sum(inv.strain_sigma[use] ** 2) / N ** 2
    vd = np.sum(inv.density_sigma[use] ** 2) / N ** 2
    assert r.strain_sigma_shifts == pytest.approx(np.sqrt(vs), rel=1e-12)
    assert r.density_sigma_shifts == pytest.approx(np.sqrt(vd), rel=1e-12)
    # reference part: one pixel with shift errors (0.2, 0.3)
    one = model.invert(0.0, 0.0, 0.2, 0.3)
    assert r.strain_sigma_reference == pytest.approx(
        float(one.strain_sigma), rel=1e-12)
    assert r.density_sigma_reference == pytest.approx(
        float(one.density_sigma), rel=1e-12)
    assert r.strain_sigma_calibration == 0.0          # exact lever arms
    assert r.strain_sigma == pytest.approx(
        np.hypot(r.strain_sigma_shifts, r.strain_sigma_reference), rel=1e-12)
    cxy = (np.sum((inv.correlation * inv.strain_sigma
                   * inv.density_sigma)[use]) / N ** 2
           + float(one.correlation * one.strain_sigma * one.density_sigma))
    assert r.covariance[0, 1] == pytest.approx(cxy, rel=1e-10)
    # scatter: the sample standard deviation of the pixel maps
    assert r.strain_scatter == pytest.approx(
        np.std(inv.strain[use], ddof=1), rel=1e-12)


def test_region_homogeneity_check_is_k_free_and_chi_square_distributed():
    model, dw1, dw2, s1, s2, edge = _region_data(seed=2)
    rho = np.full(dw1.shape, 0.4)
    r = rs.region_mean(rs.mos2_a1_2la(), dw1, dw2, s1, s2, ~edge,
                       shift_correlation=rho)
    # the same chi2 from (strain, density) space with the per-pixel
    # covariances of SeparationModel.invert (independent path)
    inv = model.invert(dw1, dw2, s1, s2, shift_correlation=rho)
    m = ~edge
    X = np.column_stack([inv.strain[m], inv.density[m]])
    C = np.empty((X.shape[0], 2, 2))
    C[:, 0, 0] = inv.strain_sigma[m] ** 2
    C[:, 1, 1] = inv.density_sigma[m] ** 2
    C[:, 0, 1] = C[:, 1, 0] = (inv.correlation * inv.strain_sigma
                               * inv.density_sigma)[m]
    Wp = np.linalg.inv(C)
    xw = np.linalg.solve(Wp.sum(0), np.einsum("pij,pj->i", Wp, X))
    e = X - xw
    q = np.einsum("pi,pij,pj->", e, Wp, e)
    assert r.chi2 == pytest.approx(q, rel=1e-9)
    assert r.dof == 2 * (X.shape[0] - 1)
    # Monte Carlo on a homogeneous region: the mean chi2 is the dof and
    # about 5 percent of the p-values fall below 0.05
    rng = np.random.default_rng(8)
    H, W = 4, 6
    sa = rng.uniform(0.05, 0.2, (H, W))
    sb = rng.uniform(0.1, 0.3, (H, W))
    q, p = [], []
    for _ in range(400):
        z1 = rng.standard_normal((H, W))
        z2 = 0.5 * z1 + np.sqrt(0.75) * rng.standard_normal((H, W))
        res = rs.region_mean(rs.mos2_a1_2la(), -0.5 + sa * z1,
                             -2.0 + sb * z2, sa, sb,
                             shift_correlation=0.5)
        q.append(res.chi2)
        p.append(res.p_value)
    dof = 2 * (H * W - 1)
    assert abs(np.mean(q) - dof) < 3.0 * np.sqrt(2.0 * dof / 400)
    assert abs(np.mean(np.array(p) < 0.05) - 0.05) < 0.035
    # a region with a real strain gradient fails the check
    _, dw1, dw2, s1, s2, edge = _region_data(seed=2)
    g = rs.region_mean(rs.mos2_a1_2la(), dw1, dw2, s1, s2)   # whole map
    assert g.p_value < 1e-6


def test_region_calibration_part_and_contrast():
    model, dw1, dw2, s1, s2, edge = _region_data(seed=5)
    strain = np.array([0.0, 0.2, 0.4, 0.0, 0.2, 0.4])
    density = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0])
    rng = np.random.default_rng(6)
    shifts = np.column_stack([strain, density]) @ K_MOS2.T
    cal = rs.calibrate_lever_arms(strain, density,
                                  shifts + rng.normal(0, 0.1, shifts.shape),
                                  sigmas=0.1)
    inner = ~edge
    r = rs.region_mean(cal, dw1, dw2, s1, s2, inner)
    # the calibration part equals that of separation_with_calibration
    # (its own loop over modes and arms) at the mean shifts
    ref = rs.separation_with_calibration(cal, dw1[inner].mean(),
                                         dw2[inner].mean())
    assert r.strain_sigma_calibration == pytest.approx(
        float(ref["strain_sigma_calibration"]), rel=1e-10)
    assert r.density_sigma_calibration == pytest.approx(
        float(ref["density_sigma_calibration"]), rel=1e-10)
    # and the derivative behind it agrees with finite differences of a
    # re-solve with one lever arm perturbed
    ybar = np.array([dw1[inner].mean(), dw2[inner].mean()])
    xbar = np.linalg.solve(cal.K, ybar)
    Kinv = np.linalg.inv(cal.K)
    h = 1e-6
    for m in range(2):
        for j in range(2):
            Kp = cal.K.copy()
            Kp[m, j] += h
            Km = cal.K.copy()
            Km[m, j] -= h
            fd = (np.linalg.solve(Kp, ybar) - np.linalg.solve(Km, ybar)) / (2 * h)
            np.testing.assert_allclose(fd, -Kinv[:, m] * xbar[j], rtol=1e-6)
    # contrast = difference of the two means; shift parts add; the
    # reference error cancels exactly (a common offset leaves it alone)
    c = rs.region_contrast(cal, dw1, dw2, s1, s2, edge, inner)
    a = rs.region_mean(cal, dw1, dw2, s1, s2, edge, ref_sigma=(0.2, 0.3))
    assert c.density == pytest.approx(a.density - r.density, abs=1e-13)
    assert c.density_sigma_shifts == pytest.approx(
        np.hypot(a.density_sigma_shifts, r.density_sigma_shifts), rel=1e-12)
    assert c.strain_sigma_reference == 0.0 == c.density_sigma_reference
    c2 = rs.region_contrast(cal, dw1 + 0.37, dw2 - 0.21, s1, s2, edge, inner)
    assert c2.density == pytest.approx(c.density, abs=1e-12)
    a2 = rs.region_mean(cal, dw1 + 0.37, dw2 - 0.21, s1, s2, edge)
    assert a2.density - a.density == pytest.approx(
        float(np.linalg.solve(cal.K, [0.37, -0.21])[1]), abs=1e-12)
    dref = rs.separation_with_calibration(
        cal, dw1[edge].mean() - dw1[inner].mean(),
        dw2[edge].mean() - dw2[inner].mean())
    assert c.density_sigma_calibration == pytest.approx(
        float(dref["density_sigma_calibration"]), rel=1e-10)
    assert c.n_pixels == (int(edge.sum()), int(inner.sum()))


def test_region_error_budget_matches_monte_carlo_with_recalibration():
    """Each trial repeats everything: the calibration measurement (and
    fit), the map noise and the reference-frequency error. The spread
    of the region mean and of the contrast over 400 trials matches the
    reported total error bars."""
    rng = np.random.default_rng(12)
    H, W = 6, 8
    strain = np.full((H, W), 0.15)
    density = np.full((H, W), 0.05)
    edge = np.zeros((H, W), dtype=bool)
    edge[:, :2] = True
    density[edge] = 0.30
    clean = np.einsum("mk,khw->mhw", K_MOS2, np.stack([strain, density]))
    cs = np.array([0.0, 0.2, 0.4, 0.0, 0.2, 0.4])
    cd = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0])
    cshift = np.column_stack([cs, cd]) @ K_MOS2.T
    s1, s2 = 0.1, 0.2
    u = (0.15, 0.25)
    means, contrasts, rep_m, rep_c = [], [], [], []
    for _ in range(400):
        cal = rs.calibrate_lever_arms(
            cs, cd, cshift + rng.normal(0, 0.1, cshift.shape), sigmas=0.1)
        dref = rng.normal(0, u)                    # common reference error
        d1 = clean[0] + s1 * rng.standard_normal((H, W)) - dref[0]
        d2 = clean[1] + s2 * rng.standard_normal((H, W)) - dref[1]
        m = rs.region_mean(cal, d1, d2, s1, s2, ~edge, ref_sigma=u)
        c = rs.region_contrast(cal, d1, d2, s1, s2, edge, ~edge)
        means.append([m.strain, m.density])
        contrasts.append([c.strain, c.density])
        rep_m.append([m.strain_sigma, m.density_sigma])
        rep_c.append([c.strain_sigma, c.density_sigma])
    means, contrasts = np.array(means), np.array(contrasts)
    ratio_m = np.std(means, axis=0, ddof=1) / np.median(rep_m, axis=0)
    ratio_c = np.std(contrasts, axis=0, ddof=1) / np.median(rep_c, axis=0)
    # 400 trials: the standard deviation is known to ~3.5 percent
    np.testing.assert_allclose(ratio_m, 1.0, atol=0.12)
    np.testing.assert_allclose(ratio_c, 1.0, atol=0.12)
    # the contrast is unbiased and far better known than either mean
    err = np.std(contrasts, axis=0, ddof=1) / np.sqrt(400)
    assert np.all(np.abs(np.mean(contrasts, axis=0) - [0.0, 0.25]) < 4 * err)
    assert np.median(rep_c, axis=0)[1] < 0.5 * np.median(rep_m, axis=0)[1]


def test_region_refusals():
    model, dw1, dw2, s1, s2, edge = _region_data()
    coef = rs.mos2_a1_2la()
    with pytest.raises(ValueError, match="share pixels"):
        rs.region_contrast(coef, dw1, dw2, s1, s2, edge, np.ones_like(edge))
    with pytest.raises(ValueError, match="no pixel"):
        rs.region_mean(coef, dw1, dw2, s1, s2, np.zeros_like(edge))
    with pytest.raises(ValueError, match="boolean"):
        rs.region_mean(coef, dw1, dw2, s1, s2, edge.astype(int))
    with pytest.raises(ValueError, match="boolean"):
        rs.region_mean(coef, dw1, dw2, s1, s2, edge[:, :3])
    with pytest.raises(ValueError, match="positive"):
        rs.region_mean(coef, dw1, dw2, 0.0, s2)
    with pytest.raises(ValueError, match="ref_sigma"):
        rs.region_mean(coef, dw1, dw2, s1, s2, ref_sigma=(0.1, -0.1))
    with pytest.raises(ValueError, match="same shape"):
        rs.region_mean(coef, dw1, dw2[:, :3], s1, s2)
    with pytest.raises(ValueError, match="singular"):
        rs.region_mean(rs.ModeCoefficients("a", "b", 1.0, 2.0, 2.0, 4.0,
                                           "test"), dw1, dw2, s1, s2)
    with pytest.raises(ValueError, match="ModeCoefficients"):
        rs.region_mean(K_MOS2, dw1, dw2, s1, s2)
    cal3 = rs.calibrate_lever_arms([0.0, 0.2, 0.0, 0.2], [0, 0, 0.5, 0.5],
                                   np.zeros((4, 3)) + [[0, 0, 0]],
                                   sigmas=0.1)
    with pytest.raises(ValueError, match="exactly two modes"):
        rs.region_mean(cal3, dw1, dw2, s1, s2)
    with pytest.raises(ValueError, match="homogeneity"):
        rs.region_mean(coef, dw1, dw2, s1, s2, shift_correlation=1.0)
    # density in cm^-2 is accepted (unit-free rank test) and gives the
    # same physics
    cm = rs.ModeCoefficients("A'1", "2LA(M)", -5.1, -2.2 / CM2, -20.9,
                             0.0, "test")
    a = rs.region_mean(coef, dw1, dw2, s1, s2, edge)
    b = rs.region_mean(cm, dw1, dw2, s1, s2, edge)
    assert b.density / CM2 == pytest.approx(a.density, rel=1e-12)
    assert b.chi2 == pytest.approx(a.chi2, rel=1e-12)
    single = np.zeros_like(edge)
    single[4, 4] = True
    one = rs.region_mean(coef, dw1, dw2, s1, s2, single)
    assert one.n_pixels == 1 and one.chi2 is None and one.dof is None


# ----------------------------------------------------------------------
# Tutorials


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_tutorial(name):
    path = os.path.join(ROOT, "examples", name)
    if not os.path.exists(path):          # e.g. tests run from an sdist
        pytest.skip("examples/ is not present")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, path], capture_output=True,
                         text=True, env=env, timeout=120, check=True)
    return out.stdout


def _docstring_output(name):
    """The output pasted at the top of the tutorial's docstring."""
    with open(os.path.join(ROOT, "examples", name), encoding="utf-8") as fh:
        text = fh.read()
    block = re.split(r"Output \(ramansep [0-9.]+\):\n\n", text,
                     maxsplit=1)[1]
    return block.split('"""', 1)[0]


@pytest.mark.parametrize("name", ["tutorial_edge_charge.py",
                                  "tutorial_overlapping_peaks.py",
                                  "tutorial_edge_profile.py"])  # v0.15
def test_tutorials_print_what_their_docstrings_and_the_readme_show(name):
    out = _run_tutorial(name)
    assert out == _docstring_output(name)
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    assert out in readme


def test_tutorial_key_numbers():
    edge = _run_tutorial("tutorial_edge_charge.py")
    assert "Edge minus interior: strain -0.0005 +/- 0.0010 %, density " \
           "+0.242 +/- 0.010 (1e13 cm^-2)" in edge
    assert "reference 0 (cancels)" in edge
    over = _run_tutorial("tutorial_overlapping_peaks.py")
    assert "correlation +0.41" in over
    assert "density spread 0.0284, error bar 0.0361 (independent) / " \
           "0.0287 (correlated)" in over
