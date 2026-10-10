"""v0.13 anchors.

1. Fixes (each test fails on 0.12.0 and passes now):
   * `load_spectrum_csv` returned a wavenumber axis containing NaN or
     infinity, although it promises to refuse an axis that is not
     strictly increasing (a NaN compares False, so `diff <= 0` let it
     through).
   * `fit_map` with a NaN reference frequency reported every pixel as
     fitted (`ok` True, `n_masked` 0) while every shift was NaN;
     `fit_two_modes` returned NaN shifts without an error.
   * `bayesian_map_inversion` with a NaN or infinite smoothness weight
     returned all-NaN maps with only SciPy's "Matrix is exactly
     singular" warning.
2. `load_spectrum_table` / `load_map_table`: files written here, line by
   line, from arrays built independently; the loaders must give those
   arrays back exactly, whatever the line order, axis direction,
   separator and column order.
3. `calibrate_lever_arms(..., fit_offset=True)`: checked against
   `numpy.linalg.lstsq`, against the closed-form covariance of the
   Frisch-Waugh-Lovell theorem, against the closed-form bias of the
   through-origin fit, and against seeded Monte Carlo.
"""
import warnings

import numpy as np
import pytest

import ramansep as rs
from ramansep import lorentzian

# ----------------------------------------------------------------------
# 1. Fixes


def test_load_spectrum_csv_refuses_non_finite_axis(tmp_path):
    p = tmp_path / "s.csv"
    # 0.12.0 returned the axis [400, nan, 402, 403] without an error
    p.write_text("wavenumber_cm1,counts\n400,1\nnan,2\n402,3\n403,4\n")
    with pytest.raises(ValueError, match="NaN or infinite"):
        rs.load_spectrum_csv(p)
    # 0.12.0 returned [400, 401, inf]
    p.write_text("wavenumber_cm1,counts\n400,1\n401,2\ninf,3\n")
    with pytest.raises(ValueError, match="NaN or infinite"):
        rs.load_spectrum_csv(p)
    # finite, increasing axes still load exactly
    p.write_text("wavenumber_cm1,counts\n400.5,1\n401.5,nan\n")
    x, y = rs.load_spectrum_csv(p)
    assert np.array_equal(x, [400.5, 401.5]) and np.isnan(y[1])


def test_load_map_csv_non_finite_axis_has_a_clear_message(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("row,col,wavenumber_cm1,counts\n0,0,400,1\n0,0,nan,2\n"
                 "0,0,402,3\n")
    # 0.12.0 refused this too, but said that pixel (0, 0) "carries a
    # different wavenumber axis than pixel (0, 0)"
    with pytest.raises(ValueError, match="NaN or infinite"):
        rs.load_map_csv(p)
    p.write_text("row,col,wavenumber_cm1,counts\n0,0,400,1\n0,x,401,2\n")
    with pytest.raises(ValueError, match="line 3"):
        rs.load_map_csv(p)


def _two_peak_cube():
    x = np.linspace(380.0, 480.0, 401)
    y = (lorentzian(x, 404.9, 3.0, 800.0, 50.0)
         + lorentzian(x, 451.7, 4.0, 500.0, 0.0))
    return x, np.tile(y, (2, 3, 1))


def test_fit_map_refuses_a_non_finite_reference():
    x, cube = _two_peak_cube()
    # 0.12.0: ok was True at all 6 pixels, n_masked 0, every dw1 NaN
    with pytest.raises(ValueError, match="ref1 and ref2 must be finite"):
        rs.fit_map(x, cube, (395, 415), (440, 465), np.nan, 452.0)
    with pytest.raises(ValueError, match="ref1 and ref2 must be finite"):
        rs.fit_map(x, cube, (395, 415), (440, 465), 404.7, np.inf)
    # 0.12.0: fit_two_modes returned dw1 = nan without an error
    with pytest.raises(ValueError, match="ref1 and ref2 must be finite"):
        rs.fit_two_modes(x, cube[0, 0], (395, 415), (440, 465), np.nan,
                         452.0)
    # finite references: unchanged, and every pixel equals the direct fit
    res = rs.fit_map(x, cube, (395, 415), (440, 465), 404.7, 452.0)
    d1, d2, s1, s2, _, _ = rs.fit_two_modes(x, cube[1, 2], (395, 415),
                                            (440, 465), 404.7, 452.0)
    assert res.ok.all()
    assert res.dw1[1, 2] == d1 and res.dw2[1, 2] == d2
    assert abs(d1 - 0.2) < 1e-2 and abs(d2 + 0.3) < 1e-2


def test_bayesian_refuses_non_finite_smoothness_weights():
    K = np.array([[-2.3, 1.1], [-0.9, -1.7], [0.4, 2.2]])
    shifts = np.random.default_rng(0).normal(size=(3, 4, 4))
    sig = [0.1, 0.1, 0.1]
    # 0.12.0: all-NaN maps and a SciPy MatrixRankWarning, no error
    for lam in (np.nan, np.inf):
        with pytest.raises(ValueError, match="must be finite"):
            rs.bayesian_map_inversion(K, shifts, sig, lam)
        with pytest.raises(ValueError, match="must be finite"):
            rs.bayesian_map_inversion(K, shifts, sig, 1.0, lam)
    # a large finite weight still works and approaches the pooled value
    r = rs.bayesian_map_inversion(K, shifts, sig, 1e9)
    assert np.all(np.isfinite(r.strain))


def test_condition_number_is_that_of_the_unweighted_matrix():
    """Documentation fix: `MultiModeResult.condition_number` is cond(K)
    whatever the sigmas (the docstring said "weighted")."""
    K = np.array([[-2.3, 1.1], [-0.9, -1.7], [0.4, 2.2]])
    res = rs.MultiModeModel(K).invert([0.1, 0.2, 0.3],
                                      sigmas=[0.01, 1.0, 1.0])
    s = np.linalg.svd(K, compute_uv=False)
    assert res.condition_number == pytest.approx(s[0] / s[-1], rel=1e-12)


def test_clear_messages_for_wrong_numbers_of_maps_and_sigmas():
    K = np.array([[-2.3, 1.1], [-0.9, -1.7], [0.4, 2.2]])
    mm = rs.MultiModeModel(K)
    shifts = [np.zeros((2, 2))] * 3
    with pytest.raises(ValueError, match="expected 3 sigmas"):
        mm.invert(shifts, sigmas=[0.1, 0.1])
    with pytest.raises(ValueError, match="mode 2 has shape"):
        mm.invert(shifts, sigmas=[0.1, np.ones(3), 0.1])
    with pytest.raises(ValueError, match="same shape"):
        mm.invert([np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(3)])
    K3 = np.array([[-2.1, -0.8, -0.011], [-1.0, -2.3, -0.016],
                   [-3.2, -0.4, -0.007], [-0.6, -1.1, -0.021]])
    with pytest.raises(ValueError, match="expected 4 sigmas"):
        rs.ThreeCauseModel(K3).invert([0.0] * 4, sigmas=[0.1] * 3)


# ----------------------------------------------------------------------
# 2. Plain text tables


def _raster(H=3, W=4, L=9, seed=5):
    """Independent truth: stage positions, axis and spectra."""
    rng = np.random.default_rng(seed)
    xpos = -1.5 + 0.5 * np.arange(W)                # micrometres
    ypos = 2.0 + 0.5 * np.arange(H)
    wn = 380.0 + 2.5 * np.arange(L)
    cube = rng.uniform(0.0, 1000.0, (H, W, L))
    return xpos, ypos, wn, cube


def _write_long(path, lines, header=None, sep="\t"):
    text = "" if header is None else header + "\n"
    text += "\n".join(sep.join(repr(float(v)) for v in rec)
                      for rec in lines) + "\n"
    path.write_text(text)


def test_map_table_rebuilds_a_serpentine_descending_scan(tmp_path):
    xpos, ypos, wn, cube = _raster()
    H, W, L = cube.shape
    lines = []
    for i in range(H - 1, -1, -1):                  # y from top to bottom
        cols = range(W) if i % 2 else range(W - 1, -1, -1)   # serpentine
        for j in cols:
            for k in range(L - 1, -1, -1):          # axis high -> low
                lines.append((xpos[j], ypos[i], wn[k], cube[i, j, k]))
    p = tmp_path / "scan.txt"
    _write_long(p, lines, header="#X\t#Y\t#Wave\t#Intensity")
    wn2, cube2, x2, y2 = rs.load_map_table(p)
    assert np.array_equal(wn2, wn) and np.array_equal(cube2, cube)
    assert np.array_equal(x2, xpos) and np.array_equal(y2, ypos)

    # comma-separated, text header, columns in another order
    q = tmp_path / "scan.csv"
    _write_long(q, [(rec[2], rec[3], rec[0], rec[1]) for rec in lines],
                header="wavenumber,counts,x_um,y_um", sep=",")
    wn3, cube3, x3, y3 = rs.load_map_table(q, usecols=(2, 3, 0, 1))
    assert np.array_equal(wn3, wn) and np.array_equal(cube3, cube)


def test_map_table_feeds_fit_map_bit_for_bit(tmp_path):
    x = np.linspace(380.0, 480.0, 201)
    rng = np.random.default_rng(9)
    cube = np.empty((2, 3, x.size))
    for i in range(2):
        for j in range(3):
            cube[i, j] = (lorentzian(x, 404.7 + 0.1 * j, 3.0, 800.0, 50.0)
                          + lorentzian(x, 452.0 - 0.2 * i, 4.0, 500.0, 0.0)
                          + rng.normal(0.0, 3.0, x.size))
    lines = [(10.0 * j, 10.0 * i, x[k], cube[i, j, k])
             for i in range(2) for j in range(3) for k in range(x.size)]
    p = tmp_path / "map.txt"
    _write_long(p, lines[::-1], sep=" ")             # reversed line order
    wn, c2, _, _ = rs.load_map_table(p)
    a = rs.fit_map(x, cube, (395, 415), (440, 465), 404.7, 452.0)
    b = rs.fit_map(wn, c2, (395, 415), (440, 465), 404.7, 452.0)
    for f in ("dw1", "dw2", "sigma1", "sigma2", "ok"):
        assert np.array_equal(getattr(a, f), getattr(b, f))


def test_map_table_refusals(tmp_path):
    xpos, ypos, wn, cube = _raster(H=2, W=2, L=4)
    full = [(xpos[j], ypos[i], wn[k], cube[i, j, k])
            for i in range(2) for j in range(2) for k in range(4)]
    p = tmp_path / "bad.txt"

    _write_long(p, [r for r in full if not (r[0] == xpos[1]
                                             and r[1] == ypos[1])])
    with pytest.raises(ValueError, match="complete rectangular grid"):
        rs.load_map_table(p)
    _write_long(p, full[:-1])                         # one point short
    with pytest.raises(ValueError, match="different numbers"):
        rs.load_map_table(p)
    shifted = [r if r[0] != xpos[1] or r[1] != ypos[0]
               else (r[0], r[1], r[2] + 0.5, r[3]) for r in full]
    _write_long(p, shifted)
    with pytest.raises(ValueError, match="different wavenumber axis"):
        rs.load_map_table(p)
    dup = [r if r[2] != wn[1] else (r[0], r[1], wn[0], r[3]) for r in full]
    _write_long(p, dup)
    with pytest.raises(ValueError, match="no repeated wavenumber"):
        rs.load_map_table(p)
    _write_long(p, [(np.nan,) + r[1:] if n == 0 else r
                    for n, r in enumerate(full)])
    with pytest.raises(ValueError, match="stage x position"):
        rs.load_map_table(p)
    _write_long(p, [(r[0], r[1], np.inf if n == 2 else r[2], r[3])
                    for n, r in enumerate(full)])
    with pytest.raises(ValueError, match="wavenumber column"):
        rs.load_map_table(p)
    p.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError, match="could not read 4 numeric"):
        rs.load_map_table(p)
    p.write_text("# nothing here\n\n")
    with pytest.raises(ValueError, match="no data lines"):
        rs.load_map_table(p)
    with pytest.raises(ValueError, match="usecols"):
        rs.load_map_table(p, usecols=(0, 1, 2, 2))

    # positions that jitter in the last digits: refused exactly, merged
    # with `decimals`
    jitter = [(r[0] + (1e-9 if r[2] == wn[0] else 0.0),) + r[1:]
              for r in full]
    _write_long(p, jitter)
    with pytest.raises(ValueError, match="decimals="):
        rs.load_map_table(p)
    wn2, cube2, x2, _ = rs.load_map_table(p, decimals=6)
    assert np.array_equal(cube2, cube) and np.array_equal(x2, xpos)


def test_spectrum_table_reads_descending_exports(tmp_path):
    x = np.linspace(380.0, 480.0, 41)
    y = lorentzian(x, 404.7, 3.0, 800.0, 50.0)
    p = tmp_path / "spec.txt"
    _write_long(p, list(zip(x[::-1], y[::-1])),
                header="Raman shift (cm-1)\tIntensity")
    x2, y2 = rs.load_spectrum_table(p)
    assert np.array_equal(x2, x) and np.array_equal(y2, y)
    q = tmp_path / "spec.csv"
    _write_long(q, list(zip(np.zeros(x.size), x, y)), sep=",")
    x3, y3 = rs.load_spectrum_table(q, usecols=(1, 2))
    assert np.array_equal(x3, x) and np.array_equal(y3, y)
    for bad in ([400.0, 401.0, 401.0], [400.0, 402.0, 401.0],
                [400.0, np.nan, 402.0], [400.0]):
        _write_long(p, [(v, 1.0) for v in bad])
        with pytest.raises(ValueError, match="wavenumber axis"):
            rs.load_spectrum_table(p)


# ----------------------------------------------------------------------
# 3. Calibration with a free offset per mode

K_TRUE = np.array([[-5.1, -2.2], [-20.9, 0.0], [-8.0, -1.1]])
C_TRUE = np.array([0.7, -1.9, 0.25])      # unknown starting-state shifts
EPS = np.array([0.0, 0.2, 0.4, 0.6, 0.0, 0.3, 0.1])
RHO = np.array([0.0, 0.0, 0.0, 0.1, 0.8, 0.5, 1.1])


def _shifts(noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.column_stack([EPS, RHO]) @ K_TRUE.T + C_TRUE
    return y + noise * rng.normal(size=y.shape)


def test_offset_fit_recovers_noise_free_arms_and_offsets_exactly():
    cal = rs.calibrate_lever_arms(EPS, RHO, _shifts(), sigmas=0.1,
                                  fit_offset=True)
    assert np.allclose(cal.K, K_TRUE, atol=1e-12, rtol=0)
    assert np.allclose(cal.offset, C_TRUE, atol=1e-12, rtol=0)
    assert cal.chi2_dof == EPS.size - 3
    assert np.all(cal.chi2 < 1e-20)
    assert cal.full_cov.shape == (3, 3, 3)
    assert np.array_equal(cal.cov, cal.full_cov[:, :2, :2])
    # the default fit is untouched and carries no offset
    plain = rs.calibrate_lever_arms(EPS, RHO, _shifts() - C_TRUE,
                                    sigmas=0.1)
    assert plain.offset is None and plain.full_cov is None
    assert np.allclose(plain.K, K_TRUE, atol=1e-12, rtol=0)


def test_through_origin_bias_is_the_closed_form_projection():
    """Forcing data with offsets c through zero gives
    K + c (X^T W X)^-1 X^T W 1 per mode (normal equations); the offset
    fit removes the bias."""
    y = _shifts()
    sig = 0.1
    plain = rs.calibrate_lever_arms(EPS, RHO, y, sigmas=sig)
    X = np.column_stack([EPS, RHO])
    proj = np.linalg.solve(X.T @ X, X.T @ np.ones(EPS.size))
    expected = K_TRUE + C_TRUE[:, None] * proj[None, :]
    assert np.allclose(plain.K, expected, atol=1e-12, rtol=0)
    assert np.abs(plain.K - K_TRUE).max() > 1.0          # a real bias
    off = rs.calibrate_lever_arms(EPS, RHO, y, sigmas=sig, fit_offset=True)
    assert np.allclose(off.K, K_TRUE, atol=1e-12, rtol=0)


def test_offset_fit_agrees_with_lstsq_and_the_fwl_closed_form():
    rng = np.random.default_rng(3)
    y = _shifts(noise=0.1, seed=3)
    sig = rng.uniform(0.05, 0.2, y.shape)              # per point and mode
    cal = rs.calibrate_lever_arms(EPS, RHO, y, sigmas=sig, fit_offset=True)
    X = np.column_stack([EPS, RHO])
    for k in range(3):
        w = 1.0 / sig[:, k] ** 2
        # independent solve on the whitened augmented system
        A = np.column_stack([X, np.ones(EPS.size)]) / sig[:, k:k + 1]
        beta, *_ = np.linalg.lstsq(A, y[:, k] / sig[:, k], rcond=None)
        assert np.allclose(cal.K[k], beta[:2], atol=1e-10, rtol=0)
        assert abs(cal.offset[k] - beta[2]) < 1e-10
        # Frisch-Waugh-Lovell: arms from the weighted-centred design
        xbar = (w[:, None] * X).sum(axis=0) / w.sum()
        Xc = X - xbar
        cov_arms = np.linalg.inv(Xc.T @ (w[:, None] * Xc))
        assert np.allclose(cal.cov[k], cov_arms, rtol=1e-10, atol=0)
        ybar = (w * y[:, k]).sum() / w.sum()
        assert abs(cal.offset[k] - (ybar - xbar @ cal.K[k])) < 1e-10
        var_c = 1.0 / w.sum() + xbar @ cov_arms @ xbar
        assert cal.offset_sigma[k] == pytest.approx(np.sqrt(var_c),
                                                    rel=1e-10)
        assert np.allclose(cal.full_cov[k, :2, 2], -cov_arms @ xbar,
                           rtol=1e-9, atol=1e-14)
        resid = (y[:, k] - X @ cal.K[k] - cal.offset[k]) / sig[:, k]
        assert cal.chi2[k] == pytest.approx(resid @ resid, rel=1e-9)


def test_offset_fit_without_sigmas_uses_the_residual_variance():
    y = _shifts(noise=0.1, seed=4)
    cal = rs.calibrate_lever_arms(EPS, RHO, y, fit_offset=True)
    A = np.column_stack([EPS, RHO, np.ones(EPS.size)])
    for k in range(3):
        beta, rss, *_ = np.linalg.lstsq(A, y[:, k], rcond=None)
        s2 = float(rss[0]) / (EPS.size - 3)
        cov = s2 * np.linalg.inv(A.T @ A)
        assert np.allclose(cal.K_sigma[k], np.sqrt(np.diag(cov))[:2],
                           rtol=1e-9, atol=0)
        assert cal.offset_sigma[k] == pytest.approx(np.sqrt(cov[2, 2]),
                                                    rel=1e-9)
    assert cal.chi2 is None and cal.chi2_dof is None


def test_offset_fit_error_bars_match_monte_carlo():
    sig = 0.1
    n_draw = 400
    rng = np.random.default_rng(11)
    clean = _shifts()
    Ks, cs, chi = [], [], []
    for _ in range(n_draw):
        y = clean + rng.normal(0.0, sig, clean.shape)
        cal = rs.calibrate_lever_arms(EPS, RHO, y, sigmas=sig,
                                      fit_offset=True)
        Ks.append(cal.K)
        cs.append(cal.offset)
        chi.append(cal.chi2)
    ref = rs.calibrate_lever_arms(EPS, RHO, clean, sigmas=sig,
                                  fit_offset=True)
    assert np.allclose(np.std(Ks, axis=0) / ref.K_sigma, 1.0, atol=0.15)
    assert np.allclose(np.std(cs, axis=0) / ref.offset_sigma, 1.0,
                       atol=0.15)
    # mean chi-square equals its degrees of freedom (n - 3 = 4)
    assert abs(np.mean(chi) - (EPS.size - 3)) < 0.4


def test_offset_fit_refusals_and_single_mode_sigmas():
    line_eps = np.array([0.0, 0.2, 0.4, 0.6])
    line_rho = 0.5 - line_eps                 # a line NOT through zero
    y = np.column_stack([line_eps, line_rho]) @ K_TRUE[:1].T + 0.3
    # the through-origin fit accepts it (rank 2) ...
    rs.calibrate_lever_arms(line_eps, line_rho, y, sigmas=0.1)
    # ... the offset fit cannot separate the offset from the arms
    with pytest.raises(ValueError, match="one straight line"):
        rs.calibrate_lever_arms(line_eps, line_rho, y, sigmas=0.1,
                                fit_offset=True)
    with pytest.raises(ValueError, match="n >= 4"):
        rs.calibrate_lever_arms(EPS[:3], RHO[[0, 1, 4]],
                                np.zeros(3), fit_offset=True)
    with pytest.raises(ValueError, match="one straight line"):
        rs.calibrate_lever_arms(np.ones(4), np.ones(4), np.zeros(4),
                                sigmas=0.1, fit_offset=True)
    # one mode given as (n,) with one sigma per point (new in v0.13)
    rng = np.random.default_rng(2)
    sig = rng.uniform(0.05, 0.2, EPS.size)
    y1 = _shifts(noise=0.1, seed=2)[:, 0]
    for fo in (False, True):
        a = rs.calibrate_lever_arms(EPS, RHO, y1, sigmas=sig, fit_offset=fo)
        cols = [EPS, RHO] + ([np.ones(EPS.size)] if fo else [])
        A = np.column_stack(cols) / sig[:, None]
        beta, *_ = np.linalg.lstsq(A, y1 / sig, rcond=None)
        assert np.allclose(a.K[0], beta[:2], atol=1e-10, rtol=0)
    with pytest.raises(ValueError, match="do not fit 7 reference points"):
        rs.calibrate_lever_arms(EPS, RHO, _shifts(), sigmas=[0.1] * 7)


def test_plan_with_offset_equals_the_calibration_covariance():
    sig = 0.15
    plan = rs.plan_calibration(EPS, RHO, sig, fit_offset=True)
    cal = rs.calibrate_lever_arms(EPS, RHO, _shifts(), sigmas=sig,
                                  fit_offset=True)
    assert plan["identifiable"]
    assert np.allclose(plan["cov"], cal.cov[0], rtol=0, atol=1e-15)
    assert plan["condition_number"] == cal.condition_number
    # the offset costs precision: never better than the origin fit
    plain = rs.plan_calibration(EPS, RHO, sig)
    assert np.all(plan["K_sigma"] >= plain["K_sigma"])
    # repeating the design r times divides the covariance by r exactly
    r, rep = rs.repeats_for_sigma(0.05, EPS, RHO, sig, fit_offset=True)
    assert np.allclose(rep["cov"], plan["cov"] / r, rtol=1e-12, atol=0)
    assert rep["K_sigma"].max() <= 0.05 < plan["K_sigma"].max() / np.sqrt(
        r - 1)
    line = rs.plan_calibration([0.0, 0.2, 0.4], [0.5, 0.3, 0.1], sig,
                               fit_offset=True)
    assert not line["identifiable"] and line["cov"] is None
    with pytest.raises(ValueError, match="collinear"):
        rs.repeats_for_sigma(0.05, [0.0, 0.2, 0.4], [0.5, 0.3, 0.1], sig,
                             fit_offset=True)


def test_offset_calibration_propagates_into_the_maps():
    """The arms' marginal covariance is what `separation_with_calibration`
    needs: the spread of the strain over re-drawn offset calibrations
    matches its calibration error bar."""
    K2 = K_TRUE[:2]
    eps, rho = EPS, RHO
    sig = 0.2
    clean = np.column_stack([eps, rho]) @ K2.T + C_TRUE[:2]
    ref = rs.calibrate_lever_arms(eps, rho, clean, sigmas=sig,
                                  fit_offset=True)
    dw1, dw2 = -2.0, -8.0
    out = rs.separation_with_calibration(ref, dw1, dw2)
    rng = np.random.default_rng(21)
    strains, dens = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(400):
            cal = rs.calibrate_lever_arms(
                eps, rho, clean + rng.normal(0.0, sig, clean.shape),
                sigmas=sig, fit_offset=True)
            x = np.linalg.solve(cal.K, [dw1, dw2])
            strains.append(x[0])
            dens.append(x[1])
    assert np.std(strains) == pytest.approx(
        float(out["strain_sigma_calibration"]), rel=0.2)
    assert np.std(dens) == pytest.approx(
        float(out["density_sigma_calibration"]), rel=0.2)
