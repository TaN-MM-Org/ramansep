"""v0.16 anchors.

Fixed (each test fails on 0.15.2):
* `MultiModeModel.invert` gave a mode with an infinite sigma zero
  weight, as it should, but still checked the pixel with m - 2 degrees
  of freedom: the p-value came out too large, and 1 where no check is
  possible. A pixel left with one usable mode was refused or not
  depending on rounding;
* `compare_mode_sets` accepted a `mode_names` list longer than the
  number of modes and used its first names (a shorter one failed with an
  IndexError);
* `design_references` cut a fractional `n_pick` down silently.

Added: correlated shift errors between modes in `MultiModeModel`,
`ThreeCauseModel` and `multimode_with_calibration` (generalized least
squares with the full shift covariance), checked against a whitened
`numpy.linalg.lstsq` solve built here, the closed form K^-1 S K^-T of the
two-mode inversion, finite differences of a re-solve and seeded Monte
Carlo draws of correlated noise; and `fit_voigt(..., noise="varying")`,
checked against the sandwich formula built from an independent
finite-difference Jacobian and against seeded shot-noise spectra.
"""
import numpy as np
import pytest
from scipy.stats import chi2 as chi2_dist

import ramansep as rs

K3 = np.array([[-5.1, -2.2], [-20.9, 0.0], [-8.0, -1.1]])
SIG3 = np.array([0.15, 0.10, 0.12])
K4 = np.array([[-5.1, -2.2], [-20.9, 0.0], [-8.0, -1.1], [-3.3, -0.7]])
SIG4 = np.array([0.1, 0.2, 0.15, 0.3])
T4 = np.array([-0.011, -0.016, -0.007, -0.021])     # illustrative, per K


def _random_correlations(m, n, seed):
    """n random, well-conditioned correlation matrices of size m, built
    from random Gram matrices."""
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, m, m + 2))
    C = G @ G.transpose(0, 2, 1)
    d = np.sqrt(np.einsum("pii->pi", C))
    return C / d[:, :, None] / d[:, None, :]


def _whitened_lstsq(K, s, sig, R):
    """Generalized least squares for one pixel, computed independently of
    the package: whiten with the Cholesky factor of the covariance
    S = D R D and solve the ordinary least-squares problem with
    numpy.linalg.lstsq. Returns (x, cov, chi2)."""
    S = np.diag(sig) @ R @ np.diag(sig)
    L = np.linalg.cholesky(S)
    Kw = np.linalg.solve(L, K)
    sw = np.linalg.solve(L, s)
    x = np.linalg.lstsq(Kw, sw, rcond=None)[0]
    cov = np.linalg.inv(Kw.T @ Kw)
    res = sw - Kw @ x
    return x, cov, float(res @ res)


# ----------------------------------------------------------------------
# Fixed: infinite sigmas in MultiModeModel


def test_infinite_sigma_checks_only_the_remaining_modes():
    """A mode with an infinite sigma carries no information. The estimate
    already ignored it; now the model check does too, and equals that of
    the model without the mode. 0.15.2 used m - 2 degrees of freedom: on
    this pixel p = 1.26e-5 instead of 2.03e-6, and p = 1 (instead of no
    check) when only two modes remained."""
    mm = rs.MultiModeModel(K4)
    s = mm.forward(0.1, 0.2)
    s[3] = s[3] + 0.5                                # mode 4 disagrees
    sig = [0.1, 0.1, np.inf, 0.1]
    got = mm.invert(s, sigmas=sig)
    keep = [0, 1, 3]
    ref = rs.MultiModeModel(K4[keep]).invert([s[k] for k in keep],
                                             sigmas=[0.1, 0.1, 0.1])
    assert got.dof == 2 and ref.dof == 1
    assert float(got.chi2_map) == pytest.approx(float(ref.chi2_map),
                                                rel=1e-12)
    assert float(got.p_value) == pytest.approx(float(ref.p_value),
                                               rel=1e-10)
    assert float(got.p_value) == pytest.approx(2.0266e-6, rel=1e-4)
    for f in ("strain", "density", "strain_sigma", "density_sigma"):
        assert float(getattr(got, f)) == pytest.approx(
            float(getattr(ref, f)), rel=1e-12, abs=1e-15)

    # three modes, one of them infinite: no redundancy is left
    m3 = rs.MultiModeModel(K3)
    r = m3.invert(m3.forward(0.1, 0.2), sigmas=[0.1, 0.1, np.inf])
    assert np.isnan(r.chi2_map) and np.isnan(r.p_value)
    assert float(r.strain) == pytest.approx(0.1, abs=1e-12)

    # on a map, only the pixel with the infinite sigma is affected
    rng = np.random.default_rng(1)
    shifts = [v + rng.normal(0.0, e, (4, 5))
              for v, e in zip(mm.forward(np.full((4, 5), 0.1),
                                         np.full((4, 5), 0.2)), SIG4)]
    smap = [np.full((4, 5), e) for e in SIG4]
    base = mm.invert(shifts, sigmas=smap)
    smap[2] = smap[2].copy()
    smap[2][1, 3] = np.inf
    mixed = mm.invert(shifts, sigmas=smap)
    other = np.ones((4, 5), bool)
    other[1, 3] = False
    for f in ("strain", "chi2_map", "p_value"):
        np.testing.assert_allclose(getattr(mixed, f)[other],
                                   getattr(base, f)[other], rtol=1e-12)
    one = rs.MultiModeModel(K4[[0, 1, 3]]).invert(
        [shifts[k][1, 3] for k in (0, 1, 3)],
        sigmas=[SIG4[k] for k in (0, 1, 3)])
    assert mixed.p_value[1, 3] == pytest.approx(float(one.p_value),
                                                rel=1e-10)


def test_a_single_usable_mode_is_always_refused():
    """With one finite sigma left, strain and density are not determined.
    0.15.2 refused that only when the rounding of a11 a22 - a12^2 came out
    <= 0: for 669 of these 2000 random matrices it returned finite values
    instead (for example strain -0.125 +/- 1.0e6)."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        K = np.round(rng.normal(0.0, 5.0, (3, 2)), 2)
        if not np.linalg.matrix_rank(K) == 2:        # pragma: no cover
            continue
        with pytest.raises(ValueError, match="singular"):
            rs.MultiModeModel(K).invert([1.0, 2.0, 3.0],
                                        sigmas=[0.1, np.inf, np.inf])
    # a masked pixel (NaN) next to it is still passed through as NaN
    r = rs.MultiModeModel(K3).invert(
        [np.array([0.0, np.nan])] * 3,
        sigmas=[np.array([0.1, np.nan]), np.array([0.1, np.inf]), 0.1])
    assert np.isfinite(r.strain[0]) and np.isnan(r.strain[1])


# ----------------------------------------------------------------------
# Fixed: argument checks


def test_compare_mode_sets_refuses_names_of_another_length():
    """0.15.2 used the first three of four names (mislabelling every
    subset when the list does not match the rows of K) and failed with
    an IndexError on a list that is too short."""
    with pytest.raises(ValueError, match="one name per mode"):
        rs.compare_mode_sets(K3, SIG3, mode_names=["E'", "A'1", "2LA(M)",
                                                   "X"])
    with pytest.raises(ValueError, match="one name per mode"):
        rs.compare_mode_sets(K3, SIG3, mode_names=["A'1"])
    ok = rs.compare_mode_sets(K3, SIG3, mode_names=["a", "b", "c"])
    assert [r["names"] for r in ok][0] == ["a", "b"]


def test_design_references_refuses_a_fractional_n_pick():
    """0.15.2 picked 2 states for n_pick=2.7."""
    eps = np.array([0.0, 0.1, 0.2, 0.3, 0.0, 0.25])
    rho = np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.8])
    for bad in (2.7, 3.5, None):
        with pytest.raises(ValueError, match="n_pick"):
            rs.design_references(eps, rho, bad, sigmas=0.1)
    a = rs.design_references(eps, rho, 3, sigmas=0.1)
    b = rs.design_references(eps, rho, 3.0, sigmas=0.1)
    c = rs.design_references(eps, rho, np.int64(3), sigmas=0.1)
    assert a["indices"] == b["indices"] == c["indices"]


# ----------------------------------------------------------------------
# Added: correlated shift errors between modes


def test_two_correlated_modes_equal_the_closed_form():
    """With two modes the estimate is K^-1 s whatever the weights, and its
    covariance (K^T S^-1 K)^-1 equals K^-1 S K^-T (the two-mode closed
    form, computed here and by SeparationModel)."""
    K = K3[:2]
    rng = np.random.default_rng(2)
    d1, d2 = rng.normal(size=(2, 6, 7))
    s1 = rng.uniform(0.05, 0.3, (6, 7))
    s2 = rng.uniform(0.05, 0.3, (6, 7))
    r = rng.uniform(-0.9, 0.9, (6, 7))
    coeffs = rs.ModeCoefficients("a", "b", *K.ravel(), reference="test")
    ref = rs.SeparationModel(coeffs).invert(d1, d2, s1, s2,
                                            shift_correlation=r)
    got = rs.MultiModeModel(K).invert([d1, d2], [s1, s2],
                                      shift_correlation={(0, 1): r})
    np.testing.assert_allclose(got.strain, ref.strain, rtol=1e-12,
                               atol=1e-14)
    np.testing.assert_allclose(got.density, ref.density, rtol=1e-12,
                               atol=1e-14)
    for f in ("strain_sigma", "density_sigma", "correlation"):
        np.testing.assert_allclose(getattr(got, f), getattr(ref, f),
                                   rtol=1e-10)
    Ki = np.linalg.inv(K)
    S = np.array([[s1[2, 3] ** 2, r[2, 3] * s1[2, 3] * s2[2, 3]],
                  [r[2, 3] * s1[2, 3] * s2[2, 3], s2[2, 3] ** 2]])
    C = Ki @ S @ Ki.T
    assert got.strain_sigma[2, 3] == pytest.approx(np.sqrt(C[0, 0]),
                                                   rel=1e-10)
    assert got.density_sigma[2, 3] == pytest.approx(np.sqrt(C[1, 1]),
                                                    rel=1e-10)
    assert got.dof == 0 and got.chi2_map is None


@pytest.mark.parametrize("ncause", [2, 3])
def test_correlated_fit_equals_a_whitened_lstsq(ncause):
    """Per-pixel correlation matrices and sigmas: the estimate, its
    covariance, chi2 and p equal an independent whitened lstsq solve.
    The dict form equals the matrix form."""
    m = 5
    rng = np.random.default_rng(3 + ncause)
    K = np.column_stack([K4[:, 0].tolist() + [-12.0],
                         K4[:, 1].tolist() + [-0.4],
                         T4.tolist() + [-0.004]])[:, :ncause]
    shape = (3, 4)
    P = 12
    R = _random_correlations(m, P, seed=7).reshape(shape + (m, m))
    sig = rng.uniform(0.05, 0.3, (m,) + shape)
    s = rng.normal(0.0, 1.0, (m,) + shape)
    model = (rs.MultiModeModel(K) if ncause == 2
             else rs.ThreeCauseModel(K))
    got = model.invert(list(s), list(sig), shift_correlation=R)
    est = ([got.strain, got.density] if ncause == 2
           else [got.strain, got.density, got.temperature])
    sds = ([got.strain_sigma, got.density_sigma] if ncause == 2
           else [got.strain_sigma, got.density_sigma,
                 got.temperature_sigma])
    for i in range(shape[0]):
        for j in range(shape[1]):
            x, cov, q = _whitened_lstsq(K, s[:, i, j], sig[:, i, j],
                                        R[i, j])
            for k in range(ncause):
                assert est[k][i, j] == pytest.approx(x[k], rel=1e-9,
                                                     abs=1e-12)
                assert sds[k][i, j] == pytest.approx(np.sqrt(cov[k, k]),
                                                     rel=1e-9)
            assert got.chi2_map[i, j] == pytest.approx(q, rel=1e-9)
            assert got.p_value[i, j] == pytest.approx(
                chi2_dist.sf(q, m - ncause), rel=1e-8)
            if ncause == 3:
                np.testing.assert_allclose(got.covariance[i, j], cov,
                                           rtol=1e-9, atol=1e-18)
    # one matrix for the whole map, as an (m, m) array and as a dict
    R0 = 0.5 * (R[0, 0] + R[0, 0].T)        # exactly symmetric
    np.fill_diagonal(R0, 1.0)
    pairs = {(a, b): R0[a, b] for a in range(m) for b in range(a + 1, m)}
    by_matrix = model.invert(list(s), list(sig), shift_correlation=R0)
    by_dict = model.invert(list(s), list(sig), shift_correlation=pairs)
    for f in ("strain", "density", "strain_sigma", "p_value"):
        assert np.array_equal(getattr(by_matrix, f), getattr(by_dict, f))


def test_identity_correlation_equals_the_independent_path():
    """Zero correlations give the results of the independent-error code
    (a different arithmetic path), for the three models that take
    shift_correlation."""
    rng = np.random.default_rng(4)
    s = list(rng.normal(size=(4, 3, 5)))
    sig = list(rng.uniform(0.05, 0.3, (4, 3, 5)))
    mm = rs.MultiModeModel(K4)
    a, b = mm.invert(s, sig), mm.invert(s, sig, shift_correlation=np.eye(4))
    for f in ("strain", "density", "strain_sigma", "density_sigma",
              "correlation", "chi2_map", "p_value"):
        np.testing.assert_allclose(getattr(b, f), getattr(a, f),
                                   rtol=1e-11, atol=1e-14)
    tc = rs.ThreeCauseModel(np.column_stack([K4, T4]))
    a = tc.invert(s, sig)
    b = tc.invert(s, sig, shift_correlation={(0, 3): 0.0})
    for f in ("strain", "temperature", "temperature_sigma", "p_value"):
        np.testing.assert_allclose(getattr(b, f), getattr(a, f),
                                   rtol=1e-9, atol=1e-12)
    eps = np.array([0.0, 0.4, 0.8, 0.0, 0.3, 0.6])
    rho = np.array([0.0, 0.0, 0.2, 0.9, 0.6, 0.3])
    cal = rs.calibrate_lever_arms(
        eps, rho, np.column_stack([eps, rho]) @ K4.T
        + rng.normal(0.0, 0.1, (6, 4)), sigmas=0.1)
    a = rs.multimode_with_calibration(cal, s, sig)
    b = rs.multimode_with_calibration(cal, s, sig,
                                      shift_correlation=np.eye(4))
    for k in ("strain", "density", "strain_sigma", "density_sigma",
              "strain_sigma_calibration", "density_sigma_calibration",
              "p_value"):
        np.testing.assert_allclose(b[k], a[k], rtol=1e-11, atol=1e-14)


def test_correlated_errors_match_monte_carlo():
    """20000 seeded draws of noise with a known correlation matrix (built
    by an explicit Cholesky factor). With the correlation the reported
    error bars match the spread and the chi-square check is calibrated;
    the independent-error estimate has a larger variance (Gauss-Markov),
    equal to its sandwich formula, and misstates both."""
    R = np.array([[1.0, 0.3, 0.6], [0.3, 1.0, -0.2], [0.6, -0.2, 1.0]])
    S = np.diag(SIG3) @ R @ np.diag(SIG3)
    n = 20000
    rng = np.random.default_rng(11)
    noise = np.linalg.cholesky(S) @ rng.standard_normal((3, n))
    mm = rs.MultiModeModel(K3)
    shifts = [v + e for v, e in zip(mm.forward(np.full(n, 0.1),
                                               np.full(n, 0.3)), noise)]
    gls = mm.invert(shifts, sigmas=SIG3, shift_correlation=R)
    ind = mm.invert(shifts, sigmas=SIG3)
    # reported error bars (one design for every pixel) against the spread
    for f in ("strain", "density"):
        spread = np.std(getattr(gls, f), ddof=1)
        assert spread == pytest.approx(getattr(gls, f + "_sigma")[0],
                                       rel=0.03)
    assert abs(np.mean(gls.strain) - 0.1) < 4 * gls.strain_sigma[0] / np.sqrt(n)
    assert abs(np.mean(gls.chi2_map) - 1.0) < 3 * np.sqrt(2.0 / n)
    assert abs(np.mean(gls.p_value < 0.05) - 0.05) < 0.005
    # the independent-error estimator: its true covariance is the
    # sandwich V = A^-1 K^T W S W K A^-1, at least the GLS covariance
    W = np.diag(1.0 / SIG3 ** 2)
    Ai = np.linalg.inv(K3.T @ W @ K3)
    V = Ai @ K3.T @ W @ S @ W @ K3 @ Ai
    G = np.linalg.inv(K3.T @ np.linalg.inv(S) @ K3)
    assert np.min(np.linalg.eigvalsh(V - G)) > -1e-12 * np.max(np.diag(V))
    assert np.std(ind.density, ddof=1) == pytest.approx(np.sqrt(V[1, 1]),
                                                        rel=0.03)
    assert np.sqrt(V[1, 1]) > 1.15 * ind.density_sigma[0]   # understated
    assert np.mean(ind.p_value < 0.05) < 0.02                # check too lax


def test_correlated_masked_pixels_and_infinite_sigma():
    """A NaN in the correlation matrix or the sigmas of one pixel masks
    only that pixel; an infinite sigma drops that mode, as the model of
    the remaining modes with their sub-matrix of correlations does."""
    rng = np.random.default_rng(5)
    s = list(rng.normal(size=(4, 3, 3)))
    sig = [np.full((3, 3), e) for e in SIG4]
    R = np.broadcast_to(_random_correlations(4, 1, seed=9)[0],
                        (3, 3, 4, 4)).copy()
    mm = rs.MultiModeModel(K4)
    base = mm.invert(s, sig, shift_correlation=R)
    Rn = R.copy()
    Rn[0, 1, 0, 2] = Rn[0, 1, 2, 0] = np.nan
    sn = [v.copy() for v in sig]
    sn[1][2, 2] = np.nan
    got = mm.invert(s, sn, shift_correlation=Rn)
    masked = np.zeros((3, 3), bool)
    masked[0, 1] = masked[2, 2] = True
    assert np.all(np.isnan(got.strain[masked]))
    assert np.all(np.isnan(got.p_value[masked]))
    for f in ("strain", "density_sigma", "p_value"):
        np.testing.assert_allclose(getattr(got, f)[~masked],
                                   getattr(base, f)[~masked], rtol=1e-12)
    si = [v.copy() for v in sig]
    si[2][1, 1] = np.inf
    got = mm.invert(s, si, shift_correlation=R)
    keep = [0, 1, 3]
    sub = rs.MultiModeModel(K4[keep]).invert(
        [s[k][1, 1] for k in keep], [SIG4[k] for k in keep],
        shift_correlation=R[1, 1][np.ix_(keep, keep)])
    for f in ("strain", "density", "strain_sigma", "chi2_map", "p_value"):
        assert getattr(got, f)[1, 1] == pytest.approx(
            float(getattr(sub, f)), rel=1e-10)
    np.testing.assert_allclose(got.strain[~np.eye(3, dtype=bool)],
                               base.strain[~np.eye(3, dtype=bool)],
                               rtol=1e-12)


def test_multimode_with_calibration_correlated_derivative():
    """The calibration part with correlated shift errors equals the one
    built from a central finite-difference Jacobian of the MultiModeModel
    re-solve with perturbed lever arms, at pixels with a non-zero
    residual; with two modes it equals separation_with_calibration."""
    rng = np.random.default_rng(6)
    eps = np.array([0.0, 0.4, 0.8, 0.0, 0.3, 0.6, 0.2])
    rho = np.array([0.0, 0.0, 0.2, 0.9, 0.6, 0.3, 1.1])
    cal = rs.calibrate_lever_arms(
        eps, rho, np.column_stack([eps, rho]) @ K4.T
        + rng.normal(0.0, 0.1, (7, 4)), sigmas=0.1)
    s = list(rng.normal(-2.0, 1.0, (4, 2, 3)))
    sig = list(rng.uniform(0.05, 0.3, (4, 2, 3)))
    R = _random_correlations(4, 6, seed=12).reshape(2, 3, 4, 4)
    out = rs.multimode_with_calibration(cal, s, sig, shift_correlation=R)
    h = 1e-6
    var = np.zeros((2, 2, 3))
    for m in range(4):
        g = []
        for j in range(2):
            Kp, Km = cal.K.copy(), cal.K.copy()
            Kp[m, j] += h
            Km[m, j] -= h
            a = rs.MultiModeModel(Kp).invert(s, sig, shift_correlation=R)
            b = rs.MultiModeModel(Km).invert(s, sig, shift_correlation=R)
            g.append(np.stack([(a.strain - b.strain) / (2 * h),
                               (a.density - b.density) / (2 * h)]))
        for j in range(2):
            for k in range(2):
                var += g[j] * cal.cov[m, j, k] * g[k]
    np.testing.assert_allclose(out["strain_sigma_calibration"],
                               np.sqrt(var[0]), rtol=1e-6)
    np.testing.assert_allclose(out["density_sigma_calibration"],
                               np.sqrt(var[1]), rtol=1e-6)
    base = rs.MultiModeModel(cal.K).invert(s, sig, shift_correlation=R)
    np.testing.assert_allclose(out["strain_sigma_shifts"],
                               base.strain_sigma, rtol=1e-14)
    assert np.all(out["chi2_map"] > 1.0)        # residuals are non-zero
    # two modes: the separation tool with the same correlation
    cal2 = rs.calibrate_lever_arms(
        eps, rho, np.column_stack([eps, rho]) @ K4[:2].T
        + rng.normal(0.0, 0.1, (7, 2)), sigmas=0.1)
    r = rng.uniform(-0.8, 0.8, (2, 3))
    a = rs.separation_with_calibration(cal2, s[0], s[1], sig[0], sig[1],
                                       shift_correlation=r)
    b = rs.multimode_with_calibration(cal2, s[:2], sig[:2],
                                      shift_correlation={(0, 1): r})
    for k in ("strain", "density", "strain_sigma", "density_sigma",
              "strain_sigma_calibration", "density_sigma_calibration"):
        np.testing.assert_allclose(b[k], a[k], rtol=1e-10, atol=1e-13)


def test_correlation_refusals():
    mm = rs.MultiModeModel(K3)
    s = mm.forward(0.1, 0.2)
    good = np.eye(3)
    with pytest.raises(ValueError, match="needs sigmas"):
        mm.invert(s, shift_correlation=good)
    bad = good.copy()
    bad[0, 1] = 0.5                                   # not symmetric
    with pytest.raises(ValueError, match="symmetric"):
        mm.invert(s, SIG3, shift_correlation=bad)
    bad = good.copy()
    bad[1, 1] = 0.9
    with pytest.raises(ValueError, match="diagonal"):
        mm.invert(s, SIG3, shift_correlation=bad)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        mm.invert(s, SIG3, shift_correlation={(0, 1): 1.2})
    with pytest.raises(ValueError, match="infinite"):
        mm.invert(s, SIG3, shift_correlation={(0, 1): np.inf})
    with pytest.raises(ValueError, match="positive definite"):
        mm.invert(s, SIG3, shift_correlation={(0, 1): 1.0})
    with pytest.raises(ValueError, match="positive definite"):
        mm.invert(s, SIG3, shift_correlation={(0, 1): 0.9, (0, 2): 0.9,
                                              (1, 2): -0.9})
    with pytest.raises(ValueError, match="positive definite"):
        rs.MultiModeModel(K3[:2]).invert(s[:2], SIG3[:2],
                                         shift_correlation={(0, 1): -1.0})
    for key in ((0, 0), (0, 3), (0.5, 1), (True, 1), "ab", 1):
        with pytest.raises(ValueError, match="key"):
            mm.invert(s, SIG3, shift_correlation={key: 0.1})
    with pytest.raises(ValueError, match="twice"):
        mm.invert(s, SIG3, shift_correlation={(0, 1): 0.1, (1, 0): 0.1})
    with pytest.raises(ValueError, match="must be an"):
        mm.invert(s, SIG3, shift_correlation=np.eye(2))
    with pytest.raises(ValueError, match="broadcast"):
        mm.invert(s, SIG3, shift_correlation={(0, 1): np.zeros(4)})
    tc = rs.ThreeCauseModel(np.column_stack([K4, T4]))
    with pytest.raises(ValueError, match="needs sigmas"):
        tc.invert(tc.forward(0.1, 0.2, 5.0), shift_correlation=np.eye(4))
    with pytest.raises(ValueError, match="positive definite"):
        tc.invert(tc.forward(0.1, 0.2, 5.0), SIG4,
                  shift_correlation={(1, 2): 1.0})


# ----------------------------------------------------------------------
# Added: shot-noise error bars for the Voigt fit

XV = np.linspace(380.0, 430.0, 201)


def _fd_voigt_jacobian(p, h=1e-6):
    """Central finite differences of `voigt` at p, step h * max(1, |p|)."""
    J = np.empty((XV.size, 5))
    for k in range(5):
        dp = np.zeros(5)
        dp[k] = h * max(1.0, abs(p[k]))
        J[:, k] = (rs.voigt(XV, *(p + dp)) - rs.voigt(XV, *(p - dp))) \
            / (2.0 * dp[k])
    return J


def test_voigt_varying_noise_is_the_sandwich_of_an_independent_jacobian():
    rng = np.random.default_rng(8)
    clean = rs.voigt(XV, 404.0, 2.0, 2.0, 400.0, 30.0)
    y = rng.poisson(clean).astype(float)
    a = rs.fit_voigt(XV, y)
    b = rs.fit_voigt(XV, y, noise="varying")
    p = np.array([b.center, b.sigma, b.gamma, b.amplitude, b.offset])
    J = _fd_voigt_jacobian(p)
    r = y - rs.voigt(XV, *p)
    B = np.linalg.inv(J.T @ J)
    n = XV.size
    C = n / (n - 5) * B @ (J.T * r ** 2) @ J @ B
    got = [b.center_sigma, b.sigma_sigma, b.gamma_sigma, b.amplitude_sigma,
           b.offset_sigma]
    np.testing.assert_allclose(got, np.sqrt(np.diag(C)), rtol=1e-5)
    # the fitted values do not depend on the setting (same computation)
    assert (a.center, a.sigma, a.gamma) == (b.center, b.sigma, b.gamma)
    with pytest.raises(ValueError, match="noise"):
        rs.fit_voigt(XV, y, noise="poisson")


def test_voigt_varying_noise_error_bars_match_shot_noise_scatter():
    """600 seeded shot-noise (Poisson) spectra of a Voigt peak 400 counts
    high on a 30-count baseline: the spread of the fitted center is 1.6
    times the default error bar, and within 2 % of the "varying" one."""
    clean = rs.voigt(XV, 404.0, 2.0, 2.0, 400.0, 30.0)
    rng = np.random.default_rng(3)
    c, s_const, s_vary = [], [], []
    for y in rng.poisson(clean, (600, XV.size)).astype(float):
        a = rs.fit_voigt(XV, y)
        b = rs.fit_voigt(XV, y, noise="varying")
        assert a.converged and a.center == b.center
        c.append(a.center)
        s_const.append(a.center_sigma)
        s_vary.append(b.center_sigma)
    spread = np.std(c, ddof=1)
    r_const = spread / np.sqrt(np.mean(np.square(s_const)))
    r_vary = spread / np.sqrt(np.mean(np.square(s_vary)))
    assert 1.5 < r_const < 1.9            # the default underestimate
    assert abs(r_vary - 1.0) < 0.10
