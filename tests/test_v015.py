"""v0.15 anchors.

Fixed (each test fails on 0.14.0):
* `bayesian_map_inversion` gave unit-dependent maps: with the density in
  cm^-2 its sparse solve lost about four digits (a relative 6e-4 in the
  density map against the same problem in 1e13 cm^-2);
* `MultiModeModel`, `ThreeCauseModel` and `bayesian_map_inversion`
  refused a well-separating lever-arm matrix as "rank < 2" when the
  density was in m^-2 (the SI unit), and `compare_mode_sets` silently
  left out every subset.

Performance: the exact posterior sigmas of `bayesian_map_inversion` come
from a sparse LU factorization instead of a dense inverse; the dense
inverse is kept here as the reference.

Added: `region_profile` (means in bins of a coordinate, with the joint
covariance of all bins), `RegionProfile.combination`,
`integrated_excess` and `contrast`. Each part of the covariance is
checked against an independent calculation (per-pixel inversions,
finite differences of a re-solve with perturbed lever arms, a
pixel-level linear form) and the totals against seeded Monte Carlo
runs that repeat the calibration, the map noise and the reference
error.

Tutorial: examples/tutorial_edge_profile.py prints the numbers pasted
in the README.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import ramansep as rs
from ramansep.bayesian import _inverse_diagonal

EPS = np.finfo(float).eps
K3 = np.array([[-5.1, -2.2], [-20.9, 0.0], [-3.0, -1.0]])
SIG3 = np.array([0.1, 0.2, 0.15])
K_MOS2 = np.array(rs.mos2_a1_2la().matrix())


# ----------------------------------------------------------------------
# Fixed: unit dependence of the Bayesian map inversion


def _laplacian(h, w):
    """Graph Laplacian of the 4-neighbour h x w lattice, built here
    independently of the package (dense)."""
    n = h * w
    L = np.zeros((n, n))
    for i in range(h):
        for j in range(w):
            p = i * w + j
            for di, dj in ((0, 1), (1, 0)):
                if i + di < h and j + dj < w:
                    q = (i + di) * w + j + dj
                    L[p, p] += 1.0
                    L[q, q] += 1.0
                    L[p, q] -= 1.0
                    L[q, p] -= 1.0
    return L


def _dense_precision(K, sig, h, w, lam_s, lam_n):
    """The posterior precision of the smoothness-prior problem, dense,
    unknowns ordered (all strains, all densities); scalar sigmas."""
    n = h * w
    A2 = K.T @ np.diag(1.0 / sig ** 2) @ K
    L = _laplacian(h, w)
    eye = np.eye(n)
    return np.block([[A2[0, 0] * eye + lam_s * L, A2[0, 1] * eye],
                     [A2[1, 0] * eye, A2[1, 1] * eye + lam_n * L]])


def _shift_map(seed=0, h=6, w=6):
    rng = np.random.default_rng(seed)
    truth = np.stack([0.2 + 0.01 * rng.standard_normal((h, w)),
                      0.5 + 0.01 * rng.standard_normal((h, w))])
    return np.einsum("mk,khw->mhw", K3, truth) + rng.normal(0, 0.1,
                                                             (3, h, w))


def test_bayesian_map_does_not_depend_on_the_density_unit():
    """With the density in cm^-2 (scale 1e13) the maps of 0.14.0 differed
    from the dense solve by a relative 6.1e-4 (density) and 1.4e-4
    (strain); with the density in m^-2 (scale 1e17) it refused K as
    "rank < 2"; with scale 1e-6 it was off by 8e-13. The tolerance is set
    from the conditioning of the well-scaled problem: a backward-stable
    solve is accurate to about eps * cond(A) relative (cond(A) = 34
    here), and the test allows 100 times that, 7.6e-13."""
    shifts = _shift_map()
    h, w = shifts.shape[1:]
    A = _dense_precision(K3, SIG3, h, w, 1.0, 1.0)
    cond = np.linalg.cond(A)
    tol = 100.0 * EPS * cond
    assert tol < 1e-8
    b = np.concatenate(K3.T @ np.diag(SIG3 ** -2) @ shifts.reshape(3, -1))
    x = np.linalg.solve(A, b)                    # independent dense solve
    s_ref, n_ref = x[:h * w].reshape(h, w), x[h * w:].reshape(h, w)
    sd = np.sqrt(np.diag(np.linalg.inv(A)))
    for scale in (1.0, 1e13, 1e17, 1e-6):        # 1e13 cm^-2, cm^-2, m^-2
        r = rs.bayesian_map_inversion(K3 / [1.0, scale], shifts, SIG3, 1.0,
                                      1.0 / scale ** 2, posterior_sigma=True)
        np.testing.assert_allclose(r.strain, s_ref, rtol=tol, atol=0)
        np.testing.assert_allclose(r.density / scale, n_ref, rtol=tol, atol=0)
        np.testing.assert_allclose(r.strain_sigma.ravel(), sd[:h * w],
                                   rtol=tol, atol=0)
        np.testing.assert_allclose(r.density_sigma.ravel() / scale,
                                   sd[h * w:], rtol=tol, atol=0)
        assert r.lam_density == 1.0 / scale ** 2   # reported as given
    # the per-pixel-sigma path and the fill_masked path as well
    sig_map = np.broadcast_to(SIG3[:, None, None], shifts.shape).copy()
    holed = shifts.copy()
    holed[:, 2, 3] = np.nan
    for scale in (1e13, 1e17):
        for sh, kw in ((shifts, {}), (holed, dict(fill_masked=True))):
            a = rs.bayesian_map_inversion(K3, sh, sig_map, 1.0, 1.0, **kw)
            c = rs.bayesian_map_inversion(K3 / [1.0, scale], sh, sig_map,
                                          1.0, 1.0 / scale ** 2, **kw)
            np.testing.assert_allclose(c.strain, a.strain, rtol=tol, atol=0)
            np.testing.assert_allclose(c.density / scale, a.density,
                                       rtol=tol, atol=0)


def test_bayesian_unchanged_path_for_well_scaled_matrices():
    """Columns within 2^10 of each other in scale take the 0.14 code
    path; its result equals the independent dense solve (tolerance as
    above). The lever arms of README example 5 and of the MoS2 sets are
    such matrices."""
    shifts = _shift_map(seed=3)
    h, w = shifts.shape[1:]
    for K in (K3, K3 * [1.0, 900.0], K3 * [1.0, 1.0 / 900.0]):
        A = _dense_precision(K, SIG3, h, w, 2.0, 3.0)
        tol = 100.0 * EPS * np.linalg.cond(A)
        b = np.concatenate(K.T @ np.diag(SIG3 ** -2) @ shifts.reshape(3, -1))
        x = np.linalg.solve(A, b)
        r = rs.bayesian_map_inversion(K, shifts, SIG3, 2.0, 3.0)
        np.testing.assert_allclose(r.strain.ravel(), x[:h * w], rtol=tol,
                                   atol=0)
        np.testing.assert_allclose(r.density.ravel(), x[h * w:], rtol=tol,
                                   atol=0)


# ----------------------------------------------------------------------
# Fixed: unit-dependent rank tests


def test_rank_tests_do_not_depend_on_the_units():
    """With the density in m^-2 (lever arms near 1e-17 cm^-1 per m^-2),
    0.14.0 refused these matrices as "rank < 2" / "rank < 3" and
    compare_mode_sets returned []. The results must be those in
    1e13 cm^-2, rescaled. Tolerances: the 2 x 2 normal equations are
    solved in closed form; the normalized K has condition number below
    10, so rounding stays near 1e-15; 1e-10 leaves a wide margin."""
    rng = np.random.default_rng(2)
    truth = [rng.normal(0.1, 0.05, (4, 5)), rng.normal(0.5, 0.1, (4, 5))]
    shifts = [K3[m, 0] * truth[0] + K3[m, 1] * truth[1]
              + rng.normal(0, 0.1, (4, 5)) for m in range(3)]
    a = rs.MultiModeModel(K3).invert(shifts, list(SIG3))
    for scale in (1e17, 1e15):
        b = rs.MultiModeModel(K3 / [1.0, scale]).invert(shifts, list(SIG3))
        np.testing.assert_allclose(b.strain, a.strain, rtol=1e-10, atol=0)
        np.testing.assert_allclose(b.density / scale, a.density, rtol=1e-10,
                                   atol=0)
        np.testing.assert_allclose(b.density_sigma / scale, a.density_sigma,
                                   rtol=1e-10, atol=0)
        np.testing.assert_allclose(b.chi2_map, a.chi2_map, rtol=1e-9, atol=0)
        # two modes as well
        rs.MultiModeModel(K3[:2] / [1.0, scale])
        # compare_mode_sets: the same three subsets, the same variances
        ca = rs.compare_mode_sets(K3, SIG3)
        cb = rs.compare_mode_sets(K3 / [1.0, scale], SIG3)
        assert len(cb) == 3                             # [] in 0.14.0
        va = {d["indices"]: d for d in ca}
        for d in cb:
            ref = va[d["indices"]]
            assert d["var_strain"] == pytest.approx(ref["var_strain"],
                                                    rel=1e-10)
            assert d["var_density"] / scale ** 2 == pytest.approx(
                ref["var_density"], rel=1e-10)
        # ThreeCauseModel with a temperature column (illustrative values)
        KT = np.column_stack([K3, [-0.013, -0.02, 0.004]])
        KT = np.vstack([KT, [-1.0, -0.5, -0.01]])
        sh4 = shifts + [rng.normal(0, 0.1, (4, 5))]
        ta = rs.ThreeCauseModel(KT).invert(sh4, [0.1] * 4)
        tb = rs.ThreeCauseModel(KT / [1.0, scale, 1.0]).invert(sh4,
                                                              [0.1] * 4)
        # 3 x 3 systems with condition number 3.1e6 in mixed units:
        # rounding up to about eps * 3.1e6 = 7e-10, hence 1e-8
        np.testing.assert_allclose(tb.temperature, ta.temperature, rtol=1e-8)
        np.testing.assert_allclose(tb.density / scale, ta.density, rtol=1e-8)
        # and the Bayesian inversion accepts it
        rs.bayesian_map_inversion(K3 / [1.0, scale], np.stack(shifts), SIG3,
                                  1.0, 1.0 / scale ** 2)
    # a matrix that really is rank-deficient is refused in any unit
    bad = np.column_stack([K3[:, 0], 3.0 * K3[:, 0]])
    for scale in (1.0, 1e17):
        Kb = bad / [1.0, scale]
        with pytest.raises(ValueError, match="rank < 2"):
            rs.MultiModeModel(Kb)
        with pytest.raises(ValueError, match="rank < 2"):
            rs.bayesian_map_inversion(Kb, np.zeros((3, 2, 2)), SIG3, 1.0)
        assert rs.compare_mode_sets(Kb, SIG3) == []
        with pytest.raises(ValueError, match="rank < 3"):
            rs.ThreeCauseModel(np.column_stack([Kb, [1.0, 2.0, 0.5]]))


# ----------------------------------------------------------------------
# Performance: posterior sigmas from a sparse LU factorization


def test_inverse_diagonal_equals_the_dense_inverse():
    """The 0.14 dense inverse, kept as the reference. Both are
    backward-stable computations of the same diagonal; they may differ
    by about eps * cond(A) relative, and the tolerance is 100 times
    that."""
    rng = np.random.default_rng(1)
    from scipy.sparse import csc_matrix
    for n, block in ((50, 7), (61, 256), (40, 1)):
        B = rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.1)
        A = B @ B.T + n * np.eye(n)
        ref = np.diag(np.linalg.inv(A))
        got = _inverse_diagonal(csc_matrix(A), block=block)
        tol = 100.0 * EPS * np.linalg.cond(A)
        np.testing.assert_allclose(got, ref, rtol=tol, atol=0)
    # through the public function, with per-pixel sigmas and a hole
    shifts = _shift_map(seed=4, h=5, w=7)
    sig_map = np.broadcast_to(SIG3[:, None, None], shifts.shape).copy()
    sig_map[1, 2, 2] = 0.4
    shifts[:, 4, 6] = np.nan
    r = rs.bayesian_map_inversion(K3, shifts, sig_map, 0.5, 2.0,
                                  posterior_sigma=True, fill_masked=True)
    # the dense precision, built here: zero data weight at the hole
    h, w = 5, 7
    wts = np.where(np.isfinite(shifts), sig_map ** -2.0, 0.0).reshape(3, -1)
    a11 = (K3[:, 0] ** 2) @ wts
    a12 = (K3[:, 0] * K3[:, 1]) @ wts
    a22 = (K3[:, 1] ** 2) @ wts
    L = _laplacian(h, w)
    A = np.block([[np.diag(a11) + 0.5 * L, np.diag(a12)],
                  [np.diag(a12), np.diag(a22) + 2.0 * L]])
    sd = np.sqrt(np.diag(np.linalg.inv(A)))
    tol = 100.0 * EPS * np.linalg.cond(A)
    np.testing.assert_allclose(r.strain_sigma.ravel(), sd[:h * w], rtol=tol)
    np.testing.assert_allclose(r.density_sigma.ravel(), sd[h * w:], rtol=tol)
    with pytest.raises(ValueError, match="max_dense"):
        rs.bayesian_map_inversion(K3, shifts, sig_map, 0.5, 2.0,
                                  posterior_sigma=True, fill_masked=True,
                                  max_dense=69)


def test_lattice_posterior_sigmas_equal_the_dense_inverse():
    """One sigma per mode and no missing values: the posterior sigmas
    come from the eigen-decomposition of the lattice Laplacian (a
    Kronecker sum). The reference is the 0.14 dense inverse of the
    precision, built independently here; tolerance 100 eps cond(A) as
    above. Shapes include single rows and columns, unequal smoothing
    weights, lam = 0 and a nearly collinear lever-arm matrix."""
    rng = np.random.default_rng(5)
    Kc = np.array([[-5.1, -2.2], [-5.0, -2.3], [-4.9, -2.1]])
    cases = [(K3, 1, 9, 1.0, 1.0), (K3, 7, 1, 0.3, 4.0), (K3, 4, 6, 0.0, 0.0),
             (K3, 5, 5, 2.0, 0.0), (Kc, 6, 4, 0.5, 0.05)]
    for K, h, w, ls, ln in cases:
        shifts = rng.normal(0.0, 0.1, (3, h, w))
        r = rs.bayesian_map_inversion(K, shifts, SIG3, ls, ln,
                                      posterior_sigma=True)
        A = _dense_precision(K, SIG3, h, w, ls, ln)
        sd = np.sqrt(np.diag(np.linalg.inv(A)))
        tol = 100.0 * EPS * np.linalg.cond(A)
        np.testing.assert_allclose(r.strain_sigma.ravel(), sd[:h * w],
                                   rtol=tol, atol=0)
        np.testing.assert_allclose(r.density_sigma.ravel(), sd[h * w:],
                                   rtol=tol, atol=0)


# ----------------------------------------------------------------------
# Added: region_profile


def _ribbon(seed=0, H=6, W=12, noise=0.05, nan_at=(2, 3)):
    """Shift maps of a ribbon with an edge profile, the distance of each
    pixel from the nearer edge (in pixels), and the shift errors."""
    rng = np.random.default_rng(seed)
    col = np.arange(W)
    d = np.broadcast_to((np.minimum(col, W - 1 - col) + 0.5), (H, W))
    strain = 0.1 - 0.03 * np.exp(-d / 2.0)
    dens = 0.03 + 0.4 * np.exp(-d / 1.5)
    dw = np.einsum("mk,khw->mhw", K_MOS2, np.stack([strain, dens]))
    dw = dw + rng.normal(0.0, noise, dw.shape)
    s1 = np.full((H, W), noise)
    s2 = np.full((H, W), 1.2 * noise)
    if nan_at is not None:
        s1[nan_at] = np.nan
    return dw, s1, s2, np.array(d), strain, dens


def _calibration(seed=0, sig=0.1):
    rng = np.random.default_rng(seed)
    st = np.array([0.0, 0.2, 0.4, 0.0, 0.2, 0.4])
    dn = np.array([0.0, 0.0, 0.0, 0.6, 0.6, 1.2])
    y = np.column_stack([st, dn]) @ K_MOS2.T + rng.normal(0, sig, (6, 2))
    return rs.calibrate_lever_arms(st, dn, y, sigmas=sig), (st, dn)


BINS = [0.0, 1.0, 2.0, 3.0, 6.0]
REF_SIGMA = (0.2, 0.3)


def _bin_masks(d, bins):
    R = len(bins) - 1
    return [(d >= bins[k]) & ((d < bins[k + 1]) if k < R - 1
                              else (d <= bins[k + 1])) for k in range(R)]


def test_profile_bins_are_the_region_means():
    cal, _ = _calibration()
    dw, s1, s2, d, _, _ = _ribbon()
    rho = np.full(d.shape, 0.3)
    p = rs.region_profile(cal, dw[0], dw[1], s1, s2, d, BINS,
                          shift_correlation=rho, ref_sigma=REF_SIGMA)
    inv = rs.SeparationModel(cal.coefficients("test")).invert(
        dw[0], dw[1], s1, s2, shift_correlation=rho)
    for k, m in enumerate(_bin_masks(d, BINS)):
        use = m & np.isfinite(s1)
        # independent: the mean of the per-pixel inversion over the bin
        assert p.strain[k] == pytest.approx(inv.strain[use].mean(), abs=1e-14)
        assert p.density[k] == pytest.approx(inv.density[use].mean(),
                                             abs=1e-14)
        assert p.position[k] == pytest.approx(d[use].mean(), rel=1e-14)
        assert p.n_pixels[k] == use.sum()
        assert p.n_excluded[k] == (m & ~np.isfinite(s1)).sum()
        # shift part: the per-pixel covariances summed, over N^2
        N = use.sum()
        cs = (inv.correlation[use] * inv.strain_sigma[use]
              * inv.density_sigma[use]).sum() / N ** 2
        blk = p.covariance_shifts[k, :, k, :]
        assert blk[0, 0] == pytest.approx(
            (inv.strain_sigma[use] ** 2).sum() / N ** 2, rel=1e-12)
        assert blk[1, 1] == pytest.approx(
            (inv.density_sigma[use] ** 2).sum() / N ** 2, rel=1e-12)
        assert blk[0, 1] == pytest.approx(cs, rel=1e-12)
        # and each bin's error parts and homogeneity check are those of
        # region_mean on that bin
        r = rs.region_mean(cal, dw[0], dw[1], s1, s2, m,
                           shift_correlation=rho, ref_sigma=REF_SIGMA)
        for a, b in ((p.strain_sigma[k], r.strain_sigma),
                     (p.density_sigma[k], r.density_sigma),
                     (p.chi2[k], r.chi2), (p.p_value[k], r.p_value)):
            assert a == pytest.approx(b, rel=1e-12)
        assert p.dof[k] == r.dof
    # bins of other pixels are uncorrelated in their shift noise
    off = p.covariance_shifts.copy()
    for k in range(len(BINS) - 1):
        off[k, :, k, :] = 0.0
    assert np.all(off == 0.0)


def test_profile_cross_bin_covariance_by_finite_differences():
    cal, _ = _calibration()
    dw, s1, s2, d, _, _ = _ribbon()
    p = rs.region_profile(cal, dw[0], dw[1], s1, s2, d, BINS,
                          ref_sigma=REF_SIGMA)
    masks = [m & np.isfinite(s1) for m in _bin_masks(d, BINS)]
    ybar = np.array([[dw[0][m].mean(), dw[1][m].mean()] for m in masks])

    def bins_at(K):                           # independent re-solve
        return np.linalg.solve(K, ybar.T).T  # (R, 2)

    R = len(masks)
    J = np.zeros((R, 2, 2, 2))               # d x_k / d K[m, j]
    for m in range(2):
        for j in range(2):
            h = 1e-6 * abs(cal.K[m, j]) + 1e-9
            Kp, Km = cal.K.copy(), cal.K.copy()
            Kp[m, j] += h
            Km[m, j] -= h
            J[:, :, m, j] = (bins_at(Kp) - bins_at(Km)) / (2 * h)
    C = np.einsum("iamj,mjl,cbml->iacb", J, cal.cov, J)
    np.testing.assert_allclose(p.covariance_calibration, C, rtol=1e-6,
                               atol=1e-6 * np.abs(C).max())
    # reference: the same block for every pair, K^-1 U K^-T; and an
    # offset added to all dw1 moves every bin by K^-1 (offset, 0)
    Ki = np.linalg.inv(cal.K)
    B = Ki @ np.diag(np.square(REF_SIGMA)) @ Ki.T
    for i in range(R):
        for j in range(R):
            np.testing.assert_allclose(p.covariance_reference[i, :, j, :], B,
                                       rtol=1e-12)
    q = rs.region_profile(cal, dw[0] + 0.01, dw[1], s1, s2, d, BINS)
    np.testing.assert_allclose(q.strain - p.strain, 0.01 * Ki[0, 0],
                               rtol=1e-8)
    np.testing.assert_allclose(q.density - p.density, 0.01 * Ki[1, 0],
                               rtol=1e-8)
    # the total is the sum of the parts; the sigmas are its diagonal
    tot = p.covariance_shifts + p.covariance_calibration \
        + p.covariance_reference
    np.testing.assert_allclose(p.covariance, tot, rtol=1e-15)
    np.testing.assert_allclose(p.density_sigma ** 2,
                               np.einsum("iaia->ia", tot)[:, 1], rtol=1e-12)


def test_profile_combinations_excess_and_contrast():
    cal, _ = _calibration()
    dw, s1, s2, d, _, _ = _ribbon()
    rho = np.full(d.shape, -0.2)
    p = rs.region_profile(cal, dw[0], dw[1], s1, s2, d, BINS,
                          shift_correlation=rho, ref_sigma=REF_SIGMA)
    R = len(BINS) - 1
    # a general combination equals c^T C c from the stored arrays
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=R), rng.normal(size=R)
    est = p.combination(a, b)
    c = np.column_stack([a, b])
    for part, sig in ((p.covariance, est.sigma),
                      (p.covariance_shifts, est.sigma_shifts),
                      (p.covariance_calibration, est.sigma_calibration),
                      (p.covariance_reference, est.sigma_reference)):
        var = np.einsum("ia,iajb,jb->", c, part, c)
        assert sig == pytest.approx(np.sqrt(var), rel=1e-10)
    assert est.value == pytest.approx(a @ p.strain + b @ p.density,
                                      rel=1e-14)
    # integrated excess: the value by hand, the reference part cancels
    widths = np.diff(BINS)
    e = p.integrated_excess(bulk=R - 1)
    hand = np.sum(widths[:-1] * (p.density[:-1] - p.density[-1]))
    assert e.value == pytest.approx(hand, rel=1e-13)
    assert e.sigma_reference <= 1e-12 * np.sqrt(p.covariance_reference[
        0, 1, 0, 1])
    assert p.integrated_excess(bulk=-1) == e
    # its shift part from the pixel-level linear form: the excess is
    # sum_p a_p . y_p with a_p = (w_k / N_k) row 1 of K^-1 for pixels of
    # bin k (and -W / N_bulk for the bulk bin)
    Ki = np.linalg.inv(cal.K)
    masks = [m & np.isfinite(s1) for m in _bin_masks(d, BINS)]
    wk = np.append(widths[:-1], -widths[:-1].sum())
    var = 0.0
    for k, m in enumerate(masks):
        g = wk[k] / m.sum() * Ki[1]
        S = np.zeros((m.sum(), 2, 2))
        S[:, 0, 0], S[:, 1, 1] = s1[m] ** 2, s2[m] ** 2
        S[:, 0, 1] = S[:, 1, 0] = rho[m] * s1[m] * s2[m]
        var += np.einsum("i,pij,j->", g, S, g)
    assert e.sigma_shifts == pytest.approx(np.sqrt(var), rel=1e-10)
    # its calibration part from finite differences of a re-solve
    ybar = np.array([[dw[0][m].mean(), dw[1][m].mean()] for m in masks])

    def excess_at(K):
        n = np.linalg.solve(K, ybar.T)[1]
        return np.sum(widths[:-1] * (n[:-1] - n[-1]))

    var = 0.0
    for m in range(2):
        g = np.empty(2)
        for j in range(2):
            h = 1e-6 * abs(cal.K[m, j]) + 1e-9
            Kp, Km = cal.K.copy(), cal.K.copy()
            Kp[m, j] += h
            Km[m, j] -= h
            g[j] = (excess_at(Kp) - excess_at(Km)) / (2 * h)
        var += g @ cal.cov[m] @ g
    assert e.sigma_calibration == pytest.approx(np.sqrt(var), rel=1e-6)
    # strain excess, custom widths
    es = p.integrated_excess(0, quantity="strain", widths=[1, 2, 3, 4])
    assert es.value == pytest.approx(
        np.sum(np.array([2, 3, 4]) * (p.strain[1:] - p.strain[0])), rel=1e-13)
    # contrast: region_contrast on the two bins' pixels
    for i, j in ((0, 3), (1, 2), (3, 0)):
        con = p.contrast(i, j)
        mk = _bin_masks(d, BINS)
        rc = rs.region_contrast(cal, dw[0], dw[1], s1, s2, mk[i], mk[j],
                                shift_correlation=rho)
        for f in ("strain", "density", "strain_sigma", "density_sigma",
                  "density_sigma_shifts", "density_sigma_calibration",
                  "correlation"):
            assert getattr(con, f) == pytest.approx(getattr(rc, f),
                                                    rel=1e-12, abs=1e-15)
        assert con.density_sigma_reference == 0.0
        assert con.n_pixels == rc.n_pixels


def test_profile_error_budget_matches_monte_carlo():
    """300 seeded trials repeat the calibration measurement and fit, the
    map noise and the reference error. The spread of the integrated
    excess and of a bin mean must match the reported error bars within
    15 % (the standard error of a spread from 300 trials is 4 %; the
    calibration part is first order), the excess must be unbiased within
    4 standard errors, and the correlation of two bins' density errors
    must match the reported one within 0.1."""
    rng = np.random.default_rng(42)
    st = np.array([0.0, 0.2, 0.4, 0.0, 0.2, 0.4])
    dn = np.array([0.0, 0.0, 0.0, 0.6, 0.6, 1.2])
    _, s1, s2, d, strain, dens = _ribbon(nan_at=None)
    clean = np.einsum("mk,khw->mhw", K_MOS2, np.stack([strain, dens]))
    masks = _bin_masks(d, BINS)
    true_bins = np.array([dens[m].mean() for m in masks])
    widths = np.diff(BINS)[:-1]
    true_excess = np.sum(widths * (true_bins[:-1] - true_bins[-1]))
    exc, sig, b0, b3, s0, rep = [], [], [], [], [], None
    for _ in range(300):
        y = np.column_stack([st, dn]) @ K_MOS2.T + rng.normal(0, 0.1, (6, 2))
        cal = rs.calibrate_lever_arms(st, dn, y, sigmas=0.1)
        ref_err = rng.normal(0.0, REF_SIGMA)
        dw = clean + rng.normal(0.0, 1.0, clean.shape) * np.stack([s1, s2])
        dw = dw - ref_err[:, None, None]
        p = rs.region_profile(cal, dw[0], dw[1], s1, s2, d, BINS,
                              ref_sigma=REF_SIGMA)
        e = p.integrated_excess(bulk=-1)
        exc.append(e.value)
        sig.append(e.sigma)
        b0.append(p.density[0])
        b3.append(p.density[3])
        s0.append(p.density_sigma[0])
        rep = p
    exc, sig = np.array(exc), np.array(sig)
    assert abs(np.std(exc, ddof=1) / np.mean(sig) - 1.0) < 0.15
    assert abs(np.mean(exc) - true_excess) < 4 * np.std(exc) / np.sqrt(300)
    assert abs(np.std(b0, ddof=1) / np.mean(s0) - 1.0) < 0.15
    c = rep.covariance
    rho_rep = c[0, 1, 3, 1] / np.sqrt(c[0, 1, 0, 1] * c[3, 1, 3, 1])
    assert abs(np.corrcoef(b0, b3)[0, 1] - rho_rep) < 0.1
    # the excess is far better known than a bin mean times its width
    assert np.mean(sig) < 0.5 * np.sum(widths) * rep.density_sigma[-1]


def test_profile_refusals():
    cal, _ = _calibration()
    dw, s1, s2, d, _, _ = _ribbon()
    args = (cal, dw[0], dw[1], s1, s2)
    for bad in ([1.0], [0.0, 0.0, 1.0], [0.0, np.nan, 2.0], [2.0, 1.0]):
        with pytest.raises(ValueError, match="bins"):
            rs.region_profile(*args, d, bad)
    with pytest.raises(ValueError, match="coordinate"):
        rs.region_profile(*args, d[:, :3], BINS)
    with pytest.raises(ValueError, match="holds no pixel"):
        rs.region_profile(*args, d, [0.0, 1.0, 1.2, 6.0])
    with pytest.raises(ValueError, match="mask"):
        rs.region_profile(*args, d, BINS, mask=np.ones(d.shape))
    with pytest.raises(ValueError, match="ref_sigma"):
        rs.region_profile(*args, d, BINS, ref_sigma=(0.1, -0.1))
    # a bin whose pixels are all masked by the fit
    s1b = s1.copy()
    s1b[d < 1.0] = np.nan
    with pytest.raises(ValueError, match="no pixel with finite"):
        rs.region_profile(cal, dw[0], dw[1], s1b, s2, d, BINS)
    # NaN coordinates and pixels outside the bins are simply not used
    dn = d.copy()
    dn[0, 0] = np.nan
    p = rs.region_profile(*args, dn, BINS)
    q = rs.region_profile(*args, d, BINS)
    assert p.n_pixels[0] == q.n_pixels[0] - 1
    assert rs.region_profile(*args, d, [0.0, 1.0, 2.0]).n_pixels.sum() \
        == np.sum((d <= 2.0) & np.isfinite(s1))
    p = rs.region_profile(*args, d, BINS)
    for bad in (4, -5, 1.5, True):
        with pytest.raises(ValueError, match="bulk"):
            p.integrated_excess(bad)
    with pytest.raises(ValueError, match="quantity"):
        p.integrated_excess(0, quantity="charge")
    with pytest.raises(ValueError, match="widths"):
        p.integrated_excess(0, widths=[1.0, 2.0])
    with pytest.raises(ValueError, match="weights"):
        p.combination([1.0, np.nan, 0.0, 0.0])
    with pytest.raises(ValueError, match="different bins"):
        p.contrast(1, -3)
    with pytest.raises(ValueError, match="bin indices"):
        p.contrast(0, 7)


# ----------------------------------------------------------------------
# Tutorial


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_edge_profile_tutorial_key_numbers():
    path = os.path.join(ROOT, "examples", "tutorial_edge_profile.py")
    if not os.path.exists(path):          # e.g. tests run from an sdist
        pytest.skip("examples/ is not present")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, path], capture_output=True,
                         text=True, env=env, timeout=120,
                         check=True).stdout
    assert "Map: 12 x 24 pixels, 0 masked" in out
    assert "reference 0.00)" in out
    assert "= 1360 +/- 72 extra electrons per micrometer of edge " \
           "(true: 1440)" in out
    assert "error bar if the bins were treated as independent: 14.13" in out
