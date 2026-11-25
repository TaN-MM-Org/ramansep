"""v0.17 anchors.

Fixed (each test fails on 0.16.0):
* `SeparationModel` refused only a lever-arm matrix whose determinant
  came out exactly zero; a matrix with proportional rows whose
  determinant rounds to 3.9e-17 gave a strain of 3.9e16 with a warning;
* `bayesian_map_inversion(..., posterior_sigma=True)` refused maps above
  `max_dense` unknowns even where the exact sigmas come from the cheap
  lattice formulas.

Added: correlated shift errors in `bayesian_map_inversion`, the evidence
of the smoothness weights (`bayesian_log_evidence`) and their choice by
it (`choose_smoothing`), an infinite sigma as "no information" in
`ThreeCauseModel`, `design_references(..., fit_offset=True)`, and
`decimal=","` in the text-table loaders. Each is checked against an
independent calculation built here: a dense whitened least-squares solve,
the dense covariance form of the Gaussian evidence, an exhaustive search,
the model without the dropped mode, or the same table written with
decimal points.
"""
import itertools
import os

import numpy as np
import pytest

import ramansep as rs
from ramansep.bayesian import _grid_laplacian, _path_laplacian

K3 = np.array([[-2.3, 1.1], [-0.9, -1.7], [0.4, 2.2]])   # README ex. 5
SIG3 = np.array([0.03, 0.05, 0.02])
R3 = np.array([[1.0, 0.4, 0.1],
               [0.4, 1.0, -0.3],
               [0.1, -0.3, 1.0]])


def _random_correlations(m, n, seed):
    """n random, well-conditioned correlation matrices of size m."""
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, m, m + 2))
    C = G @ G.transpose(0, 2, 1)
    d = np.sqrt(np.einsum("pii->pi", C))
    return C / d[:, :, None] / d[:, None, :]


def _smooth_map(H, W, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    strain = 0.002 * xx + 0.001 * np.sin(yy)
    density = 0.01 + 0.001 * yy
    shifts = np.einsum("mj,jhw->mhw", K3, np.stack([strain, density]))
    return shifts + rng.normal(size=shifts.shape) * SIG3[:, None, None]


def _dense_map_solve(K, shifts, sig_full, R_pix, lam_s, lam_n):
    """The map problem solved independently of the package: whiten the
    data of each pixel with the Cholesky factor of its shift covariance
    (measured modes only), append sqrt(lam) times the differences along
    every lattice edge as extra rows, and solve with numpy.linalg.lstsq.
    Returns (strain, density, strain_sigma, density_sigma)."""
    m, H, W = shifts.shape
    P = H * W
    rows, rhs = [], []
    for p in range(P):
        i, j = divmod(p, W)
        meas = [k for k in range(m) if np.isfinite(shifts[k, i, j])
                and np.isfinite(sig_full[k, i, j])]
        if not meas:
            continue
        d = sig_full[meas, i, j]
        S = np.diag(d) @ R_pix[p][np.ix_(meas, meas)] @ np.diag(d)
        Lc = np.linalg.cholesky(S)
        Kp = np.zeros((len(meas), 2 * P))
        Kp[:, p] = K[meas, 0]
        Kp[:, P + p] = K[meas, 1]
        rows.append(np.linalg.solve(Lc, Kp))
        rhs.append(np.linalg.solve(Lc, shifts[meas, i, j]))
    for i in range(H):
        for j in range(W):
            for di, dj in ((0, 1), (1, 0)):
                if i + di < H and j + dj < W:
                    a, b = i * W + j, (i + di) * W + (j + dj)
                    for off, lam in ((0, lam_s), (P, lam_n)):
                        r = np.zeros(2 * P)
                        r[off + a], r[off + b] = np.sqrt(lam), -np.sqrt(lam)
                        rows.append(r[None, :])
                        rhs.append(np.zeros(1))
    J = np.vstack(rows)
    y = np.concatenate(rhs)
    x = np.linalg.lstsq(J, y, rcond=None)[0]
    sd = np.sqrt(np.diag(np.linalg.inv(J.T @ J)))
    return (x[:P].reshape(H, W), x[P:].reshape(H, W),
            sd[:P].reshape(H, W), sd[P:].reshape(H, W))


def _dense_log_evidence(K, shifts, sig_full, R, lam_s, lam_n, delta):
    """log N(s; 0, S + Kb Q^-1 Kb^T) with the proper prior precision
    Q = blockdiag(lam_s L, lam_n L) + delta I: the covariance form of the
    Gaussian evidence, independent of the precision form the package
    uses. Missing entries (NaN) are left out of s and of the covariance."""
    m, H, W = shifts.shape
    P = H * W
    L = _grid_laplacian(H, W).toarray()
    Q = np.zeros((2 * P, 2 * P))
    Q[:P, :P] = lam_s * L
    Q[P:, P:] = lam_n * L
    Q += delta * np.eye(2 * P)
    Kb = np.zeros((m * P, 2 * P))
    S = np.zeros((m * P, m * P))
    for k in range(m):
        Kb[k * P:(k + 1) * P, :P] = K[k, 0] * np.eye(P)
        Kb[k * P:(k + 1) * P, P:] = K[k, 1] * np.eye(P)
    sf = sig_full.reshape(m, P)
    for p in range(P):
        Sp = np.diag(sf[:, p]) @ R @ np.diag(sf[:, p])
        idx = [k * P + p for k in range(m)]
        S[np.ix_(idx, idx)] = Sp
    s = shifts.reshape(-1)
    keep = np.isfinite(s) & np.isfinite(sf.reshape(-1))
    C = (S + Kb @ np.linalg.solve(Q, Kb.T))[np.ix_(keep, keep)]
    sk = s[keep]
    return -0.5 * np.linalg.slogdet(C)[1] - 0.5 * sk @ np.linalg.solve(C, sk)


# ----------------------------------------------------------------------
# Fixed: a lever-arm matrix that is singular up to rounding


def test_separation_model_refuses_a_numerically_singular_matrix():
    """Rows (0.1, 0.3) and (0.7, 2.1) are proportional (factor 7), but the
    determinant rounds to 3.9e-17, not 0. 0.16.0 accepted the matrix with
    a condition-number warning and returned a strain of 3.9e16 for shifts
    of 1 and 2 cm^-1. MultiModeModel refused the same matrix."""
    c = rs.ModeCoefficients("a", "b", 0.1, 0.3, 0.7, 2.1, "test")
    assert np.linalg.det(c.matrix()) != 0.0
    with pytest.raises(ValueError, match="singular"):
        rs.SeparationModel(c)
    with pytest.raises(ValueError, match="rank"):
        rs.MultiModeModel(c.matrix())
    # exactly singular, as before
    with pytest.raises(ValueError, match="singular"):
        rs.SeparationModel(rs.ModeCoefficients("a", "b", 1, 2, 3, 6, "t"))
    # a good matrix in very different units is still accepted, and gives
    # the same result as in the usual units, rescaled
    k = rs.mos2_a1_2la()
    si = rs.ModeCoefficients("A'1", "2LA(M)", k.k1_strain,
                             k.k1_density / 1e17, k.k2_strain,
                             k.k2_density / 1e17, "m^-2")
    with pytest.warns(UserWarning):
        m_si = rs.SeparationModel(si)
    a = rs.SeparationModel(k).invert(-0.5, 0.0)
    b = m_si.invert(-0.5, 0.0)
    assert float(b.density) == pytest.approx(float(a.density) * 1e17,
                                             rel=1e-12)   # measured 1.8e-16


# ----------------------------------------------------------------------
# Added: infinite sigma in ThreeCauseModel


KT = np.array([[-2.1, -0.8, -0.011],
               [-1.0, -2.3, -0.016],
               [-3.2, -0.4, -0.007],
               [-0.6, -1.1, -0.021],
               [-1.5, -0.3, -0.012]])     # illustrative, as README ex. 10


def test_three_cause_infinite_sigma_leaves_the_mode_out():
    """An infinite sigma now means "no information", as in MultiModeModel:
    the result equals that of the model without the mode (0.16.0 returned
    NaN at such a pixel)."""
    m = rs.ThreeCauseModel(KT)
    s = m.forward(0.2, 0.5, 40.0)
    s[4] = s[4] + 0.3                      # mode 5 disagrees
    keep = [0, 1, 2, 4]
    sig = [0.05, 0.05, 0.05, np.inf, 0.05]
    got = m.invert(s, sigmas=sig)
    ref = rs.ThreeCauseModel(KT[keep]).invert([s[k] for k in keep],
                                              sigmas=[0.05] * 4)
    for f in ("strain", "density", "temperature", "temperature_sigma",
              "chi2_map", "p_value"):
        assert float(getattr(got, f)) == pytest.approx(
            float(getattr(ref, f)), rel=1e-10)          # measured 0
    assert got.dof == 2 and ref.dof == 1
    # with correlated errors
    R = np.eye(5)
    R[0, 1] = R[1, 0] = 0.3
    R[3, 4] = R[4, 3] = 0.5                # mode 4 is dropped
    got = m.invert(s, sigmas=sig, shift_correlation=R)
    ref = rs.ThreeCauseModel(KT[keep]).invert(
        [s[k] for k in keep], sigmas=[0.05] * 4,
        shift_correlation=R[np.ix_(keep, keep)])
    for f in ("strain", "density", "temperature", "p_value"):
        assert float(getattr(got, f)) == pytest.approx(
            float(getattr(ref, f)), rel=1e-10)          # measured 0
    # four modes, one infinite: no check left, but the estimate exists
    r4 = rs.ThreeCauseModel(KT[:4]).invert(s[:4],
                                           sigmas=[0.05, np.inf, 0.05, 0.05])
    assert np.isnan(r4.chi2_map) and np.isnan(r4.p_value)
    assert float(r4.temperature) == pytest.approx(40.0, rel=1e-9)  # 1.0e-14
    # a NaN sigma still masks the pixel; fewer than three usable modes
    # is refused
    assert np.isnan(m.invert(s, sigmas=[0.05, np.nan, 0.05, 0.05,
                                        0.05]).strain)
    with pytest.raises(ValueError, match="singular"):
        m.invert(s, sigmas=[0.05, np.inf, np.inf, np.inf, 0.05])
    # on a map, the other pixels do not change
    rng = np.random.default_rng(3)
    smap = [v + rng.normal(0, 0.05, (3, 4)) for v in
            m.forward(np.full((3, 4), 0.2), np.full((3, 4), 0.5),
                      np.full((3, 4), 40.0))]
    sg = [np.full((3, 4), 0.05) for _ in range(5)]
    base = m.invert(smap, sigmas=sg)
    sg[2] = sg[2].copy()
    sg[2][1, 2] = np.inf
    mixed = m.invert(smap, sigmas=sg)
    other = np.ones((3, 4), bool)
    other[1, 2] = False
    for f in ("strain", "temperature", "p_value"):
        np.testing.assert_array_equal(getattr(mixed, f)[other],
                                      getattr(base, f)[other])
    assert np.isfinite(mixed.strain[1, 2])


# ----------------------------------------------------------------------
# Added: correlated shift errors in bayesian_map_inversion


def test_bayesian_correlated_at_lam_zero_equals_multimode():
    """At lam = 0 the map problem splits into pixels, so it must give the
    per-pixel generalized least squares of MultiModeModel, for every form
    of shift_correlation."""
    shifts = _smooth_map(5, 6, 1)
    mm = rs.MultiModeModel(K3)
    Rpix = _random_correlations(3, 30, 2).reshape(5, 6, 3, 3)
    sig_pix = SIG3[:, None, None] * np.random.default_rng(4).uniform(
        0.5, 2.0, (3, 5, 6))
    rho01 = np.random.default_rng(5).uniform(-0.6, 0.6, (5, 6))
    for corr, sig in ((R3, SIG3), (Rpix, sig_pix),
                      ({(0, 1): rho01, (1, 2): 0.2}, sig_pix)):
        b = rs.bayesian_map_inversion(K3, shifts, sig, 0.0,
                                      posterior_sigma=True,
                                      shift_correlation=corr)
        sm = sig if sig.ndim == 3 else [np.full((5, 6), v) for v in sig]
        r = mm.invert(list(shifts), sigmas=list(sm), shift_correlation=corr)
        np.testing.assert_allclose(b.strain, r.strain, rtol=1e-10,
                                   atol=1e-14)          # measured 6.5e-15
        np.testing.assert_allclose(b.density, r.density, rtol=1e-10,
                                   atol=1e-14)
        np.testing.assert_allclose(b.strain_sigma, r.strain_sigma,
                                   rtol=1e-10)          # measured 8.5e-16
        np.testing.assert_allclose(b.density_sigma, r.density_sigma,
                                   rtol=1e-10)


def test_bayesian_correlated_matches_a_dense_whitened_solve():
    """lam > 0, per-pixel sigmas, per-pixel correlation matrices and
    missing values (one mode at one pixel, a whole pixel at another):
    maps and posterior sigmas equal an independent dense least-squares
    solve of the same objective."""
    H, W = 4, 5
    shifts = _smooth_map(H, W, 7)
    sig = SIG3[:, None, None] * np.random.default_rng(8).uniform(
        0.5, 2.0, (3, H, W))
    Rpix = _random_correlations(3, H * W, 9)
    shifts[1, 2, 3] = np.nan
    shifts[:, 0, 1] = np.nan
    b = rs.bayesian_map_inversion(K3, shifts, sig, 300.0, 40.0,
                                  posterior_sigma=True, fill_masked=True,
                                  shift_correlation=Rpix.reshape(H, W, 3, 3))
    ref = _dense_map_solve(K3, shifts, sig, Rpix, 300.0, 40.0)
    for got, want in zip((b.strain, b.density, b.strain_sigma,
                          b.density_sigma), ref):
        np.testing.assert_allclose(got, want, rtol=1e-9,
                                   atol=1e-13)          # measured 1.3e-13
    # a scalar sigma per mode and one matrix: the lattice formulas
    shifts = _smooth_map(H, W, 10)
    b = rs.bayesian_map_inversion(K3, shifts, SIG3, 300.0, 40.0,
                                  posterior_sigma=True,
                                  shift_correlation=R3)
    ref = _dense_map_solve(K3, shifts,
                           np.broadcast_to(SIG3[:, None, None], (3, H, W)),
                           np.broadcast_to(R3, (H * W, 3, 3)), 300.0, 40.0)
    for got, want in zip((b.strain, b.density, b.strain_sigma,
                          b.density_sigma), ref):
        np.testing.assert_allclose(got, want, rtol=1e-9,
                                   atol=1e-13)          # measured 2.6e-14


def test_bayesian_zero_correlation_and_units():
    """The identity correlation gives the independent-error result, on
    both the lattice and the sparse path; with the density column in
    cm^-2 (K / 1e13) the result is the same, rescaled."""
    shifts = _smooth_map(6, 7, 11)
    sig_pix = SIG3[:, None, None] * np.random.default_rng(12).uniform(
        0.5, 2.0, (3, 6, 7))
    for sig in (SIG3, sig_pix):
        a = rs.bayesian_map_inversion(K3, shifts, sig, 500.0,
                                      posterior_sigma=True)
        b = rs.bayesian_map_inversion(K3, shifts, sig, 500.0,
                                      posterior_sigma=True,
                                      shift_correlation=np.eye(3))
        for f in ("strain", "density", "strain_sigma", "density_sigma"):
            np.testing.assert_allclose(getattr(b, f), getattr(a, f),
                                       rtol=1e-12, atol=1e-15)  # 1.9e-14
    a = rs.bayesian_map_inversion(K3, shifts, SIG3, 500.0, 50.0,
                                  posterior_sigma=True,
                                  shift_correlation=R3)
    c = rs.bayesian_map_inversion(K3 / np.array([1.0, 1e13]), shifts, SIG3,
                                  500.0, 50.0 / 1e26, posterior_sigma=True,
                                  shift_correlation=R3)
    np.testing.assert_allclose(c.density, a.density * 1e13,
                               rtol=1e-10)              # measured 1.1e-14
    np.testing.assert_allclose(c.density_sigma, a.density_sigma * 1e13,
                               rtol=1e-10)
    np.testing.assert_allclose(c.strain, a.strain, rtol=1e-10)


def test_bayesian_correlated_refusals():
    shifts = _smooth_map(4, 4, 13)
    Rnan = np.broadcast_to(R3, (4, 4, 3, 3)).copy()
    Rnan[1, 1, 0, 1] = Rnan[1, 1, 1, 0] = np.nan
    with pytest.raises(ValueError, match="fill_masked"):
        rs.bayesian_map_inversion(K3, shifts, SIG3, 1.0,
                                  shift_correlation=Rnan)
    # with fill_masked the pixel is missing as a whole, like a NaN pixel
    a = rs.bayesian_map_inversion(K3, shifts, SIG3, 1.0, fill_masked=True,
                                  shift_correlation=Rnan)
    s2 = shifts.copy()
    s2[:, 1, 1] = np.nan
    b = rs.bayesian_map_inversion(K3, s2, SIG3, 1.0, fill_masked=True,
                                  shift_correlation=R3)
    np.testing.assert_allclose(a.strain, b.strain, rtol=1e-12)  # measured 0
    bad = R3.copy()
    bad[0, 1] = bad[1, 0] = 1.0
    bad[0, 2] = bad[2, 0] = -1.0
    with pytest.raises(ValueError, match="positive definite"):
        rs.bayesian_map_inversion(K3, shifts, SIG3, 1.0,
                                  shift_correlation=bad)
    with pytest.raises(ValueError):
        rs.bayesian_map_inversion(K3, shifts, SIG3, 1.0,
                                  shift_correlation=np.eye(2))


# ----------------------------------------------------------------------
# Fixed: exact posterior sigmas of large maps on the lattice path


def test_lattice_posterior_sigmas_need_no_max_dense():
    """70 x 70 pixels, 9800 unknowns, above the default max_dense = 4096:
    0.16.0 refused; the lattice formulas now give the sigmas, and they
    equal entries of the inverse posterior precision computed one by one
    with scipy's sparse solver (an independent route)."""
    from scipy.sparse.linalg import spsolve
    from ramansep.bayesian import _assemble, _data_part
    H = W = 70
    shifts = np.zeros((3, H, W))
    r = rs.bayesian_map_inversion(K3, shifts, SIG3, 200.0, 30.0,
                                  posterior_sigma=True)
    data = _data_part(K3, shifts, SIG3, False, False, None)
    A = _assemble(data, _grid_laplacian(H, W), 200.0, 30.0)
    P = H * W
    for p in (0, 35 * 70 + 35, P - 1, 7):
        e = np.zeros(2 * P)
        e[p] = 1.0
        vs = spsolve(A, e)[p]
        e[p], e[P + p] = 0.0, 1.0
        vn = spsolve(A, e)[P + p]
        i, j = divmod(p, W)
        assert r.strain_sigma[i, j] == pytest.approx(np.sqrt(vs),
                                                     rel=1e-10)  # 4.4e-16
        assert r.density_sigma[i, j] == pytest.approx(np.sqrt(vn),
                                                      rel=1e-10)
    # the sparse path keeps its limit
    with pytest.raises(ValueError, match="max_dense"):
        rs.bayesian_map_inversion(
            K3, shifts, np.broadcast_to(SIG3[:, None, None], (3, H, W)),
            200.0, posterior_sigma=True)


# ----------------------------------------------------------------------
# Added: the evidence of the smoothness weights


@pytest.mark.parametrize("case", ["lattice", "correlated", "missing"])
def test_evidence_equals_the_dense_covariance_form(case):
    """Differences of the log evidence between two choices of the weights
    equal those of the dense Gaussian density N(s; 0, S + Kb Q^-1 Kb^T),
    with the improper prior made proper by delta I and the limit
    delta -> 0 taken by Richardson extrapolation from delta, 2 delta and
    4 delta (error of order delta^3)."""
    H, W = 4, 5
    rng = np.random.default_rng(0)
    shifts = rng.normal(0, 0.05, (3, H, W)) + 0.3
    sig_full = np.broadcast_to(SIG3[:, None, None], (3, H, W)).copy()
    R = np.eye(3)
    kw = {}
    sig = SIG3
    if case == "correlated":
        R = R3
        kw = {"shift_correlation": R3}
    if case == "missing":
        sig_full = sig_full * rng.uniform(0.5, 2.0, (3, H, W))
        sig = sig_full
        shifts[2, 1, 1] = np.nan
        shifts[:, 3, 4] = np.nan
        kw = {"fill_masked": True}

    def pkg(a, b):
        return rs.bayesian_log_evidence(K3, shifts, sig, a, b, **kw)

    def dense(d):
        return (_dense_log_evidence(K3, shifts, sig_full, R, 5.0, 20.0, d)
                - _dense_log_evidence(K3, shifts, sig_full, R, 500.0, 3.0,
                                      d))
    d = 1e-3
    want = (8 * dense(d) - 6 * dense(2 * d) + dense(4 * d)) / 3
    got = pkg(5.0, 20.0) - pkg(500.0, 3.0)
    assert got == pytest.approx(want, abs=2e-6)   # measured 9e-9 to 3.5e-7


def test_evidence_lattice_and_sparse_paths_agree_and_are_invariant():
    """The lattice eigenvalue formula and the sparse LU factorization give
    the same log evidence (per-pixel sigmas that are the same everywhere
    take the sparse path). Adding a constant strain and density to the
    whole map changes nothing (the prior says nothing about the mean),
    and differences between weights do not depend on the density unit."""
    shifts = _smooth_map(6, 8, 21)
    full = np.broadcast_to(SIG3[:, None, None], shifts.shape).copy()
    for corr in (None, R3):
        a = rs.bayesian_log_evidence(K3, shifts, SIG3, 80.0, 9.0,
                                     shift_correlation=corr)
        b = rs.bayesian_log_evidence(K3, shifts, full, 80.0, 9.0,
                                     shift_correlation=corr)
        assert a == pytest.approx(b, rel=1e-12, abs=1e-9)  # measured 5.7e-14
    moved = shifts + (K3 @ np.array([0.7, -0.2]))[:, None, None]
    assert rs.bayesian_log_evidence(K3, moved, SIG3, 80.0, 9.0) \
        == pytest.approx(rs.bayesian_log_evidence(K3, shifts, SIG3, 80.0,
                                                  9.0), abs=1e-9)  # 0
    Kc = K3 / np.array([1.0, 1e13])

    def diff(K, f):
        return (rs.bayesian_log_evidence(K, shifts, SIG3, 80.0, 9.0 * f)
                - rs.bayesian_log_evidence(K, shifts, SIG3, 3.0, 400.0 * f))
    assert diff(Kc, 1e-26) == pytest.approx(diff(K3, 1.0),
                                            abs=1e-8)       # measured 5.7e-14


def _draw_prior(lam, Uh, Uw, mu, rng):
    """A draw of the intrinsic smoothness prior (mean level 0): random
    normal coefficients on the eigenvectors of the lattice Laplacian with
    variance 1 / (lam mu), and none on the constant one (mu = 0)."""
    z = rng.standard_normal(mu.shape)
    sd = np.zeros(mu.shape)
    pos = mu > 1e-12
    sd[pos] = 1.0 / np.sqrt(lam * mu[pos])
    return Uh @ (z * sd) @ Uw.T


def test_evidence_recovers_the_weights_of_maps_drawn_from_the_prior():
    """Maps drawn from the smoothness prior with known weights (300 for
    the strain, 80 for the density) and measured with noise. The score
    (the derivative of the log evidence with respect to log lam) of a
    correctly normalized likelihood has mean zero at the true value: over
    200 draws it does, within 4 standard errors, while at four times the
    true weight it is far from zero. `choose_smoothing(separate=True)`
    recovers log lam without bias over 60 draws."""
    H, W = 8, 10
    lh, Uh = np.linalg.eigh(_path_laplacian(H))
    lw, Uw = np.linalg.eigh(_path_laplacian(W))
    mu = lh[:, None] + lw[None, :]
    ls, ln = 300.0, 80.0
    rng = np.random.default_rng(11)
    h = 1e-4
    score_s, score_n, wrong, est = [], [], [], []
    for k in range(200):
        xs = _draw_prior(ls, Uh, Uw, mu, rng) + 0.1
        xn = _draw_prior(ln, Uh, Uw, mu, rng) - 0.3
        sh = (np.einsum("mj,jhw->mhw", K3, np.stack([xs, xn]))
              + rng.standard_normal((3, H, W)) * SIG3[:, None, None])

        def f(a, b):
            return rs.bayesian_log_evidence(K3, sh, SIG3, a, b)
        score_s.append((f(ls * np.exp(h), ln) - f(ls * np.exp(-h), ln))
                       / (2 * h))
        score_n.append((f(ls, ln * np.exp(h)) - f(ls, ln * np.exp(-h)))
                       / (2 * h))
        wrong.append((f(4 * ls * np.exp(h), ln) - f(4 * ls * np.exp(-h), ln))
                     / (2 * h))
        if k < 60:
            c = rs.choose_smoothing(K3, sh, SIG3, separate=True)
            assert not (c.at_lower_bound or c.at_upper_bound)
            est.append((np.log(c.lam_strain / ls),
                        np.log(c.lam_density / ln)))
    for sc in (score_s, score_n):
        sc = np.asarray(sc)
        se = sc.std(ddof=1) / np.sqrt(sc.size)
        assert abs(sc.mean()) < 4 * se      # measured 1.8 and 1.5 se
    w = np.asarray(wrong)
    assert w.mean() < -10 * w.std(ddof=1) / np.sqrt(w.size)  # measured -58
    e = np.asarray(est)
    se = e.std(axis=0, ddof=1) / np.sqrt(e.shape[0])
    assert np.all(np.abs(e.mean(axis=0)) < 4 * se)   # measured 1.7, 0.9 se


def test_choose_smoothing_finds_the_maximum_and_reports_bounds():
    shifts = _smooth_map(10, 12, 31)
    for separate in (False, True):
        c = rs.choose_smoothing(K3, shifts, SIG3, separate=separate)
        if not separate:
            assert c.lam_strain == c.lam_density
        best = rs.bayesian_log_evidence(K3, shifts, SIG3, c.lam_strain,
                                        c.lam_density)
        assert c.log_evidence == pytest.approx(best, abs=1e-9)
        for fs, fn in ((1.05, 1.0), (1 / 1.05, 1.0), (1.0, 1.05),
                       (1.0, 1 / 1.05)):
            if not separate:
                fn = fs
            assert rs.bayesian_log_evidence(
                K3, shifts, SIG3, c.lam_strain * fs,
                c.lam_density * fn) <= best + 1e-9
    # a truth that does not vary at all: the most smoothing is best
    flat = (np.einsum("mj,j->m", K3, [0.01, 0.02])[:, None, None]
            + np.random.default_rng(2).normal(size=(3, 10, 12))
            * SIG3[:, None, None])
    c = rs.choose_smoothing(K3, flat, SIG3)
    assert c.at_upper_bound and not c.at_lower_bound
    # the units of the density do not change the choice
    a = rs.choose_smoothing(K3, shifts, SIG3, separate=True)
    b = rs.choose_smoothing(K3 / np.array([1.0, 1e13]), shifts, SIG3,
                            separate=True)
    # (the search stops at a step of 1e-4 in log lam)
    assert b.lam_strain == pytest.approx(a.lam_strain, rel=1e-3)  # 3.2e-5
    assert b.lam_density == pytest.approx(a.lam_density / 1e26,
                                          rel=1e-3)     # measured 2.1e-5
    # refusals
    with pytest.raises(ValueError, match="positive"):
        rs.bayesian_log_evidence(K3, shifts, SIG3, 0.0)
    with pytest.raises(ValueError, match="search_range"):
        rs.choose_smoothing(K3, shifts, SIG3, search_range=1.0)
    with pytest.raises(ValueError, match="2 pixels"):
        rs.choose_smoothing(K3, shifts[:, :1, :1], SIG3)


# ----------------------------------------------------------------------
# Added: design_references for the calibration with a fitted offset


EPS7 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.25])  # README 7
RHO7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 0.8])


def _offset_det(idx):
    plan = rs.plan_calibration(EPS7[list(idx)], RHO7[list(idx)], 0.15,
                               fit_offset=True)
    return np.linalg.det(plan["cov"]) if plan["identifiable"] else np.inf


def test_design_references_with_a_fitted_offset():
    """On the candidates of README example 7, the greedy picks for the
    offset fit equal the best subset found by an exhaustive search over
    all subsets (smallest determinant of the planned lever-arm
    covariance), for 3 to 6 picks; the picks made for the through-origin
    fit are 2.05 to 4.31 times worse for the offset fit. The reported
    covariance is that of plan_calibration(..., fit_offset=True)."""
    for n_pick in (3, 4, 5, 6):
        d = rs.design_references(EPS7, RHO7, n_pick, sigmas=0.15,
                                 fit_offset=True)
        best = min(itertools.combinations(range(EPS7.size), n_pick),
                   key=_offset_det)
        # (several subsets can share the best value, so the values are
        # compared)
        assert _offset_det(d["indices"]) == pytest.approx(
            _offset_det(best), rel=1e-12)               # measured 0
        o = rs.design_references(EPS7, RHO7, n_pick, sigmas=0.15)
        assert _offset_det(o["indices"]) > 2.0 * _offset_det(d["indices"])
        plan = rs.plan_calibration(EPS7[d["indices"]], RHO7[d["indices"]],
                                   0.15, fit_offset=True)
        np.testing.assert_array_equal(d["cov"], plan["cov"])
    # a strain sweep at a fixed, non-zero density lies on a line that
    # misses the origin: enough for the through-origin fit, not with an
    # offset
    e = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    r = np.full(5, 0.5)
    assert rs.design_references(e, r, 3)["cov"] is not None
    with pytest.raises(ValueError, match="straight line"):
        rs.design_references(e, r, 3, fit_offset=True)
    d = rs.design_references(np.append(e, 0.3), np.append(r, 0.9), 3,
                             fit_offset=True)
    assert 5 in d["indices"]
    with pytest.raises(ValueError, match="between 3"):
        rs.design_references(EPS7, RHO7, 2, fit_offset=True)


# ----------------------------------------------------------------------
# Added: decimal commas in instrument tables


def test_tables_with_decimal_commas(tmp_path):
    """The same table written with decimal points and with decimal
    commas (tab- and semicolon-separated) loads to the same arrays, bit
    for bit."""
    rng = np.random.default_rng(1)
    wn = np.linspace(480.0, 380.0, 21)
    xs = [0.0, 0.5, 1.25]
    ys = [10.0, 10.5]
    lines = []
    for y in ys:
        for x in xs:
            for w in wn:
                lines.append((x, y, w, round(float(rng.normal(100, 5)), 3)))
    dot = tmp_path / "dot.txt"
    dot.write_text("x\ty\twave\tcounts\n" + "\n".join(
        "\t".join(f"{v}" for v in ln) for ln in lines) + "\n")
    a = rs.load_map_table(dot)
    for name, sep in (("tab.txt", "\t"), ("semi.txt", ";")):
        p = tmp_path / name
        p.write_text("x;y;wave;counts\n".replace(";", sep) + "\n".join(
            sep.join(f"{v}".replace(".", ",") for v in ln) for ln in lines)
            + "\n")
        b = rs.load_map_table(p, decimal=",")
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)
        # without decimal="," such a file is refused, not misread
        with pytest.raises(ValueError):
            rs.load_map_table(p)
    spec = tmp_path / "spec.txt"
    spec.write_text("\n".join(f"{w:.2f};{c:.1f}".replace(".", ",")
                              for w, c in zip(wn, np.arange(21.0))) + "\n")
    x, y = rs.load_spectrum_table(spec, decimal=",")
    np.testing.assert_array_equal(x, np.round(wn[::-1], 2))
    np.testing.assert_array_equal(y, np.arange(21.0)[::-1])
    with pytest.raises(ValueError, match="delimiter"):
        rs.load_spectrum_table(spec, decimal=",", delimiter=",")
    with pytest.raises(ValueError, match="decimal"):
        rs.load_spectrum_table(spec, decimal=";")
    assert os.path.exists(spec)
