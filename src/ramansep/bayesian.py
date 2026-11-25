"""Joint Bayesian inversion of a whole map with spatial smoothness
priors -- the v0.6 roadmap item.

The per-pixel GLS inversion of `multimode` treats every pixel alone,
so pixel noise lands directly in the strain and density maps.
Physically, strain and doping fields vary smoothly on the pixel scale
far more often than they jump, and that knowledge is worth variance.
This module makes it explicit as a Gaussian Markov random field prior
(Rue and Held, Gaussian Markov Random Fields, Chapman and Hall, 2005):
minimize

    sum_j (s_j - K x_j)^T W (s_j - K x_j)
        + lam_strain  * sum_edges (strain_i  - strain_j)^2
        + lam_density * sum_edges (density_i - density_j)^2

over both fields jointly -- a sparse linear (MAP) problem, solved
exactly, with the exact posterior covariance available on request.

Because everything is linear-Gaussian, the estimator's behavior is
provable and the tests assert it rather than trust it:

* lam = 0 reproduces the per-pixel GLS maps AND their per-pixel
  sigmas of `MultiModeModel.invert` to machine precision;
* a spatially constant truth measured without noise is recovered
  exactly at every lam (the prior costs nothing on the truth);
* lam -> infinity drives the solution to the spatially constant
  precision-weighted pooled GLS estimate, computed independently in
  the tests;
* smoothing never increases the posterior variance: adding a positive
  semidefinite precision term shrinks the covariance in the Loewner
  order, so every per-pixel posterior sigma at lam > 0 is at or below
  its lam = 0 value -- asserted numerically pixel by pixel.

The prior graph is the 4-neighbor pixel lattice with natural (Neumann)
boundaries. Shift uncertainties are either one scalar per mode for the
whole map, or (since v0.12) one value per mode and pixel, which makes
the data term pixel-dependent without changing anything structural.

Masked pixels (since v0.12, on request): a missing measurement is a
zero data weight, so with ``fill_masked=True`` the same linear system
fills the gaps from the neighbors. At a pixel without any data the
optimality condition reduces to lam (L x)_i = 0, i.e. the value is the
exact average of its neighbors (a discrete harmonic interpolation);
the tests assert this, recovery of a constant truth through the gaps,
and agreement with an independent dense least-squares solve of the
same objective.

Since v0.15 the answer does not depend on the units: a lever-arm matrix
whose columns differ in scale by more than 2^10 (density in cm^-2 or
m^-2) is solved in an exactly rescaled unit (before, the maps lost up
to four digits, and a density in m^-2 was refused as "rank < 2"); and
the exact posterior sigmas no longer need a dense inverse (see
`posterior_sigma`).

New in v0.17:

* correlated shift errors (`shift_correlation`), with the inverse of
  the full shift covariance of each pixel in the data term, so that
  lam = 0 gives the per-pixel result of `MultiModeModel` with the same
  correlation (asserted in the tests, as is agreement with a dense
  whitened least-squares solve at lam > 0);
* when every pixel has the same data precision and nothing is missing,
  the maps are solved in the eigenvectors of the pixel lattice (about
  200 times faster than the sparse solve for 200 x 200 pixels), and the
  exact posterior sigmas are no longer limited by `max_dense`;
* `bayesian_log_evidence` and `choose_smoothing`: the probability of
  the measured shifts given the smoothness weights, with the fields
  integrated out, and the weights that maximize it (D. J. C. MacKay,
  "Bayesian interpolation", Neural Computation 4, 415 (1992)). The
  tests check the evidence against the dense covariance form of the
  same Gaussian density, and that on maps drawn from the prior the
  weights are recovered without bias.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.sparse import (eye as speye, kron as spkron, csc_matrix,
                          csr_matrix, bmat)
from scipy.sparse.linalg import spsolve

from .multimode import _has_full_column_rank

__all__ = ["BayesianMapResult", "bayesian_map_inversion",
           "bayesian_log_evidence", "choose_smoothing", "SmoothingChoice"]


@dataclasses.dataclass
class BayesianMapResult:
    """Result of `bayesian_map_inversion`.

    strain, density : (H, W) maximum a posteriori maps, in the units of
        the lever-arm matrix columns.
    strain_sigma, density_sigma : (H, W) exact posterior standard
        deviations when `posterior_sigma=True` was requested, else None.
    lam_strain, lam_density : the smoothness weights used.
    """

    strain: np.ndarray
    density: np.ndarray
    strain_sigma: np.ndarray | None
    density_sigma: np.ndarray | None
    lam_strain: float
    lam_density: float


def _grid_laplacian(h, w):
    """Combinatorial Laplacian of the 4-neighbor h x w pixel lattice
    (Neumann boundaries): x^T L x = sum over edges (x_i - x_j)^2."""
    def path(n):
        d = np.zeros(n)
        d[:-1] += 1.0
        d[1:] += 1.0
        L = np.diag(d)
        off = -np.ones(n - 1)
        L += np.diag(off, 1) + np.diag(off, -1)
        return csr_matrix(L)

    Lh, Lw = path(h), path(w)
    return spkron(Lh, speye(w)) + spkron(speye(h), Lw)


def _path_laplacian(n):
    """Dense Laplacian of a path of n pixels (Neumann boundaries), the
    1D factor of `_grid_laplacian`."""
    L = np.zeros((n, n))
    i = np.arange(n - 1)
    L[i, i] += 1.0
    L[i + 1, i + 1] += 1.0
    L[i, i + 1] = L[i + 1, i] = -1.0
    return L


def _lattice_posterior_sd(A2, lam_s, lam_n, h, w):
    """Exact posterior sigmas of the scalar-sigma problem without
    missing values (new in v0.15).

    The precision is then A = [[a I + lam_s L, c I], [c I, d I + lam_n L]]
    with (a, c, d) from the 2 x 2 data precision A2 = K^T W K and
    L = L_h (x) I + I (x) L_w the lattice Laplacian, a Kronecker sum.
    With the eigen-decompositions L_h = U_h diag(l_h) U_h^T and
    L_w = U_w diag(l_w) U_w^T, the eigenvectors of L are the products
    U_h[:, i] (x) U_w[:, j] with eigenvalues mu_ij = l_h[i] + l_w[j], and
    on each of them A acts as the 2 x 2 matrix
    M_ij = [[a + lam_s mu_ij, c], [c, d + lam_n mu_ij]]. Hence the
    diagonal of A^-1 is, for pixel (p, q),
    sum_ij U_h[p, i]^2 U_w[q, j]^2 (M_ij^-1)_00 for the strain (and _11
    for the density): two small matrix products instead of a dense
    inverse of A. The tests compare it with that dense inverse, kept as
    the reference."""
    lh, Uh = np.linalg.eigh(_path_laplacian(h))
    lw, Uw = np.linalg.eigh(_path_laplacian(w))
    mu = lh[:, None] + lw[None, :]
    p = A2[0, 0] + lam_s * mu
    q = A2[1, 1] + lam_n * mu
    det = p * q - A2[0, 1] * A2[1, 0]
    Ph, Pw = Uh * Uh, Uw * Uw
    var_s = Ph @ (q / det) @ Pw.T
    var_n = Ph @ (p / det) @ Pw.T
    return (np.sqrt(np.maximum(var_s, 0.0)),
            np.sqrt(np.maximum(var_n, 0.0)))


def _lattice_eig(h, w):
    """Eigenvectors of the two path Laplacians and the eigenvalues
    mu_ij = l_h[i] + l_w[j] of the lattice Laplacian (see
    `_lattice_posterior_sd`). Returns (U_h, U_w, mu)."""
    lh, Uh = np.linalg.eigh(_path_laplacian(h))
    lw, Uw = np.linalg.eigh(_path_laplacian(w))
    return Uh, Uw, lh[:, None] + lw[None, :]


def _lattice_solve(A2, b, lam_s, lam_n, h, w, eig=None):
    """The MAP maps when every pixel has the same 2 x 2 data precision A2
    and no value is missing (new in v0.17). In the eigenvectors of the
    lattice Laplacian the posterior precision falls apart into the 2 x 2
    blocks M_ij = [[a + lam_s mu_ij, c], [c, d + lam_n mu_ij]] (see
    `_lattice_posterior_sd`), so the right-hand side is transformed with
    U_h^T B U_w, each block is solved in closed form, and the result is
    transformed back: two small matrix products per field instead of a
    sparse factorization of all 2 H W unknowns. Returns (strain,
    density, det) with det the (h, w) array of det M_ij."""
    Uh, Uw, mu = _lattice_eig(h, w) if eig is None else eig
    a, c, d = A2[0, 0], A2[0, 1], A2[1, 1]
    p = a + lam_s * mu
    q = d + lam_n * mu
    det = p * q - c * c
    n = h * w
    t0 = Uh.T @ b[:n].reshape(h, w) @ Uw
    t1 = Uh.T @ b[n:].reshape(h, w) @ Uw
    xs = Uh @ ((q * t0 - c * t1) / det) @ Uw.T
    xn = Uh @ ((p * t1 - c * t0) / det) @ Uw.T
    return xs, xn, det


def _inverse_diagonal(A, block: int = 256) -> np.ndarray:
    """The diagonal of A^-1 for a sparse, invertible A (new in v0.15).

    A is factorized once (`scipy.sparse.linalg.splu`) and solved against
    the unit vectors, `block` at a time; entry i of the diagonal is
    entry i of the solution for unit vector i. This is exact up to
    rounding, like the dense `numpy.linalg.inv(A.toarray())` used before
    v0.15 (kept in the tests as the reference), and cheaper for the
    sparse posterior precision of a map: 2 to 4 times faster at 4050
    unknowns on the test machine. Used when the sigmas are given per
    pixel or values are missing; otherwise `_lattice_posterior_sd`
    applies."""
    from scipy.sparse.linalg import splu
    lu = splu(csc_matrix(A))
    n = A.shape[0]
    out = np.empty(n)
    for start in range(0, n, block):
        k = min(block, n - start)
        rows = np.arange(start, start + k)
        E = np.zeros((n, k))
        E[rows, np.arange(k)] = 1.0
        out[start:start + k] = lu.solve(E)[rows, np.arange(k)]
    return out


def _check_inputs(K, shifts, sigmas, lam_strain, lam_density):
    """The argument checks shared by `bayesian_map_inversion`,
    `bayesian_log_evidence` and `choose_smoothing` (those of
    `bayesian_map_inversion` before v0.17, unchanged and in the same
    order). Returns (K, shifts, sig, per_pixel, lam_s, lam_n)."""
    K = np.asarray(K, dtype=float)
    if K.ndim != 2 or K.shape[1] != 2:
        raise ValueError("K must be (m, 2)")
    if not np.all(np.isfinite(K)):
        raise ValueError("K must be finite")
    if not _has_full_column_rank(K):             # unit-free, since v0.15
        raise ValueError("lever-arm matrix has rank < 2: the modes "
                         "cannot separate strain from density")
    m = K.shape[0]
    shifts = np.asarray(shifts, dtype=float)
    if shifts.ndim != 3 or shifts.shape[0] != m:
        raise ValueError("shifts must be (m, H, W)")
    sig = np.asarray(sigmas, dtype=float)
    per_pixel = sig.ndim == 3
    if per_pixel:
        if sig.shape != shifts.shape:
            raise ValueError("per-pixel sigmas must have the shape of "
                             f"shifts {shifts.shape}; got {sig.shape}")
    elif sig.shape != (m,):
        raise ValueError("sigmas must be m positive scalars or an "
                         "(m, H, W) array")
    lam_s = float(lam_strain)
    lam_n = lam_s if lam_density is None else float(lam_density)
    if not (np.isfinite(lam_s) and np.isfinite(lam_n)):
        # before v0.13 a NaN or infinite weight gave all-NaN maps with
        # only SciPy's "Matrix is exactly singular" warning
        raise ValueError(
            "smoothness weights must be finite; for the fully pooled "
            "(spatially constant) limit use a large finite value")
    if lam_s < 0.0 or lam_n < 0.0:
        raise ValueError("smoothness weights must be non-negative")
    return K, shifts, sig, per_pixel, lam_s, lam_n


def _unit_scale(K):
    """The power of two t by which the density column of K is
    multiplied before solving, or None (v0.15). When the strain and
    density columns of K differ in scale by more than 2^10 (for example
    the density in cm^-2 or m^-2), the sparse solve loses digits: with
    the density in cm^-2 the maps of 0.14 differed from those in
    1e13 cm^-2 by a relative 6e-4. The problem is then solved for
    density / t with the density column times t and lam_density times
    t^2. Multiplying by a power of two is exact in floating point, so
    this is the same problem in a better unit, and the answer is scaled
    back exactly. Matrices whose columns are within 2^10 of each other
    in scale take the unchanged path."""
    norms = np.sqrt(np.sum(K * K, axis=0))
    ratio = float(np.log2(norms[0] / norms[1]))
    if abs(ratio) > 10.0:
        return 2.0 ** round(ratio)
    return None


def _data_part(K, shifts, sig, per_pixel, fill_masked, shift_correlation):
    """The data term of the map problem, without the smoothness prior.

    Returns a dict with the map shape (H, W), npix, n_missing, the
    per-pixel 2 x 2 data precision (a11, a12, a22, each (npix,)), the
    right-hand side b (2 npix,), `A2` (the common 2 x 2 data precision
    when every pixel has the same one and nothing is missing, which
    allows the lattice formulas; else None) and `misfit`, a function of
    the (strain, density) maps that returns the weighted data misfit
    sum_p (s_p - K x_p)^T W_p (s_p - K x_p). Refuses non-finite or
    non-positive inputs as `bayesian_map_inversion` documents."""
    m, H_, W_ = shifts.shape
    npix = H_ * W_
    sig_full = np.broadcast_to(sig[:, None, None] if not per_pixel
                               else sig, shifts.shape)
    present = np.isfinite(shifts) & np.isfinite(sig_full)
    if not fill_masked and not np.all(np.isfinite(shifts)):
        raise ValueError(
            "shifts contain non-finite values; the smoothness prior "
            "couples pixels, so masked pixels (e.g. from fit_map) "
            "must be excluded or infilled deliberately before a joint "
            "map inversion -- pass fill_masked=True to let the prior "
            "fill them, or use the per-pixel inversion, which "
            "propagates them as NaN")
    if not fill_masked and not np.all(np.isfinite(sig_full)):
        raise ValueError("sigmas contain non-finite values; pass "
                         "fill_masked=True to treat those entries as "
                         "missing")
    if np.any(sig_full[present] <= 0.0):
        raise ValueError("sigmas must be positive")
    if shift_correlation is not None:
        return _data_part_correlated(K, shifts, sig_full, present,
                                     per_pixel, fill_masked,
                                     shift_correlation)
    n_missing = int(present.size - present.sum())
    # per-pixel data weights, zero where a measurement is missing
    w = np.zeros(shifts.shape)
    w[present] = 1.0 / sig_full[present] ** 2
    s0 = np.where(present, shifts, 0.0)
    w_flat = w.reshape(m, npix)
    s_flat = s0.reshape(m, npix)
    out = dict(H=H_, W=W_, npix=npix, n_missing=n_missing)
    if per_pixel or n_missing:
        out["a11"] = np.einsum("k,kp->p", K[:, 0] ** 2, w_flat)
        out["a12"] = np.einsum("k,kp->p", K[:, 0] * K[:, 1], w_flat)
        out["a22"] = np.einsum("k,kp->p", K[:, 1] ** 2, w_flat)
        out["b"] = np.concatenate([K[:, 0] @ (w_flat * s_flat),
                                   K[:, 1] @ (w_flat * s_flat)])
        out["A2"] = None
    else:
        # the historical scalar-sigma path, kept operation for operation
        Wmat = np.diag(1.0 / sig ** 2)
        A2 = K.T @ Wmat @ K                      # (2, 2) data precision
        b2 = K.T @ Wmat @ shifts.reshape(m, npix)  # (2, npix)
        out["A2"] = A2
        out["b"] = np.concatenate([b2[0], b2[1]])

    def misfit(xs, xn):
        r = s_flat - (K[:, 0][:, None] * xs.ravel()[None, :]
                      + K[:, 1][:, None] * xn.ravel()[None, :])
        return float(np.sum(w_flat * r * r))
    out["misfit"] = misfit
    out["pooled"] = np.einsum("ki,kp,kj->ij", K, w_flat, K)
    return out


def _data_part_correlated(K, shifts, sig_full, present, per_pixel,
                          fill_masked, shift_correlation):
    """`_data_part` with correlated shift errors (new in v0.17).

    The shift covariance of pixel p is S_p = D_p R_p D_p (D_p the
    diagonal matrix of the sigmas, R_p the correlation matrix, in any
    form `MultiModeModel.invert` takes). Its inverse W_p replaces the
    diagonal weights: the data precision is K^T W_p K and the
    right-hand side K^T W_p s_p. A missing measurement (NaN shift or
    sigma, with `fill_masked=True`) is removed from the pixel as an
    infinite sigma is in `MultiModeModel`: its correlations with the
    other modes are set to zero and its weight to zero, so W_p is the
    inverse covariance of the measured modes alone. A pixel whose
    correlation matrix holds a NaN is missing as a whole (refused
    without `fill_masked=True`)."""
    from .multimode import _correlation_stack
    m, H_, W_ = shifts.shape
    npix = H_ * W_
    R, masked = _correlation_stack(shift_correlation, m, (H_, W_))
    if np.any(masked):
        if not fill_masked:
            raise ValueError(
                "shift_correlation contains NaN at some pixels; pass "
                "fill_masked=True to treat those pixels as missing")
        present = present & ~masked.reshape(H_, W_)[None, :, :]
    n_missing = int(present.size - present.sum())
    pres = present.reshape(m, npix).T                    # (P, m)
    eye = np.eye(m, dtype=bool)
    Rz = np.where((pres[:, :, None] & pres[:, None, :]) | eye, R, 0.0)
    Rz[masked] = np.eye(m)
    ev = np.linalg.eigvalsh(Rz)
    if np.any(ev[:, 0] <= 10.0 * m * np.finfo(float).eps * ev[:, -1]):
        raise ValueError(
            "the correlation matrix of the shift errors is not positive "
            "definite at some pixels (a correlation of exactly +1 or -1, "
            "or correlations that contradict one another)")
    sig_flat = sig_full.reshape(m, npix).T
    dinv = np.zeros((npix, m))
    dinv[pres] = 1.0 / sig_flat[pres]
    Wp = np.linalg.inv(Rz) * dinv[:, :, None] * dinv[:, None, :]
    s = np.where(pres, shifts.reshape(m, npix).T, 0.0)   # (P, m)
    A = np.einsum("ki,pkl,lj->pij", K, Wp, K)
    bb = np.einsum("ki,pkl,pl->ip", K, Wp, s)            # (2, P)
    out = dict(H=H_, W=W_, npix=npix, n_missing=n_missing,
               b=np.concatenate([bb[0], bb[1]]),
               pooled=A.sum(axis=0))
    if not per_pixel and not n_missing and np.all(A == A[0]):
        out["A2"] = A[0]
    else:
        out["A2"] = None
        out["a11"], out["a12"], out["a22"] = (A[:, 0, 0], A[:, 0, 1],
                                              A[:, 1, 1])

    def misfit(xs, xn):
        r = s - (xs.ravel()[:, None] * K[:, 0][None, :]
                 + xn.ravel()[:, None] * K[:, 1][None, :])
        return float(np.einsum("pk,pkl,pl->", r, Wp, r))
    out["misfit"] = misfit
    return out


def _check_pooled(data):
    """Refuse missing values that leave both fields undetermined: the
    pooled information sum_p K^T W_p K must have rank 2. Judged on its
    unit-free (correlation) form: before v0.14 the raw eigenvalue ratio
    was used, which refused any K whose strain and density columns
    differ in scale by about 1e6 (e.g. density in cm^-2) as
    "singular"."""
    pooled = data["pooled"]
    d = np.sqrt(np.diag(pooled))
    if np.all(d > 0.0):
        ev = np.linalg.eigvalsh(pooled / np.outer(d, d))
    else:
        ev = np.zeros(2)
    if not ev[0] > 1e-12 * ev[-1]:
        raise ValueError(
            "the measured (unmasked) shifts do not determine both "
            "fields even with the prior: their pooled information "
            "matrix is singular")


def _assemble(data, L, lam_s, lam_n):
    """The sparse posterior precision A of the map problem (2 npix
    square), built as before v0.17."""
    npix = data["npix"]
    if data["A2"] is None:
        from scipy.sparse import diags
        return bmat([[diags(data["a11"]) + lam_s * L, diags(data["a12"])],
                     [diags(data["a12"]), diags(data["a22"]) + lam_n * L]],
                    format="csc")
    A2 = data["A2"]
    ident = speye(npix, format="csr")
    return bmat([[A2[0, 0] * ident + lam_s * L, A2[0, 1] * ident],
                 [A2[1, 0] * ident, A2[1, 1] * ident + lam_n * L]],
                format="csc")


def bayesian_map_inversion(K, shifts, sigmas, lam_strain: float,
                           lam_density: float | None = None,
                           posterior_sigma: bool = False,
                           max_dense: int = 4096,
                           fill_masked: bool = False, *,
                           shift_correlation=None) -> BayesianMapResult:
    """Joint MAP inversion of shift maps with spatial smoothness priors.

    K : (m, 2) lever-arm matrix of rank 2 (as in `MultiModeModel`);
        a rank-deficient K is refused, as `MultiModeModel` refuses it.
    shifts : (m, H, W) measured shift maps.
    sigmas : (m,) per-mode shift uncertainties, one value per mode for
        the whole map; or (since v0.12) (m, H, W), one value per mode
        and pixel, e.g. the `sigma1`, `sigma2` maps of `fit_map`
        stacked. Must be positive where the shift is finite.
    lam_strain, lam_density : smoothness weights (finite, >= 0);
        lam_density defaults to lam_strain. lam = 0 is exactly the
        per-pixel GLS. `choose_smoothing` (new in v0.17) picks them from
        the data, if you accept its assumptions.
    posterior_sigma : also return exact per-pixel posterior sigmas
        (the diagonal of the inverse posterior precision). They are
        computed exactly, without a dense inverse: from the eigenvectors
        of the pixel lattice when every pixel has the same data
        precision and no value is missing (one sigma per mode, and one
        correlation matrix for the map if any), and from one sparse LU
        factorization and a solve per unknown otherwise. Only the second
        way is refused above ``max_dense`` unknowns (two per pixel),
        rather than run for a long time silently. Before v0.17 the limit
        applied to both, although the lattice formulas cost little at
        any map size (7 ms for 200 x 200 pixels on the machine where
        v0.17 was made).
    fill_masked : False (default) refuses non-finite shifts. True (new
        in v0.12) treats every non-finite shift -- or non-finite sigma
        -- as a missing measurement of that mode at that pixel: it gets
        zero weight, and the smoothness prior fills the gap from the
        neighbors (a pixel with no data at all ends up at the average
        of its neighbors, for both fields). Needs lam_strain > 0 and
        lam_density > 0, and enough data overall to fix both fields
        (the pooled information sum_p K^T W_p K of rank 2); otherwise
        refused. The returned maps then have values at the masked
        pixels too: they are interpolations, and their posterior sigmas
        are larger than those of measured pixels.
    shift_correlation : optional correlations between the shift errors
        of different modes (new in v0.17), in any form
        `MultiModeModel.invert` takes: an (m, m) correlation matrix for
        the whole map, one matrix per pixel (shape (H, W, m, m)), or a
        dict {(i, j): r} of 0-based mode pairs with r a number or an
        (H, W) map, for example `{(0, 1): fits.shift_correlation}` after
        `fit_map(..., joint=True)`. The data term of each pixel then uses
        the inverse of the full shift covariance S_p = D_p R_p D_p in
        place of diag(1/sigma^2), so that lam = 0 gives the per-pixel
        result of `MultiModeModel.invert(..., shift_correlation=...)`.
        A NaN in a pixel's matrix marks that pixel as missing (needs
        `fill_masked=True`); a missing shift of one mode removes only
        that mode there. A matrix that is not positive definite is
        refused. None (default) treats the errors as independent, with
        the results of earlier versions unchanged.

    Returns a `BayesianMapResult`.
    """
    K, shifts, sig, per_pixel, lam_s, lam_n = _check_inputs(
        K, shifts, sigmas, lam_strain, lam_density)
    _, H_, W_ = shifts.shape
    npix = H_ * W_

    t = _unit_scale(K)                            # unit balance, v0.15
    if t is not None and np.isfinite(lam_n * t * t):
        res = bayesian_map_inversion(
            K * np.array([1.0, t]), shifts, sigmas, lam_s,
            lam_n * t * t, posterior_sigma=posterior_sigma,
            max_dense=max_dense, fill_masked=fill_masked,
            shift_correlation=shift_correlation)
        return BayesianMapResult(
            strain=res.strain, density=res.density * t,
            strain_sigma=res.strain_sigma,
            density_sigma=(None if res.density_sigma is None
                           else res.density_sigma * t),
            lam_strain=lam_s, lam_density=lam_n)

    data = _data_part(K, shifts, sig, per_pixel, fill_masked,
                      shift_correlation)
    n_missing = data["n_missing"]
    if n_missing and (lam_s <= 0.0 or lam_n <= 0.0):
        raise ValueError(
            f"{n_missing} shift values are missing; filling them needs "
            "the smoothness prior on both fields (lam_strain > 0 and "
            "lam_density > 0)")
    if n_missing:
        _check_pooled(data)

    if data["A2"] is not None:
        # the same pixel data everywhere: solved in the eigenvectors of
        # the lattice (v0.17; 0.16 used the sparse solve below, about
        # 500 times slower for 200 x 200 pixels)
        strain, density, _ = _lattice_solve(data["A2"], data["b"], lam_s,
                                            lam_n, H_, W_)
    else:
        A = _assemble(data, _grid_laplacian(H_, W_), lam_s, lam_n)
        x = spsolve(A, data["b"])
        strain = x[:npix].reshape(H_, W_)
        density = x[npix:].reshape(H_, W_)

    ssig = nsig = None
    if posterior_sigma:
        if data["A2"] is not None:
            # lattice formulas: cheap at any size, so no limit (v0.17)
            ssig, nsig = _lattice_posterior_sd(data["A2"], lam_s, lam_n,
                                               H_, W_)
        else:
            if 2 * npix > int(max_dense):
                raise ValueError(
                    f"exact posterior sigmas need one solve per unknown "
                    f"here (the data precision differs between pixels, "
                    f"or values are missing): {2 * npix} unknowns "
                    f"(> max_dense = {max_dense}); raise max_dense "
                    "explicitly if that cost is intended")
            d = np.sqrt(np.maximum(_inverse_diagonal(A), 0.0))
            ssig = d[:npix].reshape(H_, W_)
            nsig = d[npix:].reshape(H_, W_)
    return BayesianMapResult(strain=strain, density=density,
                             strain_sigma=ssig, density_sigma=nsig,
                             lam_strain=lam_s, lam_density=lam_n)


# ----------------------------------------------------------------------
# Choosing the smoothness weights from the data (new in v0.17)


def _log_evidence(data, L, lam_s, lam_n):
    """log p(shifts | lam_s, lam_n) up to a constant that does not
    depend on the weights (see `bayesian_log_evidence`)."""
    H_, W_, npix = data["H"], data["W"], data["npix"]
    b = data["b"]
    if data["A2"] is not None:
        # lattice: A is block diagonal in the eigenvectors of L, with the
        # 2 x 2 blocks M_ij, so log det A = sum_ij log det M_ij
        if "eig" not in data:
            data["eig"] = _lattice_eig(H_, W_)
        xs, xn, det = _lattice_solve(data["A2"], b, lam_s, lam_n, H_, W_,
                                     data["eig"])
        logdet = float(np.sum(np.log(det)))
    else:
        from scipy.sparse.linalg import splu
        lu = splu(_assemble(data, L, lam_s, lam_n))
        logdet = float(np.sum(np.log(np.abs(lu.U.diagonal()))))
        x = lu.solve(b)
        xs, xn = x[:npix].reshape(H_, W_), x[npix:].reshape(H_, W_)
    fs, fn = xs.ravel(), xn.ravel()
    penalty = lam_s * float(fs @ (L @ fs)) + lam_n * float(fn @ (L @ fn))
    F = data["misfit"](xs, xn) + penalty
    return (0.5 * (npix - 1) * (np.log(lam_s) + np.log(lam_n))
            - 0.5 * logdet - 0.5 * F)


def bayesian_log_evidence(K, shifts, sigmas, lam_strain: float,
                          lam_density: float | None = None, *,
                          fill_masked: bool = False,
                          shift_correlation=None) -> float:
    """The log evidence of the smoothness weights: log p(shifts | lam),
    up to a constant that does not depend on lam (new in v0.17).

    Inputs as for `bayesian_map_inversion`; both weights must be
    positive. The model behind `bayesian_map_inversion` is
    shifts = K x + noise, with Gaussian noise of the stated sigmas (and
    correlations), and the smoothness prior read as a probability:
    each field has the density exp(-lam/2 sum_edges (x_i - x_j)^2), an
    "intrinsic" Gaussian Markov random field (Rue and Held 2005), which
    says nothing about the mean level of the field. Integrating the
    fields out gives

        log p = (N - 1)/2 (log lam_s + log lam_n) - 1/2 log det A
                - 1/2 F + const,

    with N the number of pixels, A the posterior precision of the map
    problem and F its objective at the solution (the weighted misfit
    plus the smoothness penalty). (N - 1)/2 log lam comes from the
    prior's normalization: the lattice Laplacian of a connected map has
    N - 1 non-zero eigenvalues. Comparing two choices of the weights by
    their evidence is the method of D. J. C. MacKay, "Bayesian
    interpolation", Neural Computation 4, 415 (1992),
    doi:10.1162/neco.1992.4.3.415, which sets "regularizing constants"
    by their posterior probability. `choose_smoothing` maximizes it.

    The constant depends on the units of the lever arms, so compare
    values only within one problem. log det A comes from the lattice
    eigenvalues when every pixel has the same data precision and
    nothing is missing, and from a sparse LU factorization otherwise.
    """
    K, shifts, sig, per_pixel, lam_s, lam_n = _check_inputs(
        K, shifts, sigmas, lam_strain, lam_density)
    if not (lam_s > 0.0 and lam_n > 0.0):
        raise ValueError("the evidence needs positive smoothness weights "
                         "(at lam = 0 the prior is flat and the evidence "
                         "is not defined)")
    _, H_, W_ = shifts.shape
    if H_ * W_ < 2:
        raise ValueError("the evidence needs a map of at least 2 pixels")
    t = _unit_scale(K)
    if t is not None and np.isfinite(lam_n * t * t):
        # the same problem in an exactly rescaled unit; the flat prior
        # on the mean density then differs by the factor t, which is
        # added back so that the value equals that of the direct solve
        return float(np.log(t)) + bayesian_log_evidence(
            K * np.array([1.0, t]), shifts, sigmas, lam_s, lam_n * t * t,
            fill_masked=fill_masked, shift_correlation=shift_correlation)
    data = _data_part(K, shifts, sig, per_pixel, fill_masked,
                      shift_correlation)
    if data["n_missing"]:
        _check_pooled(data)
    return float(_log_evidence(data, _grid_laplacian(H_, W_), lam_s,
                               lam_n))


@dataclasses.dataclass
class SmoothingChoice:
    """Result of `choose_smoothing` (new in v0.17).

    lam_strain, lam_density : the weights with the largest evidence,
        to pass to `bayesian_map_inversion`.
    log_evidence : the evidence there (up to a constant; see
        `bayesian_log_evidence`).
    at_lower_bound, at_upper_bound : True when the evidence at the
        lower (upper) edge of the searched range is within 0.001 of the
        best value, so the data cannot tell the chosen weight from
        weights beyond that edge. At the upper edge the data are
        consistent with a field that does not vary at all; at the lower
        edge they show no sign of smoothness at the pixel scale. In
        either case the returned weight is not a measurement of
        anything.
    n_evaluations : how many times the evidence was computed.
    """

    lam_strain: float
    lam_density: float
    log_evidence: float
    at_lower_bound: bool
    at_upper_bound: bool
    n_evaluations: int


def choose_smoothing(K, shifts, sigmas, *, separate: bool = False,
                     fill_masked: bool = False, shift_correlation=None,
                     search_range: float = 1e6) -> SmoothingChoice:
    """Choose the smoothness weights of `bayesian_map_inversion` by
    their evidence (new in v0.17).

    Inputs as for `bayesian_map_inversion`. separate : False (default)
    looks for one weight used for both fields (lam_density =
    lam_strain, the default of `bayesian_map_inversion`); True looks
    for two. search_range : the weights searched lie within this factor
    above and below the typical data precision of one pixel for that
    field (the mean diagonal of K^T W_p K), a range that does not
    depend on the units.

    The weights returned maximize `bayesian_log_evidence`: a grid of 8
    points per decade, refined with a bounded scalar search (one weight)
    or a Nelder-Mead search (two weights). This is type-II maximum
    likelihood, MacKay's evidence method. It is only as good as its two
    assumptions, which this function cannot check:

    * the shift error bars are right: error bars that are too small
      make noise look like structure, and the chosen weight comes out
      too small (less smoothing);
    * the fields look like a draw from the smoothness prior, that is,
      they vary gradually from pixel to pixel. A field with sharp steps
      (an edge, a grain boundary) is not such a draw; the chosen weight
      is then a compromise that blurs the steps.

    The tests check the evidence against an independent dense
    computation, and that on maps drawn from the prior the weights are
    recovered without bias.
    """
    K, shifts, sig, per_pixel, _, _ = _check_inputs(K, shifts, sigmas,
                                                    1.0, None)
    R_ = float(search_range)
    if not (np.isfinite(R_) and R_ > 1.0):
        raise ValueError("search_range must be a finite number above 1")
    _, H_, W_ = shifts.shape
    if H_ * W_ < 2:
        raise ValueError("choosing a smoothness weight needs a map of at "
                         "least 2 pixels")
    # in an exactly rescaled unit when the columns of K differ in scale
    # by more than 2^10 (as in `bayesian_map_inversion`); the search
    # itself runs in the caller's units
    t = _unit_scale(K)
    tt = 1.0 if t is None else t
    Kc = K if t is None else K * np.array([1.0, t])
    data = _data_part(Kc, shifts, sig, per_pixel, fill_masked,
                      shift_correlation)
    if data["n_missing"]:
        _check_pooled(data)
    L = _grid_laplacian(H_, W_)
    if data["A2"] is not None:
        a_s, a_n = float(data["A2"][0, 0]), float(data["A2"][1, 1])
    else:
        a_s, a_n = float(np.mean(data["a11"])), float(np.mean(data["a22"]))
    a_n = a_n / (tt * tt)
    n_eval = [0]

    def evidence(lam_s, lam_n):
        return float(np.log(tt)) + _log_evidence(data, L, lam_s,
                                                 lam_n * tt * tt)

    def neg(log_s, log_n):
        n_eval[0] += 1
        return -evidence(float(np.exp(log_s)), float(np.exp(log_n)))

    from scipy.optimize import minimize, minimize_scalar
    span = float(np.log(R_))
    if not separate:
        lo = float(np.log(min(a_s, a_n))) - span
        hi = float(np.log(max(a_s, a_n))) + span
        grid = np.linspace(lo, hi, int(np.ceil((hi - lo) / np.log(10.0)
                                               * 8)) + 1)
        vals = [neg(g, g) for g in grid]
        k = int(np.argmin(vals))
        a, b = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
        r = minimize_scalar(lambda g: neg(g, g), bounds=(a, b),
                            method="bounded", options={"xatol": 1e-7})
        best = (float(r.x), float(r.x)) if r.fun <= vals[k] \
            else (float(grid[k]), float(grid[k]))
        bounds = [(lo, hi), (lo, hi)]
    else:
        ls, ln_ = float(np.log(a_s)), float(np.log(a_n))
        g = np.linspace(-span, span, int(np.ceil(2 * span / np.log(10.0)
                                                 * 8)) + 1)
        vals = [neg(ls + v, ln_ + v) for v in g]
        k = int(np.argmin(vals))
        bounds = [(ls - span, ls + span), (ln_ - span, ln_ + span)]
        r = minimize(lambda v: neg(v[0], v[1]),
                     x0=[ls + g[k], ln_ + g[k]], method="Nelder-Mead",
                     bounds=bounds,
                     options={"xatol": 1e-4, "fatol": 1e-6,
                              "maxfev": 400})
        best = (float(r.x[0]), float(r.x[1])) if r.fun <= vals[k] \
            else (ls + float(g[k]), ln_ + float(g[k]))
    lam_s, lam_n = float(np.exp(best[0])), float(np.exp(best[1]))
    top = evidence(lam_s, lam_n)
    # a bound is flagged when the evidence there is within 1e-3 of the
    # best value: the data then cannot tell the chosen weight from any
    # weight beyond that bound
    if separate:
        ends = {"low": [(bounds[0][0], best[1]), (best[0], bounds[1][0])],
                "high": [(bounds[0][1], best[1]), (best[0], bounds[1][1])]}
    else:
        ends = {"low": [(bounds[0][0], bounds[0][0])],
                "high": [(bounds[0][1], bounds[0][1])]}
    flag = {k: any(-neg(a, b) >= top - 1e-3 for a, b in v)
            for k, v in ends.items()}
    return SmoothingChoice(
        lam_strain=lam_s, lam_density=lam_n, log_evidence=float(top),
        at_lower_bound=flag["low"], at_upper_bound=flag["high"],
        n_evaluations=n_eval[0] + 1)
