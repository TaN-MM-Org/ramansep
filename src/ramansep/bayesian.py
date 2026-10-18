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
fills the gaps from the neighbours. At a pixel without any data the
optimality condition reduces to lam (L x)_i = 0, i.e. the value is the
exact average of its neighbours (a discrete harmonic interpolation);
the tests assert this, recovery of a constant truth through the gaps,
and agreement with an independent dense least-squares solve of the
same objective.

Since v0.15 the answer does not depend on the units: a lever-arm matrix
whose columns differ in scale by more than 2^10 (density in cm^-2 or
m^-2) is solved in an exactly rescaled unit (before, the maps lost up
to four digits, and a density in m^-2 was refused as "rank < 2"); and
the exact posterior sigmas no longer need a dense inverse (see
`posterior_sigma`).
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.sparse import (eye as speye, kron as spkron, csc_matrix,
                          csr_matrix, bmat)
from scipy.sparse.linalg import spsolve

from .multimode import _has_full_column_rank

__all__ = ["BayesianMapResult", "bayesian_map_inversion"]


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


def bayesian_map_inversion(K, shifts, sigmas, lam_strain: float,
                           lam_density: float | None = None,
                           posterior_sigma: bool = False,
                           max_dense: int = 4096,
                           fill_masked: bool = False) -> BayesianMapResult:
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
        per-pixel GLS.
    posterior_sigma : also return exact per-pixel posterior sigmas
        (the diagonal of the inverse posterior precision; refused above
        ``max_dense`` unknowns rather than approximated silently). Since
        v0.15 that diagonal is no longer taken from a dense inverse,
        but computed exactly in a cheaper way: from the eigenvectors of
        the pixel lattice when the sigmas are one per mode and no value
        is missing (about 100 times faster at 4050 unknowns), and from
        one sparse LU factorization and a solve per unknown otherwise
        (2 to 4 times faster). The numbers are the same up to rounding;
        the limit is kept.
    fill_masked : False (default) refuses non-finite shifts. True (new
        in v0.12) treats every non-finite shift -- or non-finite sigma
        -- as a missing measurement of that mode at that pixel: it gets
        zero weight, and the smoothness prior fills the gap from the
        neighbours (a pixel with no data at all ends up at the average
        of its neighbours, for both fields). Needs lam_strain > 0 and
        lam_density > 0, and enough data overall to fix both fields
        (the pooled information sum_p K^T W_p K of rank 2); otherwise
        refused. The returned maps then have values at the masked
        pixels too: they are interpolations, and their posterior sigmas
        are larger than those of measured pixels.

    Returns a `BayesianMapResult`.
    """
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
    _, H_, W_ = shifts.shape
    npix = H_ * W_
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

    # Unit balance (new in v0.15). When the strain and density columns
    # of K differ in scale by more than 2^10 (for example the density in
    # cm^-2 or m^-2), the sparse solve loses digits: with the density in
    # cm^-2 the maps of 0.14 differed from those in 1e13 cm^-2 by a
    # relative 6e-4. The problem is then solved for density / t with the
    # density column times t and lam_density times t^2, t a power of two
    # close to the ratio of the column norms. Multiplying by a power of
    # two is exact in floating point, so this is the same problem in a
    # better unit, and the answer is scaled back exactly. Matrices whose
    # columns are within 2^10 of each other in scale take the unchanged
    # path below.
    norms = np.sqrt(np.sum(K * K, axis=0))
    ratio = float(np.log2(norms[0] / norms[1]))
    if abs(ratio) > 10.0:
        t = 2.0 ** round(ratio)
        if np.isfinite(lam_n * t * t):
            res = bayesian_map_inversion(
                K * np.array([1.0, t]), shifts, sigmas, lam_s,
                lam_n * t * t, posterior_sigma=posterior_sigma,
                max_dense=max_dense, fill_masked=fill_masked)
            return BayesianMapResult(
                strain=res.strain, density=res.density * t,
                strain_sigma=res.strain_sigma,
                density_sigma=(None if res.density_sigma is None
                               else res.density_sigma * t),
                lam_strain=lam_s, lam_density=lam_n)

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
    n_missing = int(present.size - present.sum())
    if n_missing and (lam_s <= 0.0 or lam_n <= 0.0):
        raise ValueError(
            f"{n_missing} shift values are missing; filling them needs "
            "the smoothness prior on both fields (lam_strain > 0 and "
            "lam_density > 0)")

    # per-pixel data weights, zero where a measurement is missing
    w = np.zeros(shifts.shape)
    w[present] = 1.0 / sig_full[present] ** 2
    s0 = np.where(present, shifts, 0.0)
    w_flat = w.reshape(m, npix)
    s_flat = s0.reshape(m, npix)
    if n_missing:
        pooled = np.einsum("ki,kp,kj->ij", K, w_flat, K)
        # judged on the unit-free (correlation) form of the pooled
        # information: before v0.14 the raw eigenvalue ratio was used,
        # which refused any K whose strain and density columns differ in
        # scale by about 1e6 (e.g. density in cm^-2) as "singular"
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

    L = _grid_laplacian(H_, W_)
    if per_pixel or n_missing:
        from scipy.sparse import diags
        a11 = np.einsum("k,kp->p", K[:, 0] ** 2, w_flat)
        a12 = np.einsum("k,kp->p", K[:, 0] * K[:, 1], w_flat)
        a22 = np.einsum("k,kp->p", K[:, 1] ** 2, w_flat)
        A = bmat([[diags(a11) + lam_s * L, diags(a12)],
                  [diags(a12), diags(a22) + lam_n * L]], format="csc")
        b = np.concatenate([K[:, 0] @ (w_flat * s_flat),
                            K[:, 1] @ (w_flat * s_flat)])
    else:
        # the historical scalar-sigma path, kept operation for operation
        Wmat = np.diag(1.0 / sig ** 2)
        A2 = K.T @ Wmat @ K                      # (2, 2) data precision
        b2 = K.T @ Wmat @ shifts.reshape(m, npix)  # (2, npix)
        ident = speye(npix, format="csr")
        A = bmat([[A2[0, 0] * ident + lam_s * L, A2[0, 1] * ident],
                  [A2[1, 0] * ident, A2[1, 1] * ident + lam_n * L]],
                 format="csc")
        b = np.concatenate([b2[0], b2[1]])
    x = spsolve(A, b)
    strain = x[:npix].reshape(H_, W_)
    density = x[npix:].reshape(H_, W_)

    ssig = nsig = None
    if posterior_sigma:
        if 2 * npix > int(max_dense):
            raise ValueError(
                f"exact posterior sigmas need one solve per unknown, "
                f"{2 * npix} here (> max_dense = {max_dense}); "
                "raise max_dense explicitly if that cost is intended")
        if per_pixel or n_missing:
            d = np.sqrt(np.maximum(_inverse_diagonal(A), 0.0))
            ssig = d[:npix].reshape(H_, W_)
            nsig = d[npix:].reshape(H_, W_)
        else:
            ssig, nsig = _lattice_posterior_sd(A2, lam_s, lam_n, H_, W_)
    return BayesianMapResult(strain=strain, density=density,
                             strain_sigma=ssig, density_sigma=nsig,
                             lam_strain=lam_s, lam_density=lam_n)
