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
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.sparse import eye as speye, kron as spkron, csr_matrix, bmat
from scipy.sparse.linalg import spsolve

__all__ = ["BayesianMapResult", "bayesian_map_inversion"]


@dataclasses.dataclass
class BayesianMapResult:
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


def bayesian_map_inversion(K, shifts, sigmas, lam_strain, lam_density=None,
                           posterior_sigma=False, max_dense=4096,
                           fill_masked=False):
    """Joint MAP inversion of shift maps with spatial smoothness priors.

    K : (m, 2) lever-arm matrix of rank 2 (as in `MultiModeModel`);
        a rank-deficient K is refused, as `MultiModeModel` refuses it.
    shifts : (m, H, W) measured shift maps.
    sigmas : (m,) per-mode shift uncertainties, one value per mode for
        the whole map; or (since v0.12) (m, H, W), one value per mode
        and pixel, e.g. the `sigma1`, `sigma2` maps of `fit_map`
        stacked. Must be positive where the shift is finite.
    lam_strain, lam_density : smoothness weights (>= 0); lam_density
        defaults to lam_strain. lam = 0 is exactly the per-pixel GLS.
    posterior_sigma : also return exact per-pixel posterior sigmas
        (dense inverse; refused above ``max_dense`` unknowns rather
        than approximated silently).
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
    if np.linalg.matrix_rank(K) < 2:
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
    if lam_s < 0.0 or lam_n < 0.0:
        raise ValueError("smoothness weights must be non-negative")

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
        ev = np.linalg.eigvalsh(pooled)
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
                f"posterior covariance needs a dense inverse of "
                f"{2 * npix} unknowns (> max_dense = {max_dense}); "
                "raise max_dense explicitly if that cost is intended")
        cov = np.linalg.inv(A.toarray())
        d = np.sqrt(np.maximum(np.diag(cov), 0.0))
        ssig = d[:npix].reshape(H_, W_)
        nsig = d[npix:].reshape(H_, W_)
    return BayesianMapResult(strain=strain, density=density,
                             strain_sigma=ssig, density_sigma=nsig,
                             lam_strain=lam_s, lam_density=lam_n)
