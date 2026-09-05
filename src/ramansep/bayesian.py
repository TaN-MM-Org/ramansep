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
  sigmas of `MultiModeSeparation.invert` to machine precision;
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
boundaries. Per-mode shift uncertainties are scalars here (one
uncertainty per mode across the map); per-pixel weights would make the
data term pixel-dependent but change nothing structural.
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
                           posterior_sigma=False, max_dense=4096):
    """Joint MAP inversion of shift maps with spatial smoothness priors.

    K : (m, 2) lever-arm matrix (as in `MultiModeSeparation`).
    shifts : (m, H, W) measured shift maps.
    sigmas : (m,) per-mode shift uncertainties (scalars across the map).
    lam_strain, lam_density : smoothness weights (>= 0); lam_density
        defaults to lam_strain. lam = 0 is exactly the per-pixel GLS.
    posterior_sigma : also return exact per-pixel posterior sigmas
        (dense inverse; refused above ``max_dense`` unknowns rather
        than approximated silently).

    Returns a `BayesianMapResult`.
    """
    K = np.asarray(K, dtype=float)
    if K.ndim != 2 or K.shape[1] != 2:
        raise ValueError("K must be (m, 2)")
    m = K.shape[0]
    shifts = np.asarray(shifts, dtype=float)
    if shifts.ndim != 3 or shifts.shape[0] != m:
        raise ValueError("shifts must be (m, H, W)")
    sig = np.asarray(sigmas, dtype=float)
    if sig.shape != (m,) or np.any(sig <= 0.0):
        raise ValueError("sigmas must be m positive scalars")
    lam_s = float(lam_strain)
    lam_n = lam_s if lam_density is None else float(lam_density)
    if lam_s < 0.0 or lam_n < 0.0:
        raise ValueError("smoothness weights must be non-negative")

    _, H_, W_ = shifts.shape
    npix = H_ * W_
    Wmat = np.diag(1.0 / sig ** 2)
    A2 = K.T @ Wmat @ K                          # (2, 2) data precision
    s_flat = shifts.reshape(m, npix)
    b2 = K.T @ Wmat @ s_flat                     # (2, npix)

    L = _grid_laplacian(H_, W_)
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
