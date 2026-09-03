"""Overdetermined separation: any number of Raman modes, with model checking.

Two modes make the (strain, density) inversion exactly determined, so any
measurement error lands silently in the answer.  A third (or fourth) mode
makes the system overdetermined, and that redundancy buys two things this
module implements:

* a generalized-least-squares inversion, weighting each mode by its shift
  uncertainty: with the m x 2 lever-arm matrix K, per-pixel weights
  W = diag(1/sigma_k^2) and shifts s,
      x_hat = (K^T W K)^{-1} K^T W s,     Cov(x_hat) = (K^T W K)^{-1},
  the minimum-variance unbiased linear estimator (Gauss-Markov); and
* a per-pixel goodness-of-fit: the weighted residual sum of squares is
  chi-square distributed with (m - 2) degrees of freedom when the
  two-cause model holds, so its p-value map flags pixels where strain and
  density alone cannot explain the measured shifts (a third latent
  variable, a phase boundary, a bad fit), something no exactly determined
  inversion can ever detect.

For m = 2 the estimator reduces exactly to the package's
:class:`~ramansep.core.SeparationModel` inversion (zero residual, no
model check); the test suite asserts that equivalence to machine
precision.  :func:`compare_mode_sets` ranks candidate mode subsets by the
uncertainty they deliver (D- and A-optimality of the information matrix)
so the choice of modes becomes a computation instead of a habit.
"""
from __future__ import annotations

import dataclasses
import itertools

import numpy as np
from scipy.stats import chi2


@dataclasses.dataclass
class MultiModeResult:
    """Result of an m-mode generalized-least-squares separation.

    strain, density: estimate maps (units set by the coefficient matrix).
    strain_sigma, density_sigma: propagated one-standard-deviation maps.
    correlation: pixelwise correlation of the two estimates.
    chi2_map: weighted residual sum of squares per pixel (m - 2 dof).
    p_value: probability of a chi2 at least this large under the
        two-cause model; small values flag model violations.
    dof: residual degrees of freedom (m - 2).
    condition_number: of the weighted design matrix (scalar weights) or
        of the unweighted design (pixel-dependent weights).
    """

    strain: np.ndarray
    density: np.ndarray
    strain_sigma: np.ndarray
    density_sigma: np.ndarray
    correlation: np.ndarray
    chi2_map: np.ndarray | None
    p_value: np.ndarray | None
    dof: int
    condition_number: float


class MultiModeModel:
    """Generalized-least-squares inversion of m >= 2 peak-shift maps.

    Parameters: K is the (m, 2) lever-arm matrix, one row per mode,
    columns (strain, density), in the same convention as
    :class:`~ramansep.core.SeparationModel`; mode_names labels the rows
    for reporting.
    """

    def __init__(self, K, mode_names=None):
        self.K = np.asarray(K, dtype=float)
        if self.K.ndim != 2 or self.K.shape[1] != 2 or self.K.shape[0] < 2:
            raise ValueError("K must have shape (m >= 2, 2)")
        if np.linalg.matrix_rank(self.K) < 2:
            raise ValueError("lever-arm matrix has rank < 2: the modes "
                             "cannot separate strain from density")
        self.m = self.K.shape[0]
        self.mode_names = (list(mode_names) if mode_names is not None
                          else [f"mode{i+1}" for i in range(self.m)])
        if len(self.mode_names) != self.m:
            raise ValueError("one name per mode required")
        self.condition_number = float(np.linalg.cond(self.K))

    def invert(self, shifts, sigmas=None) -> MultiModeResult:
        """Invert shift maps into strain and density with uncertainties.

        shifts: sequence of m arrays (or an array with leading axis m),
        one shift map per mode, all the same shape.  sigmas: matching
        one-standard-deviation uncertainties (scalars or arrays,
        broadcastable to the map shape); omitted means unit weights, in
        which case chi2_map is still returned but calibrated only if the
        shift noise really is unit variance.
        """
        S = np.stack([np.asarray(s, dtype=float) for s in shifts])
        if S.shape[0] != self.m:
            raise ValueError(f"expected {self.m} shift maps, got {S.shape[0]}")
        map_shape = S.shape[1:]
        if sigmas is None:
            sig = np.ones_like(S)
        else:
            sig = np.stack([np.broadcast_to(np.asarray(s, dtype=float),
                                            map_shape).copy()
                            for s in sigmas])
            if np.any(sig <= 0):
                raise ValueError("sigmas must be positive")
        # flatten pixels: (m, P)
        P = int(np.prod(map_shape)) if map_shape else 1
        s_flat = S.reshape(self.m, P)
        w_flat = 1.0 / sig.reshape(self.m, P) ** 2
        # per-pixel normal equations, vectorized over pixels
        K = self.K
        # A = K^T W K: components via sums over modes
        a11 = np.einsum("k,kp->p", K[:, 0] ** 2, w_flat)
        a12 = np.einsum("k,kp->p", K[:, 0] * K[:, 1], w_flat)
        a22 = np.einsum("k,kp->p", K[:, 1] ** 2, w_flat)
        b1 = np.einsum("k,kp->p", K[:, 0], w_flat * s_flat)
        b2 = np.einsum("k,kp->p", K[:, 1], w_flat * s_flat)
        det = a11 * a22 - a12 ** 2
        if np.any(det <= 0):
            raise ValueError("weighted design matrix is singular at some "
                             "pixels; check the coefficients and sigmas")
        strain = (a22 * b1 - a12 * b2) / det
        density = (a11 * b2 - a12 * b1) / det
        # covariance = inverse of the information matrix
        var_s = a22 / det
        var_d = a11 / det
        cov_sd = -a12 / det
        with np.errstate(invalid="ignore"):
            corr = cov_sd / np.sqrt(var_s * var_d)
        # weighted residuals and model check
        pred = K[:, 0][:, None] * strain[None, :] \
            + K[:, 1][:, None] * density[None, :]
        resid2 = np.einsum("kp,kp->p", w_flat, (s_flat - pred) ** 2)
        dof = self.m - 2
        if dof > 0:
            chi2_map = resid2.reshape(map_shape)
            p_value = chi2.sf(resid2, dof).reshape(map_shape)
        else:
            chi2_map = None
            p_value = None
        return MultiModeResult(
            strain=strain.reshape(map_shape),
            density=density.reshape(map_shape),
            strain_sigma=np.sqrt(var_s).reshape(map_shape),
            density_sigma=np.sqrt(var_d).reshape(map_shape),
            correlation=corr.reshape(map_shape),
            chi2_map=chi2_map, p_value=p_value, dof=dof,
            condition_number=self.condition_number)

    def forward(self, strain, density):
        """Predict the m peak shifts from strain and density maps."""
        strain = np.asarray(strain, dtype=float)
        density = np.asarray(density, dtype=float)
        return [self.K[k, 0] * strain + self.K[k, 1] * density
                for k in range(self.m)]


def compare_mode_sets(K, sigmas, mode_names=None, subset_size: int = 2):
    """Rank all mode subsets of a given size by delivered uncertainty.

    K: (m, 2) lever arms; sigmas: per-mode shift uncertainties (scalars);
    returns a list of dicts sorted best-first by A-optimality (the trace
    of the estimate covariance, i.e. var_strain + var_density), each with
    the subset's names, indices, var_strain, var_density, trace, and
    d_optimality = det of the information matrix (larger is better).
    """
    K = np.asarray(K, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    m = K.shape[0]
    if sigmas.shape != (m,):
        raise ValueError("one sigma per mode required")
    names = (list(mode_names) if mode_names is not None
             else [f"mode{i+1}" for i in range(m)])
    out = []
    for idx in itertools.combinations(range(m), subset_size):
        Ks = K[list(idx)]
        if np.linalg.matrix_rank(Ks) < 2:
            continue
        W = np.diag(1.0 / sigmas[list(idx)] ** 2)
        info = Ks.T @ W @ Ks
        cov = np.linalg.inv(info)
        out.append(dict(names=[names[i] for i in idx], indices=idx,
                        var_strain=float(cov[0, 0]),
                        var_density=float(cov[1, 1]),
                        trace=float(np.trace(cov)),
                        d_optimality=float(np.linalg.det(info))))
    out.sort(key=lambda d: d["trace"])
    return out
