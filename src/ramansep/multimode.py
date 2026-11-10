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

Since v0.15 the rank test that refuses a lever-arm matrix which cannot
separate the causes is made on K with unit-length columns
(`_has_full_column_rank`), so it does not depend on the units: before,
a density in m^-2 made a well-separating K look like rank 1. Note that
the A-optimality ranking of `compare_mode_sets` (the sum of the strain
and density variances) does depend on the units, as any sum of
quantities in different units does; the D-optimality value scales by
the same factor for every subset and ranks them alike in any unit.

Correlated shift errors (new in v0.16): two overlapping peaks fitted
together have correlated centers, so the errors of their shifts are
not independent. `MultiModeModel.invert(..., shift_correlation=...)`
then uses the full shift covariance S = D R D (D the diagonal matrix of
the sigmas, R the correlation matrix) in place of W^-1:

    x_hat = (K^T S^-1 K)^{-1} K^T S^-1 s,   Cov(x_hat) = (K^T S^-1 K)^{-1},

and the chi-square check r^T S^-1 r, still with (m - 2) degrees of
freedom. The tests check it against a whitened ordinary least-squares
solve (`numpy.linalg.lstsq` after a Cholesky factor of S, computed
independently), against the closed form K^-1 S K^-T of the two-mode
inversion, and against seeded Monte Carlo draws of correlated noise.
"""
from __future__ import annotations

import dataclasses
import itertools

import numpy as np
from scipy.stats import chi2


def _has_full_column_rank(K) -> bool:
    """True if the lever-arm matrix K (m, c) has rank c, judged without
    regard to the units of its columns (new in v0.15).

    Each column is divided by its Euclidean norm (a zero column stays
    zero) before `numpy.linalg.matrix_rank` is applied with its default
    tolerance. Rank does not depend on the scale of a column, so
    neither may this test. Before v0.15 the rank was taken of K as
    given: with the density in m^-2 (the SI unit; arms near 1e-17 cm^-1
    per m^-2 against strain arms near 5 cm^-1 per percent)
    `MultiModeModel`, `ThreeCauseModel` and `bayesian_map_inversion`
    refused a well-separating matrix as "rank < 2", and
    `compare_mode_sets` silently left out every subset."""
    K = np.asarray(K, dtype=float)
    norms = np.sqrt(np.sum(K * K, axis=0))
    norms[norms == 0.0] = 1.0
    return bool(np.linalg.matrix_rank(K / norms) == K.shape[1])


def _stack_sigmas(sigmas, m, map_shape):
    """Stack one sigma (scalar or map) per mode into (m, *map_shape),
    with a clear message when the count or the shapes do not fit."""
    try:
        n = len(sigmas)
    except TypeError:
        raise ValueError(f"sigmas must be a sequence of {m} values or "
                         "maps, one per mode") from None
    if n != m:
        raise ValueError(f"expected {m} sigmas (one per mode), got {n}")
    out = []
    for k, s in enumerate(sigmas):
        try:
            out.append(np.broadcast_to(np.asarray(s, dtype=float),
                                       map_shape).copy())
        except ValueError:
            raise ValueError(
                f"sigma of mode {k + 1} has shape "
                f"{np.shape(s)}, which does not broadcast to the map "
                f"shape {map_shape}") from None
    return np.stack(out)


def _stack_shifts(shifts, m):
    """Stack m shift maps into (m, *map_shape), refusing a wrong count
    or maps of different shapes with a clear message."""
    maps = [np.asarray(s, dtype=float) for s in shifts]
    if len(maps) != m:
        raise ValueError(f"expected {m} shift maps, got {len(maps)}")
    if any(a.shape != maps[0].shape for a in maps):
        raise ValueError("all shift maps must have the same shape; got "
                         f"{[a.shape for a in maps]}")
    return np.stack(maps)


def _whole_number(v) -> bool:
    """True for an integer value given as an int, a NumPy integer or a
    float with no fractional part (not for a bool)."""
    if isinstance(v, (bool, np.bool_)):
        return False
    try:
        return float(v).is_integer()
    except (TypeError, ValueError):
        return False


def _correlation_stack(shift_correlation, m, map_shape):
    """The correlation matrices of the m shift errors at every pixel, as
    a (P, m, m) array, from either form that `MultiModeModel.invert` and
    `ThreeCauseModel.invert` accept (new in v0.16):

    * an array of shape (m, m), one matrix for the whole map, or of
      shape map_shape + (m, m), one matrix per pixel;
    * a dict {(i, j): r} of 0-based mode pairs i != j, each r a scalar
      or a map that broadcasts to map_shape; pairs not named have
      correlation 0. `{(0, 1): fits.shift_correlation}` passes the
      correlation of a joint fit of modes 1 and 2.

    Refuses infinite entries, off-diagonal entries outside [-1, 1], a
    diagonal other than 1 and an asymmetric matrix (both to 1e-12). A
    pixel whose matrix holds a NaN is masked (it comes out NaN).
    Returns (R, masked) with R symmetrized and its diagonal exactly 1.
    """
    P = int(np.prod(map_shape)) if map_shape else 1
    if isinstance(shift_correlation, dict):
        R = np.broadcast_to(np.eye(m), (P, m, m)).copy()
        seen = set()
        for key, val in shift_correlation.items():
            try:
                ok = len(key) == 2 and all(_whole_number(v) for v in key)
            except TypeError:
                ok = False
            if not ok:
                raise ValueError("shift_correlation keys must be pairs "
                                 f"(i, j) of mode indices; got {key!r}")
            i, j = int(key[0]), int(key[1])
            if not (0 <= i < m and 0 <= j < m) or i == j:
                raise ValueError(
                    f"shift_correlation key {key!r} must name two "
                    f"different modes, 0-based indices from 0 to {m - 1}")
            if frozenset((i, j)) in seen:
                raise ValueError(f"the mode pair {key!r} is given twice "
                                 "in shift_correlation")
            seen.add(frozenset((i, j)))
            try:
                v = np.broadcast_to(np.asarray(val, dtype=float),
                                    map_shape).reshape(P)
            except ValueError:
                raise ValueError(
                    f"shift_correlation[{key!r}] of shape "
                    f"{np.shape(val)} does not broadcast to the map "
                    f"shape {map_shape}") from None
            R[:, i, j] = v
            R[:, j, i] = v
    else:
        a = np.asarray(shift_correlation, dtype=float)
        if a.shape == (m, m):
            R = np.broadcast_to(a, (P, m, m)).copy()
        elif a.shape == tuple(map_shape) + (m, m):
            R = a.reshape(P, m, m).copy()
        else:
            raise ValueError(
                f"shift_correlation must be an ({m}, {m}) correlation "
                f"matrix, an array of shape {tuple(map_shape) + (m, m)} "
                "(one matrix per pixel) or a dict {(i, j): r}; got "
                f"shape {a.shape}")
    if np.any(np.isinf(R)):
        raise ValueError("shift_correlation must not be infinite (NaN "
                         "marks a masked pixel)")
    masked = np.any(np.isnan(R), axis=(1, 2))
    Rg = R[~masked]
    off = ~np.eye(m, dtype=bool)
    if np.any(np.abs(Rg[:, off]) > 1.0):
        raise ValueError("shift correlations must lie in [-1, 1]")
    if np.any(np.abs(np.diagonal(Rg, axis1=1, axis2=2) - 1.0) > 1e-12):
        raise ValueError("a correlation matrix must have 1 on its "
                         "diagonal")
    if np.any(np.abs(Rg - Rg.transpose(0, 2, 1)) > 1e-12):
        raise ValueError("a correlation matrix must be symmetric")
    Rg = 0.5 * (Rg + Rg.transpose(0, 2, 1))
    Rg[:, np.eye(m, dtype=bool)] = 1.0
    R[~masked] = Rg
    return R, masked


def _gls_correlated(K, S, sig, R, masked, drop_infinite):
    """Per-pixel generalized least squares with correlated shift errors
    (new in v0.16).

    K (m, c) lever arms; S (m, P) shifts; sig (m, P) sigmas; R (P, m, m)
    correlation matrices and masked (P,) from `_correlation_stack`. The
    shift covariance of pixel p is Sigma_p = D_p R_p D_p with
    D_p = diag(sig[:, p]); with W_p = Sigma_p^-1 the estimate is

        x_p = (K^T W_p K)^-1 K^T W_p s_p,   Cov(x_p) = (K^T W_p K)^-1,
        chi2_p = r_p^T W_p r_p,   r_p = s_p - K x_p,

    the generalized least-squares estimate, which is the minimum-variance
    linear unbiased estimate for correlated errors (the Gauss-Markov
    theorem with the errors whitened), and chi2_p is chi-square distributed
    with m - c degrees of freedom for Gaussian errors. W_p is formed as
    D_p^-1 R_p^-1 D_p^-1, so the matrix inverted is the correlation
    matrix, whose conditioning does not depend on the units of the
    shifts.

    With drop_infinite, an infinite sigma removes that mode at that
    pixel: its correlations with the other modes are set to zero before
    the inverse, and 1/inf = 0 makes its row and column of W_p zero, so
    W_p is the inverse covariance of the other modes alone (the limit
    sigma -> infinity). A pixel with a non-finite shift, a NaN sigma (or
    an infinite one without drop_infinite) or a masked correlation
    matrix comes out NaN. Refuses a correlation matrix that is not
    positive definite at a pixel used, and a singular weighted design.

    Returns (x (P, c), cov (P, c, c), chi2 (P,), n_used (P,), W (P, m, m)),
    with n_used the number of modes with a finite sigma.
    """
    m, c = K.shape
    P = S.shape[1]
    finite_sig = np.isfinite(sig)
    inf_sig = np.isinf(sig)
    usable = (finite_sig | inf_sig) if drop_infinite else finite_sig
    good = (~masked & np.all(np.isfinite(S), axis=0)
            & np.all(usable, axis=0))
    n_used = np.sum(finite_sig, axis=0)
    x = np.full((P, c), np.nan)
    cov = np.full((P, c, c), np.nan)
    chi2_ = np.full(P, np.nan)
    W = np.full((P, m, m), np.nan)
    if not np.any(good):
        return x, cov, chi2_, n_used, W
    singular = ValueError("weighted design matrix is singular at some "
                          "pixels; check the coefficients and sigmas")
    if np.any(n_used[good] < c):
        raise singular
    Rg = R[good]
    dropped = inf_sig[:, good].T                       # (Pg, m)
    if np.any(dropped):
        keep = ~dropped
        Rg = np.where((keep[:, :, None] & keep[:, None, :])
                      | np.eye(m, dtype=bool), Rg, 0.0)
    ev = np.linalg.eigvalsh(Rg)
    if np.any(ev[:, 0] <= 10.0 * m * np.finfo(float).eps * ev[:, -1]):
        raise ValueError(
            "the correlation matrix of the shift errors is not positive "
            "definite at some pixels (a correlation of exactly +1 or -1, "
            "or correlations that contradict one another), so the "
            "weighted fit is undefined there")
    dinv = 1.0 / sig[:, good].T                         # 0 where infinite
    Wg = np.linalg.inv(Rg) * dinv[:, :, None] * dinv[:, None, :]
    sg = S[:, good].T                                   # (Pg, m)
    A = np.einsum("ki,pkl,lj->pij", K, Wg, K)
    b = np.einsum("ki,pkl,pl->pi", K, Wg, sg)
    det = np.linalg.det(A)
    if np.any(det <= 0) or not np.all(np.isfinite(det)):
        raise singular
    xg = np.linalg.solve(A, b[..., None])[..., 0]
    r = sg - xg @ K.T
    x[good] = xg
    cov[good] = np.linalg.inv(A)
    chi2_[good] = np.einsum("pk,pkl,pl->p", r, Wg, r)
    W[good] = Wg
    return x, cov, chi2_, n_used, W


def _chi2_check(resid2, n_used, m, c):
    """chi2_map and p_value of the model check with c causes, or
    (None, None) when m == c. A pixel where only n_used < m modes have
    a finite sigma (an infinite sigma drops a mode there) is checked
    with n_used - c degrees of freedom, and gets NaN when that is zero
    (new in v0.16; before, it was checked with m - c, which made the
    p-value too large, and 1 where no check is possible)."""
    dof = m - c
    if dof <= 0:
        return None, None
    chi2_map = np.array(resid2, dtype=float, copy=True)
    p_value = chi2.sf(resid2, dof)
    fewer = n_used < m
    if np.any(fewer):
        d = n_used[fewer] - c
        has = d > 0
        q = chi2_map[fewer]
        chi2_map[fewer] = np.where(has, q, np.nan)
        p_value[fewer] = np.where(has, chi2.sf(q, np.maximum(d, 1)),
                                  np.nan)
    return chi2_map, p_value


@dataclasses.dataclass
class MultiModeResult:
    """Result of an m-mode generalized-least-squares separation.

    strain, density: estimate maps (units set by the coefficient matrix).
    strain_sigma, density_sigma: propagated one-standard-deviation maps.
    correlation: pixelwise correlation of the two estimates.
    chi2_map: weighted residual sum of squares per pixel (m - 2 dof).
    p_value: probability of a chi2 at least this large under the
        two-cause model; small values flag model violations.
    dof: residual degrees of freedom (m - 2). A pixel where some modes
        have an infinite sigma is checked with fewer: the number of
        modes with a finite sigma, minus 2; where that is 0, its
        chi2_map and p_value are NaN (since v0.16).
    condition_number: of the unweighted lever-arm matrix K, whatever
        sigmas were given (the docstring before v0.13 said "of the
        weighted design matrix" for scalar weights, which the code never
        computed; the value itself is unchanged).
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

    def __init__(self, K, mode_names: list[str] | None = None) -> None:
        self.K = np.asarray(K, dtype=float)
        if self.K.ndim != 2 or self.K.shape[1] != 2 or self.K.shape[0] < 2:
            raise ValueError("K must have shape (m >= 2, 2)")
        if not np.all(np.isfinite(self.K)):
            raise ValueError("K must be finite")
        if not _has_full_column_rank(self.K):    # unit-free, since v0.15
            raise ValueError("lever-arm matrix has rank < 2: the modes "
                             "cannot separate strain from density")
        self.m = self.K.shape[0]
        self.mode_names = (list(mode_names) if mode_names is not None
                          else [f"mode{i+1}" for i in range(self.m)])
        if len(self.mode_names) != self.m:
            raise ValueError("one name per mode required")
        self.condition_number = float(np.linalg.cond(self.K))

    def invert(self, shifts, sigmas=None,
               shift_correlation=None) -> MultiModeResult:
        """Invert shift maps into strain and density with uncertainties.

        shifts: sequence of m arrays (or an array with leading axis m),
        one shift map per mode, all the same shape.  sigmas: matching
        one-standard-deviation uncertainties (scalars or arrays,
        broadcastable to the map shape); omitted means unit weights, in
        which case chi2_map and the sigma maps are still returned but
        mean something only if the shift noise really is unit variance.
        A NaN shift or sigma (a pixel masked by `fit_map`) gives NaN at
        that pixel. An infinite sigma means that mode carries no
        information at that pixel: it gets zero weight, and the model
        check there counts only the other modes (since v0.16; before,
        the p-value of such a pixel was computed with m - 2 degrees of
        freedom, too many, and a pixel left with a single usable mode
        was not always refused).

        shift_correlation: optional correlations between the shift
        errors of different modes (new in v0.16), for example from a
        joint fit of two overlapping peaks
        (`fit_map(..., joint=True).shift_correlation`). Give either an
        (m, m) correlation matrix for the whole map, an array of shape
        map_shape + (m, m) with one matrix per pixel, or a dict
        {(i, j): r} with 0-based mode indices and r a number or a map,
        for example `{(0, 1): fits.shift_correlation}` (pairs not named
        are uncorrelated). It needs `sigmas`. The estimate is then the
        generalized least-squares solution with the full shift
        covariance S = D R D (D the diagonal matrix of the sigmas):
        x = (K^T S^-1 K)^-1 K^T S^-1 s, covariance (K^T S^-1 K)^-1, and
        chi2 = r^T S^-1 r with r = s - K x. A NaN in a matrix marks a
        masked pixel; a matrix that is not positive definite (for
        example a correlation of exactly +1 or -1) is refused. None
        (default) treats the errors as independent, with the results of
        earlier versions unchanged.
        """
        S = _stack_shifts(shifts, self.m)
        map_shape = S.shape[1:]
        if sigmas is None:
            if shift_correlation is not None:
                raise ValueError("shift_correlation needs sigmas")
            sig = np.ones_like(S)
        else:
            sig = _stack_sigmas(sigmas, self.m, map_shape)
            if np.any(sig <= 0):
                raise ValueError("sigmas must be positive")
        # flatten pixels: (m, P)
        P = int(np.prod(map_shape)) if map_shape else 1
        s_flat = S.reshape(self.m, P)
        sig_flat = sig.reshape(self.m, P)
        # modes with a finite sigma at each pixel (an infinite sigma is
        # a zero weight; since v0.16 the model check counts only these)
        n_used = np.sum(np.isfinite(sig_flat), axis=0)
        K = self.K
        if shift_correlation is not None:            # new in v0.16
            R, masked = _correlation_stack(shift_correlation, self.m,
                                           map_shape)
            x, cv, resid2, _, _ = _gls_correlated(K, s_flat, sig_flat, R,
                                                  masked, True)
            strain, density = x[:, 0], x[:, 1]
            var_s, var_d, cov_sd = cv[:, 0, 0], cv[:, 1, 1], cv[:, 0, 1]
        else:
            w_flat = 1.0 / sig_flat ** 2
            # per-pixel normal equations, vectorized over pixels
            # A = K^T W K: components via sums over modes
            a11 = np.einsum("k,kp->p", K[:, 0] ** 2, w_flat)
            a12 = np.einsum("k,kp->p", K[:, 0] * K[:, 1], w_flat)
            a22 = np.einsum("k,kp->p", K[:, 1] ** 2, w_flat)
            b1 = np.einsum("k,kp->p", K[:, 0], w_flat * s_flat)
            b2 = np.einsum("k,kp->p", K[:, 1], w_flat * s_flat)
            det = a11 * a22 - a12 ** 2
            # a pixel with fewer than two finite sigmas has a singular
            # design; its det is zero only up to rounding, and before
            # v0.16 it was refused or not depending on that rounding
            if np.any(det <= 0) or np.any(np.isfinite(det) & (n_used < 2)):
                raise ValueError("weighted design matrix is singular at "
                                 "some pixels; check the coefficients "
                                 "and sigmas")
            strain = (a22 * b1 - a12 * b2) / det
            density = (a11 * b2 - a12 * b1) / det
            # covariance = inverse of the information matrix
            var_s = a22 / det
            var_d = a11 / det
            cov_sd = -a12 / det
            # weighted residuals
            pred = K[:, 0][:, None] * strain[None, :] \
                + K[:, 1][:, None] * density[None, :]
            resid2 = np.einsum("kp,kp->p", w_flat, (s_flat - pred) ** 2)
        with np.errstate(invalid="ignore"):
            corr = cov_sd / np.sqrt(var_s * var_d)
        dof = self.m - 2
        chi2_map, p_value = _chi2_check(resid2, n_used, self.m, 2)
        if chi2_map is not None:
            chi2_map = chi2_map.reshape(map_shape)
            p_value = p_value.reshape(map_shape)
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


def compare_mode_sets(K, sigmas, mode_names=None,
                      subset_size: int = 2) -> list[dict]:
    """Rank all mode subsets of a given size by delivered uncertainty.

    K: (m, 2) lever arms; sigmas: per-mode shift uncertainties (scalars);
    mode_names: optional, one label per row of K (a list of another
    length is refused since v0.16);
    subset_size: modes per subset, a whole number from 2 to m (refused
    otherwise since v0.14; before, it returned an empty list); returns a list of dicts sorted best-first by A-optimality (the trace
    of the estimate covariance, i.e. var_strain + var_density), each with
    the subset's names, indices, var_strain, var_density, trace, and
    d_optimality = det of the information matrix (larger is better).
    """
    K = np.asarray(K, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    if K.ndim != 2 or K.shape[1] != 2:
        raise ValueError("K must have shape (m, 2)")
    if not np.all(np.isfinite(K)):
        raise ValueError("K must be finite")
    m = K.shape[0]
    if sigmas.shape != (m,):
        raise ValueError("one sigma per mode required")
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("sigmas must be finite and positive")
    # before v0.14 a subset size below 2 or above m returned an empty
    # list without saying why (one mode cannot separate two causes)
    if (isinstance(subset_size, bool) or not float(subset_size).is_integer()
            or not 2 <= int(subset_size) <= m):
        raise ValueError(f"subset_size must be a whole number from 2 to "
                         f"the number of modes ({m}); got {subset_size!r}")
    subset_size = int(subset_size)
    names = (list(mode_names) if mode_names is not None
             else [f"mode{i+1}" for i in range(m)])
    # before v0.16 a longer list was accepted and its first m names were
    # used, which mislabels every subset if the list does not match the
    # rows of K; a shorter one failed with an IndexError
    if len(names) != m:
        raise ValueError(f"one name per mode required: K has {m} rows, "
                         f"mode_names has {len(names)}")
    out = []
    for idx in itertools.combinations(range(m), subset_size):
        Ks = K[list(idx)]
        if not _has_full_column_rank(Ks):        # unit-free, since v0.15
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
