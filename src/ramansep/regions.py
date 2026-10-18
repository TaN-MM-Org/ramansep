"""Average strain and density over a region of a map, and compare two
regions, with every source of error accounted for (new in v0.14).

A map is usually summarized by regions: the mean strain of a flake, the
charge at its edge against its interior. Taking `numpy.mean` of the
strain map gives the right value but no honest error bar, because the
errors of a region mean come from three sources that behave
differently:

* shift noise: independent from pixel to pixel, so it averages down
  (the variance of a mean of N pixels is the sum of the pixel
  variances over N^2);
* the lever-arm matrix K, when it comes from a calibration with error
  bars (`calibrate_lever_arms`): one K serves every pixel, so this part
  does not average down at all;
* the reference frequencies ref1, ref2 (the zero points of the
  shifts): an error in them shifts every pixel alike, so it does not
  average down either.

`region_mean` reports the mean and each part separately, plus a check
of whether the pixels agree with one another within their error bars.
`region_contrast` reports the difference between two regions (for
example edge minus interior). In a difference the reference-frequency
error cancels exactly, because both regions subtract the same
reference, and the calibration error is proportional to the difference
itself rather than to the absolute values, so a contrast is often known
far better than either mean.

The algebra, with the two-mode inverse x = K^-1 y (y the two shifts):

* the region mean is xbar = K^-1 ybar, with ybar the plain (unweighted)
  mean of the shifts over the N pixels used; it equals the mean of the
  per-pixel strain and density maps of `SeparationModel.invert`;
* shift part: Cov = K^-1 S K^-T with S = (1/N^2) sum_p S_p and S_p the
  2 x 2 covariance of the shifts at pixel p (sigma1^2, sigma2^2 and,
  with `shift_correlation`, r sigma1 sigma2 off the diagonal);
* reference part: Cov = K^-1 diag(u1^2, u2^2) K^-T for reference
  uncertainties (u1, u2);
* calibration part: from the exact derivative dxbar/dK_mj =
  -K^-1 E_mj xbar (the identity `separation_with_calibration` uses),
  Cov = sum_m (xbar^T C_m xbar) k_m k_m^T, with C_m the 2 x 2
  calibration covariance of mode m (modes are calibrated independently)
  and k_m column m of K^-1; first order in the calibration errors.

These are the law of propagation of uncertainty (JCGM 100:2008, the
GUM, 5.2.2, Eq. (13), which includes correlated inputs;
doi:10.59161/JCGM100-2008E) applied to independent error sources; the
total covariance is their sum. A contrast d = xbar_A - xbar_B is linear
in the same way: its shift part sums the two regions' parts (the
regions must not share pixels), its calibration part uses d in place
of xbar, and its reference part is zero.

Homogeneity check (`region_mean`): under the hypothesis that every
pixel in the region has the same strain and density, the pixel shift
vectors y_p scatter about a common value with their own covariances
S_p, and

    chi2 = sum_p (y_p - y_w)^T S_p^-1 (y_p - y_w),

with y_w the inverse-covariance-weighted mean, is chi-square
distributed with 2 (N - 1) degrees of freedom for Gaussian errors.
Because x = K^-1 y is linear and invertible, this is the same number
computed in (strain, density) space with the per-pixel covariances,
so the check does not depend on K at all. A small p-value means the
region is not uniform (or the shift error bars are too small): the
mean is then still the mean of those pixels, but the pixel-to-pixel
spread (`strain_scatter`, `density_scatter`) is real variation, not
noise.

The tests check each part against an independent calculation (the
mean and per-pixel covariances of `SeparationModel.invert`,
`separation_with_calibration`, finite differences of a re-solve with
perturbed K) and the totals against seeded Monte Carlo runs that
repeat the shift noise, the reference error and the whole calibration.

Profiles (new in v0.15): `region_profile` treats every bin of a
coordinate (for example the distance from an edge) as a region and
also returns the covariance BETWEEN bins. Different bins hold different
pixels, so their shift noise is independent, but they share the
lever-arm matrix and the reference frequencies: bins i and j have the
calibration covariance sum_m (x_i^T C_m x_j) k_m k_m^T (the same
first-order derivative as above, applied to both bins) and the
reference covariance K^-1 diag(u1^2, u2^2) K^-T. A linear combination
sum_k c_k . x_k of the bins (`RegionProfile.combination`) then has the
shift variance sum_k c_k^T S_k c_k, the reference variance u^T B u with
u = sum_k c_k (zero when the weights of each quantity sum to zero, as
in any difference), and the calibration variance sum_m g_m^T C_m g_m
with g_m = sum_k (c_k . k_m) x_k. `integrated_excess` is the
combination sum_k w_k (q_k - q_bulk): across a straight edge, with the
bins tiling the strip next to it, the excess of q per unit length of
edge. The tests check every part against an independent calculation
(per-pixel inversions, finite differences of a re-solve with perturbed
K, a pixel-level linear form) and the totals against seeded Monte Carlo
runs that repeat the calibration, the map noise and the reference
error.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.stats import chi2 as _chi2

from .calibration import _spans_plane
from .core import _check_shift_correlation

__all__ = ["RegionEstimate", "region_mean", "region_contrast",
           "RegionProfile", "LinearEstimate", "region_profile"]


@dataclasses.dataclass
class RegionEstimate:
    """Mean (or difference of means) of strain and density over map
    regions, with its uncertainty split by source.

    strain, density : the region mean (`region_mean`) or the difference
        region A minus region B (`region_contrast`), in the units of
        the lever arms.
    strain_sigma, density_sigma : total 1-sigma uncertainty, the
        quadrature sum of the three parts below.
    correlation : correlation of the strain and density estimates
        (from the total covariance).
    covariance : (2, 2) total covariance of (strain, density).
    strain_sigma_shifts, density_sigma_shifts : part from the shift
        noise of the pixels (averages down over pixels).
    strain_sigma_calibration, density_sigma_calibration : part from the
        uncertainty of the lever arms (0 for exact `ModeCoefficients`).
    strain_sigma_reference, density_sigma_reference : part from the
        uncertainty of the reference frequencies (0 when `ref_sigma` is
        not given, and always 0 for a contrast, where it cancels).
    n_pixels : pixels used (a tuple (n_A, n_B) for a contrast).
    n_excluded : pixels inside the region(s) left out because a shift,
        sigma or correlation there is NaN (masked by `fit_map`).
    chi2, dof, p_value : homogeneity check of `region_mean` (see the
        module docstring); None for a contrast or a single pixel.
    strain_scatter, density_scatter : sample standard deviation
        (ddof=1) of the per-pixel strain and density over the region;
        None for a contrast or a single pixel.
    """

    strain: float
    density: float
    strain_sigma: float
    density_sigma: float
    correlation: float
    covariance: np.ndarray
    strain_sigma_shifts: float
    density_sigma_shifts: float
    strain_sigma_calibration: float
    density_sigma_calibration: float
    strain_sigma_reference: float
    density_sigma_reference: float
    n_pixels: int | tuple
    n_excluded: int | tuple
    chi2: float | None = None
    dof: int | None = None
    p_value: float | None = None
    strain_scatter: float | None = None
    density_scatter: float | None = None


def _lever_arms(coefficients):
    """K (2, 2) and the per-mode calibration covariance (2, 2, 2) or
    None, from a ModeCoefficients or a two-mode CalibrationResult."""
    if hasattr(coefficients, "cov") and hasattr(coefficients, "K"):
        K = np.asarray(coefficients.K, dtype=float)
        cov = np.asarray(coefficients.cov, dtype=float)
        if K.shape != (2, 2) or cov.shape != (2, 2, 2):
            raise ValueError("a calibration passed here must carry "
                             "exactly two modes (K (2, 2), cov (2, 2, 2))")
        if not np.all(np.isfinite(cov)):
            raise ValueError("calibration cov must be finite")
    elif hasattr(coefficients, "matrix"):
        K = np.asarray(coefficients.matrix(), dtype=float)
        cov = None
    else:
        raise ValueError("coefficients must be a ModeCoefficients or a "
                         "two-mode CalibrationResult")
    if K.shape != (2, 2) or not np.all(np.isfinite(K)):
        raise ValueError("the lever-arm matrix must be a finite 2 x 2 "
                         "matrix")
    if not _spans_plane(K):                       # unit-free rank test
        raise ValueError("the lever-arm matrix is singular: the two "
                         "modes cannot separate strain from density")
    return K, cov


def _pixels(dw1, dw2, sigma1, sigma2, shift_correlation, mask, name):
    """Stack the shifts, sigmas and correlations of the usable pixels of
    one region; returns (y (N, 2), s (N, 2), r (N,), n_excluded)."""
    d1 = np.asarray(dw1, dtype=float)
    d2 = np.asarray(dw2, dtype=float)
    if d1.shape != d2.shape:
        raise ValueError("the two shift maps must have the same shape; "
                         f"got {d1.shape} and {d2.shape}")
    shape = d1.shape
    try:
        s1 = np.broadcast_to(np.asarray(sigma1, dtype=float), shape)
        s2 = np.broadcast_to(np.asarray(sigma2, dtype=float), shape)
    except ValueError:
        raise ValueError("sigma1 and sigma2 must broadcast to the map "
                         f"shape {shape}") from None
    if shift_correlation is None:
        rho = np.zeros(shape)
    else:
        rho = _check_shift_correlation(shift_correlation, shape)
    m = np.asarray(mask)
    if m.dtype != bool or m.shape != shape:
        raise ValueError(f"{name} must be a boolean array of the map "
                         f"shape {shape}")
    usable = (np.isfinite(d1) & np.isfinite(d2) & np.isfinite(s1)
              & np.isfinite(s2) & np.isfinite(rho))
    use = m & usable
    if not np.any(use):
        raise ValueError(f"{name} contains no pixel with finite shifts "
                         "and sigmas")
    s = np.column_stack([s1[use], s2[use]])
    if np.any(s <= 0.0):
        raise ValueError("shift sigmas must be positive in the region")
    y = np.column_stack([d1[use], d2[use]])
    return y, s, rho[use], int(np.sum(m & ~usable))


def _mean_shift_cov(s, r):
    """(1/N^2) sum_p S_p for per-pixel shift sigmas s (N, 2) and
    correlations r (N,)."""
    N = s.shape[0]
    c12 = np.sum(r * s[:, 0] * s[:, 1])
    return np.array([[np.sum(s[:, 0] ** 2), c12],
                     [c12, np.sum(s[:, 1] ** 2)]]) / N ** 2


def _calibration_cov(Kinv, cov, x):
    """sum_m (x^T C_m x) k_m k_m^T: first-order covariance of K^-1 y
    from the per-mode lever-arm covariances C_m."""
    out = np.zeros((2, 2))
    if cov is None:
        return out
    for m in range(2):
        out += float(x @ cov[m] @ x) * np.outer(Kinv[:, m], Kinv[:, m])
    return out


def _reference_cov(Kinv, ref_sigma):
    """K^-1 diag(u1^2, u2^2) K^-T for the reference-frequency errors
    ref_sigma = (u1, u2), or zeros for None; refuses anything but two
    finite, non-negative numbers (moved out of `region_mean` unchanged
    in v0.15)."""
    if ref_sigma is None:
        return np.zeros((2, 2))
    u = np.asarray(ref_sigma, dtype=float)
    if u.shape != (2,) or not np.all(np.isfinite(u)) or np.any(u < 0):
        raise ValueError("ref_sigma must be two finite, non-negative "
                         "numbers (u1, u2) in cm^-1")
    return Kinv @ np.diag(u ** 2) @ Kinv.T


def _homogeneity(y, s, r, Kinv):
    """The homogeneity check of `region_mean` for the usable pixels
    (y, s, r) of one region: dict(chi2, dof, p_value, strain_scatter,
    density_scatter), or {} for a single pixel. Moved here from
    `region_mean` unchanged in v0.15, so that `region_profile` runs the
    same check on every bin."""
    N = y.shape[0]
    extra = {}
    if N >= 2:
        # weighted mean and chi2 in shift space, per-pixel 2 x 2 inverse
        c12 = r * s[:, 0] * s[:, 1]
        det = (s[:, 0] * s[:, 1]) ** 2 - c12 ** 2
        if np.any(det <= 0.0):
            raise ValueError("a shift correlation of +1 or -1 leaves the "
                             "homogeneity check undefined; use |r| < 1")
        Winv = np.empty((N, 2, 2))
        Winv[:, 0, 0] = s[:, 1] ** 2 / det
        Winv[:, 1, 1] = s[:, 0] ** 2 / det
        Winv[:, 0, 1] = Winv[:, 1, 0] = -c12 / det
        A = Winv.sum(axis=0)
        yw = np.linalg.solve(A, np.einsum("pij,pj->i", Winv, y))
        e = y - yw
        q = float(np.einsum("pi,pij,pj->", e, Winv, e))
        dof = 2 * (N - 1)
        xs = y @ Kinv.T                             # per-pixel estimates
        extra = dict(chi2=q, dof=dof, p_value=float(_chi2.sf(q, dof)),
                     strain_scatter=float(np.std(xs[:, 0], ddof=1)),
                     density_scatter=float(np.std(xs[:, 1], ddof=1)))
    return extra


def _estimate(x, parts, n_pixels, n_excluded, **extra):
    total = parts[0] + parts[1] + parts[2]
    sd = [np.sqrt(np.maximum(np.diag(c), 0.0)) for c in (total,) + parts]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = float(total[0, 1] / (sd[0][0] * sd[0][1]))
    return RegionEstimate(
        strain=float(x[0]), density=float(x[1]),
        strain_sigma=float(sd[0][0]), density_sigma=float(sd[0][1]),
        correlation=corr, covariance=total,
        strain_sigma_shifts=float(sd[1][0]),
        density_sigma_shifts=float(sd[1][1]),
        strain_sigma_calibration=float(sd[2][0]),
        density_sigma_calibration=float(sd[2][1]),
        strain_sigma_reference=float(sd[3][0]),
        density_sigma_reference=float(sd[3][1]),
        n_pixels=n_pixels, n_excluded=n_excluded, **extra)


def region_mean(coefficients, dw1, dw2, sigma1, sigma2, mask=None, *,
                shift_correlation=None,
                ref_sigma: tuple[float, float] | None = None
                ) -> RegionEstimate:
    """Mean strain and density over a region of a map, with its error
    split into shift-noise, calibration and reference parts.

    coefficients : a `ModeCoefficients` (lever arms taken as exact) or
        a two-mode `CalibrationResult` from `calibrate_lever_arms`
        (lever arms with their covariance, which adds the calibration
        part).
    dw1, dw2 : shift maps (cm^-1), as for `SeparationModel.invert`,
        e.g. `fits.dw1`, `fits.dw2` from `fit_map`.
    sigma1, sigma2 : their 1-sigma errors (maps or scalars; required,
        positive in the pixels used).
    mask : boolean map of the region; None uses the whole map. Pixels
        whose shift, sigma or correlation is NaN (masked by `fit_map`)
        are left out and counted in `n_excluded`.
    shift_correlation : optional correlation of the two shift errors
        (scalar or map), e.g. `fits.shift_correlation` after
        `fit_map(..., joint=True)`.
    ref_sigma : optional (u1, u2), the 1-sigma uncertainties of the
        reference frequencies ref1, ref2 (cm^-1) from which the shifts
        were measured. They are common to all pixels and do not
        average down.

    Returns a `RegionEstimate`. Its `strain` and `density` equal the
    mean of the per-pixel `SeparationModel.invert` maps over the pixels
    used. The homogeneity check (`chi2`, `dof`, `p_value`) needs at
    least two pixels.
    """
    K, cov = _lever_arms(coefficients)
    shape = np.shape(np.asarray(dw1))
    if mask is None:
        mask = np.ones(shape, dtype=bool)
    y, s, r, n_exc = _pixels(dw1, dw2, sigma1, sigma2,
                                shift_correlation, mask, "mask")
    N = y.shape[0]
    Kinv = np.linalg.inv(K)
    ybar = y.mean(axis=0)
    xbar = Kinv @ ybar
    c_shift = Kinv @ _mean_shift_cov(s, r) @ Kinv.T
    c_cal = _calibration_cov(Kinv, cov, xbar)
    c_ref = _reference_cov(Kinv, ref_sigma)
    extra = _homogeneity(y, s, r, Kinv)
    return _estimate(xbar, (c_shift, c_cal, c_ref), N, n_exc, **extra)


def region_contrast(coefficients, dw1, dw2, sigma1, sigma2, mask_a,
                    mask_b, *, shift_correlation=None) -> RegionEstimate:
    """Difference of the mean strain and density of region A minus
    region B, with its error split into shift-noise and calibration
    parts.

    Inputs as for `region_mean`, with two boolean region maps that must
    not share a pixel. The reference-frequency error cancels exactly in
    the difference (both regions subtract the same references), so its
    part is 0 and there is no `ref_sigma`. The calibration part is
    proportional to the difference itself, so it is small for a small
    contrast even when each mean carries a large calibration error.

    Returns a `RegionEstimate` with `n_pixels = (n_A, n_B)`; the
    homogeneity check and the scatter are left None (run `region_mean`
    on each region for those).
    """
    K, cov = _lever_arms(coefficients)
    ma = np.asarray(mask_a)
    mb = np.asarray(mask_b)
    if ma.shape == mb.shape and ma.dtype == bool and mb.dtype == bool \
            and np.any(ma & mb):
        raise ValueError("the two regions share pixels; a contrast needs "
                         "disjoint regions")
    ya, sa, ra, na_exc = _pixels(dw1, dw2, sigma1, sigma2,
                                    shift_correlation, ma, "mask_a")
    yb, sb, rb, nb_exc = _pixels(dw1, dw2, sigma1, sigma2,
                                    shift_correlation, mb, "mask_b")
    Kinv = np.linalg.inv(K)
    d = Kinv @ (ya.mean(axis=0) - yb.mean(axis=0))
    c_shift = Kinv @ (_mean_shift_cov(sa, ra)
                      + _mean_shift_cov(sb, rb)) @ Kinv.T
    c_cal = _calibration_cov(Kinv, cov, d)
    return _estimate(d, (c_shift, c_cal, np.zeros((2, 2))),
                     (ya.shape[0], yb.shape[0]), (na_exc, nb_exc))


# ----------------------------------------------------------------------
# Profiles: many regions at once, with their joint covariance (v0.15)


@dataclasses.dataclass
class LinearEstimate:
    """A linear combination of the bin values of a `RegionProfile`
    (for example the excess charge per unit length of an edge), with
    its 1-sigma error split by source, as in `RegionEstimate`.

    value : the combination, in the units of the weights times the
        units of the quantity.
    sigma : total 1-sigma error, the quadrature sum of the three parts.
    sigma_shifts, sigma_calibration, sigma_reference : the parts from
        the shift noise, the lever-arm calibration and the reference
        frequencies.
    """

    value: float
    sigma: float
    sigma_shifts: float
    sigma_calibration: float
    sigma_reference: float


@dataclasses.dataclass
class RegionProfile:
    """Mean strain and density in each bin of a coordinate (for example
    the distance from an edge), with the joint covariance of all bins
    (new in v0.15). Made by `region_profile`.

    edges : (R + 1,) the bin edges along the coordinate.
    position : (R,) mean coordinate of the pixels used in each bin.
    strain, density : (R,) the mean of each bin, as `region_mean` gives
        it for that bin alone.
    strain_sigma, density_sigma : (R,) total 1-sigma errors.
    covariance : (R, 2, R, 2) total covariance; covariance[i, a, j, b]
        is the covariance of quantity a of bin i with quantity b of bin
        j (a, b = 0 for strain, 1 for density). Bins are correlated
        through the calibration and the reference frequencies, which
        all bins share.
    covariance_shifts, covariance_calibration, covariance_reference :
        (R, 2, R, 2) the three parts of `covariance`. The shift part is
        zero between different bins (they hold different pixels); the
        reference part is the same 2 x 2 block for every pair of bins.
    n_pixels, n_excluded : (R,) pixels used in each bin, and pixels of
        the bin left out because a shift, sigma or correlation there is
        NaN.
    chi2, dof, p_value : (R,) the homogeneity check of `region_mean`
        for each bin (NaN, 0 and NaN for a bin with one pixel).

    `combination`, `integrated_excess` and `contrast` turn the profile
    into single numbers with the full error budget.
    """

    edges: np.ndarray
    position: np.ndarray
    strain: np.ndarray
    density: np.ndarray
    strain_sigma: np.ndarray
    density_sigma: np.ndarray
    covariance: np.ndarray
    covariance_shifts: np.ndarray
    covariance_calibration: np.ndarray
    covariance_reference: np.ndarray
    n_pixels: np.ndarray
    n_excluded: np.ndarray
    chi2: np.ndarray
    dof: np.ndarray
    p_value: np.ndarray
    _Kinv: np.ndarray = dataclasses.field(default=None, repr=False)
    _cal: np.ndarray = dataclasses.field(default=None, repr=False)
    _shift_blocks: np.ndarray = dataclasses.field(default=None,
                                                  repr=False)
    _ref_block: np.ndarray = dataclasses.field(default=None, repr=False)

    def _values(self):
        return np.column_stack([self.strain, self.density])   # (R, 2)

    def combination(self, strain_weights=None,
                    density_weights=None) -> LinearEstimate:
        """sum_k (a_k strain_k + b_k density_k) for weights a = strain_
        weights and b = density_weights (each (R,) or None for zeros),
        with its error by source.

        The parts are computed from their exact forms rather than from
        the (R, 2, R, 2) arrays, so that a cancellation stays exact: the
        shift part is sum_k c_k^T S_k c_k with c_k = (a_k, b_k) and S_k
        the shift block of bin k; the reference part is u^T B u with
        u = sum_k c_k and B the common reference block, which is zero
        when the weights of each quantity sum to zero; the calibration
        part is sum_m g_m^T C_m g_m with g_m = sum_k (c_k . k_m) x_k,
        the first-order derivative of the combination with respect to
        row m of K (k_m is column m of K^-1, x_k the bin values).
        """
        R = self.strain.size
        c = np.zeros((R, 2))
        for col, w in ((0, strain_weights), (1, density_weights)):
            if w is not None:
                w = np.asarray(w, dtype=float)
                if w.shape != (R,) or not np.all(np.isfinite(w)):
                    raise ValueError(f"weights must be {R} finite "
                                     "numbers, one per bin")
                c[:, col] = w
        x = self._values()
        value = float(np.sum(c * x))
        v_shift = float(np.einsum("ka,kab,kb->", c, self._shift_blocks,
                                  c))
        u = np.array([math.fsum(c[:, 0]), math.fsum(c[:, 1])])
        v_ref = float(u @ self._ref_block @ u)
        v_cal = 0.0
        if self._cal is not None:
            for m in range(2):
                g = (c @ self._Kinv[:, m]) @ x            # (2,)
                v_cal += float(g @ self._cal[m] @ g)
        parts = [np.sqrt(max(v, 0.0)) for v in (v_shift, v_cal, v_ref)]
        return LinearEstimate(
            value=value, sigma=float(np.sqrt(max(v_shift + v_cal + v_ref,
                                                 0.0))),
            sigma_shifts=float(parts[0]),
            sigma_calibration=float(parts[1]),
            sigma_reference=float(parts[2]))

    def integrated_excess(self, bulk: int, quantity: str = "density",
                          widths=None) -> LinearEstimate:
        """sum over the bins k other than `bulk` of
        widths_k (q_k - q_bulk), with q the strain or the density.

        For a profile across a straight edge, with the coordinate the
        distance from the edge and bins that tile the strip next to it,
        this is the excess of q per unit length of edge (for the
        density: the extra carriers per unit length of edge, in density
        units times coordinate units; for example 1e13 cm^-2 times
        micrometers is 1e9 carriers per cm of edge). `bulk` is the index
        of the bin taken as the value far from the edge; `widths`
        defaults to the bin widths, numpy.diff(edges). The reference
        error cancels (each term is a difference), so `sigma_reference`
        is zero up to rounding.
        """
        R = self.strain.size
        if quantity not in ("strain", "density"):
            raise ValueError('quantity must be "strain" or "density"')
        if isinstance(bulk, bool) or not float(bulk).is_integer() \
                or not -R <= int(bulk) < R:
            raise ValueError(f"bulk must be a bin index from 0 to {R - 1}")
        bulk = int(bulk) % R
        if R < 2:                                  # pragma: no cover
            raise ValueError("an excess needs at least two bins")
        w = (np.diff(self.edges) if widths is None
             else np.asarray(widths, dtype=float))
        if w.shape != (R,) or not np.all(np.isfinite(w)):
            raise ValueError(f"widths must be {R} finite numbers, one "
                             "per bin")
        weights = w.copy()
        weights[bulk] = 0.0
        weights[bulk] = -math.fsum(weights)
        if quantity == "density":
            return self.combination(density_weights=weights)
        return self.combination(strain_weights=weights)

    def contrast(self, i: int, j: int) -> RegionEstimate:
        """Bin i minus bin j, as `region_contrast` gives it for the two
        bins' pixels: a `RegionEstimate` whose reference part is zero
        (it cancels exactly) and whose calibration part scales with the
        difference."""
        R = self.strain.size
        idx = []
        for k in (i, j):
            if isinstance(k, bool) or not float(k).is_integer() \
                    or not -R <= int(k) < R:
                raise ValueError(f"bin indices must be from 0 to {R - 1}")
            idx.append(int(k) % R)
        i, j = idx
        if i == j:
            raise ValueError("a contrast needs two different bins")
        x = self._values()
        d = x[i] - x[j]
        c_shift = self._shift_blocks[i] + self._shift_blocks[j]
        c_cal = _calibration_cov(self._Kinv, self._cal, d)
        return _estimate(d, (c_shift, c_cal, np.zeros((2, 2))),
                         (int(self.n_pixels[i]), int(self.n_pixels[j])),
                         (int(self.n_excluded[i]),
                          int(self.n_excluded[j])))


def region_profile(coefficients, dw1, dw2, sigma1, sigma2, coordinate,
                   bins, *, mask=None, shift_correlation=None,
                   ref_sigma: tuple[float, float] | None = None
                   ) -> RegionProfile:
    """Mean strain and density in bins of a coordinate, for example the
    distance from an edge, with the joint covariance of all bins (new
    in v0.15).

    coefficients, dw1, dw2, sigma1, sigma2, shift_correlation,
    ref_sigma : as for `region_mean`.
    coordinate : a map of the map shape giving the coordinate of every
        pixel (for example its distance from the nearest edge, in
        micrometers or in pixels). Pixels whose coordinate is NaN, or
        outside the bins, are not used.
    bins : the R + 1 bin edges, finite and strictly increasing. Bin k
        holds the pixels with edges[k] <= coordinate < edges[k + 1];
        the last bin also holds coordinate == edges[-1] (the
        convention of `numpy.histogram`).
    mask : optional boolean map; only pixels where it is True are used
        (for example the pixels on the flake).

    Each bin is a region as in `region_mean`: its values are the mean
    of the per-pixel `SeparationModel.invert` maps over its usable
    pixels, and its error splits into shift noise, calibration and
    reference parts. What a list of `region_mean` calls cannot give is
    the covariance BETWEEN bins: one lever-arm matrix and one pair of
    reference frequencies serve all bins, so their errors move all bins
    together. Bin i and bin j have the calibration covariance
    sum_m (x_i^T C_m x_j) k_m k_m^T (the first-order derivative
    -K^-1 E_mj x of each bin, as in `region_mean`), the reference
    covariance K^-1 diag(u1^2, u2^2) K^-T, and no shift covariance
    (different pixels). With these, `RegionProfile.combination`,
    `integrated_excess` (the excess per unit length of an edge) and
    `contrast` carry the full error budget, and the reference error
    cancels in every difference.

    Refuses: bins that are not at least two finite, strictly
    increasing edges; a coordinate map of another shape; a bin without
    a usable pixel (choose wider bins); and everything `region_mean`
    refuses.
    """
    K, cov = _lever_arms(coefficients)
    shape = np.shape(np.asarray(dw1))
    edges = np.asarray(bins, dtype=float).ravel()
    if edges.size < 2 or not np.all(np.isfinite(edges)) \
            or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bins must be at least two finite, strictly "
                         "increasing bin edges")
    coord = np.asarray(coordinate, dtype=float)
    if coord.shape != shape:
        raise ValueError(f"coordinate must be a map of the map shape "
                         f"{shape}; got {coord.shape}")
    if mask is None:
        mask = np.ones(shape, dtype=bool)
    m = np.asarray(mask)
    if m.dtype != bool or m.shape != shape:
        raise ValueError(f"mask must be a boolean array of the map "
                         f"shape {shape}")
    R = edges.size - 1
    finite = np.isfinite(coord)
    k = np.full(shape, -1)
    k[finite] = np.searchsorted(edges, coord[finite], side="right") - 1
    k[finite & (coord == edges[-1])] = R - 1
    Kinv = np.linalg.inv(K)
    ref_block = _reference_cov(Kinv, ref_sigma)
    x = np.empty((R, 2))
    blocks = np.empty((R, 2, 2))
    pos = np.empty(R)
    n_pix = np.empty(R, dtype=int)
    n_exc = np.empty(R, dtype=int)
    chi2 = np.full(R, np.nan)
    dof = np.zeros(R, dtype=int)
    pval = np.full(R, np.nan)
    for b in range(R):
        in_bin = m & (k == b)
        name = (f"bin {b} [{edges[b]:g}, {edges[b + 1]:g}"
                + ("]" if b == R - 1 else ")"))
        if not np.any(in_bin):
            raise ValueError(f"{name} holds no pixel; choose wider bins")
        y, s, r, n_exc[b] = _pixels(dw1, dw2, sigma1, sigma2,
                                    shift_correlation, in_bin, name)
        use = in_bin & _usable(dw1, dw2, sigma1, sigma2,
                               shift_correlation, shape)
        pos[b] = float(np.mean(coord[use]))
        n_pix[b] = y.shape[0]
        x[b] = Kinv @ y.mean(axis=0)
        blocks[b] = Kinv @ _mean_shift_cov(s, r) @ Kinv.T
        h = _homogeneity(y, s, r, Kinv)
        if h:
            chi2[b], dof[b], pval[b] = h["chi2"], h["dof"], h["p_value"]
    c_shift = np.zeros((R, 2, R, 2))
    for b in range(R):
        c_shift[b, :, b, :] = blocks[b]
    c_cal = np.zeros((R, 2, R, 2))
    if cov is not None:
        for mm in range(2):
            q = np.einsum("ia,ab,jb->ij", x, cov[mm], x)      # (R, R)
            kk = np.outer(Kinv[:, mm], Kinv[:, mm])
            c_cal += q[:, None, :, None] * kk[None, :, None, :]
    c_ref = np.broadcast_to(ref_block[None, :, None, :],
                            (R, 2, R, 2)).copy()
    total = c_shift + c_cal + c_ref
    diag = np.einsum("iaia->ia", total)
    sd = np.sqrt(np.maximum(diag, 0.0))
    return RegionProfile(
        edges=edges, position=pos, strain=x[:, 0], density=x[:, 1],
        strain_sigma=sd[:, 0], density_sigma=sd[:, 1], covariance=total,
        covariance_shifts=c_shift, covariance_calibration=c_cal,
        covariance_reference=c_ref, n_pixels=n_pix, n_excluded=n_exc,
        chi2=chi2, dof=dof, p_value=pval, _Kinv=Kinv, _cal=cov,
        _shift_blocks=blocks, _ref_block=ref_block)


def _usable(dw1, dw2, sigma1, sigma2, shift_correlation, shape):
    """The pixels `_pixels` would use (finite shifts, sigmas and
    correlation), as a boolean map."""
    d1 = np.asarray(dw1, dtype=float)
    d2 = np.asarray(dw2, dtype=float)
    s1 = np.broadcast_to(np.asarray(sigma1, dtype=float), shape)
    s2 = np.broadcast_to(np.asarray(sigma2, dtype=float), shape)
    rho = (np.zeros(shape) if shift_correlation is None
           else _check_shift_correlation(shift_correlation, shape))
    return (np.isfinite(d1) & np.isfinite(d2) & np.isfinite(s1)
            & np.isfinite(s2) & np.isfinite(rho))
