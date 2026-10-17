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
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.stats import chi2 as _chi2

from .calibration import _spans_plane
from .core import _check_shift_correlation

__all__ = ["RegionEstimate", "region_mean", "region_contrast"]


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
    if ref_sigma is None:
        c_ref = np.zeros((2, 2))
    else:
        u = np.asarray(ref_sigma, dtype=float)
        if u.shape != (2,) or not np.all(np.isfinite(u)) or np.any(u < 0):
            raise ValueError("ref_sigma must be two finite, non-negative "
                             "numbers (u1, u2) in cm^-1")
        c_ref = Kinv @ np.diag(u ** 2) @ Kinv.T
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
