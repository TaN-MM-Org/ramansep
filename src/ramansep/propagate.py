"""Carry the calibration's own uncertainty into the maps (new in
v0.11).

Until now every inversion in this package treated the lever-arm
matrix K as exact, even when K came from `calibrate_lever_arms` with
error bars of its own. That is a stated gap closed here for the
two-mode case: `separation_with_calibration` inverts the shift maps
AND propagates, to first order, the calibration covariance into the
strain and density error bars, reporting the two contributions
separately so you can see whether your uncertainty budget is limited
by the spectra or by the calibration.

The propagation leans on an exact identity of the two-mode inverse
x = K^-1 y: the derivative of the solution with respect to one
lever-arm entry is

    dx / dK_mj = -K^-1 E_mj x,

with E_mj the unit matrix carrying a 1 in entry (m, j) -- exact, not
approximate, and checked in the tests against finite differences of
the actual re-solve (two independent code paths). The per-mode
calibration covariances from `calibrate_lever_arms` are independent
between modes (each mode's arms are fitted from its own shifts), and
the first-order variance sums their contributions; seeded Monte
Carlo over both noise sources confirms the combined error bars.

Since v0.12, `multimode_with_calibration` does the same for m >= 2
modes and the weighted least-squares estimate of `MultiModeModel`,
using the exact derivative of that estimate (which, unlike the
two-mode inverse, involves the residual; see its docstring).

Honest limits, stated plainly: the propagation is first order in the
calibration errors (exact in the limit of a well-determined
calibration, the regime a calibration is for). The calibration part
comes from one lever-arm matrix shared by every pixel, so it is a
common (systematic) error of the whole map: unlike shift noise, it
does not average away over many pixels.
"""
from __future__ import annotations

import numpy as np

from .core import SeparationModel

__all__ = ["separation_with_calibration", "multimode_with_calibration"]


def separation_with_calibration(calibration, dw1, dw2, sigma1=None,
                                sigma2=None):
    """Two-mode inversion with the calibration's uncertainty included.

    calibration : a `CalibrationResult` from `calibrate_lever_arms`
        with exactly two modes (K is (2, 2), cov is (2, 2, 2)).
    dw1, dw2, sigma1, sigma2 : exactly as `SeparationModel.invert`
        takes them (shifts in cm^-1 relative to pristine, optional
        1-sigma shift errors).

    Returns dict(strain, density, strain_sigma, density_sigma,
    strain_sigma_shifts, strain_sigma_calibration,
    density_sigma_shifts, density_sigma_calibration,
    condition_number): total error bars are the quadrature sum of the
    shift-noise part (what `SeparationModel.invert` reports) and the
    calibration part (new here). Without sigma1/sigma2 the shift part
    is None and the totals carry the calibration part alone.
    """
    K = np.asarray(calibration.K, dtype=float)
    cov = np.asarray(calibration.cov, dtype=float)
    if K.shape != (2, 2) or cov.shape != (2, 2, 2):
        raise ValueError(
            "separation_with_calibration is the two-mode tool: the "
            "calibration must carry exactly two modes (the "
            "overdetermined multimode case couples the mode weights "
            "and is deliberately not half-shipped)")
    if not (np.all(np.isfinite(K)) and np.all(np.isfinite(cov))):
        raise ValueError("calibration K and cov must be finite")
    names = list(getattr(calibration, "mode_names", ["mode1",
                                                     "mode2"]))
    from .materials import ModeCoefficients
    coeffs = ModeCoefficients(
        mode1_name=str(names[0]), mode2_name=str(names[1]),
        k1_strain=float(K[0, 0]), k1_density=float(K[0, 1]),
        k2_strain=float(K[1, 0]), k2_density=float(K[1, 1]),
        reference="calibrated lever arms (provenance in the "
                  "CalibrationResult this was built from)")
    model = SeparationModel(coeffs)
    base = model.invert(dw1, dw2, sigma1=sigma1, sigma2=sigma2)

    Kinv = np.linalg.inv(K)
    x = np.stack([np.asarray(base.strain, dtype=float),
                  np.asarray(base.density, dtype=float)])  # (2, ...)
    # dx/dK_mj = -K^-1 E_mj x  =>  (dx/dK_mj)_i = -Kinv[i, m] * x[j]
    var_cal = np.zeros_like(x)
    for m in range(2):                    # mode (row of K)
        for j in range(2):                # arm  (column of K)
            for l in range(2):
                c_jl = cov[m, j, l]
                if c_jl == 0.0:
                    continue
                for i in range(2):
                    var_cal[i] += (Kinv[i, m] * x[j]) \
                        * c_jl * (Kinv[i, m] * x[l])
    sig_cal = np.sqrt(np.maximum(var_cal, 0.0))

    if base.strain_sigma is not None:
        s_tot = np.sqrt(np.asarray(base.strain_sigma) ** 2
                        + sig_cal[0] ** 2)
        d_tot = np.sqrt(np.asarray(base.density_sigma) ** 2
                        + sig_cal[1] ** 2)
        s_shift = np.asarray(base.strain_sigma)
        d_shift = np.asarray(base.density_sigma)
    else:
        s_tot, d_tot = sig_cal[0], sig_cal[1]
        s_shift = d_shift = None
    return {"strain": base.strain, "density": base.density,
            "strain_sigma": s_tot, "density_sigma": d_tot,
            "strain_sigma_shifts": s_shift,
            "density_sigma_shifts": d_shift,
            "strain_sigma_calibration": sig_cal[0],
            "density_sigma_calibration": sig_cal[1],
            "condition_number": base.condition_number}


def multimode_with_calibration(calibration, shifts, sigmas=None):
    """Weighted least-squares inversion of m >= 2 modes with the
    calibration's uncertainty included (new in v0.12).

    The m-mode counterpart of `separation_with_calibration`, for a
    calibration of any number of modes (for example three peaks
    calibrated together with `calibrate_lever_arms`).

    calibration : a `CalibrationResult` (K is (m, 2), cov (m, 2, 2)).
    shifts, sigmas : exactly as `MultiModeModel.invert` takes them (m
        shift maps; optional per-mode 1-sigma shift errors, scalars or
        maps). Without sigmas all modes get unit weight and the
        shift-noise part is not reported (None), as in
        `separation_with_calibration`.

    The estimate is the weighted least-squares solution
    x = A^-1 K^T W y with A = K^T W K. Its exact derivative with
    respect to one lever arm K_mj is

        dx/dK_mj = w_m A^-1 (e_j r_m - K_m^T x_j),

    where r = y - K x is the residual, w_m = 1/sigma_m^2, e_j the j-th
    unit vector and K_m the m-th row of K. For two modes r = 0 and
    w_m A^-1 K_m^T is column m of K^-1, which is the identity used by
    `separation_with_calibration`; the tests assert that reduction and
    check the formula against finite differences of the re-solve. The
    calibration part of the variance sums these derivatives against
    each mode's 2 x 2 calibration covariance (modes are calibrated
    independently), to first order, as in the two-mode tool.

    Returns dict(strain, density, strain_sigma, density_sigma,
    strain_sigma_shifts, density_sigma_shifts,
    strain_sigma_calibration, density_sigma_calibration, chi2_map,
    p_value, dof, condition_number). `chi2_map` and `p_value` are those
    of `MultiModeModel.invert`, which treats K as exact; with an
    uncertain calibration they are only approximate. Masked (NaN)
    pixels stay NaN.
    """
    from .multimode import MultiModeModel
    K = np.asarray(calibration.K, dtype=float)
    cov = np.asarray(calibration.cov, dtype=float)
    if K.ndim != 2 or K.shape[1] != 2 or K.shape[0] < 2:
        raise ValueError("calibration K must be (m >= 2, 2)")
    m = K.shape[0]
    if cov.shape != (m, 2, 2):
        raise ValueError(f"calibration cov must be ({m}, 2, 2)")
    if not (np.all(np.isfinite(K)) and np.all(np.isfinite(cov))):
        raise ValueError("calibration K and cov must be finite")
    model = MultiModeModel(K, mode_names=list(
        getattr(calibration, "mode_names", None) or
        [f"mode{i + 1}" for i in range(m)]))
    base = model.invert(shifts, sigmas=sigmas)

    S = np.stack([np.asarray(s, dtype=float) for s in shifts])
    map_shape = S.shape[1:]
    P = int(np.prod(map_shape)) if map_shape else 1
    y = S.reshape(m, P)
    if sigmas is None:
        w = np.ones((m, P))
    else:
        w = np.stack([np.broadcast_to(np.asarray(s, dtype=float),
                                      map_shape).reshape(P)
                      for s in sigmas]) ** -2.0
    x = np.stack([np.asarray(base.strain, dtype=float).reshape(P),
                  np.asarray(base.density, dtype=float).reshape(P)])
    r = y - K @ x                                         # (m, P)
    # A^-1 per pixel from its closed 2 x 2 form
    a11 = np.einsum("k,kp->p", K[:, 0] ** 2, w)
    a12 = np.einsum("k,kp->p", K[:, 0] * K[:, 1], w)
    a22 = np.einsum("k,kp->p", K[:, 1] ** 2, w)
    det = a11 * a22 - a12 ** 2
    Ainv = np.stack([np.stack([a22, -a12]), np.stack([-a12, a11])]) / det
    var_cal = np.zeros((2, P))
    for mm in range(m):
        # g[j] = dx/dK_mj, shape (2, P) each
        g = []
        for j in range(2):
            v = np.zeros((2, P))
            v[j] += r[mm]
            v -= K[mm][:, None] * x[j][None, :]
            g.append(w[mm] * np.einsum("ikp,kp->ip", Ainv, v))
        for j in range(2):
            for l in range(2):
                c_jl = cov[mm, j, l]
                if c_jl != 0.0:
                    var_cal += g[j] * c_jl * g[l]
    sig_cal = np.sqrt(np.maximum(var_cal, 0.0)).reshape((2,) + map_shape)
    sig_cal = np.where(np.isfinite(x.reshape((2,) + map_shape)),
                       sig_cal, np.nan)
    if sigmas is not None:
        s_shift = np.asarray(base.strain_sigma)
        d_shift = np.asarray(base.density_sigma)
        s_tot = np.sqrt(s_shift ** 2 + sig_cal[0] ** 2)
        d_tot = np.sqrt(d_shift ** 2 + sig_cal[1] ** 2)
    else:
        s_shift = d_shift = None
        s_tot, d_tot = sig_cal[0], sig_cal[1]
    return {"strain": base.strain, "density": base.density,
            "strain_sigma": s_tot, "density_sigma": d_tot,
            "strain_sigma_shifts": s_shift,
            "density_sigma_shifts": d_shift,
            "strain_sigma_calibration": sig_cal[0],
            "density_sigma_calibration": sig_cal[1],
            "chi2_map": base.chi2_map, "p_value": base.p_value,
            "dof": base.dof,
            "condition_number": base.condition_number}
