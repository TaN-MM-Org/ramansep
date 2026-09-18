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

Honest limits, stated plainly: the propagation is first order in the
calibration errors (exact in the limit of a well-determined
calibration, the regime a calibration is for), and it is implemented
for the two-mode exact-inverse case only -- the overdetermined
multimode case couples the mode weights and is deliberately not
half-shipped.
"""
from __future__ import annotations

import numpy as np

from .core import SeparationModel

__all__ = ["separation_with_calibration"]


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
