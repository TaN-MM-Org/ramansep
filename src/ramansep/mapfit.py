"""Fit a whole hyperspectral map, pixel by pixel, into the shift maps
the inversion consumes (new in v0.8).

`fit_two_modes` handles one spectrum; a Raman map is thousands of
them. `fit_map` runs the identical two-window fit at every pixel of an
(H, W, L) cube and returns the four (H, W) maps that
`SeparationModel.invert` takes directly -- with honest per-pixel
failure reporting instead of silent garbage: a pixel whose counts are
not finite, or whose fit fails to converge, or whose fitted center
lands outside its window (a runaway fit, not a measurement), or (since
v0.12) whose fitted peak height is not positive, is masked NaN in all
four maps, flagged False in `ok`, and given a reason code
(`MASK_REASONS`). Optional checks on the peak height against its
error bar (`min_snr`) and on the width (`fwhm_range`) catch pixels
that carry no peak at all, where a fit to pure noise can otherwise
converge on a noise spike.

The per-pixel inversion propagates NaN pixels through untouched
(elementwise algebra), so masked pixels stay masked in the strain and
density maps and never contaminate neighbors. The map-level Bayesian
inversion (`bayesian_map_inversion`) couples pixels through its
smoothness prior; it refuses NaNs unless asked to fill them from the
neighbours (`fill_masked=True`, since v0.12).

Since v0.14 the pixels are fitted together, in blocks, by a stacked
version of the same Levenberg-Marquardt loop (5 to 7 times faster on
a 1200-pixel map, 3 to 4 times with `joint=True`), with the numbers of
the per-pixel loop over `fit_two_modes` up to rounding (the tests allow
1e-4 of the error bar in a shift and a relative 1e-5 in an error bar;
before v0.15 this said "bit for bit", which holds on one machine but is
not guaranteed on another CPU). `MapFitResult.shift_correlation` carries the correlation of
the two shift errors for `SeparationModel.invert`, and `noise` selects
how the error bars are computed (see `fit_lorentzian`).

Anchors asserted in the tests rather than stated: on a synthetic cube
built from known strain and density maps through the forward model
and Lorentzian lineshapes, fit_map -> invert returns the input maps;
every unmasked pixel's result equals a direct `fit_two_modes` call on
that pixel's spectrum up to rounding (the stated tolerances); and
a deliberately corrupted pixel is masked while its neighbors are
recovered unchanged.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .fitting import (_check_noise, _check_references,
                      _check_two_mode_setup, _fit_joint_rows,
                      _fit_lorentzian_rows)

__all__ = ["MapFitResult", "fit_map", "MASK_REASONS"]

# pixels fitted together per block by `fit_map`; bounds the memory of
# the stacked Jacobians (block x points x 5 floats) without changing
# any result
_BLOCK = 1024


#: Meaning of the codes in `MapFitResult.reason` (new in v0.12).
MASK_REASONS = {
    0: "ok",
    1: "counts not finite",
    2: "fit raised an error",
    3: "fit did not converge",
    4: "fitted centre outside its window",
    5: "fitted peak height not positive (a dip or no peak)",
    6: "peak height below min_snr times its error bar",
    7: "fitted width outside fwhm_range",
}


@dataclasses.dataclass
class MapFitResult:
    """Per-pixel two-mode fit of a hyperspectral map.

    dw1, dw2 : (H, W) peak-shift maps (cm^-1, relative to the
        references); NaN where masked.
    sigma1, sigma2 : (H, W) 1-sigma shift uncertainties; NaN where
        masked.
    ok : (H, W) bool; False where the pixel was masked.
    n_masked : number of masked pixels.
    reason : (H, W) int codes saying why each pixel was masked (0 for
        kept pixels); `MASK_REASONS` maps each code to its meaning. The
        first failed check is recorded, in the order of the codes (new
        in v0.12).
    shift_correlation : (H, W) correlation of the errors of dw1 and
        dw2 at each kept pixel: the correlation of the two fitted
        centers with `joint=True`, and 0 for separate window fits (they
        use disjoint spectral points, so their errors are independent);
        NaN at masked pixels. Pass it to `SeparationModel.invert(...,
        shift_correlation=...)` (new in v0.14; None in results built by
        hand).
    """

    dw1: np.ndarray
    dw2: np.ndarray
    sigma1: np.ndarray
    sigma2: np.ndarray
    ok: np.ndarray
    n_masked: int
    reason: np.ndarray | None = None
    shift_correlation: np.ndarray | None = None


def fit_map(wavenumber, cube, window1, window2, ref1: float, ref2: float,
            baseline: str = "constant", *, joint: bool = False,
            min_snr: float | None = None,
            fwhm_range: tuple[float, float] | None = None,
            noise: str = "constant") -> MapFitResult:
    """Fit both modes at every pixel of a hyperspectral cube.

    wavenumber : (L,) shared axis (cm^-1).
    cube : (H, W, L) counts, e.g. from `load_map_csv`.
    window1, window2, ref1, ref2, baseline, joint, noise : exactly as
        in `fit_two_modes`, applied identically at every pixel (`noise`
        is new in v0.14: "varying" for shot-noise limited spectra).
    min_snr : optional; mask a pixel unless each fitted peak height is
        at least `min_snr` times its own error bar. Use it on maps with
        pixels that carry no peak (off the flake, a hole, a dead spot):
        a fit to pure noise can converge on a noise spike and return a
        plausible-looking shift. None (default) applies no such test.
    fwhm_range : optional (lo, hi) in cm^-1; mask a pixel unless both
        fitted widths (FWHM) lie within it. A width far below the
        spectral point spacing, or wider than the window, is not a
        measured phonon line.

    A pixel is masked (NaN in all four maps, False in `ok`, a non-zero
    code in `reason`) when its counts are not finite, a fit raises an
    error or does not converge, a fitted centre lies outside its
    window, a fitted peak height is zero or negative (new in v0.12:
    such a "peak" is a dip, not a Raman line), or it fails the optional
    `min_snr` / `fwhm_range` checks.

    Returns a `MapFitResult`; feed `.dw1, .dw2, .sigma1, .sigma2`
    straight into `SeparationModel.invert`.

    Raises ValueError, before any pixel is fitted, for a cube whose
    shape does not match the axis, a non-finite axis, a fit
    configuration that `fit_two_modes` refuses (baseline name,
    reversed or overlapping windows, fewer than 5 points in a window
    (6 for the linear baseline), a reference frequency that is not
    finite), an unknown `noise` name, a non-positive `min_snr`, or a
    `fwhm_range` that is not (lo, hi) with 0 <= lo < hi.
    """
    x = np.asarray(wavenumber, dtype=float).ravel()
    c = np.asarray(cube, dtype=float)
    if c.ndim != 3 or c.shape[2] != x.size:
        raise ValueError("cube must be (H, W, L) with L matching the "
                         "wavenumber axis")
    if not np.all(np.isfinite(x)):
        raise ValueError("the wavenumber axis must be finite")
    # configuration errors (bad baseline name, reversed or overlapping
    # windows, too few points in a window) are the caller's, not the
    # pixels': refuse them once here instead of masking every pixel
    _check_two_mode_setup(x, window1, window2, baseline)
    # a NaN reference made every shift NaN while `ok` stayed True
    # (before v0.13); refuse it once, like the other settings
    _check_references(ref1, ref2)
    _check_noise(noise)
    if min_snr is not None:
        min_snr = float(min_snr)
        if not (np.isfinite(min_snr) and min_snr > 0.0):
            raise ValueError("min_snr must be a positive number")
    if fwhm_range is not None:
        g_lo, g_hi = (float(v) for v in fwhm_range)
        if not (np.isfinite(g_lo) and np.isfinite(g_hi)
                and 0.0 <= g_lo < g_hi):
            raise ValueError("fwhm_range must be (lo, hi) with "
                             "0 <= lo < hi")
    H, W = c.shape[:2]
    dw1 = np.full((H, W), np.nan)
    dw2 = np.full((H, W), np.nan)
    s1 = np.full((H, W), np.nan)
    s2 = np.full((H, W), np.nan)
    rho = np.full((H, W), np.nan)
    ok = np.zeros((H, W), dtype=bool)
    reason = np.zeros((H, W), dtype=int)
    w1 = tuple(float(v) for v in window1)
    w2 = tuple(float(v) for v in window2)
    r1, r2 = float(ref1), float(ref2)
    # Since v0.14 the separate window fits of all pixels run together
    # (`_fit_lorentzian_rows`, the algorithm of `fit_lorentzian` stacked
    # over pixels), in blocks so that memory stays bounded. Each pixel's
    # numbers are those of a `fit_two_modes` call on its spectrum up to
    # rounding; the tests check it within stated tolerances.
    flat = c.reshape(H * W, x.size)
    finite = np.all(np.isfinite(flat), axis=1)
    reason.reshape(-1)[~finite] = 1
    sel1 = (x >= w1[0]) & (x <= w1[1])
    sel2 = (x >= w2[0]) & (x <= w2[1])
    good = np.flatnonzero(finite)
    for start in range(0, good.size, _BLOCK):
        block = good[start:start + _BLOCK]
        rows = flat[block]
        fits = list(zip(_fit_lorentzian_rows(x[sel1], rows[:, sel1],
                                             baseline, noise=noise),
                        _fit_lorentzian_rows(x[sel2], rows[:, sel2],
                                             baseline, noise=noise)))
        # a None fit is a spectrum `fit_lorentzian` refuses (reason 2)
        started = [k for k, (f1, f2) in enumerate(fits)
                   if f1 is not None and f2 is not None]
        if joint and started:
            try:
                pairs = _fit_joint_rows(x, rows[started], w1, w2,
                                        baseline,
                                        [fits[k] for k in started],
                                        noise=noise)
            except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
                pairs = [(None, None)] * len(started)
            for k, pair in zip(started, pairs):
                fits[k] = pair
        for k, pix in enumerate(block):
            i, j = divmod(int(pix), W)
            f1, f2 = fits[k]
            if f1 is None or f2 is None:
                reason[i, j] = 2
                continue
            if not (f1.converged and f2.converged):
                reason[i, j] = 3
                continue
            if not (w1[0] <= f1.center <= w1[1]
                    and w2[0] <= f2.center <= w2[1]):
                reason[i, j] = 4              # runaway fit, not data
                continue
            if not (f1.amplitude > 0.0 and f2.amplitude > 0.0):
                reason[i, j] = 5
                continue
            if min_snr is not None and not (
                    f1.amplitude >= min_snr * f1.amplitude_sigma
                    and f2.amplitude >= min_snr * f2.amplitude_sigma):
                reason[i, j] = 6
                continue
            if fwhm_range is not None and not (
                    g_lo <= f1.fwhm <= g_hi and g_lo <= f2.fwhm <= g_hi):
                reason[i, j] = 7
                continue
            dw1[i, j], dw2[i, j] = f1.center - r1, f2.center - r2
            s1[i, j], s2[i, j] = f1.center_sigma, f2.center_sigma
            # separate window fits use disjoint spectral points, so
            # their center errors are independent: correlation 0
            rho[i, j] = f1.center_correlation if joint else 0.0
            ok[i, j] = True
    return MapFitResult(dw1=dw1, dw2=dw2, sigma1=s1, sigma2=s2, ok=ok,
                        n_masked=int((~ok).sum()), reason=reason,
                        shift_correlation=rho)
