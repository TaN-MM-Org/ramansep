"""Fit a whole hyperspectral map, pixel by pixel, into the shift maps
the inversion consumes (new in v0.8).

`fit_two_modes` handles one spectrum; a Raman map is thousands of
them. `fit_map` runs the identical two-window fit at every pixel of an
(H, W, L) cube and returns the four (H, W) maps that
`SeparationModel.invert` takes directly -- with honest per-pixel
failure reporting instead of silent garbage: a pixel whose counts are
not finite, or whose fit fails to converge, or whose fitted center
lands outside its window (a runaway fit, not a measurement) is masked
NaN in all four maps and flagged False in `ok`.

The per-pixel inversion propagates NaN pixels through untouched
(elementwise algebra), so masked pixels stay masked in the strain and
density maps and never contaminate neighbors. The map-level Bayesian
inversion (`bayesian_map_inversion`) couples pixels through its
smoothness prior and therefore cannot accept NaNs; exclude or infill
masked pixels first, deliberately -- the docstrings on both sides say
so rather than leaving it to be discovered.

Anchors asserted in the tests rather than stated: on a synthetic cube
built from known strain and density maps through the forward model
and Lorentzian lineshapes, fit_map -> invert returns the input maps;
every unmasked pixel's result equals a direct `fit_two_modes` call on
that pixel's spectrum exactly (same code path, asserted anyway); and
a deliberately corrupted pixel is masked while its neighbors are
recovered unchanged.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .fitting import _check_two_mode_setup, fit_two_modes

__all__ = ["MapFitResult", "fit_map"]


@dataclasses.dataclass
class MapFitResult:
    """Per-pixel two-mode fit of a hyperspectral map.

    dw1, dw2 : (H, W) peak-shift maps (cm^-1, relative to the
        references); NaN where masked.
    sigma1, sigma2 : (H, W) 1-sigma shift uncertainties; NaN where
        masked.
    ok : (H, W) bool; False where the pixel was masked (non-finite
        counts, non-convergence, or a center outside its window).
    n_masked : number of masked pixels.
    """

    dw1: np.ndarray
    dw2: np.ndarray
    sigma1: np.ndarray
    sigma2: np.ndarray
    ok: np.ndarray
    n_masked: int


def fit_map(wavenumber, cube, window1, window2, ref1, ref2,
            baseline="constant") -> MapFitResult:
    """Fit both modes at every pixel of a hyperspectral cube.

    wavenumber : (L,) shared axis (cm^-1).
    cube : (H, W, L) counts, e.g. from `load_map_csv`.
    window1, window2, ref1, ref2, baseline : exactly as in
        `fit_two_modes`, applied identically at every pixel.

    Returns a `MapFitResult`; feed `.dw1, .dw2, .sigma1, .sigma2`
    straight into `SeparationModel.invert`.

    Raises ValueError, before any pixel is fitted, for a cube whose
    shape does not match the axis, a non-finite axis, or a fit
    configuration that `fit_two_modes` refuses (baseline name,
    reversed or overlapping windows, fewer than 5 points -- 6 for
    the linear baseline -- in a window).
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
    H, W = c.shape[:2]
    dw1 = np.full((H, W), np.nan)
    dw2 = np.full((H, W), np.nan)
    s1 = np.full((H, W), np.nan)
    s2 = np.full((H, W), np.nan)
    ok = np.zeros((H, W), dtype=bool)
    w1 = tuple(float(v) for v in window1)
    w2 = tuple(float(v) for v in window2)
    for i in range(H):
        for j in range(W):
            y = c[i, j]
            if not np.all(np.isfinite(y)):
                continue
            try:
                d1, d2, e1, e2, f1, f2 = fit_two_modes(
                    x, y, w1, w2, ref1, ref2, baseline=baseline)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if not (f1.converged and f2.converged):
                continue
            if not (w1[0] <= f1.center <= w1[1]
                    and w2[0] <= f2.center <= w2[1]):
                continue                      # runaway fit, not data
            dw1[i, j], dw2[i, j] = d1, d2
            s1[i, j], s2[i, j] = e1, e2
            ok[i, j] = True
    return MapFitResult(dw1=dw1, dw2=dw2, sigma1=s1, sigma2=s2, ok=ok,
                        n_masked=int((~ok).sum()))
