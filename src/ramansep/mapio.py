"""Bring your own instrument data: documented file contracts for
spectra and hyperspectral maps (new in v0.8).

Raman instruments export in many vendor formats; the honest generic
interchange is a documented plain-text contract, enforced instead of
guessed at. Two contracts, both CSV with an exact header:

Single spectrum (`load_spectrum_csv` / `save_spectrum_csv`):

    wavenumber_cm1,counts

one row per spectral point, wavenumber strictly increasing.

Hyperspectral map (`load_map_csv` / `save_map_csv`), long/tidy:

    row,col,wavenumber_cm1,counts

one line per pixel and spectral point; `row` and `col` are 0-based
pixel indices forming a complete H x W grid, and every pixel must
carry the same strictly increasing wavenumber axis (a shared axis is
what the map fitter and the map inversion assume; an instrument that
interpolates per pixel should export on its common axis). The loaders
refuse malformed files -- wrong header, ragged or missing pixels, a
non-shared axis -- with an explanation, and the save/load round trip
is exact, asserted in the tests. No pandas: the standard library's
csv module on top of NumPy.
"""
from __future__ import annotations

import csv

import numpy as np

__all__ = ["load_spectrum_csv", "save_spectrum_csv", "load_map_csv",
           "save_map_csv"]

_SPEC_HEADER = ("wavenumber_cm1", "counts")
_MAP_HEADER = ("row", "col", "wavenumber_cm1", "counts")


def save_spectrum_csv(path, wavenumber, counts):
    """Write one spectrum in the documented contract."""
    x = np.asarray(wavenumber, dtype=float).ravel()
    y = np.asarray(counts, dtype=float).ravel()
    if x.size != y.size or x.size < 2:
        raise ValueError("wavenumber and counts must be equal-length "
                         "arrays with at least 2 points")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_SPEC_HEADER)
        for a, b in zip(x, y):
            w.writerow([repr(float(a)), repr(float(b))])


def load_spectrum_csv(path):
    """Read one spectrum; returns (wavenumber, counts).

    Refusals instead of guesses: wrong header, wrong field count,
    non-numeric values, or a wavenumber axis that is not strictly
    increasing.
    """
    xs, ys = [], []
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        try:
            header = next(r)
        except StopIteration:
            raise ValueError("empty spectrum file") from None
        if tuple(h.strip() for h in header) != _SPEC_HEADER:
            raise ValueError(
                f"spectrum file header must be exactly {_SPEC_HEADER}; "
                f"got {tuple(header)}")
        for line, rec in enumerate(r, start=2):
            if not rec:
                continue
            if len(rec) != 2:
                raise ValueError(f"line {line}: expected 2 fields")
            xs.append(float(rec[0]))
            ys.append(float(rec[1]))
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2:
        raise ValueError("spectrum file contains fewer than 2 points")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("wavenumber axis must be strictly increasing")
    return x, y


def save_map_csv(path, wavenumber, cube):
    """Write a hyperspectral map in the documented contract.

    wavenumber : (L,) shared axis.  cube : (H, W, L) counts.
    """
    x = np.asarray(wavenumber, dtype=float).ravel()
    c = np.asarray(cube, dtype=float)
    if c.ndim != 3 or c.shape[2] != x.size:
        raise ValueError("cube must be (H, W, L) with L matching the "
                         "wavenumber axis")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_MAP_HEADER)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                for k in range(x.size):
                    w.writerow([i, j, repr(float(x[k])),
                                repr(float(c[i, j, k]))])


def load_map_csv(path):
    """Read a hyperspectral map; returns (wavenumber (L,), cube (H, W, L)).

    Refusals instead of guesses: wrong header or field count, a pixel
    grid with holes (every (row, col) in 0..H-1 x 0..W-1 must be
    present), a pixel whose wavenumber axis differs from the shared
    one, or duplicate (pixel, wavenumber) entries.
    """
    pixels = {}
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        try:
            header = next(r)
        except StopIteration:
            raise ValueError("empty map file") from None
        if tuple(h.strip() for h in header) != _MAP_HEADER:
            raise ValueError(
                f"map file header must be exactly {_MAP_HEADER}; got "
                f"{tuple(header)}")
        for line, rec in enumerate(r, start=2):
            if not rec:
                continue
            if len(rec) != 4:
                raise ValueError(f"line {line}: expected 4 fields")
            key = (int(rec[0]), int(rec[1]))
            pixels.setdefault(key, ([], []))
            pixels[key][0].append(float(rec[2]))
            pixels[key][1].append(float(rec[3]))
    if not pixels:
        raise ValueError("map file contains no data rows")
    rows = sorted({k[0] for k in pixels})
    cols = sorted({k[1] for k in pixels})
    H, W = len(rows), len(cols)
    if rows != list(range(H)) or cols != list(range(W)):
        raise ValueError("pixel indices must be 0-based and "
                         "consecutive (a complete H x W grid)")
    missing = [(i, j) for i in range(H) for j in range(W)
               if (i, j) not in pixels]
    if missing:
        raise ValueError(f"pixel grid has holes: missing {missing[:5]}"
                         + ("..." if len(missing) > 5 else ""))
    x0 = np.asarray(pixels[(0, 0)][0], dtype=float)
    if x0.size < 2 or np.any(np.diff(x0) <= 0.0):
        raise ValueError("wavenumber axis must be strictly increasing "
                         "with at least 2 points")
    cube = np.empty((H, W, x0.size))
    for (i, j), (xs, ys) in pixels.items():
        x = np.asarray(xs, dtype=float)
        if x.shape != x0.shape or not np.array_equal(x, x0):
            raise ValueError(
                f"pixel ({i}, {j}) carries a different wavenumber axis "
                "than pixel (0, 0); the contract requires one shared "
                "axis for the whole map")
        cube[i, j] = np.asarray(ys, dtype=float)
    return x0, cube
