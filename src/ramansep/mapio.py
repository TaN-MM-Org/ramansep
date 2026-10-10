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

Instrument text tables (new in v0.13): `load_spectrum_table` and
`load_map_table` read the plain numeric tables that mapping software
commonly exports, without a fixed header. A spectrum is two columns
(wavenumber, counts); a map is four columns (x, y, wavenumber, counts)
with x and y the stage positions (for example in micrometers), one line
per pixel and spectral point, in any order. Columns may be separated by
tabs, spaces or commas; lines starting with '#' and one leading text
header line are skipped. The wavenumber axis may run in either
direction (it is returned increasing), and the map loader rebuilds the
pixel grid from the stage positions. The same refusals apply as for
the CSV contract: a map that is not a complete rectangular raster, a
pixel with its own wavenumber axis, a duplicate point, or a non-finite
wavenumber or stage position is refused with an explanation. Check
the column order of your export before use; pass `usecols` if it
differs.
"""
from __future__ import annotations

import csv

import numpy as np

__all__ = ["load_spectrum_csv", "save_spectrum_csv", "load_map_csv",
           "save_map_csv", "load_spectrum_table", "load_map_table"]


def _check_axis(x, what="wavenumber axis"):
    """Refuse a wavenumber axis that is not finite and strictly
    increasing (a NaN compares False with everything, so the
    `diff <= 0` test alone lets it through)."""
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{what} contains NaN or infinite values")
    if x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError(f"{what} must be strictly increasing with at "
                         "least 2 points")

_SPEC_HEADER = ("wavenumber_cm1", "counts")
_MAP_HEADER = ("row", "col", "wavenumber_cm1", "counts")


def save_spectrum_csv(path, wavenumber, counts) -> None:
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


def load_spectrum_csv(path) -> tuple[np.ndarray, np.ndarray]:
    """Read one spectrum; returns (wavenumber, counts).

    Refusals instead of guesses: wrong header, wrong field count,
    non-numeric values, or a wavenumber axis that is not strictly
    increasing or contains NaN or infinity (the non-finite check is new
    in v0.13; before, such an axis was returned unchanged).
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
            try:
                xs.append(float(rec[0]))
                ys.append(float(rec[1]))
            except ValueError:
                raise ValueError(f"line {line}: fields must be numbers; "
                                 f"got {rec}") from None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2:
        raise ValueError("spectrum file contains fewer than 2 points")
    _check_axis(x)
    return x, y


def save_map_csv(path, wavenumber, cube) -> None:
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


def load_map_csv(path) -> tuple[np.ndarray, np.ndarray]:
    """Read a hyperspectral map; returns (wavenumber (L,), cube (H, W, L)).

    Refusals instead of guesses: wrong header or field count, a pixel
    grid with holes (every (row, col) in 0..H-1 x 0..W-1 must be
    present), a pixel whose wavenumber axis differs from the shared
    one, a wavenumber axis containing NaN or infinity, or duplicate
    (pixel, wavenumber) entries.
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
            try:
                key = (int(rec[0]), int(rec[1]))
                xv, yv = float(rec[2]), float(rec[3])
            except ValueError:
                raise ValueError(
                    f"line {line}: row and col must be integers and "
                    f"wavenumber and counts numbers; got {rec}") from None
            pixels.setdefault(key, ([], []))
            pixels[key][0].append(xv)
            pixels[key][1].append(yv)
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
    _check_axis(x0, "wavenumber axis of pixel (0, 0)")
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


# ----------------------------------------------------------------------
# Plain numeric tables as instruments export them (new in v0.13).

def _read_numeric_table(path, ncols, usecols, delimiter):
    """Parse a numeric text table into an (N, ncols) float array.

    Blank lines and lines starting with '#' are skipped; so is the
    first remaining line if it is not numeric (a column header). With
    `delimiter=None` the columns are split on commas if the last line
    contains one, else on any whitespace (tabs or spaces).
    """
    usecols = tuple(int(c) for c in usecols)
    if len(usecols) != ncols or len(set(usecols)) != ncols \
            or min(usecols) < 0:
        raise ValueError(f"usecols must name {ncols} distinct, "
                         "non-negative column indices")
    with open(path) as fh:
        lines = [ln.strip() for ln in fh]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise ValueError("the file contains no data lines")
    if delimiter is None:                         # judged on a data line
        delimiter = "," if "," in lines[-1] else None
    first = lines[0].split(delimiter)
    try:
        [float(first[c]) for c in usecols]
        start = 0
    except (ValueError, IndexError):
        start = 1                                 # a text header line
    body = lines[start:]
    if not body:
        raise ValueError("the file contains a header but no data lines")
    try:
        data = np.loadtxt(body, delimiter=delimiter, usecols=usecols,
                          ndmin=2, dtype=float)
    except (ValueError, IndexError) as err:
        raise ValueError(
            f"could not read {ncols} numeric columns {usecols} from "
            f"every data line ({err}); check the delimiter and usecols"
        ) from None
    return data


def load_spectrum_table(path, *, usecols=(0, 1),
                        delimiter=None) -> tuple[np.ndarray, np.ndarray]:
    """Read one spectrum from a plain two-column text table.

    path : a text file with one line per spectral point, holding the
        wavenumber (cm^-1) and the counts. Columns may be separated by
        tabs, spaces or commas (see `delimiter`); lines starting with
        '#' and one leading text header line are skipped.
    usecols : which columns hold (wavenumber, counts); default (0, 1).
    delimiter : None (default) splits on commas if the last line of the
        file has one and on any whitespace otherwise; or give it
        explicitly. A decimal comma (as in "404,7") is not supported.

    Returns (wavenumber, counts) with the wavenumber axis increasing:
    a file written from high to low wavenumber, as many instruments
    export it, is reversed (counts with it). Refuses a non-finite
    wavenumber, a repeated wavenumber, an axis that is not monotonic,
    fewer than 2 points, or a line that does not hold the requested
    numeric columns. Non-finite counts are returned as they are
    (`fit_map` masks such pixels).
    """
    data = _read_numeric_table(path, 2, usecols, delimiter)
    x, y = data[:, 0].copy(), data[:, 1].copy()
    if x.size >= 2 and np.all(np.diff(x) < 0.0):
        x, y = x[::-1].copy(), y[::-1].copy()
    if not np.all(np.isfinite(x)):
        raise ValueError("wavenumber axis contains NaN or infinite values")
    if x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError(
            "wavenumber axis must be strictly monotonic (increasing or "
            "decreasing) with at least 2 points and no repeated values")
    return x, y


def load_map_table(path, *, usecols=(0, 1, 2, 3), delimiter=None,
                   decimals=None) -> tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray]:
    """Read a hyperspectral map from a four-column text table.

    path : a text file with one line per pixel and spectral point,
        holding the stage position x, the stage position y, the
        wavenumber (cm^-1) and the counts, in any line order (for
        example a raster or serpentine scan, with the wavenumber
        running in either direction). Separators and headers as in
        `load_spectrum_table`.
    usecols : which columns hold (x, y, wavenumber, counts); default
        (0, 1, 2, 3). Check the column order of your export.
    delimiter : as in `load_spectrum_table`.
    decimals : optional; round the stage positions to this many
        decimals before grouping them into the pixel grid, for exports
        whose positions jitter in the last digits. None (default)
        groups exactly equal positions only.

    Returns (wavenumber (L,), cube (H, W, L), x_positions (W,),
    y_positions (H,)): `cube[i, j]` is the spectrum measured at stage
    position (x_positions[j], y_positions[i]); both position arrays
    and the wavenumber axis are increasing. Pass `wavenumber, cube` to
    `fit_map`.

    Refuses, with the reason: positions that do not form a complete
    rectangular grid (every combination of the distinct x and y values
    must be measured, each with the same number of points), a pixel
    whose wavenumber axis differs from the others, a repeated
    wavenumber within a pixel, a non-finite position or wavenumber,
    and lines without the requested numeric columns. Non-finite counts
    are kept (`fit_map` masks those pixels, reason code 1).
    """
    data = _read_numeric_table(path, 4, usecols, delimiter)
    px, py, wn, counts = (data[:, k] for k in range(4))
    for name, a in (("stage x position", px), ("stage y position", py),
                    ("wavenumber", wn)):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} column contains NaN or infinite "
                             "values")
    if decimals is not None:
        px = np.round(px, int(decimals))
        py = np.round(py, int(decimals))
    xs, ix = np.unique(px, return_inverse=True)
    ys, iy = np.unique(py, return_inverse=True)
    ix = ix.ravel()
    iy = iy.ravel()
    H, W = ys.size, xs.size
    pid = iy * W + ix
    per_pixel = np.bincount(pid, minlength=H * W)

    def where(k):
        return f"x = {float(xs[k % W])!r}, y = {float(ys[k // W])!r}"

    if np.any(per_pixel == 0):
        k = int(np.flatnonzero(per_pixel == 0)[0])
        raise ValueError(
            f"the stage positions do not form a complete rectangular "
            f"grid: {int((per_pixel == 0).sum())} of the {H} x {W} "
            f"combinations of distinct y and x values are missing "
            f"(first: {where(k)})"
            + ("" if decimals is not None else
               "; if the positions jitter in the last digits, pass "
               "decimals="))
    L = int(per_pixel[0])
    if np.any(per_pixel != L):
        raise ValueError(
            "pixels carry different numbers of spectral points "
            f"({int(per_pixel.min())} to {int(per_pixel.max())}); every "
            "pixel must share one wavenumber axis"
            + ("" if decimals is not None else
               "; if the stage positions jitter in the last digits, "
               "pass decimals="))
    order = np.lexsort((wn, pid))                 # by pixel, then axis
    axes = wn[order].reshape(H * W, L)
    cube = counts[order].reshape(H, W, L)
    x0 = axes[0]
    if L < 2 or np.any(np.diff(x0) <= 0.0):
        raise ValueError("each pixel needs at least 2 spectral points "
                         "with no repeated wavenumber")
    bad = np.flatnonzero(np.any(axes != x0, axis=1))
    if bad.size:
        k = int(bad[0])
        raise ValueError(
            f"the pixel at {where(k)} carries a different wavenumber "
            "axis than the first pixel; the map fitter needs one shared "
            "axis (export on a common axis)")
    return x0.copy(), cube, xs, ys
