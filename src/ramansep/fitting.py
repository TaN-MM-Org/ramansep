"""Peak-fitting front end: from raw spectra to the two-mode inversion.

This module closes the gap between a measured Raman spectrum and the
linear inversion of `SeparationModel`: it fits a Lorentzian to each of
the two modes and returns peak shifts with one-standard-deviation
uncertainties, exactly the quantities `SeparationModel.invert` consumes.

The fitter is a Levenberg-Marquardt least-squares solver with the
analytic Jacobian of the Lorentzian model:

    L(x) = A * (G/2)^2 / ((x - c)^2 + (G/2)^2) + b

where c is the center, G the full width at half maximum, A the peak
height above the baseline b. The Lorentzian is the natural lineshape of
a phonon with lifetime broadening; for instrument-dominated lines
`fit_voigt` fits the Voigt profile (Gaussian-convolved Lorentzian) via
the Faddeeva function w(z), with the analytic Jacobian built from
w'(z) = 2i/sqrt(pi) - 2 z w(z).

Algorithm: K. Levenberg, Quart. Appl. Math. 2, 164 (1944);
D. W. Marquardt, J. Soc. Ind. Appl. Math. 11, 431 (1963). Parameter
uncertainties are the standard linearized least-squares estimates,
sigma_p^2 = s^2 [(J^T J)^-1]_pp with s^2 the residual variance; they are
trustworthy when the residuals are dominated by uncorrelated noise of
the same size at every point, and that caveat is documented rather
than hidden. (Before v0.14 this said "uncorrelated noise" only. Shot
noise is uncorrelated but larger on a peak than on its baseline, and
the tests show these error bars then too small by a factor of 1.7;
use noise="varying" for such spectra.)

Exact facts the test suite asserts, rather than states:

* The analytic Jacobian matches finite differences on every parameter.
* On noiseless synthetic Lorentzians the fit recovers center and width
  to 1e-8, amplitude and baseline to 1e-6, from automatic starting
  values.
* On seeded noisy spectra the center error is compatible with the
  reported center uncertainty.
* A full round trip (strain/density maps -> forward shifts -> synthetic
  spectra -> `fit_two_modes` -> `SeparationModel.invert`) returns the
  input maps.

Since v0.14 `noise="varying"` replaces s^2 (J^T J)^-1 by the
heteroskedasticity-consistent ("sandwich") covariance of H. White,
Econometrica 48, 817 (1980), for spectra whose noise level changes
along the axis, as shot noise does; the fitted values are unchanged.

Since v0.12 `fit_two_modes(..., joint=True)` fits two overlapping
peaks together (two Lorentzians on one shared baseline, same
Levenberg-Marquardt loop, analytic Jacobian checked against finite
differences), which removes the pull of one peak's tail on the other
peak's fitted centre; on noiseless overlapping pairs it recovers both
centres to 1e-8, and on seeded noisy spectra its reported centre errors
and centre correlation match the Monte Carlo scatter.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class PeakFit:
    """Result of a single Lorentzian fit.

    center, fwhm, amplitude, offset : fitted parameters (center and fwhm
        in the x units, typically cm^-1).
    center_sigma, fwhm_sigma, amplitude_sigma, offset_sigma : 1-sigma
        uncertainties from the linearized covariance.
    residual_rms : root-mean-square residual of the fit.
    n_iter : Levenberg-Marquardt iterations used.
    converged : True if the step and cost tolerances were met.
    slope, slope_sigma : the fitted linear-baseline slope (counts per
        x unit, about the window center) and its uncertainty, when
        `baseline="linear"` was requested; 0 and NaN for the constant
        baseline (new in v0.8).
    center_correlation : correlation between the two fitted centres
        when this fit came from `fit_two_modes(..., joint=True)`; NaN
        for a single-peak fit (new in v0.12).
    """

    center: float
    fwhm: float
    amplitude: float
    offset: float
    center_sigma: float
    fwhm_sigma: float
    amplitude_sigma: float
    offset_sigma: float
    residual_rms: float
    n_iter: int
    converged: bool
    slope: float = 0.0
    slope_sigma: float = float("nan")
    center_correlation: float = float("nan")


def lorentzian(x, center, fwhm, amplitude, offset=0.0):
    """Lorentzian lineshape A * (G/2)^2 / ((x-c)^2 + (G/2)^2) + b."""
    x = np.asarray(x, dtype=float)
    h = 0.5 * fwhm
    return amplitude * h * h / ((x - center) ** 2 + h * h) + offset


def _model_and_jacobian(x, p):
    c, G, A, b = p
    h = 0.5 * G
    u = x - c
    d = u * u + h * h
    core = h * h / d
    y = A * core + b
    J = np.empty((x.size, 4))
    J[:, 0] = A * h * h * 2.0 * u / (d * d)      # d/dc
    J[:, 1] = A * h * u * u / (d * d)            # d/dG (dh/dG = 1/2)
    J[:, 2] = core                               # d/dA
    J[:, 3] = 1.0                                # d/db
    return y, J


def _model_and_jacobian_linear(x, p, x0):
    """Lorentzian plus linear baseline b + m (x - x0); x0 is the fixed
    window center, so the slope parameter stays well-conditioned."""
    c, G, A, b, mslope = p
    h = 0.5 * G
    u = x - c
    d = u * u + h * h
    core = h * h / d
    y = A * core + b + mslope * (x - x0)
    J = np.empty((x.size, 5))
    J[:, 0] = A * h * h * 2.0 * u / (d * d)      # d/dc
    J[:, 1] = A * h * u * u / (d * d)            # d/dG
    J[:, 2] = core                               # d/dA
    J[:, 3] = 1.0                                # d/db
    J[:, 4] = x - x0                             # d/dm
    return y, J


def _initial_guess(x, y):
    b0 = float(np.min(y))
    i = int(np.argmax(y))
    A0 = float(y[i] - b0)
    c0 = float(x[i])
    half = b0 + 0.5 * A0
    above = y >= half
    if above.any():
        w = x[above]
        G0 = float(w.max() - w.min())
    else:                                        # pragma: no cover
        G0 = float(x[-1] - x[0]) / 4.0
    G0 = max(G0, 2.0 * abs(float(x[1] - x[0])))
    return np.array([c0, G0, A0 if A0 > 0 else 1.0, b0])


def _check_noise(noise):
    """Refuse an unknown `noise` setting. "constant": error bars from
    the residual variance, s^2 (J^T J)^-1 (the only choice before
    v0.14). "varying": White's heteroskedasticity-consistent sandwich
    covariance (see `fit_lorentzian`). On seeded shot-noise (Poisson)
    spectra of a peak 400 counts high on a 30-count baseline, the
    tests find the actual scatter of the fitted center 1.7 times the
    "constant" error bar and within 10 percent of the "varying" one."""
    if noise not in ("constant", "varying"):
        raise ValueError('noise must be "constant" or "varying"')


def _sandwich(J, r, B):
    """(J^T J)^-1 J^T diag(r^2) J (J^T J)^-1 * n / (n - p), with B the
    inverse of J^T J (White 1980, linearized model)."""
    n, npar = J.shape
    M = (J * (r * r)[:, None]).T @ J
    return (n / (n - npar)) * (B @ M @ B)


def fit_lorentzian(x, y, p0=None, max_iter=200, tol=1e-12,
                   baseline="constant", noise: str = "constant"
                   ) -> PeakFit:
    """Fit one Lorentzian peak by Levenberg-Marquardt least squares.

    x, y : 1D arrays of equal length >= 5 (constant baseline; the
        linear baseline needs >= 6: parameters plus one degree of
        freedom for the noise estimate).
    p0 : optional starting values (center, fwhm, amplitude, offset)
        for the constant baseline, or (center, fwhm, amplitude,
        offset, slope) for the linear one; by default estimated from
        the data.
    baseline : "constant" (the historical model, unchanged) or
        "linear" (new in v0.8): adds a slope term m (x - x_mid) to the
        model, for spectra sitting on a sloped fluorescence
        background. A sloped background under a constant-baseline fit
        pulls the fitted center sideways -- the tests demonstrate the
        bias and its recovery.
    tol : relative decrease of the cost at which iteration stops.
    noise : how the error bars are computed (new in v0.14; the fitted
        values do not depend on it). "constant" (default, unchanged)
        assumes the same noise level at every spectral point:
        s^2 (J^T J)^-1 with s^2 the residual variance. "varying" makes
        no such assumption: the heteroskedasticity-consistent
        ("sandwich") covariance of H. White, Econometrica 48, 817
        (1980), (J^T J)^-1 J^T diag(r^2) J (J^T J)^-1 with r the
        residuals, applied to the linearized model at the fit, times
        n / (n - p) (n points, p parameters). Use it for shot-noise
        limited spectra, whose noise grows with the counts: there the
        "constant" error bars of a peak standing high above its
        baseline are too small, by a factor of 1.7 in the tests (see
        `_check_noise`).

    Returns a `PeakFit`. `converged` is False if `max_iter` was hit
    first; the parameters returned are then the best found, and the
    caller decides whether to trust them.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if baseline not in ("constant", "linear"):
        raise ValueError('baseline must be "constant" or "linear"')
    _check_noise(noise)
    lin = baseline == "linear"
    npar = 5 if lin else 4
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < npar + 1:
        raise ValueError(f"need at least {npar + 1} points to fit "
                         f"{npar} parameters and estimate a residual "
                         "variance")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must be finite; clean or mask the "
                         "spectrum before fitting")

    x0 = float(x.mean())
    if p0 is not None:
        p = np.asarray(p0, dtype=float)
        if p.shape != (npar,):
            raise ValueError(
                "p0 must be (center, fwhm, amplitude, offset)"
                + (" plus slope for the linear baseline" if lin else ""))
    else:
        p = _initial_guess(x, y)
        if lin:
            # remove the chord through the endpoints, re-guess the peak
            m0 = (float(y[-1]) - float(y[0])) / (float(x[-1]) - float(x[0]))
            p = np.concatenate([_initial_guess(x, y - m0 * (x - x0)),
                                [m0]])
    if p[1] <= 0:
        raise ValueError("starting fwhm must be positive")

    def mj(xv, pv):
        return (_model_and_jacobian_linear(xv, pv, x0) if lin
                else _model_and_jacobian(xv, pv))

    lam = 1e-3
    yfit, J = mj(x, p)
    r = y - yfit
    cost = float(r @ r)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        JTJ = J.T @ J
        g = J.T @ r
        try:
            step = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ)), g)
        except np.linalg.LinAlgError:            # pragma: no cover
            lam *= 10.0
            continue
        p_new = p + step
        p_new[1] = abs(p_new[1])                 # width sign is a gauge
        y_new, J_new = mj(x, p_new)
        r_new = y - y_new
        cost_new = float(r_new @ r_new)
        if cost_new < cost:
            rel = (cost - cost_new) / max(cost, 1e-300)
            p, r, J, cost = p_new, r_new, J_new, cost_new
            lam = max(lam / 10.0, 1e-12)
            if rel < tol:
                converged = True
                break
        else:
            lam *= 10.0
            if lam > 1e12:
                converged = True                 # stuck at a minimum
                break

    dof = x.size - npar
    s2 = cost / dof
    JTJ = J.T @ J
    try:
        if noise == "varying":
            cov = _sandwich(J, r, np.linalg.inv(JTJ))
        else:
            cov = s2 * np.linalg.inv(JTJ)
        sig = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:                # pragma: no cover
        sig = np.full(npar, np.nan)
    return PeakFit(
        center=float(p[0]), fwhm=float(p[1]), amplitude=float(p[2]),
        offset=float(p[3]), center_sigma=float(sig[0]),
        fwhm_sigma=float(sig[1]), amplitude_sigma=float(sig[2]),
        offset_sigma=float(sig[3]),
        residual_rms=float(np.sqrt(cost / x.size)),
        n_iter=it, converged=converged,
        slope=float(p[4]) if lin else 0.0,
        slope_sigma=float(sig[4]) if lin else float("nan"),
    )


def _check_two_mode_setup(x, window1, window2, baseline):
    """Validate the fit configuration shared by every spectrum on the
    axis x: baseline name, window order, window overlap, and enough
    points per window. Returns the two windows as float tuples.
    Used by `fit_two_modes` and, once per map, by `fit_map`, so that
    a configuration error is refused instead of masking every pixel."""
    if baseline not in ("constant", "linear"):
        raise ValueError('baseline must be "constant" or "linear"')
    w1 = tuple(float(v) for v in window1)
    w2 = tuple(float(v) for v in window2)
    for w in (w1, w2):
        if w[0] >= w[1]:
            raise ValueError(f"window {w} must be (lo, hi) with lo < hi")
    if min(w1[1], w2[1]) > max(w1[0], w2[0]):
        raise ValueError("the two windows overlap; each mode must be "
                         "fitted on its own spectral range")
    need = 6 if baseline == "linear" else 5
    for lo, hi in (w1, w2):
        m = (x >= lo) & (x <= hi)
        if m.sum() < need:
            raise ValueError(f"window ({lo}, {hi}) contains fewer than "
                             f"{need} spectral points")
    return w1, w2


def _check_references(ref1, ref2):
    """Refuse reference frequencies that are not finite numbers. A NaN
    reference turns every shift into NaN; `fit_map` used to report
    such pixels as fitted (`ok` True) before v0.13."""
    r1, r2 = float(ref1), float(ref2)
    if not (np.isfinite(r1) and np.isfinite(r2)):
        raise ValueError("reference frequencies ref1 and ref2 must be "
                         f"finite numbers (cm^-1); got {ref1!r}, {ref2!r}")
    return r1, r2


def fit_two_modes(x, y, window1, window2, ref1: float, ref2: float,
                  baseline: str = "constant", joint: bool = False,
                  noise: str = "constant"
                  ) -> tuple[float, float, float, float, PeakFit, PeakFit]:
    """Fit both modes of a spectrum and return inversion-ready shifts.

    x, y : the spectrum (wavenumber axis and counts).
    window1, window2 : (lo, hi) wavenumber windows, one per mode. The
        windows must not overlap: a shared shoulder would be counted
        twice, once per fit.
    ref1, ref2 : pristine-material reference frequencies of the two
        modes (cm^-1), the zero points of the shifts. Must be finite
        (refused otherwise since v0.13).
    baseline : "constant" or "linear", passed to `fit_lorentzian`
        (new in v0.8; use "linear" for a sloped fluorescence
        background).
    joint : False (default, unchanged behaviour) fits each peak alone
        in its own window, so the tail of the other peak that reaches
        into a window pulls that fitted centre slightly. True (new in
        v0.12) fits ONE model -- two Lorentzians on a shared baseline --
        to every point between the lower edge of the lower window and
        the upper edge of the higher one, starting from the separate
        window fits. Both tails are then part of the model, so this bias
        is gone. That span must contain no third peak. The two fitted
        centres are then slightly correlated; the correlation is
        reported in `fit1.center_correlation` (the inversion itself
        treats dw1 and dw2 as independent unless it is given that
        correlation: `SeparationModel.invert(..., shift_correlation=)`,
        new in v0.14).
    noise : "constant" (default) or "varying", passed to
        `fit_lorentzian` and used by the joint fit as well (new in
        v0.14): "varying" gives error bars that stay right when the
        noise level changes along the spectrum, as shot noise does.

    Returns (dw1, dw2, sigma1, sigma2, fit1, fit2): the two peak shifts
    relative to the references, their 1-sigma uncertainties, and the two
    full `PeakFit` results. Feed the first four straight into
    `SeparationModel.invert(dw1, dw2, sigma1, sigma2)`. With
    `joint=True` both `PeakFit`s share the baseline (`offset`, `slope`),
    `residual_rms`, `n_iter` and `converged` of the joint fit.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    w1, w2 = _check_two_mode_setup(x, window1, window2, baseline)
    ref1, ref2 = _check_references(ref1, ref2)
    _check_noise(noise)
    fits = []
    for lo, hi in (w1, w2):
        m = (x >= lo) & (x <= hi)
        fits.append(fit_lorentzian(x[m], y[m], baseline=baseline,
                                   noise=noise))
    fit1, fit2 = fits
    if joint:
        fit1, fit2 = _fit_two_lorentzians_joint(x, y, w1, w2, baseline,
                                                fit1, fit2, noise=noise)
    return (fit1.center - float(ref1), fit2.center - float(ref2),
            fit1.center_sigma, fit2.center_sigma, fit1, fit2)


# ----------------------------------------------------------------------
# Joint fit of two overlapping Lorentzians (new in v0.12).

def _two_lorentzians_model_and_jacobian(x, p, x0, lin):
    """Two Lorentzians on a shared baseline. Parameters
    (c1, G1, A1, c2, G2, A2, b[, m]); the optional slope term is
    m (x - x0) with x0 fixed, as in `_model_and_jacobian_linear`."""
    npar = 8 if lin else 7
    J = np.empty((x.size, npar))
    y = np.full(x.size, p[6])
    for k in range(2):
        c, G, A = p[3 * k], p[3 * k + 1], p[3 * k + 2]
        h = 0.5 * G
        u = x - c
        d = u * u + h * h
        core = h * h / d
        y = y + A * core
        J[:, 3 * k] = A * h * h * 2.0 * u / (d * d)       # d/dc
        J[:, 3 * k + 1] = A * h * u * u / (d * d)         # d/dG
        J[:, 3 * k + 2] = core                            # d/dA
    J[:, 6] = 1.0                                         # d/db
    if lin:
        y = y + p[7] * (x - x0)
        J[:, 7] = x - x0                                  # d/dm
    return y, J


def _fit_two_lorentzians_joint(x, y, w1, w2, baseline, start1, start2,
                               max_iter=300, tol=1e-12, noise="constant"):
    """Levenberg-Marquardt fit of two Lorentzians plus a shared
    baseline on the span covering both windows; the same damping rule
    and linearised covariance as `fit_lorentzian`. Returns two
    `PeakFit`s (mode 1, mode 2)."""
    lin = baseline == "linear"
    npar = 8 if lin else 7
    lo = min(w1[0], w2[0])
    hi = max(w1[1], w2[1])
    sel = (x >= lo) & (x <= hi)
    xs, ys = x[sel], y[sel]
    x0 = float(xs.mean())
    # the separate window fits each carry the other peak's tail in
    # their offsets, so the lower offset is the better baseline start
    b0 = min(start1.offset, start2.offset)
    p = np.array([start1.center, start1.fwhm, start1.amplitude,
                  start2.center, start2.fwhm, start2.amplitude, b0]
                 + ([0.0] if lin else []), dtype=float)

    def mj(pv):
        return _two_lorentzians_model_and_jacobian(xs, pv, x0, lin)

    lam = 1e-3
    yfit, J = mj(p)
    r = ys - yfit
    cost = float(r @ r)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        JTJ = J.T @ J
        g = J.T @ r
        try:
            step = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ)), g)
        except np.linalg.LinAlgError:            # pragma: no cover
            lam *= 10.0
            continue
        p_new = p + step
        p_new[1] = abs(p_new[1])                 # width signs are gauges
        p_new[4] = abs(p_new[4])
        y_new, J_new = mj(p_new)
        r_new = ys - y_new
        cost_new = float(r_new @ r_new)
        if cost_new < cost:
            rel = (cost - cost_new) / max(cost, 1e-300)
            p, r, J, cost = p_new, r_new, J_new, cost_new
            lam = max(lam / 10.0, 1e-12)
            if rel < tol:
                converged = True
                break
        else:
            lam *= 10.0
            if lam > 1e12:
                converged = True                 # stuck at a minimum
                break

    dof = xs.size - npar
    if dof < 1:
        raise ValueError("the joint fit span holds too few points")
    s2 = cost / dof
    try:
        if noise == "varying":
            cov = _sandwich(J, r, np.linalg.inv(J.T @ J))
        else:
            cov = s2 * np.linalg.inv(J.T @ J)
        sig = np.sqrt(np.maximum(np.diag(cov), 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = float(cov[0, 3] / (sig[0] * sig[3]))
    except np.linalg.LinAlgError:                # pragma: no cover
        sig = np.full(npar, np.nan)
        rho = float("nan")
    rms = float(np.sqrt(cost / xs.size))
    out = []
    for k in range(2):
        out.append(PeakFit(
            center=float(p[3 * k]), fwhm=float(p[3 * k + 1]),
            amplitude=float(p[3 * k + 2]), offset=float(p[6]),
            center_sigma=float(sig[3 * k]),
            fwhm_sigma=float(sig[3 * k + 1]),
            amplitude_sigma=float(sig[3 * k + 2]),
            offset_sigma=float(sig[6]), residual_rms=rms, n_iter=it,
            converged=converged,
            slope=float(p[7]) if lin else 0.0,
            slope_sigma=float(sig[7]) if lin else float("nan"),
            center_correlation=rho))
    return out[0], out[1]


# ----------------------------------------------------------------------
# Many spectra at once (new in v0.14): the engine of `fit_map`.
#
# `_fit_lorentzian_rows` runs, for every row of a (P, n) block of
# spectra on one shared axis, exactly the algorithm of `fit_lorentzian`
# with its default starting values, and `_fit_joint_rows` that of the
# joint two-peak fit: the same starting values, the same damping rule
# and stopping tests per spectrum, and the same linearized covariance.
# Only the bookkeeping is shared. Every per-spectrum reduction uses the
# same NumPy call as the single-spectrum code (a matrix product of J^T
# with J, of J^T with r, of r with r, `numpy.linalg.solve` with one
# right-hand side, `numpy.linalg.inv`), stacked over the spectra, and
# every elementwise expression keeps its order of operations, so each
# spectrum's numbers come out bit for bit as the single-spectrum fit
# gives them. The tests assert that equality on noisy maps, with
# pixels without a peak, for both baselines and both fit modes.

def _initial_guess_rows(x, Y):
    """`_initial_guess` for every row of Y, operation for operation."""
    P = Y.shape[0]
    b0 = np.min(Y, axis=1)
    i = np.argmax(Y, axis=1)
    A0 = Y[np.arange(P), i] - b0
    c0 = x[i]
    half = b0 + 0.5 * A0
    above = Y >= half[:, None]                 # never empty: the max is in
    G0 = (np.max(np.where(above, x, -np.inf), axis=1)
          - np.min(np.where(above, x, np.inf), axis=1))
    G0 = np.maximum(G0, 2.0 * abs(float(x[1] - x[0])))
    return np.column_stack([c0, G0, np.where(A0 > 0, A0, 1.0), b0])


def _lorentzian_rows(x, p, lin):
    """`_model_and_jacobian(_linear)` for every row of p (x0 in the
    last column of p when lin), same order of operations per element."""
    c, G, A, b = (p[:, k:k + 1] for k in range(4))
    h = 0.5 * G
    u = x[None, :] - c
    d = u * u + h * h
    core = h * h / d
    y = A * core + b
    J = np.empty((p.shape[0], x.size, 5 if lin else 4))
    J[:, :, 0] = A * h * h * 2.0 * u / (d * d)   # d/dc
    J[:, :, 1] = A * h * u * u / (d * d)         # d/dG
    J[:, :, 2] = core                            # d/dA
    J[:, :, 3] = 1.0                             # d/db
    return y, J


def _row_costs(r):
    """r @ r for every row of r."""
    return np.matmul(r[:, None, :], r[:, :, None])[:, 0, 0]


def _solve_rows(M, g):
    """`numpy.linalg.solve(M[k], g[k])` for every k. Returns the steps
    and None, or, if some matrix is exactly singular, the steps of the
    others and a boolean mask of the singular ones."""
    try:
        return np.linalg.solve(M, g[:, :, None])[:, :, 0], None
    except np.linalg.LinAlgError:              # pragma: no cover
        out = np.full(g.shape, np.nan)
        bad = np.zeros(g.shape[0], dtype=bool)
        for k in range(g.shape[0]):
            try:
                out[k] = np.linalg.solve(M[k], g[k][:, None])[:, 0]
            except np.linalg.LinAlgError:
                bad[k] = True
        return out, bad


def _lm_rows(Y, p, model, gauges, max_iter, tol):
    """The Levenberg-Marquardt loop of `fit_lorentzian` (and of the
    joint fit), stacked over the rows of Y from the starting values p.
    model(p) returns the model rows and Jacobians for the rows of p;
    gauges are the parameter indices whose sign is a gauge (widths).
    Returns (p, cost, J, n_iter, converged)."""
    p = p.copy()
    P = Y.shape[0]
    npar = p.shape[1]
    lam = np.full(P, 1e-3)
    yfit, J = model(p)
    r = Y - yfit
    cost = _row_costs(r)
    active = np.ones(P, dtype=bool)
    converged = np.zeros(P, dtype=bool)
    n_iter = np.full(P, max_iter)
    eye = np.eye(npar, dtype=bool)
    for it in range(1, max_iter + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        Ja = J[idx]
        JaT = Ja.transpose(0, 2, 1)
        JTJ = np.matmul(JaT, Ja)
        g = np.matmul(JaT, r[idx][:, :, None])[:, :, 0]
        D = np.where(eye, JTJ, 0.0)              # np.diag(np.diag(JTJ))
        step, singular = _solve_rows(JTJ + lam[idx][:, None, None] * D, g)
        if singular is not None:                 # pragma: no cover
            lam[idx[singular]] *= 10.0
            idx, step = idx[~singular], step[~singular]
            if idx.size == 0:
                continue
        p_new = p[idx] + step
        for k in gauges:
            p_new[:, k] = np.abs(p_new[:, k])
        y_new, J_new = model(p_new)
        r_new = Y[idx] - y_new
        cost_new = _row_costs(r_new)
        better = cost_new < cost[idx]
        ib = idx[better]
        rel = ((cost[ib] - cost_new[better])
               / np.maximum(cost[ib], 1e-300))
        p[ib], r[ib], J[ib] = p_new[better], r_new[better], J_new[better]
        cost[ib] = cost_new[better]
        lam[ib] = np.maximum(lam[ib] / 10.0, 1e-12)
        done = ib[rel < tol]
        iw = idx[~better]
        lam[iw] *= 10.0
        stuck = iw[lam[iw] > 1e12]               # stuck at a minimum
        for fin in (done, stuck):
            converged[fin] = True
            active[fin] = False
            n_iter[fin] = it
    return p, cost, J, r, n_iter, converged


def _cov_rows(J, cost, r, noise):
    """The covariance of every row, as the single-spectrum code
    computes it: s^2 (J^T J)^-1 with s^2 = cost / (n - p), or the
    sandwich of `_sandwich` for noise="varying"; NaN where J^T J is
    singular (where the single-spectrum code reports NaN)."""
    n, npar = J.shape[1:]
    JTJ = np.matmul(J.transpose(0, 2, 1), J)
    try:
        inv = np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:                # pragma: no cover
        inv = np.full(JTJ.shape, np.nan)
        for k in range(JTJ.shape[0]):
            try:
                inv[k] = np.linalg.inv(JTJ[k])
            except np.linalg.LinAlgError:
                pass
    if noise == "varying":
        M = np.matmul((J * (r * r)[:, :, None]).transpose(0, 2, 1), J)
        return (n / (n - npar)) * np.matmul(np.matmul(inv, M), inv)
    return (cost / (n - npar))[:, None, None] * inv


def _fit_lorentzian_rows(x, Y, baseline="constant", max_iter=200,
                         tol=1e-12, noise="constant"):
    """Fit one Lorentzian to every row of Y (P, n) on the shared axis x
    (n,), as `fit_lorentzian(x, Y[k], baseline=baseline)` would, and
    return a list of P `PeakFit`s. x and every row must be finite and
    n large enough (checked by the callers). A row whose starting width
    is not positive (possible only on an axis with repeated values),
    which `fit_lorentzian` refuses with a ValueError, gives None."""
    x = np.asarray(x, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float)
    lin = baseline == "linear"
    n = x.size
    x0 = float(x.mean())
    if lin:
        m0 = (Y[:, -1] - Y[:, 0]) / (float(x[-1]) - float(x[0]))
        p = np.column_stack([
            _initial_guess_rows(x, Y - m0[:, None] * (x - x0)[None, :]),
            m0])
    else:
        p = _initial_guess_rows(x, Y)
    startable = p[:, 1] > 0
    out = [None] * Y.shape[0]
    if not np.any(startable):
        return out
    dx = x - x0

    def model(pv):
        y, J = _lorentzian_rows(x, pv, lin)
        if lin:
            y = y + pv[:, 4:5] * dx[None, :]
            J[:, :, 4] = dx                      # d/dm
        return y, J

    p, cost, J, r, n_iter, conv = _lm_rows(Y[startable], p[startable],
                                           model, (1,), max_iter, tol)
    cov = _cov_rows(J, cost, r, noise)
    sig = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0.0))
    rms = np.sqrt(cost / n)
    for k, row in enumerate(np.flatnonzero(startable)):
        out[row] = PeakFit(
            center=float(p[k, 0]), fwhm=float(p[k, 1]),
            amplitude=float(p[k, 2]), offset=float(p[k, 3]),
            center_sigma=float(sig[k, 0]), fwhm_sigma=float(sig[k, 1]),
            amplitude_sigma=float(sig[k, 2]),
            offset_sigma=float(sig[k, 3]), residual_rms=float(rms[k]),
            n_iter=int(n_iter[k]), converged=bool(conv[k]),
            slope=float(p[k, 4]) if lin else 0.0,
            slope_sigma=float(sig[k, 4]) if lin else float("nan"))
    return out


def _fit_joint_rows(x, Y, w1, w2, baseline, starts, max_iter=300,
                    tol=1e-12, noise="constant"):
    """`_fit_two_lorentzians_joint` for every row of Y (P, n) on the
    shared full axis x, with starts[k] = (PeakFit of mode 1, PeakFit of
    mode 2) from the separate window fits. Returns a list of P pairs of
    `PeakFit`s, bit for bit those of the single-spectrum joint fit."""
    lin = baseline == "linear"
    npar = 8 if lin else 7
    lo = min(w1[0], w2[0])
    hi = max(w1[1], w2[1])
    sel = (x >= lo) & (x <= hi)
    xs, Ys = x[sel], Y[:, sel]
    n = xs.size
    if n - npar < 1:
        raise ValueError("the joint fit span holds too few points")
    x0 = float(xs.mean())
    dx = xs - x0
    p = np.array([[a.center, a.fwhm, a.amplitude, b.center, b.fwhm,
                   b.amplitude, min(a.offset, b.offset)]
                  + ([0.0] if lin else []) for a, b in starts],
                 dtype=float).reshape(len(starts), npar)

    def model(pv):
        J = np.empty((pv.shape[0], n, npar))
        y = np.repeat(pv[:, 6:7], n, axis=1)
        for k in range(2):
            c, G, A = (pv[:, 3 * k + q:3 * k + q + 1] for q in range(3))
            h = 0.5 * G
            u = xs[None, :] - c
            d = u * u + h * h
            core = h * h / d
            y = y + A * core
            J[:, :, 3 * k] = A * h * h * 2.0 * u / (d * d)
            J[:, :, 3 * k + 1] = A * h * u * u / (d * d)
            J[:, :, 3 * k + 2] = core
        J[:, :, 6] = 1.0
        if lin:
            y = y + pv[:, 7:8] * dx[None, :]
            J[:, :, 7] = dx
        return y, J

    p, cost, J, r, n_iter, conv = _lm_rows(Ys, p, model, (1, 4),
                                           max_iter, tol)
    cov = _cov_rows(J, cost, r, noise)
    sig = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = cov[:, 0, 3] / (sig[:, 0] * sig[:, 3])
    rms = np.sqrt(cost / n)
    out = []
    for i in range(p.shape[0]):
        pair = []
        for k in range(2):
            pair.append(PeakFit(
                center=float(p[i, 3 * k]), fwhm=float(p[i, 3 * k + 1]),
                amplitude=float(p[i, 3 * k + 2]), offset=float(p[i, 6]),
                center_sigma=float(sig[i, 3 * k]),
                fwhm_sigma=float(sig[i, 3 * k + 1]),
                amplitude_sigma=float(sig[i, 3 * k + 2]),
                offset_sigma=float(sig[i, 6]), residual_rms=float(rms[i]),
                n_iter=int(n_iter[i]), converged=bool(conv[i]),
                slope=float(p[i, 7]) if lin else 0.0,
                slope_sigma=float(sig[i, 7]) if lin else float("nan"),
                center_correlation=float(rho[i])))
        out.append(tuple(pair))
    return out


# ----------------------------------------------------------------------
# Voigt lineshape (new in v0.6): the gate named above, closed properly.

@dataclasses.dataclass
class VoigtFit:
    """Result of a single-peak Voigt fit.

    sigma is the Gaussian standard deviation, gamma the Lorentzian
    half-width; the derived widths gaussian_fwhm = 2 sqrt(2 ln 2) sigma
    and lorentzian_fwhm = 2 gamma are exact definitional conversions.
    amplitude is the peak height above offset, as in `fit_lorentzian`.
    """

    center: float
    sigma: float
    gamma: float
    amplitude: float
    offset: float
    center_sigma: float
    sigma_sigma: float
    gamma_sigma: float
    amplitude_sigma: float
    offset_sigma: float
    gaussian_fwhm: float
    lorentzian_fwhm: float
    residual_rms: float
    n_iter: int
    converged: bool


def voigt(x, center, sigma, gamma, amplitude, offset=0.0):
    """Voigt lineshape via the Faddeeva function, peak-height
    normalized: amplitude * Re w(z) / Re w(z0) + offset with
    z = ((x - center) + i gamma) / (sigma sqrt(2)) and z0 = z(x=center).

    Exact limits, asserted in the tests rather than stated: gamma -> 0
    is the Gaussian exp(-(x-c)^2 / (2 sigma^2)) exactly (Re w of a real
    argument is exp(-z^2)), and sigma -> 0 approaches the Lorentzian of
    half-width gamma.
    """
    from scipy.special import wofz
    x = np.asarray(x, dtype=float)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive (use fit_lorentzian "
                         "for a pure Lorentzian)")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    s2 = sigma * np.sqrt(2.0)
    z = ((x - center) + 1j * gamma) / s2
    z0 = 1j * gamma / s2
    return amplitude * np.real(wofz(z)) / np.real(wofz(z0)) + offset


def _voigt_model_and_jacobian(x, p):
    from scipy.special import wofz
    c, s, g, A, b = p
    s2 = s * np.sqrt(2.0)
    u = x - c
    z = (u + 1j * g) / s2
    z0 = 1j * g / s2
    w = wofz(z)
    w0 = wofz(z0)
    f = np.real(w)
    f0 = float(np.real(w0))
    dw = 2j / np.sqrt(np.pi) - 2.0 * z * w        # w'(z)
    dw0 = 2j / np.sqrt(np.pi) - 2.0 * z0 * w0
    # partials of f = Re w(z(u, s, g)) and of f0
    df_du = np.real(dw) / s2
    df_dg = -np.imag(dw) / s2
    df_ds = -np.real(z * dw) / s
    df0_dg = float(-np.imag(dw0) / s2)
    df0_ds = float(-np.real(z0 * dw0) / s)
    y = A * f / f0 + b
    J = np.empty((x.size, 5))
    J[:, 0] = -A * df_du / f0
    J[:, 1] = A * (df_ds * f0 - f * df0_ds) / (f0 * f0)
    J[:, 2] = A * (df_dg * f0 - f * df0_dg) / (f0 * f0)
    J[:, 3] = f / f0
    J[:, 4] = 1.0
    return y, J


def fit_voigt(x, y, p0=None, max_iter=300, tol=1e-12) -> VoigtFit:
    """Fit one Voigt peak by Levenberg-Marquardt with the analytic
    Faddeeva-function Jacobian (checked against finite differences in
    the tests, like the Lorentzian fitter's).

    p0 : optional (center, sigma, gamma, amplitude, offset); by default
    seeded from the Lorentzian-style guess of the full width G0: a
    Gaussian part of FWHM G0 (sigma = G0 / (2 sqrt(2 ln 2))) and a
    Lorentzian part of FWHM G0 / 2 (gamma = G0 / 4).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < 6:
        raise ValueError("need at least 6 points to fit 5 parameters "
                         "and estimate a residual variance")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must be finite; clean or mask the "
                         "spectrum before fitting")
    if p0 is None:
        c0, G0, A0, b0 = _initial_guess(x, y)
        p = np.array([c0, G0 / (4.0 * np.sqrt(2.0 * np.log(2.0))) * 2.0,
                      G0 / 4.0, A0, b0])
    else:
        p = np.asarray(p0, dtype=float)
        if p.shape != (5,):
            raise ValueError("p0 must be (center, sigma, gamma, "
                             "amplitude, offset)")
    if p[1] <= 0:
        raise ValueError("starting sigma must be positive")

    lam = 1e-3
    yfit, J = _voigt_model_and_jacobian(x, p)
    r = y - yfit
    cost = float(r @ r)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        JTJ = J.T @ J
        g = J.T @ r
        try:
            step = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ)), g)
        except np.linalg.LinAlgError:            # pragma: no cover
            lam *= 10.0
            continue
        p_new = p + step
        p_new[1] = abs(p_new[1])                 # width signs are gauges
        p_new[2] = abs(p_new[2])
        y_new, J_new = _voigt_model_and_jacobian(x, p_new)
        r_new = y - y_new
        cost_new = float(r_new @ r_new)
        if cost_new < cost:
            rel = (cost - cost_new) / max(cost, 1e-300)
            p, r, J, cost = p_new, r_new, J_new, cost_new
            lam = max(lam / 10.0, 1e-12)
            if rel < tol:
                converged = True
                break
        else:
            lam *= 10.0
            if lam > 1e12:
                converged = True                 # stuck at a minimum
                break

    dof = x.size - 5
    s2 = cost / dof
    JTJ = J.T @ J
    try:
        cov = s2 * np.linalg.inv(JTJ)
        sig = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:                # pragma: no cover
        sig = np.full(5, np.nan)
    return VoigtFit(
        center=float(p[0]), sigma=float(p[1]), gamma=float(p[2]),
        amplitude=float(p[3]), offset=float(p[4]),
        center_sigma=float(sig[0]), sigma_sigma=float(sig[1]),
        gamma_sigma=float(sig[2]), amplitude_sigma=float(sig[3]),
        offset_sigma=float(sig[4]),
        gaussian_fwhm=float(2.0 * np.sqrt(2.0 * np.log(2.0)) * p[1]),
        lorentzian_fwhm=float(2.0 * p[2]),
        residual_rms=float(np.sqrt(cost / x.size)),
        n_iter=it, converged=converged,
    )
