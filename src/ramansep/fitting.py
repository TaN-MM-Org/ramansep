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
trustworthy when the residuals are dominated by uncorrelated noise, and
that caveat is documented rather than hidden.

Exact facts the test suite asserts, rather than states:

* The analytic Jacobian matches finite differences on every parameter.
* On noiseless synthetic Lorentzians the fit recovers center, width,
  amplitude and baseline to 1e-8 from automatic starting values.
* On seeded noisy spectra the center error is compatible with the
  reported center uncertainty.
* A full round trip (strain/density maps -> forward shifts -> synthetic
  spectra -> `fit_two_modes` -> `SeparationModel.invert`) returns the
  input maps.

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


def fit_lorentzian(x, y, p0=None, max_iter=200, tol=1e-12,
                   baseline="constant") -> PeakFit:
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

    Returns a `PeakFit`. `converged` is False if `max_iter` was hit
    first; the parameters returned are then the best found, and the
    caller decides whether to trust them.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if baseline not in ("constant", "linear"):
        raise ValueError('baseline must be "constant" or "linear"')
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


def fit_two_modes(x, y, window1, window2, ref1, ref2,
                  baseline="constant", joint=False):
    """Fit both modes of a spectrum and return inversion-ready shifts.

    x, y : the spectrum (wavenumber axis and counts).
    window1, window2 : (lo, hi) wavenumber windows, one per mode. The
        windows must not overlap: a shared shoulder would be counted
        twice, once per fit.
    ref1, ref2 : pristine-material reference frequencies of the two
        modes (cm^-1), the zero points of the shifts.
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
        treats dw1 and dw2 as independent).

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
    fits = []
    for lo, hi in (w1, w2):
        m = (x >= lo) & (x <= hi)
        fits.append(fit_lorentzian(x[m], y[m], baseline=baseline))
    fit1, fit2 = fits
    if joint:
        fit1, fit2 = _fit_two_lorentzians_joint(x, y, w1, w2, baseline,
                                                fit1, fit2)
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
                               max_iter=300, tol=1e-12):
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
    seeded from the Lorentzian-style guess with the width split evenly
    between the Gaussian and Lorentzian parts.
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
