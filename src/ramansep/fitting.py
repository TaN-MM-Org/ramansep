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


def fit_lorentzian(x, y, p0=None, max_iter=200, tol=1e-12) -> PeakFit:
    """Fit one Lorentzian peak by Levenberg-Marquardt least squares.

    x, y : 1D arrays of equal length >= 5 (four parameters plus one
        degree of freedom for the noise estimate).
    p0 : optional starting values (center, fwhm, amplitude, offset);
        by default estimated from the data (peak position, half-maximum
        crossing width, minimum as baseline).
    tol : relative decrease of the cost at which iteration stops.

    Returns a `PeakFit`. `converged` is False if `max_iter` was hit
    first; the parameters returned are then the best found, and the
    caller decides whether to trust them.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < 5:
        raise ValueError("need at least 5 points to fit 4 parameters "
                         "and estimate a residual variance")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must be finite; clean or mask the "
                         "spectrum before fitting")

    p = np.asarray(p0, dtype=float) if p0 is not None else _initial_guess(x, y)
    if p.shape != (4,):
        raise ValueError("p0 must be (center, fwhm, amplitude, offset)")
    if p[1] <= 0:
        raise ValueError("starting fwhm must be positive")

    lam = 1e-3
    yfit, J = _model_and_jacobian(x, p)
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
        y_new, J_new = _model_and_jacobian(x, p_new)
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

    dof = x.size - 4
    s2 = cost / dof
    JTJ = J.T @ J
    try:
        cov = s2 * np.linalg.inv(JTJ)
        sig = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:                # pragma: no cover
        sig = np.full(4, np.nan)
    return PeakFit(
        center=float(p[0]), fwhm=float(p[1]), amplitude=float(p[2]),
        offset=float(p[3]), center_sigma=float(sig[0]),
        fwhm_sigma=float(sig[1]), amplitude_sigma=float(sig[2]),
        offset_sigma=float(sig[3]),
        residual_rms=float(np.sqrt(cost / x.size)),
        n_iter=it, converged=converged,
    )


def fit_two_modes(x, y, window1, window2, ref1, ref2):
    """Fit both modes of a spectrum and return inversion-ready shifts.

    x, y : the spectrum (wavenumber axis and counts).
    window1, window2 : (lo, hi) wavenumber windows, one per mode. The
        windows must not overlap: a shared shoulder would be counted
        twice, once per fit.
    ref1, ref2 : pristine-material reference frequencies of the two
        modes (cm^-1), the zero points of the shifts.

    Returns (dw1, dw2, sigma1, sigma2, fit1, fit2): the two peak shifts
    relative to the references, their 1-sigma uncertainties, and the two
    full `PeakFit` results. Feed the first four straight into
    `SeparationModel.invert(dw1, dw2, sigma1, sigma2)`.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    w1 = tuple(float(v) for v in window1)
    w2 = tuple(float(v) for v in window2)
    for w in (w1, w2):
        if w[0] >= w[1]:
            raise ValueError(f"window {w} must be (lo, hi) with lo < hi")
    if min(w1[1], w2[1]) > max(w1[0], w2[0]):
        raise ValueError("the two windows overlap; each mode must be "
                         "fitted on its own spectral range")
    fits = []
    for lo, hi in (w1, w2):
        m = (x >= lo) & (x <= hi)
        if m.sum() < 5:
            raise ValueError(f"window ({lo}, {hi}) contains fewer than "
                             "5 spectral points")
        fits.append(fit_lorentzian(x[m], y[m]))
    fit1, fit2 = fits
    return (fit1.center - float(ref1), fit2.center - float(ref2),
            fit1.center_sigma, fit2.center_sigma, fit1, fit2)


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
