"""Temperature as a third cause, and as a direct measurement (new in v0.9).

Raman peak positions respond to temperature as well as to strain and
carrier density: under laser heating or a temperature-controlled stage,
a two-cause inversion silently books the thermal shift as strain or
doping. This module addresses that in two independent, cross-checkable
ways.

1. `ThreeCauseModel`: the generalized-least-squares separation extended
   to a third column of lever arms -- your calibrated first-order
   temperature coefficients d(omega)/dT (e.g. cm^-1/K), which are
   material-, mode-, thickness- and substrate-specific and therefore,
   as everywhere in this package, must be supplied by you with their
   source (for monolayer/few-layer MoS2 see e.g. S. Sahoo et al.,
   J. Phys. Chem. C 117, 9042 (2013); no number is shipped here).
   Requires at least three modes; the same chi-square, covariance and
   conditioning machinery as the two-cause solver, so an unidentifiable
   mode set is refused rather than silently pseudo-inverted.

2. `temperature_from_anti_stokes`: the anti-Stokes/Stokes intensity
   ratio depends on temperature through the Bose-Einstein occupation,

       I_AS / I_S = C exp(-hbar omega / k_B T),

   the standard Raman thermometer. The prefactor C absorbs the
   frequency-dependent scattering cross-section and the spectrometer
   response, is close to but not exactly 1, and is calibrated at one
   known temperature (`calibrate_anti_stokes`) -- standard practice,
   because assuming C = 1 biases the temperature. The inversion is the
   closed form T = (hc omega_tilde / k_B) / ln(C / ratio), with
   hc/k_B the second radiation constant computed here from the exact
   SI defining constants, never typed by hand, and asserted in the
   tests against SciPy's CODATA table (two independent sources).

Measuring T with the thermometer and then feeding it to the shift
separation as a fixed, subtracted cause -- or estimating it jointly
with `ThreeCauseModel` and comparing -- is exactly the cross-validation
an experiment wants.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.stats import chi2

__all__ = ["ThreeCauseModel", "ThreeCauseResult", "HC_OVER_KB_CM_K",
           "anti_stokes_ratio", "calibrate_anti_stokes",
           "temperature_from_anti_stokes"]

# Second radiation constant c2 = h c / k_B in cm K, from the three
# EXACT SI defining constants (2019 SI): h = 6.62607015e-34 J s,
# c = 299792458 m/s (= 2.99792458e10 cm/s), k_B = 1.380649e-23 J/K.
HC_OVER_KB_CM_K = 6.62607015e-34 * 2.99792458e10 / 1.380649e-23


@dataclasses.dataclass
class ThreeCauseResult:
    """Per-pixel strain, density and temperature with uncertainties.

    strain, density, temperature : the estimated cause maps, in the
    units implied by the lever-arm matrix columns.
    strain_sigma, density_sigma, temperature_sigma : one-standard-
    deviation uncertainties propagated from the shift sigmas.
    covariance : (..., 3, 3) full per-pixel covariance of
    (strain, density, temperature).
    chi2_map, p_value : per-pixel goodness of fit against dof = m - 3
    (None when m == 3, which is exactly determined).
    condition_number : of the lever-arm matrix.
    """

    strain: np.ndarray
    density: np.ndarray
    temperature: np.ndarray
    strain_sigma: np.ndarray
    density_sigma: np.ndarray
    temperature_sigma: np.ndarray
    covariance: np.ndarray
    chi2_map: np.ndarray | None
    p_value: np.ndarray | None
    dof: int
    condition_number: float


class ThreeCauseModel:
    """Separate strain, carrier density AND temperature from m >= 3 modes.

    K : (m, 3) lever-arm matrix; columns are (d omega/d strain,
    d omega/d density, d omega/d temperature) for each mode, in
    consistent units of your choice. The third column is your
    calibrated temperature coefficient set -- cited, like every
    physical number that drives this package. K must have full rank 3:
    three modes whose responses are linearly dependent cannot separate
    three causes, and this refuses rather than pseudo-inverts.

    The estimator is exactly the weighted least squares of
    `MultiModeModel` with one more column: per pixel,
    x_hat = (K^T W K)^{-1} K^T W s with W = diag(1/sigma_k^2),
    covariance (K^T W K)^{-1}, and the weighted residual chi-square
    against dof = m - 3.
    """

    def __init__(self, K, mode_names=None):
        self.K = np.asarray(K, dtype=float)
        if self.K.ndim != 2 or self.K.shape[1] != 3 or self.K.shape[0] < 3:
            raise ValueError("K must have shape (m >= 3, 3): three causes "
                             "need at least three modes")
        if not np.all(np.isfinite(self.K)):
            raise ValueError("K must be finite")
        if np.linalg.matrix_rank(self.K) < 3:
            raise ValueError("lever-arm matrix has rank < 3: these modes "
                             "cannot separate strain, density and "
                             "temperature")
        self.m = self.K.shape[0]
        self.mode_names = (list(mode_names) if mode_names is not None
                           else [f"mode{i+1}" for i in range(self.m)])
        if len(self.mode_names) != self.m:
            raise ValueError("one name per mode required")
        self.condition_number = float(np.linalg.cond(self.K))

    def invert(self, shifts, sigmas=None) -> ThreeCauseResult:
        """Invert m shift maps into strain, density and temperature.

        shifts : sequence of m arrays (one per mode), all one shape.
        sigmas : matching one-standard-deviation shift uncertainties
        (scalars or arrays broadcastable to the map shape); omitted
        means unit weights, in which case chi2_map is calibrated only
        if the shift noise really is unit variance.
        """
        S = np.stack([np.asarray(s, dtype=float) for s in shifts])
        if S.shape[0] != self.m:
            raise ValueError(f"expected {self.m} shift maps, got {S.shape[0]}")
        map_shape = S.shape[1:]
        if sigmas is None:
            sig = np.ones_like(S)
        else:
            sig = np.stack([np.broadcast_to(np.asarray(s, dtype=float),
                                            map_shape).copy()
                            for s in sigmas])
            if np.any(sig <= 0):
                raise ValueError("sigmas must be positive")
        P = int(np.prod(map_shape)) if map_shape else 1
        s_flat = S.reshape(self.m, P)
        w_flat = 1.0 / sig.reshape(self.m, P) ** 2
        K = self.K
        # per-pixel information matrix A_p = K^T W_p K, (P, 3, 3)
        A = np.einsum("ki,kp,kj->pij", K, w_flat, K)
        b = np.einsum("ki,kp->pi", K, w_flat * s_flat)
        det = np.linalg.det(A)
        if np.any(det <= 0) or not np.all(np.isfinite(det)):
            raise ValueError("weighted design matrix is singular at some "
                             "pixels; check the coefficients and sigmas")
        x = np.linalg.solve(A, b[..., None])[..., 0]   # (P, 3)
        cov = np.linalg.inv(A)                    # (P, 3, 3)
        pred = np.einsum("ki,pi->kp", K, x)
        resid2 = np.einsum("kp,kp->p", w_flat, (s_flat - pred) ** 2)
        dof = self.m - 3
        if dof > 0:
            chi2_map = resid2.reshape(map_shape)
            p_value = chi2.sf(resid2, dof).reshape(map_shape)
        else:
            chi2_map = None
            p_value = None
        sd = np.sqrt(np.einsum("pii->pi", cov))
        return ThreeCauseResult(
            strain=x[:, 0].reshape(map_shape),
            density=x[:, 1].reshape(map_shape),
            temperature=x[:, 2].reshape(map_shape),
            strain_sigma=sd[:, 0].reshape(map_shape),
            density_sigma=sd[:, 1].reshape(map_shape),
            temperature_sigma=sd[:, 2].reshape(map_shape),
            covariance=cov.reshape(map_shape + (3, 3)),
            chi2_map=chi2_map, p_value=p_value, dof=dof,
            condition_number=self.condition_number)

    def forward(self, strain, density, temperature):
        """Predict the m peak shifts from strain, density and
        temperature maps (the exact inverse-crime partner of
        `invert`, used by the round-trip anchors)."""
        strain = np.asarray(strain, dtype=float)
        density = np.asarray(density, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        return [self.K[k, 0] * strain + self.K[k, 1] * density
                + self.K[k, 2] * temperature for k in range(self.m)]


def _check_omega(omega_cm1):
    w = float(omega_cm1)
    if not (w > 0.0 and np.isfinite(w)):
        raise ValueError("omega_cm1 must be a positive phonon frequency "
                         "in cm^-1")
    return w


def anti_stokes_ratio(temperature_K, omega_cm1, calibration=1.0):
    """Forward model: I_AS/I_S = C exp(-hc omega_tilde / k_B T).

    temperature_K : absolute temperature (K), > 0.
    omega_cm1 : phonon frequency in cm^-1 (Raman shift of the mode).
    calibration : the prefactor C from `calibrate_anti_stokes`
    (default 1.0 = the idealized textbook ratio; real spectrometers
    need the calibrated value).
    """
    T = float(temperature_K)
    if not (T > 0.0 and np.isfinite(T)):
        raise ValueError("temperature_K must be positive")
    w = _check_omega(omega_cm1)
    C = float(calibration)
    if not (C > 0.0 and np.isfinite(C)):
        raise ValueError("calibration must be positive")
    return C * float(np.exp(-HC_OVER_KB_CM_K * w / T))


def calibrate_anti_stokes(ratio_measured, omega_cm1, temperature_K):
    """The calibration constant C from one measurement at a KNOWN
    temperature: C = ratio * exp(+hc omega_tilde / k_B T_ref).

    Standard practice: measure the anti-Stokes/Stokes ratio with the
    sample held at a known temperature (e.g. an unheated spot, or a
    temperature stage), and let C absorb the omega^3/omega^4 cross-
    section factors and the spectral response of your spectrometer --
    which this package refuses to guess.
    """
    r = float(ratio_measured)
    if not (r > 0.0 and np.isfinite(r)):
        raise ValueError("ratio_measured must be positive")
    w = _check_omega(omega_cm1)
    T = float(temperature_K)
    if not (T > 0.0 and np.isfinite(T)):
        raise ValueError("temperature_K must be positive")
    return r * float(np.exp(HC_OVER_KB_CM_K * w / T))


def temperature_from_anti_stokes(ratio, omega_cm1, calibration,
                                 ratio_sigma=None):
    """Temperature from the anti-Stokes/Stokes ratio, closed form.

    T = (hc omega_tilde / k_B) / ln(C / ratio) -- the exact inversion
    of `anti_stokes_ratio`, asserted as a machine-precision round trip
    in the tests.

    ratio : measured I_AS/I_S (background-subtracted, same mode both
    sides). omega_cm1 : phonon frequency in cm^-1. calibration : C
    from `calibrate_anti_stokes`; there is no default, because C = 1
    silently biases the answer on any real spectrometer.
    ratio_sigma : optional 1-sigma uncertainty of the ratio; when
    given, the result is a dict(temperature, temperature_sigma) with
    sigma_T = T^2 / (c2 omega_tilde) * sigma_r / r by exact
    differentiation (anchored against a finite difference).

    Refuses ratio >= C: the Bose factor is strictly below 1, so such a
    ratio has no finite positive temperature -- it means the
    calibration or the background subtraction is wrong, and this says
    so instead of returning a number.
    """
    r = float(ratio)
    if not (r > 0.0 and np.isfinite(r)):
        raise ValueError("ratio must be positive")
    w = _check_omega(omega_cm1)
    C = float(calibration)
    if not (C > 0.0 and np.isfinite(C)):
        raise ValueError("calibration must be positive")
    if r >= C:
        raise ValueError("ratio >= calibration constant: the Bose factor "
                         "is strictly < 1, so no finite positive "
                         "temperature reproduces this ratio; check the "
                         "calibration and background subtraction")
    T = HC_OVER_KB_CM_K * w / float(np.log(C / r))
    if ratio_sigma is None:
        return T
    sr = float(ratio_sigma)
    if not (sr >= 0.0 and np.isfinite(sr)):
        raise ValueError("ratio_sigma must be finite and >= 0")
    sigma_T = T ** 2 / (HC_OVER_KB_CM_K * w) * (sr / r)
    return dict(temperature=T, temperature_sigma=sigma_T)
