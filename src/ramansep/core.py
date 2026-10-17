"""Linear two-mode separation of strain and carrier density."""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np

from .materials import ModeCoefficients


@dataclasses.dataclass
class SeparationResult:
    """Result of a two-mode inversion.

    Attributes
    ----------
    strain : ndarray
        Strain field, in the strain unit of the coefficient matrix
        (percent biaxial strain unless your coefficients say otherwise).
    density : ndarray
        Carrier-density field, in the density unit of the coefficient
        matrix (1e13 cm^-2 for the shipped MoS2 sets, a G-band shift in
        cm^-1 for the graphene set; whatever your own coefficients use).
    strain_sigma, density_sigma : ndarray or None
        One-standard-deviation uncertainties propagated from the peak-shift
        uncertainties, if those were provided.
    correlation : ndarray or None
        Pixelwise correlation coefficient between the strain and density
        estimates. Values near +/-1 mean the two modes barely separate the
        two causes at that pixel.
    condition_number : float
        Condition number of the coefficient matrix. Large values mean the
        mode pair is poorly chosen for separation.
    """

    strain: np.ndarray
    density: np.ndarray
    strain_sigma: np.ndarray | None
    density_sigma: np.ndarray | None
    correlation: np.ndarray | None
    condition_number: float


def _check_shift_correlation(shift_correlation, shape):
    """The correlation of the two shift errors, broadcast to the map
    shape; refuses values outside [-1, 1] and infinities (NaN passes as
    a masked pixel)."""
    try:
        rho = np.broadcast_to(np.asarray(shift_correlation, dtype=float),
                              shape)
    except ValueError:
        raise ValueError(
            f"shift_correlation of shape {np.shape(shift_correlation)} "
            f"does not broadcast to the map shape {shape}") from None
    with np.errstate(invalid="ignore"):
        bad = np.isinf(rho) | (np.abs(rho) > 1.0)
    if np.any(bad):
        raise ValueError("shift_correlation must lie in [-1, 1] (NaN "
                         "marks a masked pixel)")
    return rho


class SeparationModel:
    """Invert two peak-shift maps into strain and carrier-density maps.

    Parameters
    ----------
    coefficients : ModeCoefficients
        The 2x2 lever-arm matrix K such that

            [dw_mode1]   [k1_strain  k1_density] [strain ]
            [dw_mode2] = [k2_strain  k2_density] [density]

        where dw are peak shifts relative to the pristine reference.
    condition_warn : float
        Warn if the condition number of K exceeds this value.
    """

    def __init__(self, coefficients: ModeCoefficients, condition_warn: float = 30.0):
        self.coefficients = coefficients
        self.K = np.asarray(coefficients.matrix(), dtype=float)
        if self.K.shape != (2, 2):
            raise ValueError("coefficient matrix must be 2x2")
        if not np.all(np.isfinite(self.K)):
            raise ValueError("coefficients must be finite numbers")
        det = np.linalg.det(self.K)
        if det == 0.0:
            raise ValueError(
                "coefficient matrix is singular: the two modes respond "
                "identically and cannot separate strain from density"
            )
        self.condition_number = float(np.linalg.cond(self.K))
        if self.condition_number > condition_warn:
            warnings.warn(
                f"coefficient matrix condition number is "
                f"{self.condition_number:.1f}; the mode pair is poorly "
                "conditioned and small shift errors will produce large "
                "strain/density errors",
                stacklevel=2,
            )
        self.Kinv = np.linalg.inv(self.K)

    def invert(self, dw1, dw2, sigma1=None, sigma2=None,
               shift_correlation=None) -> SeparationResult:
        """Invert peak-shift maps (any matching numpy shapes) pixelwise.

        dw1, dw2 : peak shifts of mode 1 and mode 2 (cm^-1), relative to the
            pristine-material reference frequency of each mode.
        sigma1, sigma2 : optional 1-sigma uncertainties of the shifts,
            scalar or arrays broadcastable to the map shape. They must
            not be negative (refused since v0.12; before, a negative
            sigma was squared away silently). NaN is allowed and marks
            a masked pixel, as `fit_map` produces.
        shift_correlation : optional correlation coefficient of the two
            shift errors, scalar or broadcastable to the map shape, in
            [-1, 1] (new in v0.14). Two peaks fitted together
            (`fit_two_modes(..., joint=True)`, `fit_map(..., joint=True)`)
            have correlated centers; pass `fit1.center_correlation` or
            `MapFitResult.shift_correlation` here. The covariance of
            (strain, density) is then K^-1 S K^-T with the full shift
            covariance S = [[s1^2, r s1 s2], [r s1 s2, s2^2]] (the law of
            propagation of uncertainty with correlated inputs, JCGM
            100:2008 (GUM), 5.2.2, Eq. (13)). None (default) or 0 gives
            the independent-error result, bit for bit. Needs sigma1 and
            sigma2. NaN marks a masked pixel, like a NaN sigma.
        """
        dw1 = np.asarray(dw1, dtype=float)
        dw2 = np.asarray(dw2, dtype=float)
        if dw1.shape != dw2.shape:
            raise ValueError("the two shift maps must have the same shape; "
                             f"got {dw1.shape} and {dw2.shape}")

        strain = self.Kinv[0, 0] * dw1 + self.Kinv[0, 1] * dw2
        density = self.Kinv[1, 0] * dw1 + self.Kinv[1, 1] * dw2

        strain_sigma = density_sigma = correlation = None
        if shift_correlation is not None and (sigma1 is None
                                              or sigma2 is None):
            raise ValueError("shift_correlation needs both sigma1 and "
                             "sigma2")
        if sigma1 is not None and sigma2 is not None:
            s1 = np.broadcast_to(np.asarray(sigma1, dtype=float), dw1.shape)
            s2 = np.broadcast_to(np.asarray(sigma2, dtype=float), dw2.shape)
            if np.any(s1 < 0.0) or np.any(s2 < 0.0):
                raise ValueError("shift uncertainties sigma1, sigma2 must "
                                 "not be negative")
            var_strain = (self.Kinv[0, 0] * s1) ** 2 + (self.Kinv[0, 1] * s2) ** 2
            var_density = (self.Kinv[1, 0] * s1) ** 2 + (self.Kinv[1, 1] * s2) ** 2
            cov = (self.Kinv[0, 0] * self.Kinv[1, 0] * s1**2
                   + self.Kinv[0, 1] * self.Kinv[1, 1] * s2**2)
            if shift_correlation is not None:
                rho = _check_shift_correlation(shift_correlation, dw1.shape)
                c12 = rho * s1 * s2               # Cov(dw1, dw2)
                Ki = self.Kinv
                var_strain = var_strain + 2.0 * Ki[0, 0] * Ki[0, 1] * c12
                var_density = var_density + 2.0 * Ki[1, 0] * Ki[1, 1] * c12
                cov = cov + (Ki[0, 0] * Ki[1, 1] + Ki[0, 1] * Ki[1, 0]) * c12
            strain_sigma = np.sqrt(var_strain)
            density_sigma = np.sqrt(var_density)
            with np.errstate(invalid="ignore", divide="ignore"):
                correlation = cov / (strain_sigma * density_sigma)

        return SeparationResult(
            strain=strain,
            density=density,
            strain_sigma=strain_sigma,
            density_sigma=density_sigma,
            correlation=correlation,
            condition_number=self.condition_number,
        )

    def forward(self, strain, density):
        """Predict the two peak shifts from strain and density maps."""
        strain = np.asarray(strain, dtype=float)
        density = np.asarray(density, dtype=float)
        dw1 = self.K[0, 0] * strain + self.K[0, 1] * density
        dw2 = self.K[1, 0] * strain + self.K[1, 1] * density
        return dw1, dw2
