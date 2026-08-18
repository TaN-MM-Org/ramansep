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
        matrix (cm^-2 unless your coefficients say otherwise).
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

    def invert(self, dw1, dw2, sigma1=None, sigma2=None) -> SeparationResult:
        """Invert peak-shift maps (any matching numpy shapes) pixelwise.

        dw1, dw2 : peak shifts of mode 1 and mode 2 (cm^-1), relative to the
            pristine-material reference frequency of each mode.
        sigma1, sigma2 : optional 1-sigma uncertainties of the shifts,
            scalar or arrays broadcastable to the map shape.
        """
        dw1 = np.asarray(dw1, dtype=float)
        dw2 = np.asarray(dw2, dtype=float)
        if dw1.shape != dw2.shape:
            raise ValueError("the two shift maps must have the same shape")

        strain = self.Kinv[0, 0] * dw1 + self.Kinv[0, 1] * dw2
        density = self.Kinv[1, 0] * dw1 + self.Kinv[1, 1] * dw2

        strain_sigma = density_sigma = correlation = None
        if sigma1 is not None and sigma2 is not None:
            s1 = np.broadcast_to(np.asarray(sigma1, dtype=float), dw1.shape)
            s2 = np.broadcast_to(np.asarray(sigma2, dtype=float), dw2.shape)
            var_strain = (self.Kinv[0, 0] * s1) ** 2 + (self.Kinv[0, 1] * s2) ** 2
            var_density = (self.Kinv[1, 0] * s1) ** 2 + (self.Kinv[1, 1] * s2) ** 2
            cov = (self.Kinv[0, 0] * self.Kinv[1, 0] * s1**2
                   + self.Kinv[0, 1] * self.Kinv[1, 1] * s2**2)
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
