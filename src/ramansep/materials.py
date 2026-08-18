"""Coefficient sets for mode pairs.

IMPORTANT: this package ships NO validated material constants. Peak-shift
lever arms depend on material, mode pair, excitation wavelength and
substrate, and pasting unverified numbers into a measurement tool is how
wrong results propagate. Populate a ModeCoefficients from the literature
for your system (for the A'1 / 2LA(M) pair in 1H monolayers, see the
references in the README), or from your own calibration.

synthetic_demo() provides an arbitrary, clearly non-physical set used by
the test-suite and the example script only.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class ModeCoefficients:
    """Lever arms of two Raman modes.

    Units are your choice but must be consistent; the inversion returns
    strain and density in whatever units these coefficients imply.

    mode1_name, mode2_name : labels, e.g. "A'1" and "2LA(M)".
    k1_strain : shift of mode 1 per unit strain          (e.g. cm^-1 / %)
    k1_density : shift of mode 1 per unit carrier density (e.g. cm^-1 / 1e12 cm^-2)
    k2_strain, k2_density : same for mode 2.
    reference : where the numbers come from. Required, on purpose.
    """

    mode1_name: str
    mode2_name: str
    k1_strain: float
    k1_density: float
    k2_strain: float
    k2_density: float
    reference: str

    def matrix(self):
        return [[self.k1_strain, self.k1_density],
                [self.k2_strain, self.k2_density]]


def synthetic_demo() -> ModeCoefficients:
    """An arbitrary, NON-PHYSICAL coefficient set for tests and demos.

    The values are chosen only to be well-conditioned. Do not use for
    analysis of real spectra.
    """
    return ModeCoefficients(
        mode1_name="demo-mode-1",
        mode2_name="demo-mode-2",
        k1_strain=-2.0,
        k1_density=-0.8,
        k2_strain=-6.0,
        k2_density=-0.1,
        reference="synthetic demonstration values; not physical",
    )
