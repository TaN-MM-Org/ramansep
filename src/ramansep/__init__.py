"""ramansep: separate strain from carrier density in 2D-material Raman maps.

Two Raman modes with different strain lever arms and different doping
responses form a linear, invertible probe of the (strain, carrier density)
state at each map pixel. This package implements that inversion with full
uncertainty propagation and conditioning diagnostics.

Methodological basis: T. M. Mahim and M. M. Rahman, "Two Raman phonons
quantify the fixed edge charge left by patterning monolayer transition
metal dichalcogenides" (under review). The solver is material-agnostic:
you supply the coefficient matrix for your material and mode pair.
"""
from .core import SeparationModel, SeparationResult
from .materials import ModeCoefficients, synthetic_demo

__version__ = "0.1.0"
__all__ = ["SeparationModel", "SeparationResult", "ModeCoefficients", "synthetic_demo"]
