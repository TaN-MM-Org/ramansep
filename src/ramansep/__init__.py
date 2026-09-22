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
from .fitting import (PeakFit, VoigtFit, fit_lorentzian, fit_two_modes,
                      fit_voigt, lorentzian, voigt)
from .multimode import MultiModeModel, MultiModeResult, compare_mode_sets
from .bayesian import BayesianMapResult, bayesian_map_inversion
from .calibration import CalibrationResult, calibrate_lever_arms
from .mapfit import MapFitResult, fit_map
from .lab import (design_references, plan_calibration,
                  repeats_for_sigma)
from .propagate import separation_with_calibration
from .mapio import (load_map_csv, load_spectrum_csv, save_map_csv,
                    save_spectrum_csv)
from .materials import (ModeCoefficients, graphene_g_2d_lee2012,
                        mos2_a1_2la, mos2_eprime_a1, synthetic_demo)
from .thermal import (HC_OVER_KB_CM_K, ThreeCauseModel, ThreeCauseResult,
                      anti_stokes_ratio, calibrate_anti_stokes,
                      temperature_from_anti_stokes)

__version__ = "0.11.1"
__all__ = ["SeparationModel", "SeparationResult", "ModeCoefficients",
           "MultiModeModel", "MultiModeResult", "compare_mode_sets",
           "ThreeCauseModel", "ThreeCauseResult", "HC_OVER_KB_CM_K",
           "anti_stokes_ratio", "calibrate_anti_stokes",
           "temperature_from_anti_stokes",
           "BayesianMapResult", "bayesian_map_inversion",
           "CalibrationResult", "calibrate_lever_arms",
           "MapFitResult", "fit_map",
           "load_spectrum_csv", "save_spectrum_csv",
           "load_map_csv", "save_map_csv",
           "synthetic_demo", "mos2_a1_2la", "mos2_eprime_a1",
           "graphene_g_2d_lee2012",
           "PeakFit", "fit_lorentzian", "fit_two_modes", "lorentzian",
           "VoigtFit", "fit_voigt", "voigt",
           "plan_calibration", "design_references",
           "repeats_for_sigma", "separation_with_calibration"]
