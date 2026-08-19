"""Coefficient sets for mode pairs.

This module ships two CITED example coefficient sets for monolayer 1H-MoS2
(see mos2_a1_2la and mos2_eprime_a1 below), plus a synthetic set for the
test-suite. Every number in the cited sets is traceable to a published
measurement or to the first-principles calculation of the reference given
in its `reference` field; the docstring of each function states which
numbers come from where.

The warning of v0.1 still applies in general: peak-shift lever arms depend
on material, mode pair, excitation wavelength and substrate. Before using
either MoS2 set on your own spectra, check that your sample, excitation
and substrate match the conditions stated in the sources. For any other
material or mode pair, populate a ModeCoefficients from the literature or
from your own calibration; the mandatory `reference` field keeps the
provenance attached to the analysis.

Units used by both MoS2 sets:
    strain  : percent biaxial strain (positive = tension)
    density : electron sheet density in units of 1e13 cm^-2
    shifts  : cm^-1 (softening, i.e. redshift, is negative)
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
    k1_density : shift of mode 1 per unit carrier density (e.g. cm^-1 / 1e13 cm^-2)
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


def mos2_a1_2la() -> ModeCoefficients:
    """Monolayer 1H-MoS2, A'1 + 2LA(M) pair (the pair of the source paper).

    Strain lever arms are the frozen-phonon DFT values computed in the
    source paper: the A'1 mode softens by 5.1 cm^-1 per percent of biaxial
    tension and the disorder-activated 2LA(M) overtone by 20.9 cm^-1 per
    percent (acoustic Grueneisen parameter gamma_LA = 2.30), a lever-arm
    ratio of 4.1. The A'1 doping coefficient is the gated-Raman measurement
    of Chakraborty et al.: a softening of 2.2 cm^-1 per 1e13 cm^-2 of
    electrons, from the symmetry-selective coupling of the out-of-plane
    mode to the K-valley density; the E'-symmetry channel, and with it the
    2LA(M) overtone, couples only weakly.

    CAVEAT carried over from the source paper: no gated measurement of the
    2LA(M) doping coefficient exists, so it is set to zero here rather than
    known to be zero. The source paper bounds the consequence: letting the
    overtone renormalize at up to half the frequency-scaled A'1 rate
    changes a recovered edge charge by 16 percent.

    Units: percent biaxial strain; electron density in 1e13 cm^-2.
    Applicability: 1H monolayer, 532 nm excitation, SiO2-supported, 300 K.
    """
    return ModeCoefficients(
        mode1_name="A'1",
        mode2_name="2LA(M)",
        k1_strain=-5.1,
        k1_density=-2.2,
        k2_strain=-20.9,
        k2_density=0.0,
        reference=(
            "Strain lever arms (DFT, frozen phonon): T. M. Mahim and "
            "M. M. Rahman, 'Two Raman phonons quantify the fixed edge "
            "charge left by patterning monolayer transition metal "
            "dichalcogenides' (under review); code and data: "
            "github.com/Tanvir-Mahmud-Mahim/Width-scaling-in-monolayer-"
            "semiconductor-nanoribbon-transistors, "
            "doi:10.5281/zenodo.21778170. A'1 doping coefficient "
            "(measured): B. Chakraborty et al., Phys. Rev. B 85, "
            "161403(R) (2012). 2LA(M) doping coefficient set to zero "
            "(unmeasured; see docstring caveat)."
        ),
    )


def mos2_eprime_a1() -> ModeCoefficients:
    """Monolayer 1H-MoS2, E' + A'1 pair, from direct biaxial measurements.

    Strain lever arms follow from the measured biaxial-strain mode
    Grueneisen parameters of Michail et al. (gamma_E' = 0.56,
    gamma_A'1 = 0.31, in the convention gamma = -(1/2w) dw/deps) applied
    to the mode frequencies w(E') = 385.0 cm^-1 and w(A'1) = 403.0 cm^-1:
    dw/deps = -2 gamma w = -4.31 (E') and -2.50 (A'1) cm^-1 per percent
    of biaxial strain. Doping coefficients are the gated-Raman
    measurements of Chakraborty et al.: -0.33 (E') and -2.2 (A'1) cm^-1
    per 1e13 cm^-2 of electrons.

    This is the historically used all-optical pair (Michail et al. 2016).
    Both optical modes respond weakly to strain (lever arms of -4.31 and
    -2.50 cm^-1 per percent, against -20.9 for the 2LA(M) overtone), so
    for equal peak-shift noise this pair returns a strain uncertainty
    about five times larger than mos2_a1_2la(); propagate sigmas through
    invert() and compare strain_sigma before choosing it.

    Units: percent biaxial strain; electron density in 1e13 cm^-2.
    Applicability: 1H monolayer, SiO2-supported, 300 K.
    """
    return ModeCoefficients(
        mode1_name="E'",
        mode2_name="A'1",
        k1_strain=-4.31,
        k1_density=-0.33,
        k2_strain=-2.50,
        k2_density=-2.2,
        reference=(
            "Strain: measured biaxial Grueneisen parameters, A. Michail "
            "et al., ACS Appl. Mater. Interfaces 16, 49602 (2024) "
            "(gamma_E'=0.56, gamma_A'1=0.31), converted with "
            "w(E')=385.0 cm^-1 and w(A'1)=403.0 cm^-1 (Mignuzzi et al., "
            "Phys. Rev. B 91, 195411 (2015); Michail et al., Appl. Phys. "
            "Lett. 108, 173102 (2016)). Doping (measured): B. Chakraborty "
            "et al., Phys. Rev. B 85, 161403(R) (2012). Compiled as in "
            "T. M. Mahim and M. M. Rahman (under review), "
            "doi:10.5281/zenodo.21778170."
        ),
    )
