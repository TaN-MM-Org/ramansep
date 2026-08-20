# ramansep

[![PyPI](https://img.shields.io/pypi/v/ramansep)](https://pypi.org/project/ramansep/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22014913-blue)](https://doi.org/10.5281/zenodo.22014913) [![tests](https://github.com/TaN-MM-Org/ramansep/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/ramansep/actions)

Separate **strain** from **carrier density** in Raman maps of 2D materials,
using two phonon modes whose lever arms differ.

A single Raman frequency responds to strain and to carrier density at once,
so one mode cannot tell the two apart. Two modes with sufficiently different
responses form a linear, invertible probe: measure both shift maps, invert a
2x2 matrix per pixel, and obtain a strain map and a carrier-density map with
propagated uncertainties.

## Status

v0.3.0 (alpha). The inversion core, uncertainty propagation, conditioning
diagnostics, a synthetic end-to-end example, two cited coefficient sets
for monolayer MoS2, and a cited graphene G + 2D set are implemented and
tested. The API may change before v1.0.

## Cited coefficient sets (new in v0.2)

Two example sets for monolayer 1H-MoS2 ship with full provenance, and the
test-suite reproduces the published separation results of the source paper
from them (edge charge of 2.3e12 cm^-2 from a 0.5 cm^-1 A'1 redshift with
2LA(M) unmoved; 0.134% interior tension carrying no significant charge):

- `mos2_a1_2la()`: the A'1 + 2LA(M) pair of the source paper. Strain lever
  arms -5.1 and -20.9 cm^-1 per percent of biaxial strain (frozen-phonon
  DFT, ratio 4.1); A'1 doping coefficient -2.2 cm^-1 per 1e13 cm^-2 of
  electrons (measured, Chakraborty et al., Phys. Rev. B 85, 161403(R)
  (2012)). The 2LA(M) doping coefficient is unmeasured and set to zero;
  the source paper bounds the consequence of that choice at 16% of a
  recovered edge charge.
- `mos2_eprime_a1()`: the historically used all-optical E' + A'1 pair,
  built from the measured biaxial Grueneisen parameters of Michail et al.,
  ACS Appl. Mater. Interfaces 16, 49602 (2024). Both optical modes respond
  weakly to strain, so for equal shift noise this pair returns about five
  times the strain uncertainty of the A'1 + 2LA(M) pair; it is included
  for comparison and for workflows where the overtone is not available.

New in v0.3, `graphene_g_2d_lee2012()`: the G + 2D pair of monolayer
graphene, expressing the vector decomposition of Lee et al., Nat. Commun.
3, 1024 (2012) as a linear inversion. Strain axis: -23.5 cm^-1 per percent
of randomly oriented uniaxial strain with the measured 2D/G slope of
2.2 +/- 0.2; hole-doping axis: the measured slope 0.70 +/- 0.05. The
doping output of this set is deliberately NOT a carrier density but the
G-band shift attributable to hole doping (cm^-1): the G-mode doping
response of graphene is nonlinear and sign-dependent, so no universal
linear per-density rate exists to ship, and the docstring points to the
gated calibrations (Froehlicher and Berciaud, Phys. Rev. B 91, 205413
(2015)) a user needs for the conversion on their own substrate.

Each function's docstring states which number comes from which source and
the conditions of applicability (for MoS2: 1H monolayer, SiO2-supported,
300 K; 532 nm for the disorder-activated calibration). Check that your
sample matches before use.

## What this package deliberately does not include

No constants beyond the three documented sets are shipped. Lever arms
depend on material, mode pair, excitation wavelength and substrate; a
measurement tool that ships unverified constants propagates wrong results.
For any other system you provide a `ModeCoefficients` from the literature
or your own calibration, and the `reference` field is mandatory so the
provenance of every number travels with the analysis.

## Install

```
pip install ramansep
```

For development, clone the repository and `pip install -e .[test]`.

## Use

```python
import numpy as np
from ramansep import SeparationModel, ModeCoefficients

coeffs = ModeCoefficients(
    mode1_name="A'1", mode2_name="2LA(M)",
    k1_strain=...,   # cm^-1 per unit strain, from your calibration
    k1_density=...,  # cm^-1 per unit carrier density
    k2_strain=...,
    k2_density=...,
    reference="cite the source of these numbers",
)
model = SeparationModel(coeffs)
result = model.invert(dw_mode1, dw_mode2, sigma1=0.1, sigma2=0.1)
# result.strain, result.density, result.strain_sigma, result.density_sigma
```

`SeparationModel` warns when the mode pair is poorly conditioned, i.e. when
the two modes respond too similarly for a reliable separation.

## Method

The method of using the first-order A'1 mode together with the
disorder-activated 2LA(M) overtone, whose strain lever arms differ
severalfold while only A'1 responds appreciably to carrier density, is
developed in:

> T. M. Mahim and M. M. Rahman, "Two Raman phonons quantify the fixed edge
> charge left by patterning monolayer transition metal dichalcogenides"
> (under review). Code for the paper itself:
> https://github.com/Tanvir-Mahmud-Mahim/Width-scaling-in-monolayer-semiconductor-nanoribbon-transistors

This package is the general-purpose, material-agnostic inversion tool; the
paper repository reproduces the specific published study.

## Roadmap

- v0.2 (done): documented example coefficient sets with citations
- v0.3: peak-fitting front end (load spectra, fit the two modes, feed the
  inversion)
- v0.4: joint Bayesian inversion with spatial priors

## License

Apache-2.0
