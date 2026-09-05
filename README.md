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

v0.6.0 (alpha). Implemented and tested (44 tests, Python 3.9-3.13):
the inversion core, uncertainty propagation, conditioning diagnostics,
a synthetic end-to-end example, two cited coefficient sets for
monolayer MoS2, a cited graphene G + 2D set, a peak-fitting front end
(Lorentzian and Voigt), the overdetermined multimode GLS inversion
with its chi-square model check, and the joint Bayesian map inversion
with spatial smoothness priors. The API may change before v1.0.

## More than two modes, with model checking (new in v0.5)

`MultiModeModel` generalizes the inversion to any number of modes by
weighted (Gauss-Markov) least squares: each extra mode shrinks the
strain and density uncertainties, and the redundancy buys a per-pixel
chi-square goodness-of-fit with m - 2 degrees of freedom, so pixels where
strain and density alone cannot explain the shifts (a third latent
variable, a phase boundary, a bad fit) are flagged by their p-value map,
something no exactly determined two-mode inversion can detect.
`compare_mode_sets` ranks candidate mode subsets by the uncertainty they
deliver. For two modes the estimator reduces exactly to the 2x2 core
inversion (asserted to machine precision in the tests).

```python
from ramansep import MultiModeModel
mm = MultiModeModel(K)                     # K: (m, 2) lever arms
res = mm.invert([dw1, dw2, dw3], sigmas=[0.15, 0.10, 0.12])
suspect = res.p_value < 1e-3               # model-violation map
```

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

## Peak fitting (new in v0.4)

The roadmap's front end is in: `fit_two_modes` takes a raw spectrum,
fits a Lorentzian to each mode window with a NumPy-only
Levenberg-Marquardt solver (analytic Jacobians, linearized 1-sigma
uncertainties), and returns exactly the shifts and uncertainties that
`SeparationModel.invert` consumes.

```python
from ramansep import SeparationModel, fit_two_modes, mos2_a1_2la

model = SeparationModel(mos2_a1_2la())
dw1, dw2, s1, s2, fit1, fit2 = fit_two_modes(
    wavenumber, counts,
    window1=(395.0, 415.0), window2=(440.0, 465.0),
    ref1=404.7, ref2=452.0)   # your pristine references
result = model.invert(dw1, dw2, s1, s2)
```

Two lineshapes are offered and the choice is stated, not hidden:
`fit_lorentzian` for the lifetime lineshape of a phonon, and
`fit_voigt` (v0.6) for instrument-dominated lines, via the Faddeeva
function with an analytic Jacobian. The test suite checks both
Jacobians against finite differences, exact parameter recovery on
noiseless lines, the Voigt profile's exact Gaussian limit at zero
Lorentzian width and its convergence to the Lorentzian at zero
Gaussian width, statistical compatibility of the reported center
uncertainty with the actual scatter on noisy lines, and a full
spectrum-to-inversion round trip.

## Roadmap

- v0.2 (done): documented example coefficient sets with citations
- v0.3 (done): graphene G+2D coefficient set (Lee 2012)
- v0.4 (done): peak-fitting front end (fit the two modes, feed the
  inversion)
- v0.5 (done): overdetermined multimode GLS inversion with per-pixel
  chi-square model checking and mode-set comparison
- v0.6 (done): joint Bayesian inversion with spatial priors
  (`bayesian_map_inversion`: Gaussian Markov random field smoothness
  prior on both fields, solved exactly as a sparse linear MAP problem;
  lam = 0 reproduces the per-pixel GLS maps and sigmas to machine
  precision, asserted in the tests) and a Voigt fitter (`fit_voigt`)
  for instrument-dominated lines

The roadmap is complete. Deliberate scope, designed out rather than
overlooked: no shipped coefficient values beyond the cited example
sets (your material and mode pair need your calibration, with its
citation); the smoothness weights `lam_strain` / `lam_density` are
user-chosen regularization, not estimated hyperparameters -- full
hierarchical (evidence-maximizing) inference would need assumptions
about the noise this package refuses to invent; and the spatial prior
is the 4-neighbor lattice with Neumann boundaries, stated plainly,
not a tunable kernel zoo.

## Support and governance

The package is written and maintained by Tanvir Mahmud Mahim
(Department of Electrical and Electronic Engineering, BRAC University),
who reviews every change and takes the final decision on scope and
releases. There is no separate governance body; design questions are
discussed in the open in issues and pull requests, and the standing
rule of [CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly
as it binds contributors: a change that touches physics arrives with a
test, and a constant arrives with its source.

Support runs through the issue tracker at
https://github.com/TaN-MM-Org/ramansep/issues. Usage questions are
welcome there alongside bug reports; a docstring that left a unit or a
sign convention unclear is treated as a documentation bug, not as user
error. The maintainer aims to respond within a week.

While the version is below 1.0 the API may still move between minor
versions; such changes are called out in the release notes, and the
roadmap above states what is planned, so that a user can tell a missing
feature from an omission by design.

## License

Apache-2.0
