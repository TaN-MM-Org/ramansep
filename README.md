# ramansep

[![PyPI](https://img.shields.io/pypi/v/ramansep)](https://pypi.org/project/ramansep/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22014913-blue)](https://doi.org/10.5281/zenodo.22014913) [![tests](https://github.com/TaN-MM-Org/ramansep/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/ramansep/actions)

Tell **strain** apart from **charge** in Raman maps of 2D materials.

A Raman peak moves when the material is stretched and also when
charge is added, so one peak cannot say which happened. Two peaks
that respond differently can: measure both shift maps, invert a
small matrix at every pixel, and get a strain map and a
charge-density map with honest error bars. This package does that
inversion -- and everything around it, from the raw spectra to the
final maps.

## Install

```
pip install ramansep
```

For development: clone the repository and `pip install -e .[test]`.

## From the instrument to the maps

The whole experimental path is covered (spectrum handling new in
v0.8):

```python
import ramansep as rs

x, cube = rs.load_map_csv("measured.csv")        # documented file contract
res = rs.fit_map(x, cube, window1=(395, 415), window2=(440, 465),
                 ref1=404.7, ref2=452.0)         # per-pixel peak fits
model = rs.SeparationModel(rs.mos2_a1_2la())     # or your own coefficients
inv = model.invert(res.dw1, res.dw2, res.sigma1, res.sigma2)
# inv.strain, inv.density, inv.strain_sigma, inv.density_sigma
```

- `load_spectrum_csv` / `load_map_csv` read a documented plain-text
  contract (`wavenumber_cm1,counts`; maps add `row,col`), refuse
  malformed files with an explanation, and round-trip exactly with
  their save counterparts.
- `fit_map` runs the two-peak fit at every pixel and reports failures
  honestly: a dead pixel, a fit that did not converge, or a runaway
  center is masked NaN and flagged in `res.ok`, never silently
  guessed. Masked pixels flow through the per-pixel inversion as NaN
  without touching their neighbors.
- Peak fits are Lorentzian (the natural phonon lineshape) or Voigt
  (`fit_voigt`, for instrument-dominated lines), with analytic
  Jacobians and real uncertainties. A sloped fluorescence background
  pulls a fitted center sideways; `baseline="linear"` (new in v0.8)
  fits the slope too -- the tests demonstrate both the bias and the
  recovery.
- `SeparationModel` warns when the two modes respond too similarly
  to separate the two causes reliably.

## More modes, and a built-in lie detector

`MultiModeModel` takes any number of modes. Each extra mode shrinks
the error bars, and the redundancy buys a per-pixel model check: a
chi-square p-value map that flags pixels where strain and charge
alone cannot explain the shifts (a third cause, a phase boundary, a
bad fit) -- something no two-mode inversion can detect.
`compare_mode_sets` ranks candidate mode combinations by the
uncertainty they would deliver. For two modes the result reduces
exactly to the 2x2 inversion.

`bayesian_map_inversion` goes one step further for maps: strain and
charge usually vary smoothly from pixel to pixel, and that knowledge
is worth variance. It solves the whole map jointly with a smoothness
prior -- exactly, as one sparse linear problem. With the smoothing
turned off it reproduces the per-pixel result to machine precision;
smoothing never increases the reported uncertainty (both asserted in
the tests). It refuses NaN pixels rather than solving around them
silently -- exclude or infill masked pixels deliberately first.

## Temperature, measured and separated

Laser heating shifts Raman peaks too, and a two-cause analysis books
that shift as strain or charge. New in v0.9, two independent tools
close this gap. `ThreeCauseModel` extends the inversion to strain,
charge and temperature at once, using your calibrated temperature
coefficients (cm^-1 per kelvin, with their source -- none are
shipped); it needs at least three modes and keeps the same
uncertainty, chi-square and identifiability machinery.
`temperature_from_anti_stokes` measures temperature directly from the
anti-Stokes/Stokes intensity ratio through the Bose-Einstein
occupation factor, with a calibration constant you measure once at a
known temperature (`calibrate_anti_stokes`) instead of the biased
textbook assumption of 1. Measuring T one way and separating it the
other way is exactly the cross-check a heated-spot experiment wants.

## Calibrate your own numbers

`calibrate_lever_arms` fits the response matrix from reference states
you control (a strain-stage sweep, a gated sweep) by weighted least
squares, with uncertainties, a chi-square consistency check, and an
identifiability refusal: a strain-only sweep cannot determine the
charge response, and the fit says so instead of returning one of
infinitely many answers.

```python
res = rs.calibrate_lever_arms(strain, density, shifts, sigmas=sig,
                              mode_names=["A'1", "2LA(M)"])
model = rs.SeparationModel(
    res.coefficients(reference="stage + gate calibration, 2026-09"))
```

## Cited example sets

Three sets ship with full provenance; the tests reproduce the source
paper's published separations from them:

- `mos2_a1_2la()`: the A'1 + 2LA(M) pair for monolayer MoS2 (strain
  arms -5.1 and -20.9 cm^-1 per percent, frozen-phonon DFT; A'1
  doping response -2.2 cm^-1 per 1e13 cm^-2, measured by Chakraborty
  et al., Phys. Rev. B 85, 161403(R) (2012)). The 2LA(M) doping
  response is unmeasured and set to zero; the source paper bounds the
  consequence at 16% of a recovered edge charge.
- `mos2_eprime_a1()`: the historically used all-optical E' + A'1
  pair (Michail et al., ACS Appl. Mater. Interfaces 16, 49602
  (2024)). Both modes respond weakly to strain, so it carries about
  five times the strain uncertainty -- included for comparison.
- `graphene_g_2d_lee2012()`: the G + 2D decomposition of Lee et al.,
  Nat. Commun. 3, 1024 (2012). Its "doping" output is deliberately a
  G-band shift, not a density: graphene's doping response is
  nonlinear, so no universal linear rate exists to ship, and the
  docstring points to the gated calibrations needed for conversion.

Each docstring states which number comes from which source and the
conditions it applies under. Check that your sample matches before
use.

## What is deliberately not included

No constants beyond the three cited sets. Response coefficients
depend on material, mode pair, laser wavelength and substrate; a
measurement tool that ships unverified constants propagates wrong
results. For any other system you provide the numbers -- from the
literature or from `calibrate_lever_arms` -- and the mandatory
`reference` field makes their origin travel with the analysis. The
smoothing weights of the Bayesian inversion are likewise user-chosen,
stated plainly as such, because estimating them would need noise
assumptions this package refuses to invent.

## How it is checked

62 tests (Python 3.9-3.13, run in CI on every push), each pinned to
an exact result: noise-free recovery to machine precision throughout;
analytic Jacobians against finite differences; the Voigt profile's
exact Gaussian and Lorentzian limits; reported uncertainties checked
against actual scatter on seeded noise; the multimode estimator
reducing exactly to the 2x2 core; the Bayesian solver reproducing the
per-pixel result at zero smoothing; calibration cross-checked against
an independent QR solve; exact file-contract round trips; and full
pipeline round trips from synthetic spectra back to the generating
strain and charge maps.

## Method

> T. M. Mahim and M. M. Rahman, "Two Raman phonons quantify the fixed
> edge charge left by patterning monolayer transition metal
> dichalcogenides" (under review). Code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/Width-scaling-in-monolayer-semiconductor-nanoribbon-transistors

This package is the general-purpose, material-agnostic tool; the
paper repository reproduces the specific published study.

## Support and governance

Written and maintained by Tanvir Mahmud Mahim (Department of
Electrical and Electronic Engineering, BRAC University), who reviews
every change and takes the final decision on scope and releases.
Design questions are discussed in the open in issues and pull
requests, and the standing rule of
[CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly as it
binds contributors: a change that touches physics arrives with a
test, and a constant arrives with its source.

Support runs through the
[issue tracker](https://github.com/TaN-MM-Org/ramansep/issues).
Usage questions are welcome alongside bug reports; a docstring that
left a unit or a sign convention unclear is treated as a
documentation bug, not user error. While the version is below 1.0
the API may still move between minor versions; such changes are
called out in the release notes.

## License

Apache-2.0. Every release is archived on Zenodo under the concept DOI
[10.5281/zenodo.22014913](https://doi.org/10.5281/zenodo.22014913),
which always resolves to the latest version.
