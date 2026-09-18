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
`temperature_from_anti_stokes` measures temperature directly from
the ratio of the anti-Stokes and Stokes peak intensities -- a ratio
set by how many phonons are thermally excited, so it is a built-in
thermometer -- with a calibration constant you measure once at a
known temperature (`calibrate_anti_stokes`) instead of the textbook
assumption of 1, which biases the answer on any real spectrometer. Measuring T one way and separating it the
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

## Plan the calibration before measuring it

Because the calibration model is linear, the error bars
`calibrate_lever_arms` will report depend only on the reference
design -- which (strain, density) states you prepare and how well
your peak fitter resolves shifts -- so they can be computed exactly
before any spectrum is taken:

```python
from ramansep import plan_calibration, design_references, repeats_for_sigma

# How good would this reference set be, at 0.15 cm^-1 per shift?
plan = plan_calibration(strain=[0.0, 0.3, 0.6, 0.0],
                        density=[0.0, 0.0, 0.1, 0.8], sigmas=0.15)
print(plan["K_sigma"])              # (strain arm, density arm) error bars

# Which of the states my stage and gate can reach are worth preparing?
pick = design_references(strain_cand, density_cand, n_pick=5, sigmas=0.15)

# How many repeats to get every lever arm below 0.5 cm^-1 per unit?
r, plan_r = repeats_for_sigma(0.5, strain, density, sigmas=0.15)
```

A collinear candidate set -- a strain-only sweep, say -- is refused
with the same explanation the calibration itself gives, because no
subset of a line can identify two lever arms; and repeating a design
r times shrinks its covariance by exactly 1/r, so the repeat count is
a closed form, not a search.

## The calibration's own uncertainty, in the maps

A calibrated lever-arm matrix carries error bars of its own, and
until now every inversion treated it as exact.
`separation_with_calibration` closes that gap for the two-mode case:
it inverts the shift maps AND propagates the calibration covariance
into the strain and density error bars, reporting the two
contributions separately -- so you can see whether your budget is
limited by the spectra or by the calibration, and spend effort where
it matters:

```python
from ramansep import calibrate_lever_arms, separation_with_calibration

cal = calibrate_lever_arms(strain_refs, density_refs, ref_shifts,
                           sigmas=0.1)
out = separation_with_calibration(cal, dw1_map, dw2_map,
                                  sigma1=0.1, sigma2=0.1)
print(out["strain_sigma_shifts"], out["strain_sigma_calibration"])
```

The propagation rests on an exact derivative identity of the
two-mode inverse, checked against finite differences of the actual
re-solve; the overdetermined multimode case couples the mode weights
and is deliberately not half-shipped.

## Where to calibrate tungsten-based materials from (2024-2026)

For W-based monolayers the literature now provides verified strain
and temperature responses -- monolayer WSe2 biaxial strain rates
(Michail et al., ACS Appl. Mater. Interfaces 16, 49602 (2024)),
monolayer WS2 strain rates (Roy, Yang and Gao, Sci. Rep. 14, 3860
(2024)) and temperature coefficients (Huang et al., Sci. Rep. 6,
32236 (2016)) -- while gate-calibrated doping arms remain without a
clean linear coefficient (the measured behavior is an electron-only,
threshold-like softening of the out-of-plane modes; Sohier et al.,
PRX 9, 031019 (2019)). That is why no W-material set ships as
constants here: the strain column is citable, the doping column is
not yet, and this package does not ship half a lever-arm matrix.
Calibrate your own with `calibrate_lever_arms` -- the doping arm
from your own gated reference points -- and the provenance travels
with the analysis.

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

73 tests (Python 3.9-3.14, run in CI on every push), each pinned to
an exact result: noise-free recovery to machine precision throughout;
analytic Jacobians against finite differences; the Voigt profile's
exact Gaussian and Lorentzian limits; reported uncertainties checked
against actual scatter on seeded noise; the multimode estimator
reducing exactly to the 2x2 core; the Bayesian solver reproducing the
per-pixel result at zero smoothing; calibration cross-checked against
an independent QR solve; the three-cause inversion against an
independent per-pixel least-squares path, and its exact agreement
with the two-cause solver at a known temperature; the anti-Stokes
thermometer's exact round trip, with its physical constant checked
against SciPy's own table; exact file-contract round trips; and full
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
