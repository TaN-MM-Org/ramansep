# ramansep

Separate **strain** from **carrier density** in Raman maps of 2D materials,
using two phonon modes whose lever arms differ.

A single Raman frequency responds to strain and to carrier density at once,
so one mode cannot tell the two apart. Two modes with sufficiently different
responses form a linear, invertible probe: measure both shift maps, invert a
2x2 matrix per pixel, and obtain a strain map and a carrier-density map with
propagated uncertainties.

## Status

v0.1.0 (alpha). The inversion core, uncertainty propagation, conditioning
diagnostics and a synthetic end-to-end example are implemented and tested.
The API may change before v1.0.

## What this package deliberately does not include

No validated material constants are shipped. Lever arms depend on material,
mode pair, excitation wavelength and substrate; a measurement tool that
ships unverified constants propagates wrong results. You provide a
`ModeCoefficients` for your system from the literature or your own
calibration, and the `reference` field is mandatory so the provenance of
every number travels with the analysis.

## Install

```
pip install -e .
```

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

- v0.2: peak-fitting front end (load spectra, fit the two modes, feed the
  inversion), documented example coefficient sets with citations
- v0.3: joint Bayesian inversion with spatial priors

## License

Apache-2.0
