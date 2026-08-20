---
title: 'ramansep: two-mode separation of strain and carrier density in Raman maps of 2D materials'
tags:
  - Python
  - Raman spectroscopy
  - 2D materials
  - transition metal dichalcogenides
  - strain
  - carrier density
authors:
  - name: Tanvir M. Mahim
    orcid: 0000-0002-4550-3248
    affiliation: 1
affiliations:
  - name: Department of Electrical and Electronic Engineering, BRAC University, Dhaka, Bangladesh
    index: 1
date: DRAFT
bibliography: paper.bib
---

# Summary

A Raman-active phonon of a two-dimensional semiconductor shifts its
frequency when the lattice is strained and shifts again when charge
carriers are added. A single measured shift is therefore one equation in
two unknowns, and any map built from one mode silently attributes
everything to whichever cause the analyst assumed. `ramansep` resolves
this by the two-mode method: given peak-shift maps of two phonon modes
whose strain and doping lever arms differ, it inverts a 2x2 linear model
per pixel and returns a strain map and a carrier-density map together
with propagated one-sigma uncertainties, the strain-density correlation
coefficient, and conditioning diagnostics that warn when a mode pair
cannot separate the two causes reliably.

The package ships two documented coefficient sets for monolayer
1H-MoS$_2$ with full provenance: the A$'_1$ + 2LA(M) pair, whose
disorder-activated acoustic overtone provides a strain lever arm 4.1
times that of the optical mode while carrying negligible doping
response, and the historically used all-optical E$'$ + A$'_1$ pair built
from measured biaxial Grüneisen parameters [@Michail2024; @Mignuzzi2015]
with the gated-Raman doping coefficient of @Chakraborty2012. Every
constant travels with its citation in a mandatory `reference` field, and
the test suite reproduces published separation results from the shipped
coefficients rather than asserting stored numbers.

# Statement of need

Separating strain from doping in Raman maps is a recurring task wherever
two-dimensional devices are patterned, contacted, or transferred:
etched edges, wrinkles, and gate stacks carry both causes at once. The
optical decomposition idea is established for graphene and MoS$_2$
[@Lee2012; @Michail2016], yet groups typically re-implement the
inversion privately, without uncertainty propagation and without a
quantitative basis for choosing the mode pair. `ramansep` provides the
missing common engine: a material-agnostic linear inversion with honest
error bars, a noise-amplification comparison between candidate mode
pairs (the raw matrix condition number is scale-dependent and
misleading), and cited coefficient sets that make the MoS$_2$ workflow
reproducible end to end. The package is the general-purpose tool behind
the author's nanoribbon edge-charge study, where the A$'_1$ + 2LA(M)
pair separates an electronic edge feature from a mechanical interior
one in published tip-enhanced maps [@Krayev2026]; the study's own
repository archives the specific analysis [@MahimZenodo].

`ramansep` depends only on NumPy, installs with `pip install ramansep`,
and is released under Apache-2.0 with continuous-integration tests and
Zenodo-archived versions.

# Acknowledgements

The author thanks M. Mosaddequr Rahman for supervision and validation of
the underlying method study.

# References
