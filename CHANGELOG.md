# Changelog

## 0.6.0 (2026-09-05)

The v0.6 roadmap item -- joint Bayesian inversion with spatial priors
-- plus the Voigt fitter previously documented as out of scope. The
roadmap is now complete.

### Added

- `bayesian_map_inversion`: joint MAP inversion of whole shift maps
  with a Gaussian Markov random field smoothness prior (Rue and Held
  2005) on the strain and density fields, independently weighted
  (`lam_strain`, `lam_density`), solved exactly as one sparse linear
  system on the 4-neighbor pixel lattice with Neumann boundaries.
  Exact per-pixel posterior sigmas on request (`posterior_sigma=True`;
  the dense inverse is refused above `max_dense` unknowns rather than
  approximated silently). Anchors asserted in the tests, not stated:
  `lam = 0` reproduces `MultiModeModel.invert`'s maps and sigmas to
  machine precision; a spatially constant noiseless truth is recovered
  exactly at every `lam` (the prior vanishes on constants); the
  `lam -> infinity` limit is the independently computed pooled
  precision-weighted GLS; posterior sigmas shrink monotonically with
  `lam` (Loewner order).
- `fit_voigt` / `voigt` / `VoigtFit`: Voigt lineshape via the Faddeeva
  function `scipy.special.wofz`, peak-height normalized, fitted by the
  same Levenberg-Marquardt loop as `fit_lorentzian` with the analytic
  Jacobian from w'(z) = 2i/sqrt(pi) - 2 z w(z). Anchors: gamma = 0 is
  the Gaussian exactly (identity of the Faddeeva function, < 1e-14
  pointwise); sigma -> 0 converges linearly to the Lorentzian; the
  analytic Jacobian matches finite differences; noiseless lines are
  recovered to 1e-8 from automatic starting values.

### Changed

- README: roadmap marked complete, with the deliberate-scope
  statement (no invented hyperparameter inference, no uncited
  coefficient values, the stated prior graph rather than a kernel
  zoo); fitter section covers both lineshapes.
- `fitting` module docstring no longer claims NumPy-only dependencies
  (scipy has been a declared dependency since the finite-well test
  references) nor that Voigt is out of scope.
- CI matrix: Python 3.9, 3.11, 3.12, 3.13.

## 0.5.0 (2026-08-29)

- Overdetermined multimode GLS inversion (`MultiModeModel`) with
  per-pixel chi-square model checking and `compare_mode_sets`.

## 0.4.0

- Peak-fitting front end: `fit_lorentzian`, `fit_two_modes`.

## 0.3.0

- Graphene G + 2D coefficient set (Lee et al. 2012).

## 0.2.0

- Documented example coefficient sets with citations.

## 0.1.0

- Initial release: two-mode linear inversion with uncertainty
  propagation and conditioning diagnostics.
