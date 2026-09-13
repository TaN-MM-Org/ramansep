# Changelog

## 0.9.0 (2026-09-13)

Physics upgrade: temperature as a third separable cause, and as a
direct Bose-factor measurement.

### Added

- `ThreeCauseModel` / `ThreeCauseResult`: the weighted-least-squares
  separation extended to strain, carrier density AND temperature,
  from m >= 3 modes and your calibrated temperature coefficients
  d(omega)/dT (cited, user-supplied -- material-, mode-, thickness-
  and substrate-specific, so none are shipped; for monolayer MoS2
  see e.g. Sahoo et al., J. Phys. Chem. C 117, 9042 (2013)). Full
  per-pixel 3x3 covariance, chi-square map against dof = m - 3, and
  a rank-3 identifiability refusal. Anchors: machine-precision
  noise-free round trip; per-pixel agreement with an independent
  whitened `lstsq` path; covariance equal to the direct textbook
  inverse; exact consistency with the existing two-cause solver when
  a known temperature's shift is subtracted first.
- `temperature_from_anti_stokes` / `anti_stokes_ratio` /
  `calibrate_anti_stokes`: the anti-Stokes/Stokes Raman thermometer,
  I_AS/I_S = C exp(-hbar omega / kB T), with the prefactor C
  calibrated at one known temperature rather than assumed 1 (the
  standard-practice correction for cross-section and spectrometer
  response), exact closed-form inversion, optional exact error
  propagation, and a refusal when ratio >= C (no finite positive
  temperature exists; the calibration or background is wrong).
  Anchors: machine-precision round trip; the second radiation
  constant hc/kB computed from the exact SI defining constants and
  asserted against SciPy's CODATA table (two independent sources);
  analytic error propagation against a central finite difference;
  monotonicity; refusal tests.

## 0.8.0 (2026-09-12)

From-the-instrument release: the path from a measured hyperspectral
map to the strain and charge maps is now covered end to end, with
honest per-pixel failure reporting.

### Added

- `fit_map` / `MapFitResult`: the two-mode peak fit at every pixel of
  an (H, W, L) cube, returning the four shift/uncertainty maps
  `SeparationModel.invert` consumes. Honest masking instead of silent
  garbage: non-finite counts, non-convergence, or a fitted center
  escaping its window mask the pixel NaN and flag it in `ok`; masked
  pixels propagate through the per-pixel inversion without touching
  neighbors. Anchors: pixelwise equality with direct `fit_two_modes`
  calls; a synthetic-cube round trip back to the generating strain
  and density maps; a corrupted pixel masked while its neighbors are
  bit-identical to the clean run.
- `load_spectrum_csv` / `save_spectrum_csv` and `load_map_csv` /
  `save_map_csv`: documented plain-text contracts
  (`wavenumber_cm1,counts`; maps add 0-based `row,col` on a complete
  grid with one shared, strictly increasing wavenumber axis), exact
  round trips, and refusals with explanations for wrong headers,
  grid holes, and per-pixel axes.
- `baseline="linear"` on `fit_lorentzian` and `fit_two_modes`: a
  linear-baseline term for spectra on a sloped fluorescence
  background, with the analytic Jacobian extended. The tests
  demonstrate the constant-baseline center bias on a sloped
  background (beyond 3 sigma) and its recovery (to 1e-6) -- and that
  the historical constant-baseline path is unchanged. `PeakFit`
  gains `slope` / `slope_sigma`.

### Changed

- `bayesian_map_inversion` refuses non-finite shift maps with an
  explanation (the smoothness prior couples pixels, so masked pixels
  must be excluded or infilled deliberately), instead of solving
  around them silently.
- README rewritten: organized by workflow rather than by release
  history, in plainer language, same facts.

## 0.7.0 (2026-09-10)

### Added

- `calibrate_lever_arms` / `CalibrationResult`: fit the (m, 2)
  lever-arm matrix from the user's own reference measurements (known
  strain and density states, measured shifts) by weighted least
  squares -- exact known-noise covariances when sigmas are given,
  residual-variance estimates otherwise (refusing n < 3 in that
  case), per-mode chi-square consistency check, and an
  identifiability refusal for collinear reference states.
  `CalibrationResult.coefficients()` packages a two-mode calibration
  as a `ModeCoefficients` with a mandatory provenance string; `.K`
  feeds `MultiModeModel` for m > 2.
- Anchors: exact noise-free recovery; agreement with an independent
  QR solve to 1e-10; identity covariance on the orthogonal unit
  design; Monte-Carlo scatter matching reported sigmas;
  calibrate-then-invert round trip through `SeparationModel`.

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
