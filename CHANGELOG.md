# Changelog

## 0.12.0 (2026-09-30)

Overlapping peaks, pixels without a peak, gaps in smoothed maps, and
calibration uncertainty for any number of modes.

### Added

- `fit_two_modes(..., joint=True)` and `fit_map(..., joint=True)`: the
  two peaks are fitted together (two Lorentzians on one shared
  constant or linear baseline) over the range spanned by both windows,
  starting from the separate window fits. This removes the pull of one
  peak's tail on the other's fitted centre, listed as a limit until
  now (0.0795 and -0.0279 cm^-1 on the noiseless pair of README
  example 13; 0 to printed precision with `joint=True`). The
  correlation of the two fitted centres is reported in the new
  `PeakFit.center_correlation` (NaN for single-peak fits). Default
  `joint=False` is unchanged.
- `fit_map(..., min_snr=None, fwhm_range=None)`: optional checks that
  each fitted peak height is at least `min_snr` times its error bar,
  and that both widths lie in `fwhm_range`. They catch pixels without
  a peak, where a fit to pure noise can converge on a spike and return
  a precise-looking shift (+9.92 +/- 0.05 cm^-1 in README example 12).
- `MapFitResult.reason` and `MASK_REASONS`: why each pixel was masked.
- `bayesian_map_inversion`: `sigmas` may be an (m, H, W) array (one
  error bar per mode and pixel), lifting the "one error value per mode"
  limit; and `fill_masked=True` treats NaN shifts or error bars as
  missing measurements that the smoothness prior fills from the
  neighbours (needs both smoothness weights > 0 and unmasked data
  that fix both fields; refused otherwise). Default behaviour is
  unchanged.
- `multimode_with_calibration(calibration, shifts, sigmas)`: the
  calibration-uncertainty propagation of `separation_with_calibration`
  for any number of modes, using the exact derivative of the weighted
  least-squares estimate, dx/dK_mj = w_m A^-1 (e_j r_m - K_m^T x_j),
  which involves the residual r. Lifts the "two modes only" limit.

### Fixed

- `fit_map` kept pixels whose fitted peak height was negative (a dip,
  typically a fit to noise). They are now masked (reason code 5).
- `ThreeCauseModel.invert` failed on the whole map, with the message
  "weighted design matrix is singular", when any pixel had a NaN error
  bar (as `fit_map` gives for masked pixels). Such pixels, and pixels
  with a NaN shift, now come out as NaN, as in `MultiModeModel`.
- `SeparationModel.invert` accepted negative `sigma1`/`sigma2`, and
  `compare_mode_sets` negative `sigmas`; the sign was squared away.
  Both now raise `ValueError` (NaN still passes as a masked pixel in
  `SeparationModel`; zero is still allowed there).
- `SeparationModel` and `MultiModeModel` with a NaN or infinite lever
  arm stopped with numpy's "SVD did not converge"; they now say that
  the coefficients must be finite (still a `ValueError`).
- `compare_mode_sets` did not validate its inputs. A zero or NaN
  sigma returned NaN variances (ranked first, since NaN sorts to the
  front), an infinite sigma or a NaN lever arm stopped with numpy's
  `LinAlgError`, an infinite lever arm silently dropped every subset
  containing that mode, and a 1-D `K` returned an empty list. It now
  refuses sigmas that are not finite and positive, a non-finite `K`
  and a `K` that is not (m, 2), with a `ValueError` that says which.

### Behaviour changes

- `fit_map`, default settings: pixels with a negative fitted height
  are masked. Before/after on README example 12 (20 peak-free
  pixels): 6 kept before, 3 kept now (0 with `min_snr=3`); all 20
  pixels with real peaks are kept, with unchanged numbers. Pixels
  with positive fitted heights give bit-identical results.
- `SeparationModel.invert(0.1, 0.2, -0.1, 0.1)` with `mos2_a1_2la()`
  returned a strain error bar of 0.0048 %; it now raises `ValueError`.
  `compare_mode_sets` with a negative sigma likewise.
- `compare_mode_sets(K, sigmas)` with the three-mode `K` of README
  example 4 (checked by running 0.11.1 and 0.12.0):
  - sigmas `[0, 0.1, 0.12]` or `[nan, 0.1, 0.12]`: before, 3 subsets
    with `var_strain = var_density = nan` for the first; now
    `ValueError: sigmas must be finite and positive`.
  - sigmas `[inf, 0.1, 0.12]`: before, `LinAlgError: Singular
    matrix`; now the same `ValueError`.
  - a NaN lever arm: before, `LinAlgError: SVD did not converge`; now
    `ValueError: K must be finite`.
  - an infinite lever arm in mode 1: before, a 1-subset ranking
    (modes 2 and 3 only) with no error; now `ValueError: K must be
    finite`.
  - a 1-D `K` of length 3: before, an empty list; now
    `ValueError: K must have shape (m, 2)`.
- `ThreeCauseModel.invert` with a NaN error bar at one pixel: before,
  `ValueError` for the whole map; now NaN at that pixel only, and
  bit-identical results elsewhere.
- `bayesian_map_inversion` with a NaN in scalar `sigmas`: before, NaN
  maps without an error; now refused unless `fill_masked=True`.

### Changed

- The draft paper in `paper/` (a planned Journal of Open Source Software
  submission) is no longer kept in the repository.
- README: examples 12-15 for the new features, each with the output
  it printed; updated refusals, test descriptions, corrections and
  "Limits" (lifted: one error value per mode in the Bayesian
  inversion, the separate-window tail bias as the only option, two
  modes only for calibration propagation; added: filled pixels are
  interpolations, the joint fit needs a span without a third peak and
  its centre correlation is not used by the inversion, the
  calibration part is an error common to the whole map, the
  `fit_map` thresholds are the user's choice).
- `bayesian`, `mapfit`, `fitting`, `propagate` module docstrings
  updated accordingly.

### Tests

- New `tests/test_v012.py` (19 tests; see the README section "How the
  results are checked"). Test count: 77 -> 96. The full suite also
  passes with the oldest allowed dependencies (NumPy 1.22.0,
  SciPy 1.8.0) on Python 3.9 and 3.10.

## 0.11.1 (2026-09-22)

Two silent failures fixed, the README rewritten, and CI extended.

### Fixed

- `fit_map` caught every per-pixel `ValueError`, so a mistake in its
  own settings -- an unknown `baseline` name (e.g. `"Linear"`),
  reversed or overlapping windows, or a window with too few points --
  masked every pixel and returned an all-NaN result with no error.
  These settings are now checked once, before any pixel is fitted,
  with the same messages as `fit_two_modes` (the checks now live in
  one shared helper used by both). Valid settings give unchanged
  results.
- `fit_map` now also refuses a wavenumber axis containing NaN or
  infinity. Before, spectral points with a NaN wavenumber were
  silently left out of every pixel's fit (the fit itself still ran).
  `fit_two_modes` on its own is unchanged in this respect.
- `bayesian_map_inversion` did not check the rank of the lever-arm
  matrix. A rank-deficient `K` makes its sparse system singular: it
  returned NaN maps at `lam = 0` (with a SciPy warning) and arbitrary
  finite maps at `lam > 0` (no warning). It now refuses, with the same
  message as `MultiModeModel`, and also refuses a non-finite `K`.

### Tests

- New `tests/test_refusals_v0111.py` (4 tests): the `fit_map`
  refusals, pixelwise equality with `fit_two_modes` for valid
  settings, and the rank refusal of `bayesian_map_inversion`. Three of
  them fail on 0.11.0. Test count: 73 -> 77.
- The full suite passes with the oldest allowed dependencies
  (NumPy 1.22.0, SciPy 1.8.0) on Python 3.9 and 3.10.

### Changed

- CI: Python 3.10 added to the matrix (it now covers 3.9 to 3.14), and
  a new `oldest-dependencies` job runs the suite on Python 3.9 with
  NumPy 1.22.0 and SciPy 1.8.0.
- README rewritten for readers outside the field: a guide to the words
  used, eleven examples each with its real printed output, a list of
  every exported name, the refusals, and the actual tolerance of each
  test check.
- `calibration` module docstring: "QR solve" corrected (see below).
- `bayesian` docstrings referred to a `MultiModeSeparation` class that
  does not exist; they now name `MultiModeModel`.
- CONTRIBUTING.md: dependencies are NumPy and SciPy (it said NumPy
  only).

### Corrections to earlier notes

- The 0.11.0 README said the tests run on Python 3.9-3.14; the CI
  matrix did not include 3.10 (the 0.6.0 note lists the matrix
  correctly). It also described every test as "pinned to an exact
  result" at "machine precision", and said the tests reproduce the
  published separations of all three cited sets; many checks use
  stated tolerances (1e-8 to 1e-12 for the noise-free inversion round
  trips, 1e-4 and 1e-3 for the noiseless map-fit round trip, 5e-3 for
  the two-peak spectrum round trip, wider for statistical checks), and
  only `mos2_a1_2la` is checked against published separations.
- The 0.7.0 note (and, until now, the `calibration` module
  docstring) call the cross-check "an independent QR solve"; the test
  uses `numpy.linalg.lstsq` on the whitened system, which is not a QR
  solve. The docstring now says so.
- The 0.11.0 README attributed the E' + A'1 pair itself to Michail
  et al., ACS Appl. Mater. Interfaces 16, 49602 (2024). As the
  `mos2_eprime_a1` docstring states, that paper supplies the biaxial
  Grueneisen parameters; the pair's historical use is cited to
  Michail et al., Appl. Phys. Lett. 108, 173102 (2016). The README
  now attributes each number as the docstring does.
- The 0.9.0 "machine-precision noise-free round trip" of
  `ThreeCauseModel` is asserted to 1e-9 (1e-8 for temperature).

## 0.11.0 (2026-09-18)

The calibration's own uncertainty carried into the maps, and a
future-proofing pass.

- `propagate.separation_with_calibration`: two-mode inversion that
  also propagates the lever-arm calibration covariance into the
  strain and density error bars, reporting the shift-noise and
  calibration contributions separately. Built on the exact
  derivative identity dx/dK_mj = -K^-1 E_mj x of the two-mode
  inverse; the multimode case is deliberately not half-shipped.
- README documents the verified 2024-2026 sources for W-based
  materials (WSe2/WS2 strain and temperature responses) and states
  plainly why no W-material doping arm ships as a constant: the
  measured doping behavior has no clean linear coefficient yet, and
  this package does not ship half a lever-arm matrix.
- CI now also runs on Python 3.14.
- Anchors: zero calibration covariance reduces exactly to the plain
  inversion; the derivative identity checked against finite
  differences of the actual re-solve; 400 seeded Monte-Carlo lever-
  arm draws match the reported calibration part; the quadrature
  split exact; multimode calibrations refused by the two-mode tool.

## 0.10.0 (2026-09-17)

Lab adaptability: the calibration planned before it is measured.

- `lab.plan_calibration`: the exact covariance `calibrate_lever_arms`
  will report, computed from the reference design alone -- the model
  is linear, so this is the same (X^T W X)^-1 matrix, not an
  approximation. Collinear designs are reported non-identifiable by
  the same rank argument on which the calibration refuses.
- `lab.design_references`: greedy D-optimal choice of which reachable
  (strain, density) states to prepare (Pukelsheim, Optimal Design of
  Experiments, SIAM (2006)).
- `lab.repeats_for_sigma`: the repeat count for a target lever-arm
  uncertainty, in closed form -- r copies of a design scale its
  covariance by exactly 1/r.
- Anchors: planned covariance equals the calibration's to machine
  precision; the orthogonal unit design's exact identity covariance
  is predicted in advance; the 1/r scaling asserted by tiling the
  design; the greedy design contains the off-line point any
  identifiable subset needs and never loses to a random subset;
  collinear refusals mirrored across all three tools.

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
