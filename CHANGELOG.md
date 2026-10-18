# Changelog

## 0.15.0 (2026-10-18)

Profiles across an edge with the covariance between bins, the extra
charge per unit length of edge, exact Bayesian error bars without a
dense inverse, two unit dependences fixed, and CI-robust tests of the
stacked map fit.

### Added

- `region_profile(coefficients, dw1, dw2, sigma1, sigma2, coordinate,
  bins, *, mask=None, shift_correlation=None, ref_sigma=None)`,
  returning a `RegionProfile`: the mean strain and density in bins of a
  coordinate (for example the distance from an edge), each bin with the
  error budget of `region_mean` (shift noise, calibration, reference
  frequencies, homogeneity check), plus the covariance between all bins
  (`covariance`, `covariance_shifts`, `covariance_calibration`,
  `covariance_reference`, shape (R, 2, R, 2)). The bins share the lever
  arms and the reference frequencies, so their errors are correlated:
  calibration covariance sum_m (x_i^T C_m x_j) k_m k_m^T between bins i
  and j, reference covariance K^-1 diag(u1^2, u2^2) K^-T for every
  pair, no shift covariance (different pixels). Propagation as in JCGM
  100:2008 (GUM), 5.2.2, Eq. (13), first order in the calibration.
- `RegionProfile.combination(strain_weights, density_weights)` (any
  weighted sum of the bins, with its error parts, as a new
  `LinearEstimate`), `RegionProfile.integrated_excess(bulk,
  quantity="density", widths=None)` (sum over the other bins of width
  times value minus the bulk value: across a straight edge, the excess
  per unit length of edge; the reference error cancels) and
  `RegionProfile.contrast(i, j)` (bin i minus bin j, as
  `region_contrast` gives it).
- Validation: each bin equals the mean of the per-pixel
  `SeparationModel.invert` maps (1e-14) and `region_mean` on that bin
  (relative 1e-12); the cross-bin calibration covariance equals that
  from a finite-difference Jacobian of an independent re-solve
  (relative 1e-6); combinations equal c^T C c (relative 1e-10); the
  integrated excess equals the sum by hand (1e-13), its shift part a
  pixel-level linear form (1e-10) and its calibration part finite
  differences (1e-6); `contrast` equals `region_contrast` (1e-12). Over
  300 seeded trials that repeat calibration, map noise and reference
  error, the spread of the excess and of a bin mean is within 15 % of
  the error bars (measured 3 % and 4 %), and the measured correlation
  of two bins (0.988) matches the reported one (0.987).
- README: the profile and its refusals in the package list, a third
  tutorial, test descriptions, corrections and limits.

### Fixed

- `bayesian_map_inversion` depended on the units of the lever arms.
  Before: with the density column in cm^-2 (K / [1, 1e13]), the maps of
  the test problem differed from an independent dense solve by a
  relative 6.1e-4 (density) and 1.4e-4 (strain), and with K * [1, 1e6]
  by 8e-13. After: 1.1e-15 and 6.7e-16 (density in cm^-2). When the
  column norms of K differ by more than 2^10, the problem is solved for
  density / t with t a power of two (exact rescaling) and scaled back.
  Smaller ratios (all shipped sets, all README examples) run the
  unchanged code. Test:
  `test_bayesian_map_does_not_depend_on_the_density_unit`.
- `MultiModeModel`, `ThreeCauseModel` and `bayesian_map_inversion`
  refused a well-separating lever-arm matrix as "rank < 2" ("rank < 3")
  when the density was in m^-2 (arms near 1e-17 cm^-1 per m^-2; also
  for K / [1, 1e15]), and `compare_mode_sets` returned `[]` for it
  instead of three subsets. The rank is now judged on K with unit-length
  columns (`_has_full_column_rank`), as `calibrate_lever_arms` does
  since 0.14.0. After: the results equal those in 1e13 cm^-2, rescaled
  (measured relative differences below 1e-14); a rank-deficient matrix
  is still refused in any unit. Test:
  `test_rank_tests_do_not_depend_on_the_units`.
- Documentation and tests: `fit_map` was documented (module docstring,
  README) to give "bit for bit" the numbers of the per-pixel loop over
  `fit_two_modes`, and nine tests demanded exact equality between the
  stacked and the single-spectrum fits (and between `ThreeCauseModel`
  runs on stacks of different length). That held on the machine of
  0.14.0 but is not guaranteed on another CPU, because the stacked and
  the single products are different NumPy/BLAS calls. Demonstration:
  changing the counts of the test map by one unit in the last place
  flips the mask reason code of 1 or 2 of its 18 peak-free pixels
  between 3 and 4 (joint fits of pure noise run away to centers of
  1e20 to 1e76 cm^-1 along a chaotic path). The docs now say "up to
  rounding"; the tests compare shifts to 1e-4 of their error bar and
  error bars to a relative 1e-5 (derived from the stopping rule: a stop
  one step earlier or later moves a parameter by at most about
  1e-6 sqrt(n - p) of its error bar), compare mask decisions exactly on
  pixels with peaks (whose decisions are checked to be far from every
  threshold), and no longer compare peak-free pixels. Measured on this
  machine the outputs are still identical.

### Changed

- Results of existing calls are unchanged: a set of 21 calls covering
  fitting (both baselines, joint, `min_snr`, `noise="varying"`, Voigt),
  `fit_map`, the two-mode, multimode, three-cause and Bayesian
  inversions, `compare_mode_sets`, calibration (with and without
  offset), planning, both propagation tools, `region_mean`,
  `region_contrast` and the thermometer gives identical output under
  0.14.0 and 0.15.0, except the Bayesian posterior sigmas (new
  computation, largest relative difference 9.1e-16). README examples 1
  to 18 and the two earlier tutorials print the same output.
- `region_mean` now calls two helpers (`_homogeneity`,
  `_reference_cov`) that hold its former code unchanged, so that
  `region_profile` runs the same checks.
- The `max_dense` refusal of `bayesian_map_inversion(...,
  posterior_sigma=True)` is kept, with a message that no longer speaks
  of a dense inverse.
- Packaging: a new `MANIFEST.in` puts `examples/` (the tutorials),
  `CHANGELOG.md`, `CITATION.cff` and `CONTRIBUTING.md` into the source
  distribution. Before, the tutorial tests skipped when run from it;
  now all 148 tests pass from the unpacked sdist (installed in a fresh
  environment), none skipped. The wheel is unchanged.
- Type hints on the constructors of `MultiModeModel` and
  `ThreeCauseModel` and on the new functions and methods.
- Docstrings: `multimode` notes that the A-optimality ranking of
  `compare_mode_sets` depends on the units (the D-optimality value does
  not change the ranking).

### Performance

- `bayesian_map_inversion(..., posterior_sigma=True)` no longer takes
  the dense inverse of the posterior precision. With one sigma per mode
  and no missing value, the exact diagonal follows from the
  eigen-decomposition of the lattice Laplacian (a Kronecker sum of two
  path Laplacians): two small matrix products. Otherwise it comes from
  one sparse LU factorization and solves against the unit vectors, in
  blocks of 256. Equality with the 0.14.0 dense inverse (kept in the
  tests as the reference, built independently) is checked to
  100 eps cond(A) relative. Measured on a 45 x 45 map (4050 unknowns),
  median of interleaved runs, same machine, `OMP_NUM_THREADS=1`: one
  sigma per mode, 98 times faster (7.4 s to 0.076 s, 5 runs) and 166
  times faster in a second, more loaded session (19.9 s to 0.12 s, 7
  runs); per-pixel sigmas, 2.2 times (7.7 s to 3.5 s) and 3.4 times
  (14.2 s to 4.1 s). The machine was shared with other jobs. The maps
  themselves are unchanged (identical in both versions).

### Tutorials

- `examples/tutorial_edge_profile.py`: calibration, `fit_map` on a
  12 x 24 ribbon map (20 nm pixels), the strain and density profile
  versus the distance from the nearer edge (`region_profile`), the
  correlation between bins, and the extra electrons per micrometer of
  edge (`integrated_excess`), against the naive error bar that treats
  the bins as independent (about 1 s). Its output is pasted at the top
  of the script and in the README; the tutorial test runs it.

### Tests

- New `tests/test_v015.py` (11 tests); the tutorial test also runs
  the new tutorial; nine tests with exact comparisons in
  `test_mapfit.py`, `test_refusals_v0111.py`, `test_v012.py`,
  `test_v013.py` and `test_v014.py` now use the derived tolerances
  described above (`test_fit_map_is_bit_for_bit_the_per_pixel_loop` is
  now `test_fit_map_agrees_with_the_per_pixel_loop`). Test count:
  136 -> 148. The full suite also passes with the oldest allowed
  dependencies (NumPy 1.22.0, SciPy 1.8.0) on Python 3.9.

## 0.14.0 (2026-10-17)

Region means and contrasts with a full error budget, correlated shift
errors, error bars for shot noise, a faster `fit_map`, two tutorials,
and three fixes.

### Added

- `region_mean(coefficients, dw1, dw2, sigma1, sigma2, mask, *,
  shift_correlation=None, ref_sigma=None)` and
  `region_contrast(coefficients, dw1, dw2, sigma1, sigma2, mask_a,
  mask_b, *, shift_correlation=None)`, returning a `RegionEstimate`:
  the mean strain and density of a map region, or region A minus
  region B, with the error split into shift noise (averages down over
  pixels), calibration (from a two-mode `CalibrationResult`; does not
  average down) and reference frequencies (`ref_sigma`; does not
  average down, and cancels exactly in a contrast). `region_mean` also
  tests whether the pixels agree within their error bars (chi-square
  with 2 (N - 1) degrees of freedom, the same in shift space and in
  (strain, density) space) and reports the pixel spread. Validated
  against the mean and per-pixel covariances of
  `SeparationModel.invert`, `separation_with_calibration`, finite
  differences of a re-solve, and seeded Monte Carlo over 400 trials
  that repeat the calibration, the map noise and the reference error
  (spread within 12 % of the reported error bars). Propagation as in
  JCGM 100:2008 (GUM), 5.2.2, Eq. (13).
- `SeparationModel.invert(..., shift_correlation=None)`: the
  correlation of the two shift errors (from a joint fit) enters the
  error bars through the full 2 x 2 shift covariance. Zero gives the
  old result bit for bit. On 400 repeats of two overlapping E' + A'1
  peaks the density error bar without the correlation is 27 % above
  the actual spread; with it, it matches (tutorial 2).
  `separation_with_calibration` passes it through.
- `MapFitResult.shift_correlation`: the correlation of the two shift
  errors per pixel (the joint fit's value, 0 for separate window fits,
  NaN where masked).
- `noise="varying"` in `fit_lorentzian`, `fit_two_modes` and
  `fit_map`: error bars from the heteroskedasticity-consistent
  ("sandwich") covariance of H. White, Econometrica 48, 817 (1980),
  for spectra whose noise level changes along the axis, as shot noise
  does. Fitted values are unchanged. On 600 seeded Poisson spectra the
  center scatter is 1.7 times the default error bar and within 10 % of
  the new one; checked also against an independent finite-difference
  sandwich (relative 1e-5). The default, `noise="constant"`, is
  unchanged.
- README example 18 (shot noise) and a "Tutorials" section.

### Fixed

- `calibrate_lever_arms`, `plan_calibration` and `design_references`
  judged identifiability from the raw singular values of the design,
  which depends on the units. Before: the six-state design of the
  tests with the density in cm^-2 raised `ValueError: reference states
  are collinear ...` (`plan_calibration` returned
  `identifiable=False`). After: it gives the arms of the fit in 1e13
  cm^-2, rescaled (A'1 density arm -2.2798e-13 per cm^-2 against
  -2.2798 per 1e13 cm^-2; relative 1e-10). The test is now made on the
  design with unit-length columns (also with `fit_offset=True`). Test:
  `test_calibration_does_not_depend_on_the_density_unit`,
  `test_planning_does_not_depend_on_the_density_unit`.
- `bayesian_map_inversion(..., fill_masked=True)`: the same unit
  dependence in its pooled-information check. Before: a lever-arm
  matrix with the density column in cm^-2 raised `ValueError: ... their
  pooled information matrix is singular`. After: the filled pixel's
  density is 0.50185398920 (x 1e13 cm^-2), against 0.50185398920 with
  the density in 1e13 cm^-2 (largest relative difference over the map
  2.5e-7). Test:
  `test_bayesian_fill_masked_does_not_depend_on_the_density_unit`.
- `compare_mode_sets` with `subset_size` 0, 1 or larger than the
  number of modes returned `[]`; it now raises `ValueError` (also for
  a non-integer size, which raised `TypeError`). Test:
  `test_compare_mode_sets_refuses_a_subset_size_it_cannot_rank`.
- Documentation: the `fitting` module, the README "What is in the
  package" and "Limits" said the peak-fit error bars are reliable when
  the residuals are dominated by uncorrelated noise. With shot noise
  (uncorrelated, but larger on the peak than on the background) they
  are too small: by a factor of 1.7 for a peak 400 counts high on a
  30-count background (README example 18: scatter 0.0405 cm^-1, error
  bar 0.0239 cm^-1). The text now states the same-noise-level
  assumption, and `noise="varying"` is the remedy.

### Changed

- New refusals: `shift_correlation` outside [-1, 1], infinite, given
  without both sigmas, or of a shape that does not fit the map; an
  unknown `noise` name; the region refusals listed in the README.
- Type hints for the new `noise` parameters and return types for
  `compare_mode_sets` and `separation_with_calibration`.
- Results of all existing calls are unchanged: a set of 35 calls
  covering fitting (both baselines, joint and separate, `min_snr`,
  `fwhm_range`, Voigt), `fit_map`, the two-mode, multimode, three-cause
  and Bayesian inversions (scalar, per-pixel, `fill_masked`),
  calibration (all sigma forms, with and without offset), planning,
  both propagation tools and the thermometer gives bit-identical
  output under 0.13.1 and 0.14.0, and README examples 1 to 17 print
  the same output.

### Performance

- `fit_map` fits all pixels of a block together: the
  Levenberg-Marquardt loop of `fit_lorentzian` (and of the joint fit)
  is stacked over pixels, with each per-pixel matrix product, solve
  and inverse made by the same NumPy call as before, so every output
  (shifts, error bars, `ok`, reason codes) is bit for bit that of the
  0.13.1 per-pixel loop. The test keeps that loop as the reference and
  checks equality on noisy maps with peak-free and broken pixels, both
  baselines, joint and separate fits, and `min_snr`/`fwhm_range`.
  Measured on a 30 x 40 map with 281 points per spectrum (median of 7
  interleaved runs, same machine, `OMP_NUM_THREADS=1`): 6.6 times
  faster with the constant baseline (1.07 s to 0.16 s), 5.4 times with
  the linear baseline (1.04 s to 0.19 s) and 4.2 times with
  `joint=True` (2.16 s to 0.51 s). The machine was shared; repeated
  measurements gave 5.2 to 7.2, 4.7 to 6.8 and 3.3 to 4.2 times. Memory stays bounded (blocks of 1024 pixels).

### Tutorials

- `examples/tutorial_edge_charge.py`: calibration from an unknown
  starting state, `fit_map` on a ribbon map, then interior and edge
  means and the edge-minus-interior contrast with the full error
  budget (about 1 s).
- `examples/tutorial_overlapping_peaks.py`: separate-window bias,
  joint fit, inversion with and without the shift correlation, and a
  400-repeat check of the error bars (about 1 s).
- Their output is pasted at the top of each script and in the README;
  a test runs both and checks it.

### Tests

- New `tests/test_v014.py` (22 tests). Test count: 114 -> 136. The
  full suite also passes with the oldest allowed dependencies (NumPy
  1.22.0, SciPy 1.8.0) on Python 3.9.

## 0.13.1 (2026-10-10)

Citation corrections and packaging.

### Fixed (documentation)

- `graphene_g_2d_lee2012` docstring: the Fermi-level relation quoted "for
  holes" was the electron one. Froehlicher and Berciaud, Phys. Rev. B
  91, 205413 (2015) give `E_F = -18 dw_G - 83` (meV, holes, E_F <=
  -100 meV, their Eq. 10) and `E_F = 21 dw_G + 75` (electrons, E_F >=
  100 meV, Eq. 9). Both are now quoted. No code used the relation.
- `mos2_eprime_a1` reference and README: the mode frequencies 385.0
  and 403.0 cm^-1 used to convert the Grueneisen parameters are rounded
  monolayer values. Michail et al., Appl. Phys. Lett. 108, 173102 (2016)
  place their strain-doping origin at 384.6 and 402.7 cm^-1, which gives
  the same strain arms (-4.31 and -2.50 cm^-1 per percent) to the digits
  shipped. The numbers are unchanged.
- Packaging: the license is now given as the SPDX expression
  `license = "Apache-2.0"` with `license-files = ["LICENSE"]`, and the
  old license table and license classifier are gone. Building no longer
  prints setuptools' deprecation warnings (their support ends on
  2027-02-18). Building from source now needs setuptools 77 or newer;
  installing the package is unchanged.

## 0.13.0 (2026-10-10)

Reading instrument text exports, calibrating without knowing the
starting state, and three silent failures fixed.

### Added

- `load_map_table(path, usecols=(0, 1, 2, 3), delimiter=None,
  decimals=None)`: reads a map from a plain four-column text table
  (stage x, stage y, wavenumber, counts), the layout many mapping
  programs export, in any line order and with the wavenumber in either
  direction; rebuilds the pixel grid from the stage positions and
  returns `(wavenumber, cube, x_positions, y_positions)`. Refuses a
  scan that is not a complete rectangle, a pixel with its own axis or
  number of points, repeated or non-finite wavenumbers and non-finite
  positions. `load_spectrum_table` reads a two-column spectrum the same
  way. Tests: exact (bit for bit) recovery of independently built
  arrays from files written line by line, and `fit_map` on the loaded
  cube equal to `fit_map` on the original cube bit for bit.
- `calibrate_lever_arms(..., fit_offset=True)`: one free offset per
  mode, so strain and density need to be known only relative to the
  starting state and the shifts may be absolute peak positions. New
  `CalibrationResult` fields `offset`, `offset_sigma` and `full_cov`
  (None for the default fit). `cov` and `K_sigma` include the offset
  uncertainty, so the propagation tools take the result unchanged.
  Checked against `numpy.linalg.lstsq`, the Frisch-Waugh-Lovell closed
  form of the covariance, the closed-form bias of the through-origin
  fit, and seeded Monte Carlo. `plan_calibration` and
  `repeats_for_sigma` take `fit_offset=True` as well.
- `calibrate_lever_arms`: one error bar per point, shape (n,), for a
  single mode passed as (n,). Before, this stopped with NumPy's
  "operands could not be broadcast" error.
- README examples 16 (reading an instrument export) and 17
  (calibration with an unknown starting state), with the output they
  print.

### Fixed

- `load_spectrum_csv` accepted a wavenumber axis containing NaN or
  infinity. Before: the axis 400, nan, 402, 403 loaded as
  `[400, nan, 402, 403]` (and 400, 401, inf as `[400, 401, inf]`).
  After: `ValueError: wavenumber axis contains NaN or infinite
  values`. Test: `test_load_spectrum_csv_refuses_non_finite_axis`.
- `fit_map` with a NaN (or infinite) `ref1`/`ref2`: before, `ok` was
  True at all pixels and `n_masked` 0 while every shift was NaN (6 of
  6 pixels on the test map); `fit_two_modes` returned a NaN shift.
  After: `ValueError` before any pixel is fitted. Test:
  `test_fit_map_refuses_a_non_finite_reference`.
- `bayesian_map_inversion` with `lam_strain` or `lam_density` NaN or
  infinite: before, all-NaN maps with only SciPy's
  `MatrixRankWarning` ("Matrix is exactly singular"); after,
  `ValueError` that suggests a large finite weight. Test:
  `test_bayesian_refuses_non_finite_smoothness_weights`.
- Documentation: `MultiModeResult.condition_number` is the condition
  number of the unweighted K (the docstring said "weighted" for scalar
  sigmas; the value is unchanged); `fit_voigt`'s default start has a
  Gaussian FWHM equal to the width estimate and a Lorentzian FWHM of
  half of it (the docstring said "split evenly"); the `fitting` module
  docstring now quotes the 1e-6 tolerance the tests use for amplitude
  and baseline; `SeparationResult.density` is in the lever-arm unit
  (the docstring said cm^-2).

### Changed

- Clearer messages: `load_map_csv` with a NaN or infinite wavenumber
  (it said pixel (0, 0) "carries a different wavenumber axis than
  pixel (0, 0)"); non-numeric fields in `load_spectrum_csv` and
  `load_map_csv` (now with the line number); `MultiModeModel.invert`
  and `ThreeCauseModel.invert` with the wrong number of sigmas or
  shift maps of different shapes (before: NumPy reshape or stack
  errors); `SeparationModel.invert` with maps of different shapes (now
  with the shapes); `calibrate_lever_arms` with sigmas of a shape that
  fits neither the points nor the modes. All are still `ValueError`.
- Type hints on `fit_two_modes`, `fit_map`, `bayesian_map_inversion`,
  `calibrate_lever_arms`, the planning functions and the file loaders;
  a docstring for `BayesianMapResult`.
- Results of all existing calls are unchanged: a set of 0.12.0 calls
  covering fitting (both baselines, joint and separate, `min_snr`),
  the two-mode, multimode, three-cause and Bayesian inversions,
  calibration (all sigma forms), planning and both propagation tools
  gives bit-identical output under 0.12.0 and 0.13.0, and all README
  examples 1 to 15 print the same output.
- README: examples 16 and 17; updated package list, refusals, test
  descriptions, corrections and limits.

### Tests

- New `tests/test_v013.py` (18 tests). Test count: 96 -> 114. The
  full suite also passes with the oldest allowed dependencies (NumPy
  1.22.0, SciPy 1.8.0).

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
