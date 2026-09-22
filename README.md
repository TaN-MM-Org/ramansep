# ramansep

[![PyPI](https://img.shields.io/pypi/v/ramansep)](https://pypi.org/project/ramansep/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22014913-blue)](https://doi.org/10.5281/zenodo.22014913) [![tests](https://github.com/TaN-MM-Org/ramansep/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/ramansep/actions)

`ramansep` is a Python package that tells **strain** apart from
**charge** in Raman maps of 2D materials (sheets only one or a few atoms
thick, such as graphene or monolayer MoS2).

A Raman peak moves when the material is stretched, and it also moves
when extra electrons or holes are added. One peak alone cannot say
which of the two happened. Two peaks that respond differently can: if
you know how much each peak moves per unit of strain and per unit of
charge, the two measured shifts give two equations with two unknowns.
`ramansep` solves those equations at every pixel of a map and gives a
strain map and a charge-density map, each with an error bar. It also
covers the steps around that:

- How do I get the peak shifts out of my measured spectra, and which
  pixels failed?
- Which pair of peaks separates the two causes best?
- With three or more peaks, does "strain plus charge" actually explain
  what I measured, pixel by pixel?
- Can I trade some resolution for smaller error bars if the fields are
  smooth?
- How do I measure the response numbers on my own instrument, and how
  should I plan that measurement?
- How much of my final error bar comes from that calibration?
- What if laser heating also shifts the peaks?

The results are checked by automated tests against independent
calculations and exact limits (see
[How the results are checked](#how-the-results-are-checked)). When an
input cannot give a trustworthy answer, the package stops with an error
that says why, rather than returning a number that looks fine.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install, requirements and units](#install-requirements-and-units)
- [Examples](#examples) (each with the output it prints)
- [What is in the package](#what-is-in-the-package)
- [Cited coefficient sets](#cited-coefficient-sets)
- [When it refuses, and why](#when-it-refuses-and-why)
- [How the results are checked](#how-the-results-are-checked)
- [Corrections in earlier versions](#corrections-in-earlier-versions)
- [Limits, and what is deliberately not included](#limits-and-what-is-deliberately-not-included)
- [Where it comes from](#where-it-comes-from)
- [Citing, support and license](#citing-support-and-license)

## A short guide to the words used here

- **Raman spectrum** -- the light scattered by a material, plotted
  against its energy loss in **cm^-1** (wavenumbers). Each lattice
  vibration (**phonon**, or **mode**) gives a peak.
- **Mode names** -- labels such as A'1, E', 2LA(M) (MoS2) or G, 2D
  (graphene). They only name which peak is meant.
- **Shift** (`dw`) -- how far a peak has moved from its position in
  the undisturbed ("pristine") material, in cm^-1. Negative means the
  peak moved to lower wavenumber (the mode "softened").
- **Strain** -- stretching of the sheet, in percent. **Biaxial** means
  stretched equally in both directions; **uniaxial** means along one
  direction. Positive is tension.
- **Carrier density** (or **doping**, **charge**) -- extra electrons
  or holes (missing electrons) per area, in the MoS2 sets in units of
  1e13 cm^-2.
- **Gate, gated** -- an electrode that adds a known amount of charge to
  the sheet when a voltage is applied; a **strain stage** stretches the
  sample by a known amount. Both are used to measure how far a peak
  moves per unit of charge or strain.
- **Lever arm** -- how much one peak shifts per unit of one cause, for
  example -5.1 cm^-1 per percent strain. The lever arms of all the
  peaks form the **lever-arm matrix** `K` (one row per peak, one column
  per cause). The package assumes the shifts are linear in the causes:
  `shifts = K @ (strain, density)`.
- **Condition number** -- a single number that says how close the
  matrix is to being unsolvable. Large values mean the peaks respond
  too similarly, so small shift errors become large strain or density
  errors.
- **Sigma / error bar** -- one standard deviation of a value.
  **Correlation** between two estimates near +1 or -1 means they are
  hard to tell apart. The **covariance** matrix collects the squared
  error bars (on its diagonal) and how each pair of estimates varies
  together (off the diagonal).
- **Weighted least squares** -- the standard way to solve more
  equations than unknowns: find the values that make the weighted sum
  of squared mismatches smallest, each measurement weighted by
  1/sigma^2 so that more precise peaks count more.
- **Chi-square and p-value** -- with more peaks than causes, the
  equations cannot all be met exactly when something else is going on.
  Chi-square measures the mismatch, weighted by the error bars; the
  p-value is the probability of a mismatch at least that large if
  strain and charge alone were at work. A tiny p-value flags a pixel.
  The **degrees of freedom** (`dof`) of this check are the number of
  peaks minus the number of causes solved for.
- **Hyperspectral map (cube)** -- one spectrum per pixel, stored as an
  array of shape (rows H, columns W, spectral points L).
- **Lorentzian / Voigt** -- peak shapes. A Lorentzian is the natural
  shape of a phonon peak; a Voigt is a Lorentzian blurred by a
  Gaussian, for peaks broadened by the instrument.
- **Baseline** -- the background under a peak. A sloped background
  (for example from fluorescence) pulls a fitted peak centre sideways
  unless the slope is fitted too.
- **Stokes / anti-Stokes** -- Raman peaks on the energy-loss side and
  the energy-gain side. The ratio of their heights depends on
  temperature, so it works as a thermometer.
- **DFT (density functional theory)** -- a standard quantum-mechanical
  computer calculation of a material. "Frozen-phonon" DFT computes a
  vibration's frequency by displacing the atoms along that vibration.
  The source paper used it to compute the MoS2 strain lever arms.
- **Grueneisen parameter** (gamma) -- a number that says how strongly
  a vibration's frequency w changes when the sheet is stretched; for
  biaxial strain the package uses gamma = -(1/(2w)) dw/d(strain), with
  strain as a fraction (not percent).

## Install, requirements and units

```
pip install ramansep
```

For development: clone the repository and `pip install -e .[test]`.

It needs Python 3.9 or newer, NumPy 1.22 or newer and SciPy 1.8 or
newer. `matplotlib` is optional (`pip install ramansep[plot]`) and is
used only by the demo script `examples/synthetic_map.py`.

Units and signs:

- Shifts are in cm^-1, relative to the pristine reference frequency of
  each mode (you supply those references, e.g. `ref1`, `ref2`).
- The inversion returns strain and density in whatever units your
  lever arms use. The shipped MoS2 sets use percent biaxial strain
  (positive = tension) and electron density in 1e13 cm^-2. The
  graphene set uses percent uniaxial strain and, on purpose, a doping
  coordinate in cm^-1 (see [Cited coefficient sets](#cited-coefficient-sets)).
- Every `ModeCoefficients` must carry a `reference` string that says
  where its numbers came from; it is a required field.

## Examples

Each example below runs as written, and the output shown is what it
printed with ramansep 0.11.1. Unless a value is said to come from a
cited source, it is an illustrative value chosen for the example. The
repository also has a longer demo, `examples/synthetic_map.py`, that
inverts a noisy synthetic 128 x 128 map and plots it with matplotlib.

### 1. Strain and charge at two spots of a MoS2 ribbon

The shifts here are the ones quoted by the source paper (Mahim and
Rahman, under review) from tip-enhanced Raman maps (maps taken with a
sharp metal tip that gives nanometre resolution; Krayev et al., Appl.
Phys. Lett. 128, 203102 (2026)); the 0.1 cm^-1 shift error is
illustrative.

```python
import numpy as np
import ramansep as rs

model = rs.SeparationModel(rs.mos2_a1_2la())    # monolayer MoS2, A'1 + 2LA(M)

# Peak shifts (cm^-1) quoted by the source paper at two spots:
# the ribbon edge (A'1 -0.5, 2LA(M) 0.0) and an interior spot (-0.6, -2.8).
dw_a1 = np.array([-0.5, -0.6])
dw_2la = np.array([0.0, -2.8])
res = model.invert(dw_a1, dw_2la, sigma1=0.1, sigma2=0.1)   # 0.1 cm^-1: illustrative

for name, i in [("edge", 0), ("interior", 1)]:
    print(f"{name:8s} strain {res.strain[i]:+.3f} +/- {res.strain_sigma[i]:.3f} %   "
          f"electrons {res.density[i] * 1e13:+.2e} +/- {res.density_sigma[i] * 1e13:.1e} cm^-2")
print(f"condition number of the matrix: {res.condition_number:.1f}")
```

```
edge     strain +0.000 +/- 0.005 %   electrons +2.27e+12 +/- 4.7e+11 cm^-2
interior strain +0.134 +/- 0.005 %   electrons -3.78e+11 +/- 4.7e+11 cm^-2
condition number of the matrix: 10.1
```

The edge carries about 2.3e12 electrons per cm^2 and no strain; the
interior spot is under 0.134 % tension, and its density is within one
error bar of zero. These are the paper's published separations; the
test suite checks them (see below). `res.correlation` gives the
per-pixel correlation of the two estimates.

### 2. Choosing a mode pair

```python
import numpy as np
import ramansep as rs

sig = 0.1                                        # illustrative shift error, cm^-1
paper = rs.SeparationModel(rs.mos2_a1_2la()).invert(0.0, 0.0, sig, sig)
optical = rs.SeparationModel(rs.mos2_eprime_a1()).invert(0.0, 0.0, sig, sig)
print(f"A'1 + 2LA(M): strain error {float(paper.strain_sigma):.4f} %")
print(f"E'  + A'1   : strain error {float(optical.strain_sigma):.4f} %")
print(f"ratio: {float(optical.strain_sigma / paper.strain_sigma):.1f}")
```

```
A'1 + 2LA(M): strain error 0.0048 %
E'  + A'1   : strain error 0.0257 %
ratio: 5.4
```

Both optical modes E' and A'1 move only a little with strain, so with
the same shift errors that pair gives a strain error about five times
larger than the A'1 + 2LA(M) pair. The error bars depend only on the
lever arms and the shift errors, so this comparison can be made before
measuring anything.

### 3. From a map file to strain and charge maps

```python
import os, tempfile
import numpy as np
import ramansep as rs
from ramansep import lorentzian

coeffs = rs.mos2_a1_2la()
model = rs.SeparationModel(coeffs)

# An illustrative 4 x 5 map: known strain (%) and electron density (1e13 cm^-2).
rng = np.random.default_rng(3)
strain = rng.uniform(-0.1, 0.1, (4, 5))
density = rng.uniform(0.0, 0.2, (4, 5))
dw1, dw2 = model.forward(strain, density)

# Make a spectrum for every pixel: two Lorentzian peaks plus noise.
x = np.linspace(380.0, 480.0, 401)
cube = np.empty((4, 5, x.size))
for i in range(4):
    for j in range(5):
        cube[i, j] = (lorentzian(x, 404.7 + dw1[i, j], 3.0, 800.0, 50.0)
                      + lorentzian(x, 452.0 + dw2[i, j], 4.0, 500.0, 0.0)
                      + rng.normal(0.0, 3.0, x.size))
cube[3, 4, 100] = np.nan                         # one broken pixel

path = os.path.join(tempfile.mkdtemp(), "map.csv")
rs.save_map_csv(path, x, cube)                   # the documented file format

x2, cube2 = rs.load_map_csv(path)
fits = rs.fit_map(x2, cube2, window1=(395, 415), window2=(440, 465),
                  ref1=404.7, ref2=452.0)
print("masked pixels:", fits.n_masked, "at", np.argwhere(~fits.ok).tolist())

inv = model.invert(fits.dw1, fits.dw2, fits.sigma1, fits.sigma2)
ok = fits.ok
z_s = np.abs(inv.strain - strain)[ok] / inv.strain_sigma[ok]
z_d = np.abs(inv.density - density)[ok] / inv.density_sigma[ok]
print(f"typical error bars: strain {np.median(inv.strain_sigma[ok]):.4f} %, "
      f"density {np.median(inv.density_sigma[ok]) * 1e13:.1e} cm^-2")
print(f"pixels whose error is within 3 error bars: strain {(z_s < 3).sum()} of {ok.sum()}, "
      f"density {(z_d < 3).sum()} of {ok.sum()}")
print("strain at the broken pixel:", inv.strain[3, 4])
```

```
masked pixels: 1 at [[3, 4]]
typical error bars: strain 0.0002 %, density 1.3e+10 cm^-2
pixels whose error is within 3 error bars: strain 19 of 19, density 19 of 19
strain at the broken pixel: nan
```

The map file is plain CSV with the header `row,col,wavenumber_cm1,counts`
(0-based pixel indices, one shared wavenumber axis). A single spectrum
uses `wavenumber_cm1,counts` (`save_spectrum_csv`, `load_spectrum_csv`).
`fit_map` fits both peaks at every pixel. A pixel is masked (NaN in all
four output maps, `False` in `fits.ok`) when its counts are not finite,
a fit stops with an error or does not converge, or a fitted centre ends
up outside its window. Mistakes in the settings themselves (window
limits, `baseline` name, a wavenumber axis containing NaN) are refused
with an error before any pixel is fitted.
The per-pixel inversion keeps masked pixels as NaN and does not touch
their neighbours. For spectra on a sloped background, pass
`baseline="linear"`.

### 4. Three peaks, and a check that the model holds

```python
import numpy as np
import ramansep as rs

# Lever arms (cm^-1 per unit strain, per unit density) of three modes.
# Rows 1-2 are the MoS2 A'1 and 2LA(M) values; row 3 is an invented,
# illustrative third mode.
K = np.array([[-5.1, -2.2],
              [-20.9, 0.0],
              [-8.0, -1.1]])
sig = [0.15, 0.10, 0.12]                         # illustrative shift errors, cm^-1
mm = rs.MultiModeModel(K, mode_names=["A'1", "2LA(M)", "X"])

rng = np.random.default_rng(3)
strain = rng.normal(0, 0.2, (30, 30))
density = rng.normal(0, 1.0, (30, 30))
shifts = [s + rng.normal(0, e, s.shape) for s, e in zip(mm.forward(strain, density), sig)]
shifts[2][10:15, :] += 2.0                       # rows 10-14: a cause the model does not know

res = mm.invert(shifts, sigmas=sig)
print("degrees of freedom of the check:", res.dof)
print(f"median p-value, rows 10-14 : {np.median(res.p_value[10:15]):.1e}")
print(f"median p-value, other rows : {np.median(res.p_value[:10]):.2f}")

for r in rs.compare_mode_sets(K, sig, mode_names=["A'1", "2LA(M)", "X"]):
    print(r["names"], f"var_strain {r['var_strain']:.2e}  var_density {r['var_density']:.2e}")
print(f"all three     var_strain {res.strain_sigma[0, 0]**2:.2e}  var_density {res.density_sigma[0, 0]**2:.2e}")
```

```
degrees of freedom of the check: 1
median p-value, rows 10-14 : 1.0e-43
median p-value, other rows : 0.50
["A'1", '2LA(M)'] var_strain 2.29e-05  var_density 4.77e-03
['2LA(M)', 'X'] var_strain 2.29e-05  var_density 1.31e-02
["A'1", 'X'] var_strain 6.74e-04  var_density 1.26e-02
all three     var_strain 2.21e-05  var_density 3.65e-03
```

With more peaks than causes, `MultiModeModel` uses weighted least
squares (each peak weighted by 1/sigma^2). The extra peak shrinks the
error bars and makes the chi-square check possible: the rows with the
hidden extra shift stand out with p-values near zero, while elsewhere
the p-values scatter around 0.5 as they should. With exactly two peaks
there is no check (`dof` is 0, `chi2_map` and `p_value` are `None`).
`compare_mode_sets` ranks every subset of the modes (pairs by default)
that can separate the two causes by the sum of the two variances
(squared error bars), smallest first.

### 5. Smoothing a whole map

```python
import numpy as np
import ramansep as rs

K = np.array([[-2.3, 1.1], [-0.9, -1.7], [0.4, 2.2]])   # illustrative, non-physical
sig = np.array([0.03, 0.05, 0.02])                      # cm^-1

# A smooth illustrative truth on a 12 x 12 map, measured with noise.
yy, xx = np.mgrid[0:12, 0:12]
strain = 0.002 * xx                                     # gentle gradient
density = 0.01 + 0.001 * yy
rng = np.random.default_rng(7)
shifts = np.einsum("mj,jhw->mhw", K, np.stack([strain, density])) \
    + rng.normal(size=(3, 12, 12)) * sig[:, None, None]

for lam in (0.0, 1000.0, 10000.0):
    r = rs.bayesian_map_inversion(K, shifts, sig, lam_strain=lam, posterior_sigma=True)
    err = np.sqrt(np.mean((r.strain - strain) ** 2))
    print(f"lam = {lam:6.0f}: mean strain sigma {r.strain_sigma.mean():.4f}, "
          f"RMS strain error {err:.4f}")

shifts[0, 2, 3] = np.nan                                # a masked pixel
try:
    rs.bayesian_map_inversion(K, shifts, sig, lam_strain=100.0)
except ValueError as err:
    print("refused:", str(err)[:60], "...")
```

```
lam =      0: mean strain sigma 0.0123, RMS strain error 0.0115
lam =   1000: mean strain sigma 0.0101, RMS strain error 0.0078
lam =  10000: mean strain sigma 0.0057, RMS strain error 0.0039
refused: shifts contain non-finite values; the smoothness prior coupl ...
```

`bayesian_map_inversion` solves the whole map at once, adding a
penalty `lam` times the squared differences between neighbouring
pixels (a "smoothness prior"; `lam_strain` and `lam_density` can
differ). It is one sparse linear system, solved exactly. At `lam = 0`
it gives the same answer as the pixel-by-pixel inversion, and a larger
`lam` never increases the error bars. Smoothing helps only when the
real fields are smooth, as they are here by construction; you choose
`lam`, and the package does not estimate it for you. Masked (NaN)
pixels are refused because the prior links every pixel to its
neighbours; remove or fill them deliberately first. Exact per-pixel
error bars (`posterior_sigma=True`) need a dense matrix inverse and are
refused above `max_dense` = 4096 unknowns (two per pixel) unless you
raise that limit.

### 6. Measuring the lever arms on your own instrument

```python
import numpy as np
import ramansep as rs

# Reference states you prepared: strain (%) from a strain stage,
# electron density (1e13 cm^-2) from a gate. Illustrative values.
strain = np.array([0.0, 0.4, 0.8, 0.0, 0.3, 0.6, 0.2, 0.0])
density = np.array([0.0, 0.0, 0.0, 0.9, 0.6, 0.3, 1.2, 0.5])

# Pretend the true lever arms are the MoS2 A'1 / 2LA(M) values and
# measure the shifts with 0.1 cm^-1 noise.
K_true = np.array([[-5.1, -2.2], [-20.9, 0.0]])
rng = np.random.default_rng(1)
shifts = np.column_stack([strain, density]) @ K_true.T + rng.normal(0, 0.1, (8, 2))

cal = rs.calibrate_lever_arms(strain, density, shifts, sigmas=0.1,
                              mode_names=["A'1", "2LA(M)"])
np.set_printoptions(precision=2, suppress=True)
print("fitted lever arms:\n", cal.K)
print("their error bars:\n", cal.K_sigma)
print("chi2 / dof per mode:", cal.chi2 / cal.chi2_dof)

model = rs.SeparationModel(cal.coefficients(reference="stage + gate calibration, illustrative"))
print(model.coefficients.reference)

try:                                             # a strain-only sweep
    rs.calibrate_lever_arms(strain[:3], density[:3], shifts[:3], sigmas=0.1)
except ValueError as err:
    print("refused:", err)
```

```
fitted lever arms:
 [[ -5.    -2.27]
 [-20.9    0.03]]
their error bars:
 [[0.09 0.06]
 [0.09 0.06]]
chi2 / dof per mode: [0.09 0.56]
stage + gate calibration, illustrative
refused: reference states are collinear in the (strain, density) plane, so the strain and density lever arms are not separately identifiable; add a reference point off that line (e.g. a gated point to a strain-only sweep)
```

`calibrate_lever_arms` fits each mode's two lever arms by weighted
least squares from reference states of known strain and density. With
`sigmas` given, the error bars are the exact result for that noise
level, and `chi2 / dof` near 1 says the straight-line model and your
stated errors agree. Without `sigmas`, the error bars are estimated
from the scatter of the fit, which needs at least 3 reference points.
A sweep that changes strain only (or strain and charge in fixed
proportion) cannot tell the two lever arms apart, so it is refused.
`cal.K` of a calibration with three or more modes goes straight into
`MultiModeModel`.

### 7. Planning the calibration before measuring it

```python
import numpy as np
from ramansep import plan_calibration, design_references, repeats_for_sigma

# Four planned reference states, 0.15 cm^-1 expected shift error (illustrative).
plan = plan_calibration(strain=[0.0, 0.3, 0.6, 0.0],
                        density=[0.0, 0.0, 0.1, 0.8], sigmas=0.15)
print("identifiable:", plan["identifiable"])
print("expected error bars (strain arm, density arm):", plan["K_sigma"].round(3))

# States the stage and gate can reach; which 4 are worth measuring?
eps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.25])
rho = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 0.8])
pick = design_references(eps, rho, n_pick=4, sigmas=0.15)
print("measure candidates:", pick["indices"])

r, plan_r = repeats_for_sigma(0.1, eps[pick["indices"]], rho[pick["indices"]], sigmas=0.15)
print(f"repeat that set {r} times -> worst error bar {plan_r['K_sigma'].max():.3f}")

line = plan_calibration([0.1, 0.2, 0.4], [0.0, 0.0, 0.0], sigmas=0.15)
print("strain-only sweep identifiable:", line["identifiable"])
```

```
identifiable: True
expected error bars (strain arm, density arm): [0.225 0.187]
measure candidates: [8, 5, 7, 4]
repeat that set 6 times -> worst error bar 0.092
strain-only sweep identifiable: False
```

Because the calibration is a linear fit, the error bars it will
report depend only on which reference states you prepare and on the
shift errors, not on the measured shifts. `plan_calibration` computes
them in advance. `design_references` picks states one at a time, each
time taking the one that shrinks the joint uncertainty most (the
"D-optimal" rule; F. Pukelsheim, Optimal Design of Experiments, SIAM
(2006)); this is a good heuristic, not a guarantee of the best
possible subset. `repeats_for_sigma` uses the fact that repeating the
whole set `r` times divides the covariance by `r`. `design_references`
and `repeats_for_sigma` refuse a candidate set that lies on one line.

### 8. How much of the error bar comes from the calibration

```python
import numpy as np
from ramansep import calibrate_lever_arms, separation_with_calibration

# Five reference states (illustrative) and noise-free shifts from the
# MoS2 A'1 / 2LA(M) lever arms; the calibration is told they carry 0.2 cm^-1 errors.
K_true = np.array([[-5.1, -2.2], [-20.9, 0.0]])
eps = np.array([0.0, 0.4, 0.8, 0.0, 0.3])
rho = np.array([0.0, 0.0, 0.2, 0.9, 0.6])
cal = calibrate_lever_arms(eps, rho, np.column_stack([eps, rho]) @ K_true.T, sigmas=0.2)

out = separation_with_calibration(cal, dw1=-2.0, dw2=-8.0, sigma1=0.1, sigma2=0.1)
print(f"strain  {float(out['strain']):.4f} %")
print(f"  error from the spectra     {float(out['strain_sigma_shifts']):.4f}")
print(f"  error from the calibration {float(out['strain_sigma_calibration']):.4f}")
print(f"  total                      {float(out['strain_sigma']):.4f}")
print(f"density {float(out['density']):.4f} x 1e13 cm^-2")
print(f"  error from the spectra     {float(out['density_sigma_shifts']):.4f}")
print(f"  error from the calibration {float(out['density_sigma_calibration']):.4f}")
print(f"  total                      {float(out['density_sigma']):.4f}")
```

```
strain  0.3828 %
  error from the spectra     0.0048
  error from the calibration 0.0040
  total                      0.0063
density 0.0217 x 1e13 cm^-2
  error from the spectra     0.0468
  error from the calibration 0.0396
  total                      0.0613
```

Other inversions treat the lever arms as exact.
`separation_with_calibration` also carries the calibration's own
uncertainty into the result, and reports the two parts separately; the
total is their quadrature sum (square root of the sum of squares). It
uses the first-order (linearised) error propagation, based on the
identity dx/dK_mj = -K^-1 E_mj x for the two-mode solution x = K^-1 y
(E_mj is the matrix with a single 1 at row m, column j). It works for
two-mode calibrations only; a calibration with more modes is refused.

### 9. Temperature from the anti-Stokes / Stokes ratio

```python
from ramansep import (HC_OVER_KB_CM_K, anti_stokes_ratio,
                      calibrate_anti_stokes, temperature_from_anti_stokes)

omega = 385.0                  # the mode's Raman shift, cm^-1
print(f"hc/kB = {HC_OVER_KB_CM_K:.7f} cm K")

# Calibrate once, on a spot you know is at 295 K (illustrative ratio).
C = calibrate_anti_stokes(ratio_measured=0.13, omega_cm1=omega, temperature_K=295.0)
print(f"calibration constant C = {C:.4f}")

# Later, under stronger laser power, the ratio has risen.
out = temperature_from_anti_stokes(0.17, omega, C, ratio_sigma=0.005)
print(f"T = {out['temperature']:.1f} +/- {out['temperature_sigma']:.1f} K")
print(f"with C wrongly set to 1: T = {temperature_from_anti_stokes(0.17, omega, 1.0):.1f} K")

try:
    temperature_from_anti_stokes(1.2 * C, omega, C)
except ValueError as err:
    print("refused:", str(err)[:60], "...")
```

```
hc/kB = 1.4387769 cm K
calibration constant C = 0.8500
T = 344.2 +/- 6.3 K
with C wrongly set to 1: T = 312.6 K
refused: ratio >= calibration constant: the Bose factor is strictly < ...
```

The model is `I_AS / I_S = C exp(-hc omega / (kB T))`, where `omega`
is the mode's Raman shift. The constant `C` absorbs the instrument's
response and the scattering factors; it is close to, but not exactly,
1, and assuming 1 gives a wrong temperature (as the last printed line
shows). So `C` is measured once at a known temperature. The constant
`hc/kB` is computed from the exact SI values of h, c and kB, not typed
in by hand. A ratio at or above `C` has no finite positive temperature
and is refused; it points to a wrong calibration or background
subtraction. `anti_stokes_ratio` is the forward model.

### 10. Strain, charge and temperature together

```python
import numpy as np
from ramansep import ThreeCauseModel

# Four modes; columns: shift per % strain, per 1e13 cm^-2, per kelvin.
# Invented, illustrative values -- no temperature coefficients ship with the package.
K = np.array([[-2.1, -0.8, -0.011],
              [-1.0, -2.3, -0.016],
              [-3.2, -0.4, -0.007],
              [-0.6, -1.1, -0.021]])
model = ThreeCauseModel(K)
shifts = model.forward(strain=0.2, density=0.5, temperature=40.0)   # 40 K of heating
res = model.invert(shifts, sigmas=[0.05, 0.05, 0.05, 0.05])
print(f"strain {float(res.strain):.3f} +/- {float(res.strain_sigma):.3f} %")
print(f"density {float(res.density):.3f} +/- {float(res.density_sigma):.3f} x 1e13 cm^-2")
print(f"temperature {float(res.temperature):.1f} +/- {float(res.temperature_sigma):.1f} K")
```

```
strain 0.200 +/- 0.016 %
density 0.500 +/- 0.042 x 1e13 cm^-2
temperature 40.0 +/- 4.2 K
```

Heating shifts peaks too, and a two-cause analysis would count that
shift as strain or charge. `ThreeCauseModel` adds a third column of
lever arms, the temperature coefficients (cm^-1 per kelvin) that you
have calibrated for your material, and needs at least three modes.
Because shifts are measured from the reference frequencies, the
"temperature" it returns is the change from the temperature at which
those references hold. With four or more modes it also returns the
chi-square check (`chi2_map`, `p_value`) and the full 3 x 3 covariance
per pixel. Measuring the temperature with example 9 and comparing it
with this estimate is a direct cross-check.

### 11. Graphene: the G + 2D decomposition

```python
import warnings
import ramansep as rs

coeffs = rs.graphene_g_2d_lee2012()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    rs.SeparationModel(coeffs)                   # default condition_warn = 30
print("warning:", str(caught[0].message)[:62], "...")

model = rs.SeparationModel(coeffs, condition_warn=100)
res = model.invert(dw1=0.3, dw2=-6.84)          # G and 2D shifts, cm^-1 (illustrative)
print(f"strain {float(res.strain):.3f} %, hole-doping G shift {float(res.density):.2f} cm^-1")
```

```
warning: coefficient matrix condition number is 91.5; the mode pair is  ...
strain 0.200 %, hole-doping G shift 5.00 cm^-1
```

The G peak has barely moved here, but only because strain and doping
pushed it in opposite directions; the 2D peak separates the two. The
condition number of this set (91.5) is high mainly because its two
columns are in different units (percent and cm^-1), so the docstring
advises `condition_warn=100`. The noise amplification is moderate: a
1 cm^-1 error on both shifts costs about 0.035 % of strain and
1.6 cm^-1 of doping coordinate (asserted in the tests). The "density"
here is a G-band shift, not a carrier density; see below.

## What is in the package

Every name below is exported from `ramansep` (its `__all__`). Each
docstring (`help(ramansep.fit_map)`, for example) gives inputs, units
and conventions.

**Two-mode inversion**

- `SeparationModel(coefficients, condition_warn=30.0)` -- inverts two
  shift maps (`invert(dw1, dw2, sigma1, sigma2)`) into a
  `SeparationResult` with `strain`, `density`, `strain_sigma`,
  `density_sigma`, `correlation` and `condition_number`; `forward`
  predicts shifts. Error bars are computed only when both `sigma1` and
  `sigma2` are given.
- `ModeCoefficients` -- the four lever arms of two modes plus the
  required `reference` string.

**Coefficient sets**

- `mos2_a1_2la()`, `mos2_eprime_a1()`, `graphene_g_2d_lee2012()` --
  the cited sets described in the next section.
- `synthetic_demo()` -- arbitrary, non-physical values for tests and
  demos only.

**More modes, and whole maps**

- `MultiModeModel(K, mode_names)`, `MultiModeResult` -- weighted
  least squares for any number (two or more) of modes, with `chi2_map`,
  `p_value` and `dof`.
- `compare_mode_sets(K, sigmas, mode_names, subset_size=2)` -- ranks
  mode subsets by delivered variance.
- `bayesian_map_inversion(K, shifts, sigmas, lam_strain, lam_density,
  posterior_sigma, max_dense)`, `BayesianMapResult` -- joint inversion
  of a whole map with a smoothness prior (a Gaussian Markov random
  field, Rue and Held 2005: a standard statistical way of saying that
  each pixel is probably close to its four neighbours).

**From spectra to shifts**

- `lorentzian`, `fit_lorentzian`, `PeakFit` -- Lorentzian peak shape
  and a least-squares fit by the Levenberg-Marquardt method (a standard
  step-by-step curve-fitting method), using exact formulas for the
  derivatives; `baseline="constant"` or `"linear"`. Parameter errors
  are the standard linearised estimates, which are reliable when the
  residuals are dominated by uncorrelated noise.
- `voigt`, `fit_voigt`, `VoigtFit` -- the Voigt shape, computed
  exactly with the Faddeeva function (a standard special function,
  `scipy.special.wofz`), normalised to peak height.
- `fit_two_modes(x, y, window1, window2, ref1, ref2, baseline)` --
  fits both peaks, each in its own non-overlapping window, and returns
  `(dw1, dw2, sigma1, sigma2, fit1, fit2)`.
- `fit_map(...)`, `MapFitResult` -- the same at every pixel of a
  cube, with masking of failed pixels.
- `load_spectrum_csv`, `save_spectrum_csv`, `load_map_csv`,
  `save_map_csv` -- the documented CSV formats of example 3.

**Calibration**

- `calibrate_lever_arms(strain, density, shifts, sigmas, mode_names)`,
  `CalibrationResult` (`K`, `K_sigma`, `cov`, `chi2`, `chi2_dof`,
  `n_points`, `condition_number`, `mode_names`, and
  `coefficients(reference)` for two modes).
- `plan_calibration`, `design_references`, `repeats_for_sigma` --
  planning, example 7.
- `separation_with_calibration` -- example 8.

**Temperature**

- `ThreeCauseModel`, `ThreeCauseResult` -- example 10.
- `anti_stokes_ratio`, `calibrate_anti_stokes`,
  `temperature_from_anti_stokes`, `HC_OVER_KB_CM_K` (hc/kB in cm K) --
  example 9.

## Cited coefficient sets

Three sets ship with full provenance in their `reference` field and
docstring. Each docstring states which number comes from which source
and the conditions it applies under. Check that your sample matches
before use.

- `mos2_a1_2la()`: the A'1 + 2LA(M) pair for monolayer MoS2. Strain
  arms -5.1 and -20.9 cm^-1 per percent biaxial strain, computed by
  frozen-phonon DFT in the source paper (Mahim and Rahman, under
  review; data doi:10.5281/zenodo.21778170). A'1 doping response
  -2.2 cm^-1 per 1e13 cm^-2 of electrons, measured with a gate by
  Chakraborty et al., Phys. Rev. B 85, 161403(R) (2012). The 2LA(M)
  doping response has never been measured and is set to zero; the
  source paper bounds the consequence at 16% of a recovered edge
  charge. Stated conditions: 1H monolayer (the common semiconducting
  crystal form), 532 nm laser, on an SiO2 substrate, 300 K. The tests
  reproduce the source paper's published separations from this set
  (example 1).
- `mos2_eprime_a1()`: the historically used all-optical E' + A'1 pair
  (Michail et al., Appl. Phys. Lett. 108, 173102 (2016)). Strain arms
  -4.31 (E') and -2.50 (A'1) cm^-1 per percent, from the measured
  biaxial Grueneisen parameters 0.56 and 0.31 of Michail et al., ACS
  Appl. Mater. Interfaces 16, 49602 (2024), converted with the mode
  frequencies 385.0 and 403.0 cm^-1 (Mignuzzi et al., Phys. Rev. B 91,
  195411 (2015); Michail et al. 2016). Doping responses -0.33 (E') and
  -2.2 (A'1) cm^-1 per 1e13 cm^-2, measured by Chakraborty et al.
  (2012). Stated conditions: 1H monolayer, on SiO2, 300 K. Both modes
  respond weakly to strain, so it carries about five times the strain
  uncertainty (example 2) -- included for comparison.
- `graphene_g_2d_lee2012()`: the G + 2D decomposition of Lee et al.,
  Nat. Commun. 3, 1024 (2012). Its "doping" output is deliberately a
  G-band shift, not a density: graphene's doping response is
  nonlinear, so no universal linear rate exists to ship, and the
  docstring points to the gated calibrations needed for conversion.
  Stated for hole doping, uniaxial strain of random orientation.

**Tungsten-based materials (2024-2026).** For W-based monolayers the
literature now provides verified strain and temperature responses --
monolayer WSe2 biaxial strain rates (Michail et al., ACS Appl. Mater.
Interfaces 16, 49602 (2024)), monolayer WS2 strain rates (Roy, Yang
and Gao, Sci. Rep. 14, 3860 (2024)) and temperature coefficients
(Huang et al., Sci. Rep. 6, 32236 (2016)) -- while gate-calibrated
doping arms remain without a clean linear coefficient (the measured
behavior is an electron-only, threshold-like softening of the
out-of-plane modes, the vibrations in which atoms move perpendicular
to the sheet; Sohier et al., PRX 9, 031019 (2019)). That is why
no W-material set ships as constants here: the strain column is
citable, the doping column is not yet, and this package does not ship
half a lever-arm matrix. Calibrate your own with
`calibrate_lever_arms` -- the doping arm from your own gated reference
points -- and the provenance travels with the analysis.

For monolayer MoS2 temperature coefficients, the `thermal` module
docstring points to S. Sahoo et al., J. Phys. Chem. C 117, 9042
(2013); no temperature coefficient is shipped.

## When it refuses, and why

`ramansep` raises `ValueError` instead of guessing when:

- a two-mode lever-arm matrix is not 2 x 2 or is exactly singular (the
  two modes respond identically); a merely poorly conditioned one
  (condition number above `condition_warn`, default 30) gives a
  warning, not an error;
- a multimode or map lever-arm matrix has rank below 2 (below 3 for
  `ThreeCauseModel`, which also needs at least three modes), or the
  weighted problem is singular at some pixel;
- the shift maps do not match in shape or number;
- a shift error (`sigmas`) is zero or negative in `MultiModeModel`,
  `ThreeCauseModel`, `bayesian_map_inversion`, the calibration and
  planning tools (`SeparationModel` does not check the sign);
- `bayesian_map_inversion` gets NaN or infinite shifts, a negative
  smoothness weight, or asks for exact error bars above `max_dense`
  unknowns;
- a peak fit gets too few points (5 for a Lorentzian with a constant
  baseline, 6 with a linear baseline or for a Voigt), non-finite data,
  an unknown `baseline` name, or a non-positive starting width;
- the two fit windows are reversed, overlap, or hold too few points,
  or the `baseline` name is unknown (`fit_two_modes`, and `fit_map`
  before fitting any pixel); `fit_map` also refuses a wavenumber axis
  that contains NaN or infinity;
- a CSV file has the wrong header or field count, a wavenumber axis
  that is not strictly increasing, a pixel grid with holes, or a pixel
  with its own wavenumber axis;
- calibration reference states lie on one line through the origin of
  the (strain, density) plane (the two lever arms cannot be told
  apart), there are fewer than 2 of them, or fewer than 3 when no
  `sigmas` are given;
- `CalibrationResult.coefficients` is asked to package other than two
  modes, or `separation_with_calibration` gets such a calibration;
- the anti-Stokes ratio is at or above the calibration constant, or a
  temperature, frequency, ratio or constant is not positive.

## How the results are checked

77 automated tests run on every push and pull request, on Python 3.9,
3.10, 3.11, 3.12, 3.13 and 3.14, and once more on Python 3.9 with the
oldest NumPy (1.22.0) and SciPy (1.8.0) the package allows. The
numerical checks compare the package with an independent calculation,
a closed form, a published number, or a simulation with fixed random
seeds; the others check that refusals fire. The main checks, with the
tolerance each test really uses. "To 1e-12" means the difference may be
at most 1e-12; "plus a relative 1e-5" means that 1e-5 times the size
of the expected value is also allowed. Where a test does not set both
parts, NumPy's own defaults apply, and these are listed too:

**Two-mode inversion and the cited sets**

- Forward-then-invert round trips recover strain and density to
  1e-12 plus a relative 1e-7 (synthetic set, both MoS2 sets) and to
  1e-12 plus a relative 1e-5 (graphene set).
- Reported error bars match the scatter of 200 000 seeded noisy
  inversions to within 2 %.
- MoS2 A'1 + 2LA(M), paper edge spot: zero strain to 1e-12, density
  0.5/2.2 x 1e13 cm^-2 to a relative 1e-12, within 0.5e11 of the
  published 2.3e12 cm^-2. Interior spot: strain 0.134 % within 0.0005,
  density below 7.6e11 cm^-2 in size.
- The E' + A'1 strain arms equal -2 gamma omega / 100 from the cited
  Grueneisen parameters within 0.005; its strain error bar is more
  than 4 times that of the A'1 + 2LA(M) pair.
- Graphene: pure-strain and pure-doping trajectories decompose to zero
  doping and zero strain to 1e-12; the noise amplification equals its
  closed form to a relative 1e-12 plus 1e-8.

**Peak fitting and maps**

- Analytic derivatives match finite differences (Lorentzian: 1e-7
  plus a relative 1e-5; Voigt: 1e-6).
- A noiseless Lorentzian is recovered from automatic starting values
  (centre and width to 1e-8, height and offset to 1e-6); a noiseless
  Voigt to 1e-8.
- On 40 seeded noisy spectra, (centre error / reported sigma) has a
  standard deviation between 0.5 and 2.0 and a mean below 0.6 in size.
- The Voigt with zero Lorentzian width equals the Gaussian to 1e-14;
  as the Gaussian width shrinks tenfold (1e-3 to 1e-4), the gap to
  the Lorentzian shrinks by a factor of more than 9 (and is below 1e-5
  at width 1e-3).
- A sloped background biases a constant-baseline centre by more than
  3 of its sigmas; the linear baseline recovers the centre to 1e-6.
- `fit_map` gives exactly (bit for bit) the `fit_two_modes` result at
  the pixel checked; on a noiseless synthetic cube the maps invert back
  to the truth (strain to 1e-4, density to 1e-3); a dead pixel is
  masked and the other pixels' shifts are bit-identical to the clean
  run; with noise, every strain error is below 5 error bars + 1e-3.
- The CSV save/load round trips are exact (bit-identical arrays).
- Configuration errors in `fit_map` (unknown `baseline` name,
  reversed or overlapping windows, too few points in a window, a NaN
  in the wavenumber axis) are refused, and valid settings still give
  the `fit_two_modes` result bit for bit (new in 0.11.1).

**Multimode and map inversion**

- With two modes, `MultiModeModel` equals `SeparationModel` (maps to
  1e-12 plus a relative 1e-5, error bars to a relative 1e-12 plus
  1e-8).
- On 1600 seeded pixels the mean chi-square is within 0.1 of its
  degrees of freedom and the mean p-value within 0.05 of 0.5; a hidden
  extra shift gives median p-values below 1e-6, against above 0.1
  elsewhere.
- `bayesian_map_inversion` at `lam = 0` equals the per-pixel result
  (maps to 1e-12, error bars to 1e-13); a constant noiseless truth is
  recovered to 1e-12 at every `lam` tried (0, 1, 250); at `lam = 1e9`
  it matches an independently computed pooled estimate to 1e-5; error
  bars never grow as `lam` increases (checked at `lam` = 0, 0.5, 2
  and 10). A rank-deficient matrix is refused at `lam` = 0 and 1 (new
  in 0.11.1).

**Calibration and planning**

- Noise-free calibration recovers the lever arms to 1e-12, and agrees
  with an independent `numpy.linalg.lstsq` solve to 1e-10.
- Over 400 seeded calibrations the scatter of the fitted arms matches
  the reported error bars within 15 %.
- The planned covariance equals the one the calibration reports to
  1e-15; repeating a design 7 times divides it by 7 to a relative
  1e-12; on a test set of candidates where only one is off the line,
  the greedy design includes it, and it is never worse than any of 30
  random subsets of the same size.
- `separation_with_calibration`: with zero calibration covariance it
  equals the plain inversion to 1e-12 plus a relative 1e-5; its
  calibration part matches
  400 seeded lever-arm draws within 20 %; the derivative identity it
  uses matches finite differences to a relative 1e-5 plus 1e-10.

**Temperature**

- `ThreeCauseModel` recovers noise-free strain and density to 1e-9
  and temperature to 1e-8, agrees with an independent per-pixel
  least-squares solve to the same tolerances, and with a known
  temperature subtracted matches the two-cause solver to 1e-10.
- hc/kB agrees with SciPy's CODATA table to a relative 1e-9.
- The anti-Stokes thermometer round trip holds to a relative 1e-9,
  the calibration constant to 1e-12, and the error propagation matches
  a finite difference to a relative 1e-5.

## Corrections in earlier versions

**0.11.1 fixed two silent failures.**

- `fit_map` treated a mistake in its own settings (for example
  `baseline="Linear"`, or overlapping windows) like a bad pixel: it
  masked every pixel and returned an all-NaN result without an error.
  It now refuses such settings before fitting, as `fit_two_modes`
  does. Valid settings give the same results as before. `fit_map` now
  also refuses a wavenumber axis that contains NaN or infinity; before,
  spectral points with a NaN wavenumber were silently left out of the
  fits.
- `bayesian_map_inversion` did not check that the lever-arm matrix
  can separate the two causes. With a rank-deficient matrix it
  returned NaN maps (at `lam = 0`) or arbitrary finite maps (at
  `lam > 0`). It now refuses, as `MultiModeModel` does.

**0.8.0:** `bayesian_map_inversion` now refuses NaN shifts; before, it
solved around them silently.

**Documentation corrections in 0.11.1.** The README of 0.11.0 said the
tests ran on Python 3.9-3.14, but the CI matrix skipped 3.10 (added
now, with an oldest-dependencies job). It called every test "pinned to
an exact result" at "machine precision"; many use stated tolerances,
now listed above. It said the tests reproduce published separations
for all three cited sets; they do so for `mos2_a1_2la` only. The
calibration check described as "an independent QR solve" uses
`numpy.linalg.lstsq`. It cited the E' + A'1 pair itself to Michail et
al. (2024); that paper supplies the pair's Grueneisen parameters, and
the pair's historical use is Michail et al. (2016), as the docstring
says.

The full history is in [CHANGELOG.md](CHANGELOG.md).

## Limits, and what is deliberately not included

- The model is linear: each shift is a fixed lever arm times each
  cause. Where a material responds nonlinearly (graphene doping, the
  W-based doping behaviour above), no linear density coefficient is
  shipped.
- No constants beyond the three cited sets. Response coefficients
  depend on material, mode pair, laser wavelength and substrate; a
  measurement tool that ships unverified constants propagates wrong
  results. For any other system you provide the numbers -- from the
  literature or from `calibrate_lever_arms` -- and the mandatory
  `reference` field makes their origin travel with the analysis.
- The smoothing weights of the Bayesian inversion are user-chosen,
  because estimating them would need noise assumptions this package
  does not make. That inversion takes one error value per mode for the
  whole map, not per pixel.
- Peak-fit error bars are linearised estimates; they are reliable when
  the residuals are dominated by uncorrelated noise.
- `fit_two_modes` fits each peak in its own window; the tail of the
  other peak can shift a fitted centre slightly (the round-trip test
  allows 5e-3 cm^-1).
- `separation_with_calibration` is first-order in the calibration
  errors and covers two modes only.

## Where it comes from

> T. M. Mahim and M. M. Rahman, "Two Raman phonons quantify the fixed
> edge charge left by patterning monolayer transition metal
> dichalcogenides" (under review). Code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/Width-scaling-in-monolayer-semiconductor-nanoribbon-transistors

This package is the general-purpose, material-agnostic tool; the
paper repository reproduces the specific published study.

## Citing, support and license

If `ramansep` helps your work, please cite it with the concept DOI
[10.5281/zenodo.22014913](https://doi.org/10.5281/zenodo.22014913),
which always resolves to the latest version; every release is
archived on Zenodo. [CITATION.cff](CITATION.cff) has the details.

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

Licensed under Apache-2.0.
