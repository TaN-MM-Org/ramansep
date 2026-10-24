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

- How do I read the text files my instrument exports, and get the
  peak shifts out of the measured spectra? Which pixels failed or
  carry no peak at all?
- What if two peaks overlap, so that their fitted positions are
  correlated? Are my error bars right when the noise grows with the
  counts (shot noise)?
- Which pair of peaks separates the two causes best?
- With three or more peaks, does "strain plus charge" actually explain
  what I measured, pixel by pixel?
- Can I trade some resolution for smaller error bars if the fields are
  smooth, and fill the pixels whose fits failed?
- How do I measure the response numbers on my own instrument, even
  when the starting strain and charge of my sample are unknown, and
  how should I plan that measurement?
- How much of my final error bar comes from that calibration, with two
  peaks or more?
- What if laser heating also shifts the peaks?
- What are the mean strain and charge of a region, and how much more
  charge sits at an edge than inside, with an error bar that counts
  every source of error?
- How do strain and charge change with the distance from an edge, and
  how many extra electrons sit along each micrometer of edge?

The results are checked by automated tests against independent
calculations and exact limits (see
[How the results are checked](#how-the-results-are-checked)). When an
input cannot give a trustworthy answer, the package stops with an error
that says why, rather than returning a number that looks fine.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install, requirements and units](#install-requirements-and-units)
- [Examples](#examples) (each with the output it prints)
- [Tutorials](#tutorials) (end-to-end scripts in `examples/`)
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
- **FWHM** -- full width at half maximum, the width of a peak measured
  halfway up its height.
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
The source distribution on PyPI also carries the tests and, since
0.15.0, the tutorial scripts, so the whole test suite runs from it.

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
printed with ramansep 0.15.0. Unless a value is said to come from a
cited source, it is an illustrative value chosen for the example. The
repository also has a longer demo, `examples/synthetic_map.py`, that
inverts a noisy synthetic 128 x 128 map and plots it with matplotlib,
and three end-to-end [tutorials](#tutorials).

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
error bar of zero. These match the paper's separations; the test
suite checks them (see below). `res.correlation` gives the
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
To read the plain text tables that instruments export, see example 16.
`fit_map` fits both peaks at every pixel. A pixel is masked (NaN in all
four output maps, `False` in `fits.ok`) when its counts are not finite,
a fit stops with an error or does not converge, a fitted centre ends
up outside its window, or a fitted peak height is zero or negative (a
dip is not a Raman peak; this check is new in 0.12.0). `fits.reason`
records, per pixel, why it was masked (`rs.MASK_REASONS` spells out
the codes). Maps with pixels that carry no peak at all need one more
check, `min_snr` (example 12). Mistakes in the settings themselves
(window limits, `baseline` name, a wavenumber axis containing NaN) are
refused with an error before any pixel is fitted.
The per-pixel inversion keeps masked pixels as NaN and does not touch
their neighbours. For spectra on a sloped background, pass
`baseline="linear"`; for peaks close enough to overlap, pass
`joint=True` (example 13).

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
pixels are refused by default because the prior links every pixel to
its neighbours; `fill_masked=True` fills them from their neighbours
instead, and `sigmas` may be given per pixel (example 14). Exact
per-pixel error bars (`posterior_sigma=True`) are refused above
`max_dense` = 4096 unknowns (two per pixel) unless you raise that
limit. Since 0.15.0 they no longer need a dense matrix inverse: with
one error bar per mode and no missing values they come from the
structure of the pixel grid, about 100 times faster at 4050 unknowns,
and otherwise from a sparse factorization, 2 to 4 times faster. The
answer also no longer depends on the units of the lever arms (for
example a density in cm^-2 instead of 1e13 cm^-2; before 0.15.0 that
changed the maps in the fourth digit).

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
This fit assumes the shifts are measured from the true pristine
frequencies and that strain and density are known in absolute terms.
If they are known only relative to the starting state of the sample,
use `fit_offset=True` (example 17).
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
(E_mj is the matrix with a single 1 at row m, column j). It takes
two-mode calibrations only; for three or more modes use
`multimode_with_calibration` (example 15). The calibration part comes
from one set of lever arms shared by every pixel, so it is an error
common to the whole map: unlike the spectral noise, it does not shrink
when you average many pixels.

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
per pixel. A pixel with a NaN shift or error bar (a pixel masked by
`fit_map`) comes out as NaN without affecting the others. Measuring the temperature with example 9 and comparing it
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

### 12. Pixels with no peak at all

```python
import warnings
import numpy as np
import ramansep as rs
from ramansep import lorentzian

# An illustrative 2 x 20 map: row 0 is off the flake (background and
# noise only, no Raman peak), row 1 carries both MoS2 peaks.
x = np.linspace(380.0, 480.0, 401)
rng = np.random.default_rng(6)
cube = np.empty((2, 20, x.size))
cube[0] = 50.0 + rng.normal(0.0, 3.0, (20, x.size))
cube[1] = (lorentzian(x, 404.7, 3.0, 40.0, 50.0) + lorentzian(x, 452.0, 4.0, 25.0, 0.0)
           + rng.normal(0.0, 3.0, (20, x.size)))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)   # overflow in runaway noise fits
    plain = rs.fit_map(x, cube, (395, 415), (440, 465), 404.7, 452.0)
    checked = rs.fit_map(x, cube, (395, 415), (440, 465), 404.7, 452.0, min_snr=3.0)

for name, f in (("without min_snr", plain), ("with min_snr=3 ", checked)):
    print(f"{name}: kept off-flake {f.ok[0].sum()} of 20, on-flake {f.ok[1].sum()} of 20")
for j in np.flatnonzero(plain.ok[0]):
    print(f"  off-flake pixel {j:2d} kept without min_snr: "
          f"dw1 {plain.dw1[0, j]:+.2f} +/- {plain.sigma1[0, j]:.2f} cm^-1")
for code in np.unique(checked.reason[0]):
    print(f"  off-flake reason {code}: {(checked.reason[0] == code).sum():2d} x {rs.MASK_REASONS[code]}")
```

```
without min_snr: kept off-flake 3 of 20, on-flake 20 of 20
with min_snr=3 : kept off-flake 0 of 20, on-flake 20 of 20
  off-flake pixel 11 kept without min_snr: dw1 +9.92 +/- 0.05 cm^-1
  off-flake pixel 15 kept without min_snr: dw1 -7.72 +/- 0.11 cm^-1
  off-flake pixel 17 kept without min_snr: dw1 -0.32 +/- 1.14 cm^-1
  off-flake reason 3: 13 x fit did not converge
  off-flake reason 4:  1 x fitted centre outside its window
  off-flake reason 5:  3 x fitted peak height not positive (a dip or no peak)
  off-flake reason 6:  3 x peak height below min_snr times its error bar
```

A fit to a spectrum that holds no peak (off the flake, over a hole)
sometimes converges on a noise spike and returns a shift that looks
like a measurement: pixel 11 above reports +9.92 +/- 0.05 cm^-1 from
pure noise. `min_snr=3` masks a pixel unless each fitted peak height is
at least 3 times its own error bar (a signal-to-noise test), and
removes all such pixels here while keeping every real one; the right
threshold for your data is your choice. `fwhm_range=(lo, hi)` adds a
check on the fitted widths. Both are off by default. Before 0.12.0,
peaks with a negative fitted height were kept too: on this map 6
off-flake pixels were kept instead of 3.

### 13. Two peaks that overlap

```python
import numpy as np
import ramansep as rs
from ramansep import lorentzian

# Two broad, overlapping peaks at known positions 385 and 404 cm^-1
# (illustrative widths), no noise: both shifts should come out as zero.
x = np.linspace(360.0, 430.0, 561)
y = lorentzian(x, 385.0, 5.0, 600.0, 20.0) + lorentzian(x, 404.0, 6.0, 1000.0, 0.0)

for joint in (False, True):
    dw1, dw2, s1, s2, f1, f2 = rs.fit_two_modes(x, y, (375, 394), (395, 415),
                                                ref1=385.0, ref2=404.0, joint=joint)
    print(f"joint={joint!s:5}: dw1 {dw1:+.4f}  dw2 {dw2:+.4f} cm^-1")
print(f"correlation of the two fitted centres: {f1.center_correlation:.3f}")
```

```
joint=False: dw1 +0.0795  dw2 -0.0279 cm^-1
joint=True : dw1 +0.0000  dw2 +0.0000 cm^-1
correlation of the two fitted centres: 0.014
```

Fitting each peak in its own window leaves the tail of the other peak
in that window, and the tail pulls the fitted centre: here by 0.08
cm^-1, comparable to the 0.1 cm^-1 shift error used in the examples
above, and as a systematic offset rather than random scatter. `joint=True` fits both peaks together, as two
Lorentzians on one shared baseline, over the whole range from the lower
edge of the first window to the upper edge of the second; that range
must not contain a third peak. The two fitted centres are then slightly
correlated (`center_correlation`); the inversion treats the two shifts
as independent, which is a good approximation when this number is
small. `fit_map(..., joint=True)` does the same at every pixel.

### 14. Smoothing a map that has gaps

```python
import numpy as np
import ramansep as rs

K = np.array([[-2.3, 1.1], [-0.9, -1.7], [0.4, 2.2]])   # illustrative, non-physical
rng = np.random.default_rng(7)
yy, xx = np.mgrid[0:12, 0:12]
strain = 0.002 * xx
density = 0.01 + 0.001 * yy

# One error bar per mode AND pixel (as fit_map returns them), and a
# 3 x 3 patch of pixels whose fits failed.
sig = np.array([0.03, 0.05, 0.02])[:, None, None] * rng.uniform(0.5, 2.0, (3, 12, 12))
shifts = np.einsum("mj,jhw->mhw", K, np.stack([strain, density])) \
    + rng.normal(size=(3, 12, 12)) * sig
shifts[:, 4:7, 4:7] = np.nan

r = rs.bayesian_map_inversion(K, shifts, sig, lam_strain=1000.0,
                              fill_masked=True, posterior_sigma=True)
hole = np.zeros((12, 12), bool)
hole[4:7, 4:7] = True
print(f"RMS strain error, measured pixels: {np.sqrt(np.mean((r.strain - strain)[~hole] ** 2)):.4f}")
print(f"RMS strain error, filled pixels  : {np.sqrt(np.mean((r.strain - strain)[hole] ** 2)):.4f}")
print(f"strain sigma: hole centre {r.strain_sigma[5, 5]:.4f}, "
      f"measured pixels (median) {np.median(r.strain_sigma[~hole]):.4f}")
print("filled centre = mean of its 4 neighbours:",
      np.isclose(r.strain[5, 5], (r.strain[4, 5] + r.strain[6, 5] + r.strain[5, 4] + r.strain[5, 6]) / 4))
```

```
RMS strain error, measured pixels: 0.0077
RMS strain error, filled pixels  : 0.0064
strain sigma: hole centre 0.0197, measured pixels (median) 0.0116
filled centre = mean of its 4 neighbours: True
```

`bayesian_map_inversion` accepts (since 0.12.0) one error bar per mode
and per pixel (an array of the same shape as `shifts`), such as the
`sigma1`, `sigma2` maps of `fit_map`. With `fill_masked=True`, a NaN
shift (or NaN error bar) counts as a missing measurement: it gets zero
weight, and the smoothness prior fills the gap from the neighbouring
pixels. A pixel with no data at all ends up exactly at the average of
its neighbours, and its error bar is larger than that of a measured
pixel. The filled values are interpolations, not measurements; they
were accurate here because the true fields are smooth by construction.
Filling needs both smoothness weights above zero.

### 15. Calibration uncertainty with three or more modes

```python
import numpy as np
import ramansep as rs

# Three modes calibrated together on six reference states (illustrative);
# the shifts are generated from K_true with 0.1 cm^-1 noise.
K_true = np.array([[-5.1, -2.2], [-20.9, 0.0], [-8.0, -1.1]])
eps = np.array([0.0, 0.4, 0.8, 0.0, 0.3, 0.6])
rho = np.array([0.0, 0.0, 0.2, 0.9, 0.6, 0.3])
rng = np.random.default_rng(4)
ref_shifts = np.column_stack([eps, rho]) @ K_true.T + rng.normal(0, 0.1, (6, 3))
cal = rs.calibrate_lever_arms(eps, rho, ref_shifts, sigmas=0.1)

out = rs.multimode_with_calibration(cal, shifts=[-2.0, -8.0, -3.4],
                                    sigmas=[0.15, 0.10, 0.12])
print(f"strain  {float(out['strain']):.4f} %")
print(f"  error from the spectra     {float(out['strain_sigma_shifts']):.4f}")
print(f"  error from the calibration {float(out['strain_sigma_calibration']):.4f}")
print(f"  total                      {float(out['strain_sigma']):.4f}")
print(f"density {float(out['density']):.4f} x 1e13 cm^-2 +/- {float(out['density_sigma']):.4f}")
print(f"p-value of the model check: {float(out['p_value']):.2f}")
```

```
strain  0.3821 %
  error from the spectra     0.0046
  error from the calibration 0.0016
  total                      0.0049
density 0.1036 x 1e13 cm^-2 +/- 0.0642
p-value of the model check: 0.06
```

`multimode_with_calibration` is the counterpart of example 8 for any
number of modes: the weighted least-squares separation of example 4,
with the calibration's own uncertainty carried into the error bars, to
first order. With more modes than causes the derivative of the answer
with respect to a lever arm also depends on the leftover mismatch
(residual) of the fit; the docstring gives the formula, and the tests
check it against a direct numerical derivative. The chi-square check
(`chi2_map`, `p_value`) is the one of `MultiModeModel`, which treats the
lever arms as exact.

### 16. Reading a map exported by the instrument

```python
import os, tempfile
import numpy as np
import ramansep as rs
from ramansep import lorentzian

# Write an illustrative 3 x 4 map the way mapping software often exports
# it: tab-separated "x y wavenumber counts", stage positions in
# micrometers, a serpentine scan, and the wavenumber running high to low.
rng = np.random.default_rng(2)
x_um = np.array([0.0, 0.5, 1.0, 1.5])
y_um = np.array([10.0, 10.5, 11.0])
wn = np.linspace(480.0, 380.0, 401)
path = os.path.join(tempfile.mkdtemp(), "map_export.txt")
with open(path, "w") as fh:
    fh.write("#X\t#Y\t#Wave\t#Intensity\n")
    for i, y in enumerate(y_um):
        for x in (x_um if i % 2 == 0 else x_um[::-1]):
            counts = (lorentzian(wn, 404.7 + 0.4 * x, 3.0, 800.0, 50.0)
                      + lorentzian(wn, 452.0, 4.0, 500.0, 0.0)
                      + rng.normal(0.0, 3.0, wn.size))
            for w, c in zip(wn, counts):
                fh.write(f"{x}\t{y}\t{w:.2f}\t{c:.1f}\n")

wavenumber, cube, xs, ys = rs.load_map_table(path)
print("cube:", cube.shape, " axis:", wavenumber[0], "to", wavenumber[-1], "cm^-1")
print("x (um):", xs, " y (um):", ys)
fits = rs.fit_map(wavenumber, cube, (395, 415), (440, 465), ref1=404.7, ref2=452.0)
print("A'1 shift at y = 10 um (cm^-1):", " ".join(f"{v:+.2f}" for v in fits.dw1[0]))

with open(path) as fh:                           # a scan that stopped early
    lines = fh.readlines()
with open(path, "w") as fh:
    fh.writelines(lines[:-401])
try:
    rs.load_map_table(path)
except ValueError as err:
    print("refused:", str(err).split(";")[0])
```

```
cube: (3, 4, 401)  axis: 380.0 to 480.0 cm^-1
x (um): [0.  0.5 1.  1.5]  y (um): [10.  10.5 11. ]
A'1 shift at y = 10 um (cm^-1): -0.00 +0.20 +0.40 +0.60
refused: the stage positions do not form a complete rectangular grid: 1 of the 3 x 4 combinations of distinct y and x values are missing (first: x = 1.5, y = 11.0)
```

Mapping software rarely writes the CSV layout of example 3. A common
plain-text export has four columns, the stage position x, the stage
position y, the wavenumber and the counts, with one line per pixel and
spectral point. `load_map_table` reads such a file in any line order
(raster or serpentine scans) and with the wavenumber running in either
direction. It rebuilds the pixel grid from the stage positions:
`cube[i, j]` is the spectrum at `x[j]`, `y[i]`, and both position
arrays increase. Columns may be separated by tabs, spaces or commas.
Lines starting with `#` and one text header line are skipped. If your
export has its columns in another order, say which with `usecols`.
A scan that is not a complete rectangle (here, one stopped before the
last pixel), a pixel with its own wavenumber axis, or a repeated point
is refused rather than padded or interpolated. Stage positions that
differ only in the last digits can be merged with `decimals=`.
`load_spectrum_table` reads a single two-column spectrum the same
way. Check that your export matches this layout before relying on it;
the package does not read vendor binary formats.

### 17. Calibrating when the starting state is unknown

```python
import numpy as np
import ramansep as rs

# A stage + gate calibration (illustrative). Strain (%) and electron
# density (1e13 cm^-2) are known only relative to the starting state, and
# the peak positions are absolute frequencies (cm^-1).
eps = np.array([0.0, 0.2, 0.4, 0.6, 0.0, 0.3, 0.1])
rho = np.array([0.0, 0.0, 0.0, 0.1, 0.8, 0.5, 1.1])
K_true = np.array([[-5.1, -2.2], [-20.9, 0.0]])
start = np.array([403.6, 451.1])        # the starting state: unknown in practice
rng = np.random.default_rng(2)
freq = start + np.column_stack([eps, rho]) @ K_true.T + rng.normal(0, 0.1, (7, 2))

np.set_printoptions(precision=2, suppress=True)
old = rs.calibrate_lever_arms(eps, rho, freq - [403.0, 452.0], sigmas=0.1)
print("through zero, shifts from 403.0 and 452.0 cm^-1:\n", old.K)
print("  chi2 / dof:", old.chi2 / old.chi2_dof)
new = rs.calibrate_lever_arms(eps, rho, freq, sigmas=0.1, fit_offset=True)
print("with a fitted offset per mode:\n", new.K)
print("  error bars:\n", new.K_sigma)
print("  chi2 / dof:", new.chi2 / new.chi2_dof)
print("  starting frequencies:", new.offset, "+/-", new.offset_sigma)
print("planned error bars:", rs.plan_calibration(eps, rho, 0.1, fit_offset=True)["K_sigma"])
```

```
through zero, shifts from 403.0 and 452.0 cm^-1:
 [[ -3.8   -1.67]
 [-22.62  -0.87]]
  chi2 / dof: [13.5  34.69]
with a fitted offset per mode:
 [[ -5.05  -2.23]
 [-20.57   0.03]]
  error bars:
 [[0.2 0.1]
 [0.2 0.1]]
  chi2 / dof: [0.95 1.35]
  starting frequencies: [403.63 450.98] +/- [0.08 0.08]
planned error bars: [0.2 0.1]
```

In a real calibration, the stage reports strain relative to where it
started and the gate reports charge relative to the starting voltage.
The starting strain and doping of the flake, and so the pristine
frequencies, are unknown. The fit of example 6 goes through zero and
needs them. Here the shifts were measured from guessed references
(403.0 and 452.0 cm^-1, both off), and the lever arms come out wrong
by up to 1.7 cm^-1 per percent. The fit itself says so: chi2 / dof is
far above 1. With `fit_offset=True` each mode gets one more free
parameter, a constant offset (the shift at the calibration's own
zero). The lever arms then depend only on how the shifts change
between the reference states, and they come out within two error
bars of the true values, with chi2 / dof near 1. The offset is reported with its error
bar, but it is not the pristine frequency, and the inversions do not
use it. The error bars of the lever arms already include the
uncertainty of the offset, so `separation_with_calibration` and
`multimode_with_calibration` take the result as it is. The price of
the extra parameter: the reference states must not lie on any one
straight line (not only a line through zero), and the fit needs one
more reference point. `plan_calibration` and `repeats_for_sigma` take
`fit_offset=True` too, so the error bars can be planned in advance.

### 18. Error bars when the noise grows with the counts

```python
import numpy as np
import ramansep as rs

# 300 repeats of one illustrative peak (404 cm^-1, FWHM 6 cm^-1, 400 counts
# high on a 30-count background) with shot noise: Poisson counts.
x = np.linspace(380.0, 430.0, 201)
clean = rs.lorentzian(x, 404.0, 6.0, 400.0, 30.0)
rng = np.random.default_rng(3)
spectra = rng.poisson(clean, (300, x.size)).astype(float)

centers, err_constant, err_varying = [], [], []
for y in spectra:
    fit = rs.fit_lorentzian(x, y)                       # default error bars
    centers.append(fit.center)
    err_constant.append(fit.center_sigma)
    err_varying.append(rs.fit_lorentzian(x, y, noise="varying").center_sigma)

print(f"actual scatter of the center: {np.std(centers, ddof=1):.4f} cm^-1")
print(f'error bar, noise="constant":  {np.median(err_constant):.4f} cm^-1')
print(f'error bar, noise="varying":   {np.median(err_varying):.4f} cm^-1')
```

```
actual scatter of the center: 0.0405 cm^-1
error bar, noise="constant":  0.0239 cm^-1
error bar, noise="varying":   0.0388 cm^-1
```

With shot noise, the noise at each spectral point grows with the
counts there, so the points on the peak are noisier than the points on
the background. The default error bars (`noise="constant"`) assume one
noise level for the whole window. That level is set mostly by the
quiet background, so the error bar of the center comes out too small:
here 0.024 cm^-1 against an actual scatter of 0.041 cm^-1.
`noise="varying"` (new in 0.14.0) computes the error bars without that
assumption, with the "sandwich" estimate of H. White, Econometrica 48,
817 (1980), and matches the scatter. The fitted values are the same
with either setting; only the error bars change. `fit_two_modes` and
`fit_map` take `noise` too. On spectra whose noise is the same
everywhere, the two settings agree on average (the tests check both
cases).

## Tutorials

Three scripts in `examples/` go from "measured" data to a final result
with error bars. The data are synthetic, made in each script with a
fixed seed. Each script runs in about a second and prints the output
shown here, which is also pasted at the top of the script. A test
checks that they still print exactly this.

### From a calibration and a map to the charge at an edge

`python examples/tutorial_edge_charge.py` calibrates the A'1 and
2LA(M) lever arms of MoS2 with a strain stage and a gate, starting from
an unknown state (`fit_offset=True`). It then fits a 10 x 14 map of a
ribbon whose outer two columns on each side carry extra electrons
(`fit_map` with `min_snr=3`; row 0 lies off the flake), and compares
the edge columns with the interior (`region_mean`,
`region_contrast`). The true lever arms are those of `mos2_a1_2la()`;
all other numbers are illustrative.

```
Calibration (8 reference states, offsets fitted):
  A'1    strain arm  -5.00 +/- 0.27   density arm  -2.10 +/- 0.09
  2LA(M) strain arm -21.06 +/- 0.27   density arm  -0.05 +/- 0.09
Map: 10 x 14 pixels, 14 masked (reasons: {3: 10, 4: 1, 5: 3})
Interior (90 pixels): strain +0.0992 +/- 0.0144 %, density +0.028 +/- 0.103 (1e13 cm^-2)
  error parts, strain : shifts 0.0001, calibration 0.0013, reference 0.0143
  error parts, density: shifts 0.0007, calibration 0.0131, reference 0.1018
  homogeneity: chi2 = 201.6 for 178 dof, p = 0.11
Edge (36 pixels): strain +0.0987 +/- 0.0144 %, density +0.270 +/- 0.103 (1e13 cm^-2)
Edge minus interior: strain -0.0005 +/- 0.0010 %, density +0.242 +/- 0.010 (1e13 cm^-2)
  error parts, density: shifts 0.0013, calibration 0.0102, reference 0 (cancels)
  = 2.42e+12 +/- 1.0e+11 electrons per cm^2 more at the edge (true: 2.30e+12)
```

The mean density of each region carries an error bar of 0.103 (in
1e13 cm^-2). Almost all of it comes from the uncertainty of the
pristine reference frequencies (0.2 and 0.3 cm^-1), which shifts
every pixel alike. In the difference, edge minus interior, that part
cancels exactly, and the calibration part shrinks with the size of
the difference. The extra charge at the edge comes out as
2.42e12 +/- 0.10e12 cm^-2 (true value 2.30e12), about ten times better
known than either mean. The homogeneity check finds no variation
inside the interior beyond the noise (p = 0.11).

### Two overlapping peaks and correlated errors

`python examples/tutorial_overlapping_peaks.py` uses the MoS2 E' and
A'1 peaks with the lever arms of `mos2_eprime_a1()`, made broad (FWHM
14 and 16 cm^-1, illustrative) so that they overlap.

```
True shifts: E' -0.497, A'1 -0.690 cm^-1
Separate windows: E' +1.427 +/- 0.133, A'1 -1.631 +/- 0.080 cm^-1
Joint fit:        E' -0.506 +/- 0.064, A'1 -0.650 +/- 0.055 cm^-1, correlation +0.41
Error bars of one spectrum, independent errors vs with the correlation:
  strain  0.0163 vs 0.0154 %
  density 0.0332 vs 0.0261 (1e13 cm^-2)
400 repeats: spread of the results vs the median error bars
  strain  spread 0.0173, error bar 0.0176 (independent) / 0.0167 (correlated)
  density spread 0.0284, error bar 0.0361 (independent) / 0.0287 (correlated)
```

Fitting each peak in its own window puts the tail of the other peak
into the fit and moves the shifts by 1.9 and 0.9 cm^-1. The joint fit
(`joint=True`) recovers them within their error bars and reports a
correlation of +0.41 between the two fitted centers. Over 400 repeats,
the density error bar that treats the two shifts as independent
(0.036) is 27 % larger than the actual spread (0.028). Passing the
correlation (`shift_correlation=`) gives 0.029, in agreement with the
spread.

### The profile across an edge, and the extra charge per micrometer of edge

`python examples/tutorial_edge_profile.py` (new in 0.15.0) calibrates
the A'1 and 2LA(M) lever arms of MoS2 on six reference states, fits a
12 x 24 map of a ribbon 480 nm wide (20 nm per pixel), and averages
strain and density in bins of the distance from the nearer edge
(`region_profile`). It then integrates the extra density over that
distance (`RegionProfile.integrated_excess`), which gives the extra
electrons per unit length of edge. The true lever arms are those of
`mos2_a1_2la()`; the profiles (0.5e13 cm^-2 extra electrons at the
edge, falling off over 30 nm; tension relaxed by 0.05 % at the edge,
over 40 nm) and all other numbers are illustrative.

```
Calibration (6 reference states): A'1 arms -4.86 +/- 0.21, -2.28 +/- 0.09; 2LA(M) arms -20.86 +/- 0.21, +0.05 +/- 0.09
Map: 12 x 24 pixels, 0 masked
Profile (distance from the nearest edge, nm):
  bin   distance  pixels   strain (%)           density (1e13 cm^-2)   p
  0-20      10.0      24   +0.0628 +/- 0.0144   +0.374 +/- 0.093   0.41
  20-40     30.0      24   +0.0774 +/- 0.0143   +0.214 +/- 0.093   0.72
  40-60     50.0      24   +0.0866 +/- 0.0143   +0.120 +/- 0.093   0.75
  60-80     70.0      24   +0.0919 +/- 0.0143   +0.082 +/- 0.093   0.87
  80-100    90.0      24   +0.0952 +/- 0.0143   +0.059 +/- 0.093   0.37
  100-140  120.0      48   +0.0969 +/- 0.0143   +0.052 +/- 0.093   0.34
  140-240  190.0     120   +0.1000 +/- 0.0143   +0.039 +/- 0.093   0.75
Correlation of the density errors of the first and the last bin: +0.985
Excess density integrated over the distance (1e13 cm^-2 nm):
  13.60 +/- 0.72 (shifts 0.27, calibration 0.67, reference 0.00)
  error bar if the bins were treated as independent: 14.13
  = 1360 +/- 72 extra electrons per micrometer of edge (true: 1440)
```

Every bin carries an error bar of about 0.093 (in 1e13 cm^-2). Almost
all of it (0.092) comes from the uncertainty of the reference
frequencies, which shifts all bins alike. That is why the errors of the
first and the last bin are correlated (+0.985), and why the profile has
to carry the covariance between its bins. The integrated excess is a
sum of differences, so the reference error cancels in it, and its
error bar is 0.72. Adding up the error bars of the bins as if they were
independent gives 14.13 instead, about twenty times too large. The
result, 1360 +/- 72 extra electrons per micrometer of edge, lies 1.1
error bars from the true value of the simulation (1440, from the true
density map averaged over the same bins); most of its
error comes from the calibration (0.67 of the 0.72). The homogeneity
check (`p`) finds no variation inside any bin beyond the noise.

## What is in the package

Every name below is exported from `ramansep` (its `__all__`). Each
docstring (`help(ramansep.fit_map)`, for example) gives inputs, units
and conventions.

**Two-mode inversion**

- `SeparationModel(coefficients, condition_warn=30.0)` -- inverts two
  shift maps (`invert(dw1, dw2, sigma1, sigma2, shift_correlation)`)
  into a `SeparationResult` with `strain`, `density`, `strain_sigma`,
  `density_sigma`, `correlation` and `condition_number`; `forward`
  predicts shifts. Error bars are computed only when both `sigma1` and
  `sigma2` are given. `shift_correlation` (new in 0.14.0) is the
  correlation of the two shift errors, as a joint fit reports it; the
  error bars then use the full 2 x 2 shift covariance (the GUM, JCGM
  100:2008, 5.2.2, Eq. (13)). Leaving it out treats the two errors as
  independent, as before.
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
  mode subsets by delivered variance (`subset_size` from 2 to the
  number of modes).
- `bayesian_map_inversion(K, shifts, sigmas, lam_strain, lam_density,
  posterior_sigma, max_dense, fill_masked)`, `BayesianMapResult` --
  joint inversion of a whole map with a smoothness prior (a Gaussian Markov random
  field, Rue and Held 2005: a standard statistical way of saying that
  each pixel is probably close to its four neighbors). Since 0.15.0
  its answer does not depend on the units of the lever arms, and its
  exact error bars (`posterior_sigma=True`) are much faster (example 5).

**From spectra to shifts**

- `lorentzian`, `fit_lorentzian`, `PeakFit` -- Lorentzian peak shape
  and a least-squares fit by the Levenberg-Marquardt method (a standard
  step-by-step curve-fitting method), using exact formulas for the
  derivatives; `baseline="constant"` or `"linear"`. Parameter errors
  are the standard linearized estimates. With the default
  `noise="constant"` they assume uncorrelated noise of the same size
  at every spectral point; `noise="varying"` (new in 0.14.0) drops the
  "same size" part, for shot-noise limited spectra (example 18).
- `voigt`, `fit_voigt`, `VoigtFit` -- the Voigt shape, computed
  exactly with the Faddeeva function (a standard special function,
  `scipy.special.wofz`), normalised to peak height.
- `fit_two_modes(x, y, window1, window2, ref1, ref2, baseline,
  joint=False, noise="constant")` -- fits both peaks, each in its own
  non-overlapping window, or with `joint=True` both together (example
  13), and returns `(dw1, dw2, sigma1, sigma2, fit1, fit2)`.
- `fit_map(..., joint=False, min_snr=None, fwhm_range=None,
  noise="constant")`, `MapFitResult`, `MASK_REASONS` -- the same at
  every pixel of a cube, with masking of failed pixels and a per-pixel
  reason code (example 12). Since 0.14.0 it fits all pixels together,
  5 to 7 times faster on a 1200-pixel map (3 to 4 times with
  `joint=True`), with the same numbers as before up to rounding (on the
  machine of 0.14.0 they were identical; another CPU may differ in the
  last digits, see [How the results are checked](#how-the-results-are-checked)), and returns
  `shift_correlation`, the correlation of the two shift errors at each
  pixel (0 for separate window fits, NaN where masked).
- `load_spectrum_csv`, `save_spectrum_csv`, `load_map_csv`,
  `save_map_csv` -- the documented CSV formats of example 3.
- `load_spectrum_table(path, usecols, delimiter)`,
  `load_map_table(path, usecols, delimiter, decimals)`: plain text
  tables as instruments export them, with stage positions for maps
  (example 16, new in 0.13.0).

**Calibration**

- `calibrate_lever_arms(strain, density, shifts, sigmas, mode_names,
  fit_offset=False)`, `CalibrationResult` (`K`, `K_sigma`, `cov`,
  `chi2`, `chi2_dof`, `n_points`, `condition_number`, `mode_names`,
  `coefficients(reference)` for two modes, and, with
  `fit_offset=True`, `offset`, `offset_sigma` and `full_cov`; example
  17, new in 0.13.0).
- `plan_calibration`, `design_references`, `repeats_for_sigma` --
  planning, example 7 (`plan_calibration` and `repeats_for_sigma` also
  take `fit_offset=True`).
- `separation_with_calibration` -- example 8 (it takes
  `shift_correlation` too, since 0.14.0);
  `multimode_with_calibration` -- the same for three or more modes,
  example 15.

**Regions of a map** (new in 0.14.0)

- `region_mean(coefficients, dw1, dw2, sigma1, sigma2, mask,
  shift_correlation=None, ref_sigma=None)`: the mean strain and
  density over a region, with the error split into three parts: shift
  noise (averages down over pixels), calibration (from a
  `CalibrationResult`; one lever-arm matrix serves every pixel, so it
  does not average down) and reference frequencies (`ref_sigma`, also
  common to all pixels). It also checks whether the pixels agree with
  one another within their error bars (`chi2`, `dof`, `p_value`) and
  gives their spread (`strain_scatter`, `density_scatter`).
- `region_contrast(coefficients, dw1, dw2, sigma1, sigma2, mask_a,
  mask_b, shift_correlation=None)`: region A minus region B, for
  example edge minus interior. The reference error cancels exactly, and
  the calibration part scales with the difference.
- `RegionEstimate`: the result of both. See the first
  [tutorial](#tutorials).
- `region_profile(coefficients, dw1, dw2, sigma1, sigma2, coordinate,
  bins, mask=None, shift_correlation=None, ref_sigma=None)` (new in
  0.15.0): the mean strain and density in bins of a coordinate, for
  example the distance from an edge, each bin with the error budget of
  `region_mean`, plus the covariance between bins
  (`RegionProfile.covariance`, split into its three parts). The bins
  share the lever arms and the reference frequencies, so their errors
  are correlated, and a sum over bins needs that covariance.
  `RegionProfile.combination(strain_weights, density_weights)` gives
  any weighted sum of the bins with its error parts (a
  `LinearEstimate`); `integrated_excess(bulk, quantity="density",
  widths=None)` gives the sum over the other bins of width times
  (value minus the value of the `bulk` bin): across a straight edge,
  the excess per unit length of edge, in which the reference error
  cancels; `contrast(i, j)` gives bin i minus bin j as
  `region_contrast` would. See the third [tutorial](#tutorials).

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
  review; data doi:10.5281/zenodo.21670315). A'1 doping response
  -2.2 cm^-1 per 1e13 cm^-2 of electrons, measured with a gate by
  Chakraborty et al., Phys. Rev. B 85, 161403(R) (2012). No
  measurement of the 2LA(M) doping response of MoS2 is known, so it is
  set to zero. In the source study, with its computed parameters, a
  response of up to half the frequency-scaled A'1 rate raises a
  recovered edge charge by 16%, and only a response above 3.7 times
  that rate reverses its sign. At the low end of the range the paper
  considers for the overtone's strain response, that threshold is 0.53
  times the rate; with all its stated input uncertainties, any response
  up to half the rate still gives an electron excess in 98% of its
  samples. The paper also notes that the computed strain arms are
  likely too large: on five monolayer WSe2 devices, a direct strain
  measurement (Michail et al., ACS Appl. Mater. Interfaces 16, 49602
  (2024)) gives an overtone-to-A'1 ratio of 1.8 where the calculation
  gives 4.8. Where 2LA(M) does not shift, the recovered strain is near
  zero for any non-zero overtone arm. Stated conditions: 1H monolayer
  (the common semiconducting crystal form), 532 nm laser, on an SiO2
  substrate, 300 K. The source paper itself applies the set outside
  these conditions, to tip-enhanced maps taken with 785 nm excitation
  on gold (Krayev et al.). The tests reproduce the source paper's
  separations from this set (example 1).
- `mos2_eprime_a1()`: the historically used all-optical E' + A'1 pair
  (Michail et al., Appl. Phys. Lett. 108, 173102 (2016)). Strain arms
  -4.31 (E') and -2.50 (A'1) cm^-1 per percent, from the measured
  biaxial Grueneisen parameters 0.56 and 0.31 of Michail et al., ACS
  Appl. Mater. Interfaces 16, 49602 (2024), converted with the mode
  frequencies 385.0 and 403.0 cm^-1. These are rounded monolayer
  values (see Mignuzzi et al., Phys. Rev. B 91, 195411 (2015)); Michail
  et al. 2016 place their strain-doping origin at 384.6 and
  402.7 cm^-1, which gives the same arms to the digits shown. Doping responses -0.33 (E') and
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

- a lever-arm matrix contains NaN or infinity; a two-mode lever-arm
  matrix is not 2 x 2 or is exactly singular (the two modes respond
  identically); a merely poorly conditioned one
  (condition number above `condition_warn`, default 30) gives a
  warning, not an error;
- a multimode or map lever-arm matrix has rank below 2 (below 3 for
  `ThreeCauseModel`, which also needs at least three modes), or the
  weighted problem is singular at some pixel. Since 0.15.0 the rank is
  judged without regard to the units of the columns; before, a density
  in m^-2 made a good matrix look like rank 1, and `compare_mode_sets`
  then left out every subset without saying so;
- the shift maps do not match in shape or number;
- a shift error (`sigmas`) is zero or negative in `MultiModeModel`,
  `ThreeCauseModel`, `bayesian_map_inversion`, the calibration and
  planning tools, is zero, negative, NaN or infinite in
  `compare_mode_sets` (which also refuses a `K` that is not (m, 2)),
  or negative in
  `SeparationModel` (zero is allowed there; NaN marks a masked pixel
  and is passed through as NaN);
- `bayesian_map_inversion` gets NaN or infinite shifts or error bars
  without `fill_masked=True`, missing values with a smoothness weight
  of zero, so little data that the unmasked shifts cannot fix both
  fields, per-pixel error bars of the wrong shape, a negative, NaN or
  infinite smoothness weight (NaN and infinity are refused since
  0.13.0), or asks for exact error bars above `max_dense` unknowns;
- `MultiModeModel` or `ThreeCauseModel` gets a number of error bars
  other than one per mode, or shift maps of different shapes (since
  0.13.0 the message says which);
- a peak fit gets too few points (5 for a Lorentzian with a constant
  baseline, 6 with a linear baseline or for a Voigt), non-finite data,
  an unknown `baseline` name, or a non-positive starting width;
- the two fit windows are reversed, overlap, or hold too few points,
  the `baseline` name is unknown, or a reference frequency `ref1`,
  `ref2` is NaN or infinite (`fit_two_modes`, and `fit_map` before
  fitting any pixel; the reference check is new in 0.13.0); `fit_map` also refuses a wavenumber axis
  that contains NaN or infinity, a `min_snr` that is not positive, or
  a `fwhm_range` that is not (lo, hi) with 0 <= lo < hi;
- a CSV file has the wrong header or field count, a wavenumber axis
  that is not strictly increasing or contains NaN or infinity, a pixel
  grid with holes, or a pixel with its own wavenumber axis;
- a text table (`load_spectrum_table`, `load_map_table`) has lines
  without the requested numeric columns, a NaN or infinite wavenumber
  or stage position, a wavenumber axis that repeats a value or is not
  monotonic, stage positions that do not form a complete rectangle, or
  a pixel with its own wavenumber axis or number of points;
- calibration reference states lie on one line through the origin of
  the (strain, density) plane (the two lever arms cannot be told
  apart; since 0.14.0 this is judged without regard to the units, so
  a density in cm^-2 is accepted), there are fewer than 2 of them, or fewer than 3 when no
  `sigmas` are given; with `fit_offset=True`, the states lie on any
  one straight line, or there are fewer than 4 of them without
  `sigmas`; `sigmas` whose shape fits neither the points nor the
  modes;
- `CalibrationResult.coefficients` is asked to package other than two
  modes, or `separation_with_calibration` gets such a calibration
  (`multimode_with_calibration` takes any number);
- `shift_correlation` lies outside [-1, 1] or is infinite, is given
  without both shift errors, or does not fit the map shape (new in
  0.14.0; NaN marks a masked pixel);
- `compare_mode_sets` gets a `subset_size` that is not a whole number
  from 2 to the number of modes (before 0.14.0 it returned an empty
  list);
- a peak fit gets a `noise` setting other than `"constant"` or
  `"varying"`;
- `region_mean` or `region_contrast` gets a region that is not a
  boolean map of the map's shape, a region without a usable pixel, a
  shift error that is not positive in a pixel it uses, two regions that
  share pixels, a `ref_sigma` that is not two finite, non-negative
  numbers, a lever-arm matrix that is singular, or a calibration with
  other than two modes; and a shift correlation of exactly +1 or -1,
  for which the homogeneity check is undefined;
- `region_profile` gets bins that are not at least two finite,
  strictly increasing edges, a coordinate map of another shape than the
  shift maps, or a bin without a usable pixel (choose wider bins), and
  everything `region_mean` refuses; `RegionProfile.integrated_excess`
  gets a `bulk` index that is not a bin, a `quantity` other than
  `"strain"` or `"density"`, or widths that are not one finite number
  per bin; `combination` gets weights that are not one finite number
  per bin; `contrast` gets the same bin twice or an index that is not a
  bin (all new in 0.15.0);
- the anti-Stokes ratio is at or above the calibration constant, or a
  temperature, frequency, ratio or constant is not positive.

## How the results are checked

148 automated tests run on every push and pull request, on Python 3.9,
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
  paper's 2.3e12 cm^-2. Interior spot: strain 0.134 % within 0.0005,
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
- `fit_map` gives the `fit_two_modes` result at the pixel checked
  (shifts to 1e-4 of their error bar, error bars to a relative 1e-5;
  exactly equal before 0.15.0, see "New in 0.15.0" below); on a
  noiseless synthetic cube the maps invert back to the truth (strain
  to 1e-4, density to 1e-3); a dead pixel is masked and the other
  pixels' shifts agree with the clean run to 1e-4 of their error bar;
  with noise, every strain error is below 5 error bars + 1e-3.
- The CSV save/load round trips are exact (bit-identical arrays).
- Configuration errors in `fit_map` (unknown `baseline` name,
  reversed or overlapping windows, too few points in a window, a NaN
  in the wavenumber axis) are refused, and valid settings still give
  the `fit_two_modes` result (new in 0.11.1; to the tolerances above
  since 0.15.0).

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
- With a NaN shift at one pixel and a NaN error bar at another,
  `ThreeCauseModel` returns NaN at exactly those two pixels and, at
  every other pixel, the results of the run without them (new in
  0.12.0; to a relative 1e-9 since 0.15.0, set from the condition
  number 1.3e5 of the systems).

**New in 0.12.0** (`tests/test_v012.py`, 19 tests)

- Joint two-peak fit: its exact derivatives match central finite
  differences to 1e-6 of each derivative's largest value (plus 1e-9),
  with both baselines. On a noiseless pair 19 cm^-1 apart (widths 5
  and 6 cm^-1) the separate-window fits are off by more than 0.05 and
  0.02 cm^-1; the joint fit recovers both centres and widths to 1e-8
  cm^-1 and heights and offset to 1e-6, also on a sloped background
  (slope to 1e-8) and with the windows given in reverse order. On 600
  seeded noisy spectra of a pair 12 cm^-1 apart, the scatter of each
  fitted centre is within 12 % of the reported error bar, the mean
  error is below 4 of its standard errors, and the measured
  correlation of the two centres is within 0.12 of the reported one
  (which is above 0.05). `fit_map(..., joint=True)` equals
  `fit_two_modes(..., joint=True)` at every pixel checked (to the
  tolerances of `fit_map` above since 0.15.0).
- `fit_map` checks: on 30 seeded peak-free spectra, without `min_snr`
  at least one pixel is kept and at least one is masked for a negative
  height; with `min_snr=3` none is kept. All 10 seeded spectra with
  real (weak) peaks are kept, with shifts and error bars bit-identical
  to the run without the checks. Constructed spectra produce the
  reason codes 0, 1, 5 and 7.
- `bayesian_map_inversion`: with per-pixel error bars and `lam = 0`
  it equals `MultiModeModel` given the same error bars (maps to 1e-12,
  error bars to 1e-13); per-pixel error bars that are constant per
  mode equal the scalar path (maps to 1e-12, error bars to 1e-13).
  With a pixel without data, a missing mode at another pixel and a
  missing error bar at a third, `fill_masked=True` agrees with an
  independent dense least-squares solve of the same objective
  (`numpy.linalg.lstsq`) to 1e-10. A filled pixel equals the average
  of its neighbours to 1e-12 (interior and corner pixel), with a larger
  error bar than its measured neighbours; a constant truth is recovered
  through a 2 x 3 hole to 1e-12 at `lam` = 0.5 and 250.
- `multimode_with_calibration`: with a two-mode calibration it equals
  `separation_with_calibration` (all values and error bars, relative
  1e-10 plus 1e-13); its derivative formula matches central finite
  differences of the `MultiModeModel` re-solve to a relative 1e-6 plus
  1e-10 at a point with a non-zero residual; its calibration part
  matches 400 seeded lever-arm draws within 20 %; with zero
  calibration covariance it gives the `MultiModeModel` error bar to a
  relative 1e-12; masked pixels stay NaN.
- New refusals fire: non-finite lever arms (`SeparationModel`,
  `MultiModeModel`, `compare_mode_sets`), a negative shift error in
  `SeparationModel` (NaN still passes as a masked pixel), a zero,
  negative, NaN or infinite sigma and a `K` that is not (m, 2) in
  `compare_mode_sets`.

**New in 0.13.0** (`tests/test_v013.py`, 18 tests)

- Each fix below has a test that fails on 0.12.0 and passes now.
- Text tables: files are written line by line from arrays built
  independently in the test (a serpentine scan with the wavenumber
  running high to low; a comma-separated file with a text header and
  the columns in another order). `load_map_table` returns those arrays
  exactly (bit for bit), and `fit_map` on the loaded cube equals
  `fit_map` on the original cube bit for bit. Each refusal fires.
  `load_spectrum_table` reverses a descending export exactly.
- `calibrate_lever_arms(..., fit_offset=True)`: noise-free arms and
  offsets are recovered to 1e-12. With per-point error bars, the arms
  and offsets agree with an independent `numpy.linalg.lstsq` solve to
  1e-10, and the covariance agrees to a relative 1e-10 with the closed
  form of the Frisch-Waugh-Lovell theorem (arms from the
  weighted-centered design; offset variance 1/sum(w) plus the
  propagated mean state). Without error bars the covariance equals the
  residual-variance formula with n - 3 degrees of freedom (relative
  1e-9). The through-origin fit of offset data is off by exactly the
  closed-form projection of the offsets (to 1e-12), more than 1 cm^-1
  per percent here; the offset fit removes it. Over 400 seeded
  calibrations the scatter of the arms and offsets matches the
  reported error bars within 15 %, and the mean chi-square is within
  0.4 of its 4 degrees of freedom. The calibration part of
  `separation_with_calibration` matches 400 seeded offset calibrations
  within 20 %. `plan_calibration(..., fit_offset=True)` equals the
  reported covariance to 1e-15, and repeating the design r times
  divides it by r (relative 1e-12).

**New in 0.14.0** (`tests/test_v014.py`, 22 tests)

- Each fix below has a test that fails on 0.13.1 and passes now.
  Calibration with the density in cm^-2 gives the same lever arms as
  in 1e13 cm^-2, rescaled (relative 1e-10, with and without the
  offset fit); planning gives the same error bars (relative 1e-10)
  and `design_references` picks the same states; a truly collinear
  design is still refused. `bayesian_map_inversion(...,
  fill_masked=True)` with the density in cm^-2 agrees with the run in
  1e13 cm^-2 to a relative 1e-5 (3e-7 in fact: a sparse solve with
  columns 1e13 apart in scale loses some digits).
- `fit_map` equals the per-pixel loop over `fit_two_modes` of 0.13.1
  (kept in the test as the reference): all four maps, `ok` and the
  reason codes, on a noisy map with peak-free pixels and a broken
  pixel, with both baselines, with and without `joint=True`, `min_snr`
  and `fwhm_range`. The result does not depend on how many pixels are
  fitted together, and a spectrum that `fit_lorentzian` refuses is
  masked with reason 2, as before. (Since 0.15.0 this is checked to
  stated tolerances, and not on the peak-free pixels; see "New in
  0.15.0".)
- `shift_correlation`: the error bars and the correlation equal
  K^-1 S K^-T computed independently with the full shift covariance S
  (relative 1e-12); zero correlation gives the old result bit for bit.
  On 400 seeded noisy repeats of two overlapping peaks fitted jointly,
  the measured correlation of the two shifts is within 0.12 of the
  reported one (about 0.4), and the spread of the inverted density is
  within 12 % of the error bar with the correlation, while the error
  bar without it is more than 15 % too large.
- `noise="varying"`: on 600 seeded shot-noise (Poisson) spectra of one
  peak, the scatter of the fitted center is between 1.5 and 1.9 times
  the default error bar and within 10 % of the `"varying"` one, with
  identical fitted values. On 600-pixel maps of two overlapping peaks
  fitted jointly, the same holds for both shifts (default more than
  1.25 times too small); with noise of the same size everywhere both
  settings are within 10 % of the scatter. The error bars equal the
  sandwich formula built from an independent finite-difference
  Jacobian (relative 1e-5), and `fit_map` equals `fit_two_modes` with
  this setting too (to the tolerances of `fit_map` above since 0.15.0).
- `region_mean`: the mean equals the mean of the per-pixel
  `SeparationModel.invert` maps (1e-14); the shift part equals the sum
  of the per-pixel covariances over N^2 (relative 1e-12); the
  reference part equals the error bar of one pixel with shift errors
  (u1, u2) (relative 1e-12); the calibration part equals that of
  `separation_with_calibration` at the mean shifts (relative 1e-10),
  and the derivative behind it matches finite differences (relative
  1e-6). The homogeneity chi-square equals an independent computation
  in (strain, density) space (relative 1e-9); over 400 seeded uniform
  regions its mean is within 3 standard errors of the degrees of
  freedom and the share of p-values below 0.05 is within 0.035 of
  0.05; a region with a real gradient gives p below 1e-6.
- `region_contrast`: equals the difference of the two means (1e-13);
  the shift parts add in quadrature (relative 1e-12); a common offset
  added to all shifts leaves the contrast unchanged (1e-12) while each
  mean moves by K^-1 times it; the calibration part equals that of
  `separation_with_calibration` at the difference of the mean shifts
  (relative 1e-10).
- Over 400 seeded trials that repeat the calibration measurement and
  fit, the map noise and the reference error, the spread of the region
  mean and of the contrast is within 12 % of the reported total error
  bars, the contrast is unbiased (within 4 standard errors) and its
  density error bar is less than half that of the mean.
- Both tutorials run and print exactly the output shown above and at
  the top of each script.
- Refusals fire for each case listed under
  [When it refuses](#when-it-refuses-and-why) that is new in 0.14.0.

**New in 0.15.0** (`tests/test_v015.py`, 11 tests, and changed tests)

- Each fix below has a test that fails on 0.14.0 and passes now.
  `bayesian_map_inversion` with the density in cm^-2, in m^-2 or in
  units of 1e19 cm^-2 instead of 1e13 cm^-2 (the density column of K
  divided by 1e13 or 1e17, or multiplied by 1e6) gives the maps and
  error bars of an independent dense solve of the well-scaled
  problem to 100 eps cond(A) relative (7.6e-13 for the test map;
  0.14.0 was off by 6.1e-4 in cm^-2, refused m^-2 and was off by 8e-13
  in units of 1e19 cm^-2). Matrices whose columns are within 2^10 in scale take the
  0.14.0 code path, and equal the dense solve to the same tolerance.
  `MultiModeModel`, `ThreeCauseModel`, `compare_mode_sets` and
  `bayesian_map_inversion` accept a density in m^-2 and give the
  results in 1e13 cm^-2, rescaled (relative 1e-10, and 1e-8 for the
  three-cause model, whose systems have a condition number of 3.1e6 in
  mixed units); a truly rank-deficient matrix is still refused
  in any unit.
- Posterior sigmas: the new computations equal the 0.14.0 dense
  inverse, built independently in the test, to 100 eps cond(A)
  relative, on maps with one row, one column, unequal smoothing
  weights, `lam = 0` and a nearly collinear K (one error bar per mode),
  and with per-pixel error bars and a missing value (sparse
  factorization; also on random sparse matrices, with block sizes that
  do not divide the size).
- `region_profile`: each bin equals the mean of the per-pixel
  `SeparationModel.invert` maps over its pixels (1e-14), its shift
  covariance the sum of the per-pixel covariances over N^2 (relative
  1e-12), and its error parts and homogeneity check those of
  `region_mean` on that bin (relative 1e-12). The calibration
  covariance between all pairs of bins equals the one built from a
  finite-difference Jacobian of an independent re-solve with perturbed
  lever arms (relative 1e-6); the reference covariance is
  K^-1 diag(u1^2, u2^2) K^-T for every pair (relative 1e-12), and an
  offset added to all shifts of one mode moves every bin by K^-1 times
  it. A combination with random weights equals c^T C c from the stored
  covariance (relative 1e-10). The integrated excess equals the sum
  by hand (relative 1e-13); its reference part is zero to 1e-12 of a
  bin's; its shift part equals a pixel-level linear form (relative
  1e-10) and its calibration part finite differences of a re-solve
  (relative 1e-6). `contrast(i, j)` equals `region_contrast` on the
  two bins' pixels (relative 1e-12).
- Over 300 seeded trials that repeat the calibration measurement and
  fit, the map noise and the reference error, the spread of the
  integrated excess and of a bin mean is within 15 % of the reported
  error bars (measured: 3 % and 4 %), the excess is unbiased (within 4
  standard errors), and the measured correlation of two bins' density
  errors is within 0.1 of the reported one (0.988 against 0.987).
- The third tutorial runs and prints exactly the output shown above
  and at the top of the script.
- Refusals fire for each case listed under
  [When it refuses](#when-it-refuses-and-why) that is new in 0.15.0.
- Changed tests: the comparisons of the stacked `fit_map` with the
  single-spectrum fits (and of `ThreeCauseModel` runs on stacks of
  different length) demanded exact equality. They passed on the
  machine of 0.14.0, but the stacked and the single matrix products
  are different NumPy/BLAS calls, which another CPU may round
  differently in the last digit. They now use tolerances derived from
  the stopping rule of the fit: a converged fit stops once a step
  lowers the cost by less than a relative 1e-12, so a stop one step
  earlier or later moves a parameter by at most about
  1e-6 sqrt(n - p) of its error bar (n points, p parameters); shifts
  are compared to 1e-4 of their error bar, error bars to a relative
  1e-5. Pixels without a peak are no longer compared: a fit to pure
  noise can run away (to centers of 1e20 to 1e76 cm^-1), and changing
  the counts of the test map by one unit in the last place already
  flips the mask reason of 1 or 2 of its 18 peak-free pixels between
  3 ("did not converge") and 4 ("centre outside its window"). In 15
  such trials on that map (with `min_snr=2`) these pixels stayed
  masked every time; only the reason code changed.

## Corrections in earlier versions

**0.15.0 fixed two unit dependences, and corrected one documented
claim** (before and after values from running 0.14.0 and 0.15.0 on the
same input).

- `bayesian_map_inversion` gave maps that depended on the units of the
  lever arms. With the density in cm^-2 (lever arms near 1e-13 cm^-1
  per cm^-2) instead of 1e13 cm^-2, its sparse solve lost about four
  digits: on the test map the density differed from an independent
  dense solve by a relative 6.1e-4, and the strain by 1.4e-4 (less than
  the error bars, but a result should not depend on the unit). It now
  solves the same problem in a unit that differs by an exact power of
  two and scales the answer back: the difference is now about 1e-15.
  Lever arms whose columns are within a factor 2^10 in scale, such as
  the shipped sets and those of the README examples, take the code path
  of 0.14.0 and give the same maps; their posterior error bars, now
  computed faster, agree with 0.14.0 to a relative 1e-15.
- `MultiModeModel`, `ThreeCauseModel` and `bayesian_map_inversion`
  refused a lever-arm matrix with the density in m^-2 (the SI unit;
  arms near 1e-17 cm^-1 per m^-2) as "rank < 2" (or "rank < 3"), and
  `compare_mode_sets` returned an empty list for it. The rank is now
  judged on the matrix with unit-length columns, as `calibrate_lever_arms`
  has done since 0.14.0; the results equal those in 1e13 cm^-2,
  rescaled, to a relative 1e-14 (measured on the test inputs).
- The documentation said that `fit_map` gives "bit for bit" the
  numbers of the per-pixel loop over `fit_two_modes`, and the tests
  demanded that. It held on the machine where 0.14.0 was made, but it
  is not guaranteed on another CPU: the stacked and the single fits
  use different NumPy/BLAS calls. The text now says "up to rounding",
  and the tests use tolerances derived from the fit's stopping rule
  (see "New in 0.15.0" above). On pixels without a peak, a fit to pure
  noise can follow a chaotic path, so the reason code of such a masked
  pixel (3 or 4) can differ between machines.

**0.14.0 fixed two wrong refusals and one silent empty result, and
corrected one documented claim** (before and after values from running 0.13.1 and 0.14.0 on the
same input).

- `calibrate_lever_arms`, `plan_calibration` and `design_references`
  decided whether the reference states span the (strain, density)
  plane from the raw singular values of the design. That test depends
  on the units: with strain in percent and density in cm^-2 (values
  near 1e12 to 1e13), a well-spread set of six reference states was
  refused as "collinear" (`plan_calibration` returned
  `identifiable=False`). The test is now made on the design with each
  column scaled to unit length, which does not depend on the units.
  The same data now give the lever arms of the fit in 1e13 cm^-2,
  rescaled, to a relative 1e-10. Designs accepted before give the
  same numbers as before.
- `bayesian_map_inversion(..., fill_masked=True)` refused the same
  kind of input (a lever-arm matrix whose density column is 1e13
  times smaller) with "pooled information matrix is singular". The
  check now uses the unit-free form of that matrix.
- `compare_mode_sets` returned an empty list, without saying why, for
  a `subset_size` of 0, 1 or more than the number of modes. It now
  refuses such a value.
- The documentation said the peak-fit error bars are reliable when the
  residuals are dominated by uncorrelated noise. Shot noise is
  uncorrelated, yet with it the error bars are too small, because the
  noise is larger on the peak than on the background: by a factor of
  1.7 for the peak of example 18. The text now says that the default
  assumes the same noise level at every point, and `noise="varying"`
  gives error bars that hold for shot noise.

**0.13.0 fixed three silent failures** (before and after values from
running 0.12.0 and 0.13.0 on the same input).

- `load_spectrum_csv` returned a wavenumber axis containing NaN or
  infinity, although it promises to refuse an axis that is not
  strictly increasing: a file with the axis 400, nan, 402, 403 loaded
  as `[400, nan, 402, 403]`. It is now refused. (`load_map_csv`
  refused such an axis already, but with the message that pixel
  (0, 0) "carries a different wavenumber axis than pixel (0, 0)"; the
  message now says what is wrong.)
- `fit_map` with a NaN reference frequency reported every pixel as
  fitted (`ok` True at all pixels, `n_masked` 0) while every shift was
  NaN, and `fit_two_modes` returned a NaN shift. Both now refuse a
  reference that is not a finite number.
- `bayesian_map_inversion` with a NaN or infinite smoothness weight
  returned all-NaN maps with only SciPy's "Matrix is exactly singular"
  warning. It now refuses such a weight and suggests a large finite
  one for the fully pooled limit.

**Documentation corrections in 0.13.0.** The docstring of
`MultiModeResult` said that `condition_number` is that of the weighted
design matrix when the error bars are scalars; the code has always
returned the condition number of the unweighted lever-arm matrix (the
value is unchanged; the docstring now says so). The docstring of
`fit_voigt` said its default start splits the width evenly between the
Gaussian and Lorentzian parts; it starts with a Gaussian FWHM equal to
the estimated width and a Lorentzian FWHM of half of it. The
`fitting` module docstring said the tests recover amplitude and
baseline to 1e-8; they check 1e-6. `SeparationResult.density` was
documented as cm^-2 by default; it is in the unit of the lever arms
(1e13 cm^-2 for the MoS2 sets).

**0.12.0 fixed three silent or misleading behaviours.**

- `fit_map` kept pixels whose fitted "peak" had a negative height (a
  dip fitted to noise). Such pixels are now masked (reason code 5).
  Good pixels give the same numbers as before; on the map of example
  12, the off-flake pixels kept went from 6 of 20 to 3 of 20 (and to 0
  with `min_snr=3`).
- `ThreeCauseModel.invert` refused a whole map, with a misleading
  "singular" message, when one pixel carried a NaN error bar, which is
  how `fit_map` marks masked pixels. Such pixels now come out as NaN,
  as in `MultiModeModel`.
- `SeparationModel.invert` and `compare_mode_sets` accepted negative
  shift errors and squared the sign away (for example `sigma1=-0.1`,
  `sigma2=0.1` with `mos2_a1_2la()` gave a strain error bar of
  0.0048 %). Both now refuse them. `compare_mode_sets` also returned
  NaN variances, ranked first, for a zero or NaN sigma, and silently
  left out every subset with an infinite lever arm; it now refuses
  those inputs too.

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
  does not make. Pixels it fills (`fill_masked=True`) are
  interpolations from their neighbours, not measurements.
- Peak-fit error bars are linearized estimates. The default assumes
  uncorrelated noise of the same size at every spectral point; with
  shot noise it gives error bars that are too small (example 18), and
  `noise="varying"` should be used. Neither setting covers noise that
  is correlated between neighboring points (for example after
  smoothing or interpolation of the spectra). `fit_voigt` has the
  default error bars only.
- By default `fit_two_modes` fits each peak in its own window, and the
  tail of the other peak can shift a fitted center (by 0.08 cm^-1 in
  example 13; the round-trip test allows 5e-3 cm^-1). `joint=True`
  removes this, provided no third peak lies anywhere from the lower
  edge of the first window to the upper edge of the second; it fits
  Lorentzians only (no joint Voigt fit). The two fitted centers are
  then correlated; pass that correlation to the two-mode inversion with
  `shift_correlation` (new in 0.14.0). `MultiModeModel`,
  `ThreeCauseModel` and `bayesian_map_inversion` still treat the shift
  errors of different modes as independent.
- `min_snr` and `fwhm_range` in `fit_map` are off by default and their
  values are your choice; the package does not decide what counts as
  "no peak" for your data.
- `load_map_table` groups stage positions by exact value (or after
  rounding with `decimals`) and needs one shared wavenumber axis; it
  does not resample spectra onto a common axis, and it does not read
  vendor binary formats or decimal commas. `design_references` plans
  for the through-origin calibration only; check a set chosen for
  `fit_offset=True` with `plan_calibration(..., fit_offset=True)`.
- `separation_with_calibration` and `multimode_with_calibration` are
  first-order in the calibration errors. The calibration part is
  common to all pixels, so it does not average down over a map. The
  chi-square check of `multimode_with_calibration` ignores the
  calibration uncertainty.
- `region_mean` and `region_contrast` work with two modes. They treat
  the shift noise of different pixels as independent, which holds for
  separately measured spectra but not after spatial smoothing, and not
  for the output of `bayesian_map_inversion`. The mean is the plain
  average over the pixels used. When the homogeneity check fails, the
  spread between pixels is real variation: the error bar then
  describes the average of those pixels, not a typical pixel.
- `region_profile` has the same limits: two modes, independent pixels,
  plain averages, and a calibration part that is first order. Reading
  `integrated_excess` as the excess per unit length of an edge assumes
  a straight edge, a coordinate that is the distance from it, and bins
  that cover the whole strip next to the edge; the bin mean stands for
  the whole bin, so bins should be narrow where the profile changes
  fast. The bulk bin must really be far from the edge: whatever excess
  it still holds is subtracted from every other bin.
- The A-optimality ranking of `compare_mode_sets` (the sum of the
  strain and the density variance) depends on the units, as any sum of
  quantities in different units does; with the density in cm^-2 the
  density variance decides alone. The D-optimality value ranks the
  subsets alike in any unit.
- `fit_map` on pixels without a peak: a fit to pure noise can run away
  along a path that a change in the last digit sends elsewhere. Such
  pixels are nearly always masked, but whether the reason code says 3
  ("did not converge") or 4 ("fitted centre outside its window") can
  differ between machines, and so can, without `min_snr`, whether a
  rare one is kept (`min_snr` is the remedy, example 12).

## Where it comes from

> T. M. Mahim and M. M. Rahman, "Two Raman peaks and transistor
> thresholds quantify the fixed negative charge at etched edges of
> monolayer MoS2" (under review). Code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/Width-scaling-in-monolayer-semiconductor-nanoribbon-transistors
> Data: [10.5281/zenodo.21670315](https://doi.org/10.5281/zenodo.21670315)
> (concept DOI, resolves to the latest version)

This package is the general-purpose, material-agnostic tool; the
paper repository reproduces that specific study.

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
