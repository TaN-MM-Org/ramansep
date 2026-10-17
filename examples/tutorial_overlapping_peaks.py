"""Tutorial: two broad, overlapping peaks. Fit them together, carry
the correlation of their fitted centers into the strain and density
error bars, and check those error bars by repeating the measurement
(ramansep 0.14).

The pair is monolayer MoS2 E' + A'1 with the cited lever arms of
`mos2_eprime_a1()` (see its docstring for the sources). The peaks are
made broad (FWHM 14 and 16 cm^-1, illustrative) so that they overlap;
peak heights, baseline and noise are illustrative too. Every
"measurement" is synthetic, made here with a fixed seed, with noise of
the same size at every point (for shot noise, which grows with the
counts, pass noise="varying" to the fits). Run time: about a second.

Steps:

1. One measured spectrum: fitting each peak in its own window leaves a
   bias from the other peak's tail; `joint=True` removes it and reports
   the correlation of the two centers.
2. Invert with and without that correlation
   (`SeparationModel.invert(..., shift_correlation=...)`).
3. Repeat the measurement 400 times (a 20 x 20 "map" of the same
   spot, fitted with `fit_map(..., joint=True)`) and compare the
   actual spread of the results with the error bars.

Output (ramansep 0.14.0):

True shifts: E' -0.497, A'1 -0.690 cm^-1
Separate windows: E' +1.427 +/- 0.133, A'1 -1.631 +/- 0.080 cm^-1
Joint fit:        E' -0.506 +/- 0.064, A'1 -0.650 +/- 0.055 cm^-1, correlation +0.41
Error bars of one spectrum, independent errors vs with the correlation:
  strain  0.0163 vs 0.0154 %
  density 0.0332 vs 0.0261 (1e13 cm^-2)
400 repeats: spread of the results vs the median error bars
  strain  spread 0.0173, error bar 0.0176 (independent) / 0.0167 (correlated)
  density spread 0.0284, error bar 0.0361 (independent) / 0.0287 (correlated)
"""
import numpy as np

import ramansep as rs

model = rs.SeparationModel(rs.mos2_eprime_a1())  # cited E' + A'1 lever arms
ref = (384.6, 402.7)                             # E', A'1 (cm^-1); see docstring
true_strain, true_density = 0.10, 0.20           # %, 1e13 cm^-2 (illustrative)
dw_true = model.forward(true_strain, true_density)
x = np.linspace(350.0, 440.0, 181)
clean = (rs.lorentzian(x, ref[0] + dw_true[0], 14.0, 400.0)
         + rs.lorentzian(x, ref[1] + dw_true[1], 16.0, 500.0) + 30.0)
rng = np.random.default_rng(7)
win1, win2 = (360.0, 393.5), (394.0, 430.0)

# --- 1. one spectrum ------------------------------------------------------
y = clean + rng.normal(0.0, 10.0, x.size)
sep = rs.fit_two_modes(x, y, win1, win2, *ref)
jnt = rs.fit_two_modes(x, y, win1, win2, *ref, joint=True)
rho = jnt[4].center_correlation
print(f"True shifts: E' {float(dw_true[0]):+.3f}, A'1 {float(dw_true[1]):+.3f} cm^-1")
print(f"Separate windows: E' {sep[0]:+.3f} +/- {sep[2]:.3f}, "
      f"A'1 {sep[1]:+.3f} +/- {sep[3]:.3f} cm^-1")
print(f"Joint fit:        E' {jnt[0]:+.3f} +/- {jnt[2]:.3f}, "
      f"A'1 {jnt[1]:+.3f} +/- {jnt[3]:.3f} cm^-1, correlation {rho:+.2f}")

# --- 2. inversion with and without the correlation -------------------------
ind = model.invert(*jnt[:4])
cor = model.invert(*jnt[:4], shift_correlation=rho)
print("Error bars of one spectrum, independent errors vs with the correlation:")
print(f"  strain  {float(ind.strain_sigma):.4f} vs {float(cor.strain_sigma):.4f} %")
print(f"  density {float(ind.density_sigma):.4f} vs {float(cor.density_sigma):.4f} (1e13 cm^-2)")

# --- 3. check by repetition ------------------------------------------------
cube = clean + rng.normal(0.0, 10.0, (20, 20, x.size))
fits = rs.fit_map(x, cube, win1, win2, *ref, joint=True)
ok = fits.ok
a = model.invert(fits.dw1[ok], fits.dw2[ok], fits.sigma1[ok], fits.sigma2[ok])
b = model.invert(fits.dw1[ok], fits.dw2[ok], fits.sigma1[ok], fits.sigma2[ok],
                 shift_correlation=fits.shift_correlation[ok])
print(f"{int(ok.sum())} repeats: spread of the results vs the median error bars")
print(f"  strain  spread {np.std(a.strain, ddof=1):.4f}, error bar "
      f"{np.median(a.strain_sigma):.4f} (independent) / "
      f"{np.median(b.strain_sigma):.4f} (correlated)")
print(f"  density spread {np.std(a.density, ddof=1):.4f}, error bar "
      f"{np.median(a.density_sigma):.4f} (independent) / "
      f"{np.median(b.density_sigma):.4f} (correlated)")
