"""Tutorial: the strain and charge profile across a ribbon edge, and the
extra charge per micrometer of edge, with a full error budget
(ramansep 0.15).

The steps:

1. Calibrate the A'1 and 2LA(M) lever arms of MoS2 on six reference
   states from a strain stage and a gate (`calibrate_lever_arms`).
2. Fit both peaks at every pixel of a map of a ribbon 24 pixels wide,
   20 nm per pixel (`fit_map`).
3. Average strain and density in bins of the distance from the nearest
   edge (`region_profile`). The bins share one calibration and one pair
   of reference frequencies, so their errors are correlated; the
   profile carries the covariance of all bins with one another.
4. Integrate the excess density over the distance from the edge
   (`RegionProfile.integrated_excess`): the extra electrons per unit
   length of edge. The reference-frequency error cancels in it, and a
   naive error bar that treats the bins as independent is shown for
   comparison.

All "measured" data are synthetic, made here with a fixed seed. The
true lever arms are those of `mos2_a1_2la()` (monolayer MoS2; see its
docstring for the sources). Everything else is illustrative: the
profiles (extra electrons 0.5e13 cm^-2 at the edge falling off over
30 nm, tension relaxed by 0.05 % at the edge over 40 nm), the pixel size,
the reference frequencies, peak widths and heights, and the noise.
Run time: about a second.

Output (ramansep 0.15.0):

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
"""
import numpy as np

import ramansep as rs

rng = np.random.default_rng(15)
K_true = np.array(rs.mos2_a1_2la().matrix())     # cited lever arms (truth)
ref = np.array([404.7, 452.0])                   # illustrative pristine peaks

# --- 1. Calibration -------------------------------------------------------
# Strain (%) from a stage and electron density (1e13 cm^-2) from a gate,
# measured from the pristine state; 0.1 cm^-1 shift errors.
cal_strain = np.array([0.0, 0.2, 0.4, 0.0, 0.2, 0.4])
cal_density = np.array([0.0, 0.0, 0.0, 0.6, 0.6, 1.2])
cal_shifts = np.column_stack([cal_strain, cal_density]) @ K_true.T
cal_shifts += rng.normal(0.0, 0.1, cal_shifts.shape)
cal = rs.calibrate_lever_arms(cal_strain, cal_density, cal_shifts, sigmas=0.1,
                              mode_names=["A'1", "2LA(M)"])
print(f"Calibration ({cal.n_points} reference states): "
      f"A'1 arms {cal.K[0, 0]:+.2f} +/- {cal.K_sigma[0, 0]:.2f}, "
      f"{cal.K[0, 1]:+.2f} +/- {cal.K_sigma[0, 1]:.2f}; "
      f"2LA(M) arms {cal.K[1, 0]:+.2f} +/- {cal.K_sigma[1, 0]:.2f}, "
      f"{cal.K[1, 1]:+.2f} +/- {cal.K_sigma[1, 1]:.2f}")

# --- 2. A measured map ----------------------------------------------------
# A ribbon 24 pixels (480 nm) wide; pixel centers lie 10, 30, ... nm from
# the nearer edge. Extra electrons and a relaxed tension near both edges.
H, W, pixel_nm = 12, 24, 20.0
col = np.arange(W)
distance = (np.minimum(col, W - 1 - col) + 0.5) * pixel_nm
distance = np.broadcast_to(distance, (H, W))
strain = 0.10 - 0.05 * np.exp(-distance / 40.0)
density = 0.03 + 0.5 * np.exp(-distance / 30.0)
dw_true = np.einsum("mk,khw->mhw", K_true, np.stack([strain, density]))
x = np.linspace(380.0, 480.0, 401)
cube = (rs.lorentzian(x, ref[0] + dw_true[0][..., None], 3.0, 300.0)
        + rs.lorentzian(x, ref[1] + dw_true[1][..., None], 4.0, 200.0)
        + 50.0 + rng.normal(0.0, 15.0, (H, W, x.size)))
fits = rs.fit_map(x, cube, window1=(395, 415), window2=(440, 465),
                  ref1=ref[0], ref2=ref[1])
print(f"Map: {H} x {W} pixels, {fits.n_masked} masked")

# --- 3. The profile -------------------------------------------------------
# The pristine reference frequencies are known to 0.2 and 0.3 cm^-1
# (illustrative): an error common to every pixel and every bin.
bins = [0, 20, 40, 60, 80, 100, 140, 240]              # nm from the edge
prof = rs.region_profile(cal, fits.dw1, fits.dw2, fits.sigma1, fits.sigma2,
                         distance, bins, ref_sigma=(0.2, 0.3))
print("Profile (distance from the nearest edge, nm):")
print("  bin   distance  pixels   strain (%)           density (1e13 cm^-2)   p")
for k in range(len(bins) - 1):
    print(f"  {f'{bins[k]}-{bins[k + 1]}':8s}{prof.position[k]:6.1f}"
          f"{prof.n_pixels[k]:8d}   {prof.strain[k]:+.4f} +/- "
          f"{prof.strain_sigma[k]:.4f}   {prof.density[k]:+.3f} +/- "
          f"{prof.density_sigma[k]:.3f}   {prof.p_value[k]:.2f}")
cov = prof.covariance
rho = cov[0, 1, -1, 1] / np.sqrt(cov[0, 1, 0, 1] * cov[-1, 1, -1, 1])
print(f"Correlation of the density errors of the first and the last bin: "
      f"{rho:+.3f}")

# --- 4. Extra charge per unit length of edge ------------------------------
# sum over the near-edge bins of width x (density - density of the last
# bin, the far interior); 1e13 cm^-2 times nm = 100 electrons per micrometer.
excess = prof.integrated_excess(bulk=-1)
widths = np.diff(bins)[:-1]
naive = np.sqrt(np.sum((widths * prof.density_sigma[:-1]) ** 2)
                + (widths.sum() * prof.density_sigma[-1]) ** 2)
# the truth, binned the same way from the true density map
true_mean = [density[(distance >= bins[k]) & (distance < bins[k + 1])].mean()
             for k in range(len(bins) - 1)]
true_excess = np.sum(widths * (np.array(true_mean[:-1]) - true_mean[-1]))
print("Excess density integrated over the distance (1e13 cm^-2 nm):")
print(f"  {excess.value:.2f} +/- {excess.sigma:.2f} (shifts "
      f"{excess.sigma_shifts:.2f}, calibration {excess.sigma_calibration:.2f}, "
      f"reference {excess.sigma_reference:.2f})")
print(f"  error bar if the bins were treated as independent: {naive:.2f}")
print(f"  = {excess.value * 100:.0f} +/- {excess.sigma * 100:.0f} extra electrons "
      f"per micrometer of edge (true: {true_excess * 100:.0f})")
