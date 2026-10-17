"""Tutorial: from a calibration and a measured map to the charge at a
ribbon edge, with a full error budget (ramansep 0.14).

The steps a lab goes through, end to end:

1. Calibrate the lever arms on your own instrument from a strain stage
   and a gate, when the starting strain and charge of the flake are not
   known (`calibrate_lever_arms(..., fit_offset=True)`).
2. Fit both peaks at every pixel of a map, masking pixels that carry no
   peak (`fit_map(..., min_snr=3)`).
3. Average strain and charge over the interior and over the edges, and
   take the difference edge minus interior, each with its error split
   into shift noise, calibration and reference parts (`region_mean`,
   `region_contrast`).

All "measured" data are synthetic, made here with a fixed seed. The
true lever arms are those of `mos2_a1_2la()` (monolayer MoS2, A'1 and
2LA(M); see its docstring for the sources); everything else (reference
frequencies, peak widths and heights, noise, the map itself) is
illustrative. Run time: about a second.

Output (ramansep 0.14.0):

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
"""
import numpy as np

import ramansep as rs

rng = np.random.default_rng(2026)
K_true = np.array(rs.mos2_a1_2la().matrix())     # cited lever arms (truth)

# --- 1. Calibration -------------------------------------------------------
# Strain from a stage (%) and electron density from a gate (1e13 cm^-2),
# both known only RELATIVE to the unknown starting state of the flake.
d_strain = np.array([0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.3, 0.3])
d_density = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 1.0])
start = np.array([0.05, 0.10])                   # unknown to the analysis
ref = np.array([404.7, 452.0])                   # illustrative pristine peaks
states = np.column_stack([d_strain, d_density]) + start
positions = ref + states @ K_true.T               # absolute peak positions
positions += rng.normal(0.0, 0.1, positions.shape)   # 0.1 cm^-1 fit error
cal = rs.calibrate_lever_arms(d_strain, d_density, positions, sigmas=0.1,
                              mode_names=["A'1", "2LA(M)"], fit_offset=True)
print(f"Calibration ({cal.n_points} reference states, offsets fitted):")
for k, name in enumerate(cal.mode_names):
    print(f"  {name:6s} strain arm {cal.K[k, 0]:+6.2f} +/- {cal.K_sigma[k, 0]:.2f}"
          f"   density arm {cal.K[k, 1]:+6.2f} +/- {cal.K_sigma[k, 1]:.2f}")

# --- 2. A measured map ----------------------------------------------------
# A ribbon 14 pixels wide: uniform 0.10 % tension; 0.25e13 cm^-2 electrons
# in the two outermost columns on each side, 0.02e13 cm^-2 inside.
# Row 0 lies off the flake: background and noise only, no Raman peak.
H, W = 10, 14
strain = np.full((H, W), 0.10)
density = np.full((H, W), 0.02)
edge = np.zeros((H, W), dtype=bool)
edge[1:, [0, 1, W - 2, W - 1]] = True
density[edge] = 0.25
interior = np.zeros((H, W), dtype=bool)
interior[1:, 2:W - 2] = True
dw_true = np.einsum("mk,khw->mhw", K_true, np.stack([strain, density]))
x = np.linspace(380.0, 480.0, 401)
cube = np.empty((H, W, x.size))
for i in range(H):
    for j in range(W):
        peaks = 0.0 if i == 0 else 1.0
        cube[i, j] = (peaks * rs.lorentzian(x, ref[0] + dw_true[0, i, j], 3.0, 800.0)
                      + peaks * rs.lorentzian(x, ref[1] + dw_true[1, i, j], 4.0, 500.0)
                      + 50.0 + rng.normal(0.0, 15.0, x.size))
fits = rs.fit_map(x, cube, window1=(395, 415), window2=(440, 465),
                  ref1=ref[0], ref2=ref[1], min_snr=3)
codes, counts = np.unique(fits.reason[~fits.ok], return_counts=True)
print(f"Map: {H} x {W} pixels, {fits.n_masked} masked (reasons: "
      f"{dict(zip(codes.tolist(), counts.tolist()))})")

# --- 3. Regions -----------------------------------------------------------
# The pristine reference frequencies are known to 0.2 and 0.3 cm^-1
# (illustrative); that error is common to every pixel.
args = (cal, fits.dw1, fits.dw2, fits.sigma1, fits.sigma2)
inner = rs.region_mean(*args, interior, ref_sigma=(0.2, 0.3))
outer = rs.region_mean(*args, edge, ref_sigma=(0.2, 0.3))
diff = rs.region_contrast(*args, edge, interior)

print(f"Interior ({inner.n_pixels} pixels): strain {inner.strain:+.4f} +/- "
      f"{inner.strain_sigma:.4f} %, density {inner.density:+.3f} +/- "
      f"{inner.density_sigma:.3f} (1e13 cm^-2)")
print(f"  error parts, strain : shifts {inner.strain_sigma_shifts:.4f}, "
      f"calibration {inner.strain_sigma_calibration:.4f}, "
      f"reference {inner.strain_sigma_reference:.4f}")
print(f"  error parts, density: shifts {inner.density_sigma_shifts:.4f}, "
      f"calibration {inner.density_sigma_calibration:.4f}, "
      f"reference {inner.density_sigma_reference:.4f}")
print(f"  homogeneity: chi2 = {inner.chi2:.1f} for {inner.dof} dof, "
      f"p = {inner.p_value:.2f}")
print(f"Edge ({outer.n_pixels} pixels): strain {outer.strain:+.4f} +/- "
      f"{outer.strain_sigma:.4f} %, density {outer.density:+.3f} +/- "
      f"{outer.density_sigma:.3f} (1e13 cm^-2)")
print(f"Edge minus interior: strain {diff.strain:+.4f} +/- "
      f"{diff.strain_sigma:.4f} %, density {diff.density:+.3f} +/- "
      f"{diff.density_sigma:.3f} (1e13 cm^-2)")
print(f"  error parts, density: shifts {diff.density_sigma_shifts:.4f}, "
      f"calibration {diff.density_sigma_calibration:.4f}, reference 0 (cancels)")
print(f"  = {diff.density * 1e13:.2e} +/- {diff.density_sigma * 1e13:.1e} "
      f"electrons per cm^2 more at the edge (true: {0.23e13:.2e})")
