"""v0.8 from-the-instrument anchors: the map fitter against the
single-spectrum fitter and a known-truth round trip, honest per-pixel
masking, the sloped-baseline bias demonstrated and recovered, and the
exact file-contract round trips with their refusals."""
import numpy as np
import pytest

import ramansep as rs
from ramansep.fitting import lorentzian

W1, W2 = (395.0, 415.0), (440.0, 465.0)
REF1, REF2 = 404.7, 452.0


def _cube_from_truth(strain, density, coeffs, x, noise=0.0, rng=None):
    """Forward: truth maps -> shifts -> Lorentzian spectra."""
    model = rs.SeparationModel(coeffs)
    dw1, dw2 = model.forward(strain, density)
    H, W = strain.shape
    cube = np.empty((H, W, x.size))
    for i in range(H):
        for j in range(W):
            y = (lorentzian(x, REF1 + dw1[i, j], 3.0, 800.0, 50.0)
                 + lorentzian(x, REF2 + dw2[i, j], 4.0, 500.0, 0.0))
            if noise > 0.0:
                y = y + rng.normal(0.0, noise, x.size)
            cube[i, j] = y
    return cube


def test_fit_map_round_trip_and_pixelwise_equality():
    coeffs = rs.mos2_a1_2la()
    x = np.linspace(380.0, 480.0, 401)
    rng = np.random.default_rng(0)
    strain = rng.uniform(-0.1, 0.1, (4, 5))
    density = rng.uniform(0.0, 0.2, (4, 5))   # in the set's 1e13 cm^-2 unit
    cube = _cube_from_truth(strain, density, coeffs, x)

    res = rs.fit_map(x, cube, W1, W2, REF1, REF2)
    assert res.n_masked == 0 and res.ok.all()
    # every pixel equals a direct fit_two_modes call on its spectrum
    d1, d2, s1, s2, _, _ = rs.fit_two_modes(x, cube[2, 3], W1, W2,
                                            REF1, REF2)
    assert res.dw1[2, 3] == d1 and res.dw2[2, 3] == d2
    assert res.sigma1[2, 3] == s1 and res.sigma2[2, 3] == s2
    # the maps invert back to the generating truth
    inv = rs.SeparationModel(coeffs).invert(res.dw1, res.dw2,
                                            res.sigma1, res.sigma2)
    assert np.abs(inv.strain - strain).max() < 1e-4
    assert np.abs(inv.density - density).max() < 1e-3


def test_fit_map_masks_bad_pixels_without_contaminating_neighbors():
    coeffs = rs.mos2_a1_2la()
    x = np.linspace(380.0, 480.0, 401)
    rng = np.random.default_rng(1)
    strain = rng.uniform(-0.1, 0.1, (3, 3))
    density = rng.uniform(0.0, 0.2, (3, 3))
    cube = _cube_from_truth(strain, density, coeffs, x)
    clean = rs.fit_map(x, cube, W1, W2, REF1, REF2)
    cube_bad = cube.copy()
    cube_bad[1, 1, :] = np.nan                    # a dead pixel
    res = rs.fit_map(x, cube_bad, W1, W2, REF1, REF2)
    assert not res.ok[1, 1] and res.n_masked == 1
    assert np.isnan(res.dw1[1, 1]) and np.isnan(res.sigma2[1, 1])
    good = res.ok
    assert np.array_equal(res.dw1[good], clean.dw1[good])
    # per-pixel inversion carries the mask through untouched
    inv = rs.SeparationModel(coeffs).invert(res.dw1, res.dw2)
    assert np.isnan(inv.strain[1, 1])
    assert np.all(np.isfinite(inv.strain[good]))
    # the joint map inversion refuses NaNs instead of solving garbage
    with pytest.raises(ValueError, match="non-finite"):
        rs.bayesian_map_inversion(
            rs.SeparationModel(coeffs).K,
            np.stack([res.dw1, res.dw2]), np.array([0.1, 0.1]), 1.0)


def test_linear_baseline_removes_the_sloped_background_bias():
    """A sloped fluorescence background pulls a constant-baseline
    center sideways; the linear-baseline fit recovers it. Both halves
    demonstrated, and the analytic slope Jacobian checked against
    finite differences implicitly through the noiseless recovery."""
    x = np.linspace(390.0, 420.0, 301)
    true_c = 404.2
    y = lorentzian(x, true_c, 3.0, 400.0, 100.0) + 12.0 * (x - 405.0)
    f_const = rs.fit_lorentzian(x, y)
    f_lin = rs.fit_lorentzian(x, y, baseline="linear")
    err_const = abs(f_const.center - true_c)
    err_lin = abs(f_lin.center - true_c)
    assert err_lin < 1e-6                       # exact model, recovered
    assert err_const > 10.0 * max(err_lin, 1e-9)
    assert err_const > 3.0 * max(f_const.center_sigma, 1e-12)
    assert abs(f_lin.slope - 12.0) < 1e-6
    # noiseless recovery of every parameter with the linear model
    assert abs(f_lin.fwhm - 3.0) < 1e-6
    assert abs(f_lin.amplitude - 400.0) < 1e-4
    # the historical constant-baseline path is unchanged on flat data
    y0 = lorentzian(x, true_c, 3.0, 400.0, 100.0)
    f0 = rs.fit_lorentzian(x, y0)
    assert abs(f0.center - true_c) < 1e-8 and f0.slope == 0.0


def test_spectrum_csv_round_trip_and_refusals(tmp_path):
    x = np.linspace(380.0, 480.0, 51)
    y = lorentzian(x, 404.7, 3.0, 800.0, 50.0)
    p = tmp_path / "spec.csv"
    rs.save_spectrum_csv(p, x, y)
    x2, y2 = rs.load_spectrum_csv(p)
    assert np.array_equal(x, x2) and np.array_equal(y, y2)
    q = tmp_path / "bad.csv"
    q.write_text("wavenumber,counts\n1.0,2.0\n")
    with pytest.raises(ValueError, match="header"):
        rs.load_spectrum_csv(q)
    q.write_text("wavenumber_cm1,counts\n2.0,1.0\n1.0,1.0\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        rs.load_spectrum_csv(q)


def test_map_csv_round_trip_and_refusals(tmp_path):
    rng = np.random.default_rng(2)
    x = np.linspace(380.0, 480.0, 21)
    cube = rng.uniform(0.0, 100.0, (3, 4, 21))
    p = tmp_path / "map.csv"
    rs.save_map_csv(p, x, cube)
    x2, cube2 = rs.load_map_csv(p)
    assert np.array_equal(x, x2) and np.array_equal(cube, cube2)
    # a hole in the pixel grid is refused
    lines = p.read_text().splitlines()
    body = [ln for ln in lines[1:] if not ln.startswith("1,2,")]
    q = tmp_path / "holed.csv"
    q.write_text("\n".join([lines[0]] + body) + "\n")
    with pytest.raises(ValueError, match="holes"):
        rs.load_map_csv(q)
    # a pixel on its own axis is refused
    shifted = [ln if not ln.startswith("2,3,")
               else ",".join([ln.split(",")[0], ln.split(",")[1],
                              repr(float(ln.split(",")[2]) + 0.5),
                              ln.split(",")[3]])
               for ln in lines[1:]]
    q.write_text("\n".join([lines[0]] + shifted) + "\n")
    with pytest.raises(ValueError, match="shared axis|different wavenumber"):
        rs.load_map_csv(q)


def test_full_pipeline_file_to_maps(tmp_path):
    """The complete experimental path: map file -> fit_map -> invert,
    with noise, recovering the truth within the propagated errors."""
    coeffs = rs.mos2_a1_2la()
    x = np.linspace(380.0, 480.0, 401)
    rng = np.random.default_rng(3)
    strain = np.full((3, 3), 0.05)
    strain[0, 0] = -0.05
    density = np.full((3, 3), 0.1)
    cube = _cube_from_truth(strain, density, coeffs, x, noise=3.0,
                            rng=rng)
    p = tmp_path / "measured.csv"
    rs.save_map_csv(p, x, cube)
    x2, cube2 = rs.load_map_csv(p)
    res = rs.fit_map(x2, cube2, W1, W2, REF1, REF2)
    assert res.ok.all()
    inv = rs.SeparationModel(coeffs).invert(res.dw1, res.dw2,
                                            res.sigma1, res.sigma2)
    err = np.abs(inv.strain - strain)
    assert np.all(err < 5.0 * inv.strain_sigma + 1e-3)
