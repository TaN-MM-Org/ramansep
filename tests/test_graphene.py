import numpy as np
import pytest

from ramansep import SeparationModel, graphene_g_2d_lee2012


def test_lee2012_constants_are_locked():
    # every number traceable to Lee et al., Nat. Commun. 3, 1024 (2012);
    # a change to any of them must arrive with a new source
    c = graphene_g_2d_lee2012()
    assert (c.k1_strain, c.k1_density) == (-23.5, 1.0)
    assert (c.k2_strain, c.k2_density) == (-51.7, 0.70)
    assert abs(c.k2_strain / c.k1_strain - 2.2) < 1e-12
    assert "Nat. Commun. 3, 1024 (2012)" in c.reference


def test_vector_decomposition_roundtrip_is_exact():
    model = SeparationModel(graphene_g_2d_lee2012())
    strain = np.array([[0.0, 0.3], [-0.15, 0.42]])       # percent
    doping = np.array([[0.0, 4.0], [2.5, -1.0]])         # cm^-1 of G shift
    dw_g, dw_2d = model.forward(strain, doping)
    res = model.invert(dw_g, dw_2d)
    assert np.allclose(res.strain, strain, atol=1e-12)
    assert np.allclose(res.density, doping, atol=1e-12)


def test_pure_strain_trajectory_returns_zero_doping():
    # points along the strain axis (slope 2.2) decompose to doping = 0
    model = SeparationModel(graphene_g_2d_lee2012())
    dw_g = np.linspace(-8.0, 8.0, 9)
    res = model.invert(dw_g, 2.2 * dw_g)
    assert np.allclose(res.density, 0.0, atol=1e-12)
    assert np.allclose(res.strain, dw_g / -23.5, atol=1e-12)


def test_pure_doping_trajectory_returns_zero_strain():
    # points along the hole-doping axis (slope 0.70) decompose to
    # strain = 0 and doping coordinate = the G shift itself
    model = SeparationModel(graphene_g_2d_lee2012())
    dw_g = np.linspace(0.0, 12.0, 7)
    res = model.invert(dw_g, 0.70 * dw_g)
    assert np.allclose(res.strain, 0.0, atol=1e-12)
    assert np.allclose(res.density, dw_g, atol=1e-12)


def test_uncertainty_propagation_runs_and_is_finite():
    model = SeparationModel(graphene_g_2d_lee2012())
    res = model.invert(np.full((4, 4), 3.0), np.full((4, 4), 4.0),
                       sigma1=0.3, sigma2=0.5)
    assert np.all(np.isfinite(res.strain_sigma))
    assert np.all(np.isfinite(res.density_sigma))
    assert np.all(np.abs(res.correlation) <= 1.0 + 1e-12)


def test_noise_amplification_is_moderate():
    # the raw condition number of this set (~92) is an artifact of the
    # unit scales (percent vs cm^-1); the physically meaningful metric is
    # noise amplification, asserted here against the exact closed forms
    # |Kinv row| for unit shift noise on both modes
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SeparationModel(graphene_g_2d_lee2012())
    det = -23.5 * 0.70 - 1.0 * (-51.7)                  # = 35.25
    amp_strain = np.hypot(0.70, 1.0) / det              # %/cm^-1 of noise
    amp_doping = np.hypot(51.7, 23.5) / det             # cm^-1 per cm^-1
    res = model.invert(0.0, 0.0, sigma1=1.0, sigma2=1.0)
    assert np.isclose(float(res.strain_sigma), amp_strain, rtol=1e-12)
    assert np.isclose(float(res.density_sigma), amp_doping, rtol=1e-12)
    # unit shift noise costs under 0.04 percent strain and under
    # 1.7 cm^-1 of doping coordinate: a well-separated axis pair
    assert amp_strain < 0.04
    assert amp_doping < 1.7
