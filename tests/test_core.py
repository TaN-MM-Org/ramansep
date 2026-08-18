import numpy as np
import pytest

from ramansep import SeparationModel, ModeCoefficients, synthetic_demo


def test_round_trip_recovers_fields():
    model = SeparationModel(synthetic_demo())
    rng = np.random.default_rng(0)
    strain = rng.normal(0.0, 0.1, size=(32, 32))
    density = rng.normal(0.0, 1.0, size=(32, 32))
    dw1, dw2 = model.forward(strain, density)
    out = model.invert(dw1, dw2)
    np.testing.assert_allclose(out.strain, strain, atol=1e-12)
    np.testing.assert_allclose(out.density, density, atol=1e-12)


def test_uncertainty_propagation_scales_linearly():
    model = SeparationModel(synthetic_demo())
    out1 = model.invert(np.zeros((4, 4)), np.zeros((4, 4)), sigma1=0.1, sigma2=0.1)
    out2 = model.invert(np.zeros((4, 4)), np.zeros((4, 4)), sigma1=0.2, sigma2=0.2)
    np.testing.assert_allclose(out2.strain_sigma, 2.0 * out1.strain_sigma)
    np.testing.assert_allclose(out2.density_sigma, 2.0 * out1.density_sigma)


def test_uncertainty_matches_monte_carlo():
    model = SeparationModel(synthetic_demo())
    rng = np.random.default_rng(1)
    s1, s2 = 0.05, 0.08
    n = 200_000
    dw1 = rng.normal(0.0, s1, n)
    dw2 = rng.normal(0.0, s2, n)
    out = model.invert(dw1, dw2, sigma1=s1, sigma2=s2)
    # analytic sigma (constant across pixels) vs empirical spread
    assert np.isclose(out.strain.std(), out.strain_sigma[0], rtol=2e-2)
    assert np.isclose(out.density.std(), out.density_sigma[0], rtol=2e-2)


def test_singular_matrix_rejected():
    bad = ModeCoefficients("m1", "m2", 1.0, 2.0, 2.0, 4.0, reference="test")
    with pytest.raises(ValueError):
        SeparationModel(bad)


def test_poor_conditioning_warns():
    nearly = ModeCoefficients("m1", "m2", 1.0, 1.0, 1.0, 1.001, reference="test")
    with pytest.warns(UserWarning):
        SeparationModel(nearly)


def test_shape_mismatch_rejected():
    model = SeparationModel(synthetic_demo())
    with pytest.raises(ValueError):
        model.invert(np.zeros(3), np.zeros(4))
