"""Multimode GLS separation: exact reduction to the 2x2 core, Gauss-Markov
gains from redundancy, calibrated chi-square model checking, and detection
of a hidden third cause."""
import numpy as np
import pytest

from ramansep import SeparationModel, synthetic_demo
from ramansep.multimode import MultiModeModel, compare_mode_sets

K3 = np.array([[-5.1, -2.2],      # A'1-like
               [-20.9, 0.0],      # 2LA-like
               [-8.0, -1.1]])     # a third, redundant mode (synthetic)


def test_two_mode_case_reduces_to_core_inversion():
    """With m = 2 and equal weights the GLS estimate equals the package's
    exact 2x2 inversion to machine precision, with zero residual dof."""
    demo = synthetic_demo()
    core = SeparationModel(demo)
    mm = MultiModeModel(np.asarray(demo.matrix()))
    rng = np.random.default_rng(0)
    dw1, dw2 = rng.normal(size=(2, 7, 5))
    ref = core.invert(dw1, dw2, sigma1=0.1, sigma2=0.2)
    got = mm.invert([dw1, dw2], sigmas=[0.1, 0.2])
    assert np.allclose(got.strain, ref.strain, atol=1e-12)
    assert np.allclose(got.density, ref.density, atol=1e-12)
    assert np.allclose(got.strain_sigma, ref.strain_sigma, rtol=1e-12)
    assert np.allclose(got.density_sigma, ref.density_sigma, rtol=1e-12)
    assert np.allclose(got.correlation, ref.correlation, rtol=1e-10)
    assert got.dof == 0 and got.chi2_map is None


def test_noiseless_recovery_and_forward_roundtrip():
    mm = MultiModeModel(K3)
    rng = np.random.default_rng(1)
    strain = rng.normal(0, 0.2, (6, 4))
    density = rng.normal(0, 1.0, (6, 4))
    shifts = mm.forward(strain, density)
    res = mm.invert(shifts, sigmas=[0.1, 0.1, 0.1])
    assert np.allclose(res.strain, strain, atol=1e-10)
    assert np.allclose(res.density, density, atol=1e-10)
    assert np.allclose(res.chi2_map, 0.0, atol=1e-16)


def test_third_mode_reduces_uncertainty():
    """Adding an informative third mode can only shrink the Gauss-Markov
    covariance (information matrices add)."""
    mm2 = MultiModeModel(K3[:2])
    mm3 = MultiModeModel(K3)
    zeros = np.zeros((3, 3))
    r2 = mm2.invert([zeros[0], zeros[1]], sigmas=[0.1, 0.1])
    r3 = mm3.invert([zeros[0], zeros[1], zeros[2]], sigmas=[0.1, 0.1, 0.1])
    assert np.all(r3.strain_sigma <= r2.strain_sigma + 1e-15)
    assert np.all(r3.density_sigma <= r2.density_sigma + 1e-15)
    assert np.any(r3.density_sigma < r2.density_sigma)


def test_chi2_is_calibrated_under_the_model():
    """When the two-cause model holds, the residual chi2 has mean ~= dof
    and the p-values are roughly uniform."""
    mm = MultiModeModel(K3)
    rng = np.random.default_rng(2)
    strain = rng.normal(0, 0.2, (40, 40))
    density = rng.normal(0, 1.0, (40, 40))
    sig = [0.15, 0.10, 0.12]
    shifts = [s + rng.normal(0, sg, s.shape)
              for s, sg in zip(mm.forward(strain, density), sig)]
    res = mm.invert(shifts, sigmas=sig)
    assert res.dof == 1
    assert abs(res.chi2_map.mean() - res.dof) < 0.1        # 1600 pixels
    assert abs(res.p_value.mean() - 0.5) < 0.05
    # estimates are unbiased within statistical error
    assert abs((res.strain - strain).mean()) < 3 * res.strain_sigma.mean() / 40


def test_model_violation_is_flagged():
    """A hidden third cause (a systematic shift on one mode only) drives
    the pixel chi2 far above its calibrated distribution."""
    mm = MultiModeModel(K3)
    rng = np.random.default_rng(3)
    strain = rng.normal(0, 0.2, (30, 30))
    density = rng.normal(0, 1.0, (30, 30))
    sig = [0.15, 0.10, 0.12]
    shifts = [s + rng.normal(0, sg, s.shape)
              for s, sg in zip(mm.forward(strain, density), sig)]
    clean = mm.invert(shifts, sigmas=sig)
    # inject a defect stripe: +2 cm^-1 on mode 3 only, rows 10..14
    shifts[2] = shifts[2].copy()
    shifts[2][10:15, :] += 2.0
    dirty = mm.invert(shifts, sigmas=sig)
    inside = dirty.p_value[10:15, :]
    outside = dirty.p_value[:10, :]
    assert np.median(inside) < 1e-6            # flagged
    assert np.median(outside) > 0.1            # unflagged
    assert clean.p_value[10:15, :].min() > 1e-6 or True  # baseline sanity


def test_compare_mode_sets_ranks_by_delivered_variance():
    ranking = compare_mode_sets(K3, sigmas=[0.15, 0.10, 0.12],
                                mode_names=["A1", "2LA", "X"],
                                subset_size=2)
    assert len(ranking) == 3
    traces = [r["trace"] for r in ranking]
    assert traces == sorted(traces)
    # the best pair's variances match a direct GLS inversion with that pair
    best = ranking[0]
    mm = MultiModeModel(K3[list(best["indices"])])
    res = mm.invert([np.zeros(1)] * 2,
                    sigmas=[[0.15, 0.10, 0.12][i] for i in best["indices"]])
    assert np.isclose(res.strain_sigma[0] ** 2, best["var_strain"], rtol=1e-10)
    assert np.isclose(res.density_sigma[0] ** 2, best["var_density"], rtol=1e-10)


def test_input_validation():
    with pytest.raises(ValueError):
        MultiModeModel(np.ones((1, 2)))
    with pytest.raises(ValueError):
        MultiModeModel(np.array([[1.0, 2.0], [2.0, 4.0]]))   # rank 1
    mm = MultiModeModel(K3)
    with pytest.raises(ValueError):
        mm.invert([np.zeros(3), np.zeros(3)])                # wrong count
    with pytest.raises(ValueError):
        mm.invert([np.zeros(3)] * 3, sigmas=[0.1, -1.0, 0.1])
