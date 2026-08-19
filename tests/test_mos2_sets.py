"""Tests of the cited MoS2 coefficient sets.

The two headline tests reproduce, from the shipped coefficients and the
peak shifts reported in the source paper (Mahim and Rahman, under review;
tip-enhanced maps of Krayev et al., Appl. Phys. Lett. 128, 203102 (2026)),
the paper's published separation results:

  * at the ribbon edge, A'1 redshifts by 0.5 cm^-1 while 2LA(M) does not
    move -> an electron excess of 2.3e12 cm^-2 with strain below 0.03%;
  * at an interior spot, 2LA(M) redshifts by 2.8 cm^-1 and A'1 by
    0.6 cm^-1 -> 0.134% of tension carrying no significant charge
    (the paper's quoted uncertainty on the edge charge is 7.6e11 cm^-2).

If a transcription error ever enters the coefficient tables, these tests
fail against the published numbers.
"""
import numpy as np
import pytest

from ramansep import SeparationModel, mos2_a1_2la, mos2_eprime_a1


def test_values_locked_a1_2la():
    c = mos2_a1_2la()
    assert c.mode1_name == "A'1" and c.mode2_name == "2LA(M)"
    assert c.k1_strain == -5.1      # cm^-1 per % biaxial (DFT, source paper)
    assert c.k1_density == -2.2     # cm^-1 per 1e13 cm^-2 (Chakraborty 2012)
    assert c.k2_strain == -20.9     # cm^-1 per % biaxial (DFT, source paper)
    assert c.k2_density == 0.0      # unmeasured, set to zero (documented)
    assert "Chakraborty" in c.reference
    assert "10.5281/zenodo.21778170" in c.reference


def test_values_locked_eprime_a1():
    c = mos2_eprime_a1()
    # dw/deps = -2 gamma w / 100 per percent of biaxial strain
    assert c.k1_strain == pytest.approx(-2 * 0.56 * 385.0 / 100, abs=0.005)
    assert c.k2_strain == pytest.approx(-2 * 0.31 * 403.0 / 100, abs=0.005)
    assert c.k1_density == -0.33
    assert c.k2_density == -2.2
    assert "Michail" in c.reference and "Chakraborty" in c.reference


def test_lever_arm_ratio_of_paper_pair():
    c = mos2_a1_2la()
    # "a computed ratio of 4.1" (source paper, Sec. II)
    assert c.k2_strain / c.k1_strain == pytest.approx(4.1, abs=0.05)


def test_edge_charge_reproduces_paper():
    model = SeparationModel(mos2_a1_2la())
    res = model.invert(np.array([-0.5]), np.array([0.0]))
    density_cm2 = res.density[0] * 1e13
    strain_pct = res.strain[0]
    # 2LA(M) unmoved forces zero recovered strain in this pair
    assert strain_pct == pytest.approx(0.0, abs=1e-12)
    # exact linear inversion: 0.5 / 2.2 = 0.227e13
    assert density_cm2 == pytest.approx(0.5 / 2.2 * 1e13, rel=1e-12)
    # agrees with the published rounded value 2.3e12 cm^-2
    assert abs(density_cm2 - 2.3e12) < 0.5e11


def test_interior_strain_reproduces_paper():
    model = SeparationModel(mos2_a1_2la())
    res = model.invert(np.array([-0.6]), np.array([-2.8]))
    strain_pct = res.strain[0]
    density_cm2 = res.density[0] * 1e13
    # published: 0.134% tension
    assert strain_pct == pytest.approx(0.134, abs=0.0005)
    # published: no significant charge (edge-charge 1-sigma is 7.6e11)
    assert abs(density_cm2) < 7.6e11


def test_optical_pair_amplifies_strain_noise_more():
    """With equal shift noise, the all-optical pair is the noisier probe
    of strain, because both optical modes respond weakly to strain. The
    raw matrix condition number does not capture this (it is sensitive to
    row scaling), so the comparison is made on the propagated sigma."""
    dw = np.array([0.0])
    sig = 0.1
    r_paper = SeparationModel(mos2_a1_2la()).invert(dw, dw, sig, sig)
    r_optical = SeparationModel(mos2_eprime_a1()).invert(dw, dw, sig, sig)
    ratio = r_optical.strain_sigma[0] / r_paper.strain_sigma[0]
    assert ratio > 4.0


def test_forward_inverse_roundtrip_cited_sets():
    for coeffs in (mos2_a1_2la(), mos2_eprime_a1()):
        model = SeparationModel(coeffs)
        strain = np.array([0.05, -0.02, 0.134])
        density = np.array([0.23, 0.0, 0.038])
        dw1, dw2 = model.forward(strain, density)
        res = model.invert(dw1, dw2)
        np.testing.assert_allclose(res.strain, strain, atol=1e-12)
        np.testing.assert_allclose(res.density, density, atol=1e-12)
