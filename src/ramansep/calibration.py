"""Calibrate the lever-arm matrix from your own reference measurements.

Every analysis in this package stands on a coefficient matrix, and the
shipped sets carry the caveat that lever arms depend on material, mode
pair, excitation and substrate.  The rigorous response to that caveat is
not a larger library of literature sets but a calibration: measure the
shifts of your modes on reference states you control -- a strain stage
sweep at fixed (nominally zero) doping, a gated sweep at fixed strain,
or any other set of points with KNOWN (strain, density) -- and fit the
lever arms from those measurements, on your instrument, with
uncertainties.  This module implements that fit.

The model is the same linear response the rest of the package inverts,

    dw_m(j) = k_m_strain * strain_j + k_m_density * density_j,

one row per mode m, so each mode's pair of lever arms is a weighted
linear least-squares problem on the n reference points with the shared
2-column design X = [strain, density].  With per-point shift
uncertainties sigma the estimator and its covariance are the standard
exact results for known Gaussian noise,

    beta_hat = (X^T W X)^{-1} X^T W y,   Cov(beta_hat) = (X^T W X)^{-1},
    W = diag(1/sigma^2),

(Gauss-Markov; the same algebra as the package's own multimode
inversion, applied to the transposed problem).  Without provided sigmas
the covariance uses the residual variance s^2 = RSS/(n-2), which is why
that path refuses to run with fewer than three reference points: with
n = 2 the fit is exact and carries no internal error estimate, and this
package does not invent one.

Identifiability is checked, not assumed: if the reference states are
collinear in the (strain, density) plane -- for example strain-only
references, or gate voltages that also strain the flake proportionally
-- the two lever arms of a mode are not separately determined by the
data, and the fit REFUSES with an explanation instead of returning one
of the infinitely many minimizers.  A strain-only sweep calibrates
strain arms only; add at least one reference that moves off that line
(a gated point) to calibrate both.

Exact facts the test suite asserts, rather than states: noise-free
shifts generated from a known matrix are recovered to machine
precision; the hand-built normal-equation solution agrees with an
independent QR solve (`numpy.linalg.lstsq` on the whitened system) to
1e-10; on the orthogonal unit design {(1,0), (0,1)} with unit sigmas
the parameter covariance is exactly the identity; a seeded Monte-Carlo
run finds the empirical scatter of the fitted arms compatible with the
reported sigmas; degenerate reference sets raise; and a full round trip
-- calibrate on synthetic references, build the ModeCoefficients, and
run the package's own inversion on forward-generated maps -- returns
the input maps.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .materials import ModeCoefficients


@dataclasses.dataclass
class CalibrationResult:
    """Fitted lever arms with uncertainties and diagnostics.

    K : (m, 2) fitted lever-arm matrix, rows in the mode order of the
        `shifts` columns, columns (strain, density) -- exactly the
        matrix `MultiModeModel` consumes.
    K_sigma : (m, 2) one-standard-deviation uncertainties.
    cov : (m, 2, 2) per-mode parameter covariance matrices.
    chi2, chi2_dof : per-mode weighted residual sum of squares and its
        degrees of freedom (n - 2), when sigmas were provided; the
        chi2/dof ratio near 1 says the linear model and your stated
        sigmas are mutually consistent.  None when sigmas were not
        provided (the residuals then SET the error scale and cannot
        also test it).
    n_points : number of reference measurements used.
    condition_number : of the (unweighted) reference design; large
        values mean the reference states barely span the plane and the
        arms, though identifiable, are poorly determined.
    mode_names : labels carried through to `coefficients`.
    """

    K: np.ndarray
    K_sigma: np.ndarray
    cov: np.ndarray
    chi2: np.ndarray | None
    chi2_dof: int | None
    n_points: int
    condition_number: float
    mode_names: list

    def coefficients(self, reference: str) -> ModeCoefficients:
        """Package a two-mode calibration as a ModeCoefficients.

        The mandatory `reference` string should record what the
        reference states were and how they were known (stage, gate,
        instrument, date) -- calibration provenance is provenance.
        """
        if self.K.shape[0] != 2:
            raise ValueError(
                "coefficients() packages exactly two modes; for "
                f"{self.K.shape[0]} modes pass result.K to MultiModeModel"
            )
        return ModeCoefficients(
            mode1_name=str(self.mode_names[0]),
            mode2_name=str(self.mode_names[1]),
            k1_strain=float(self.K[0, 0]),
            k1_density=float(self.K[0, 1]),
            k2_strain=float(self.K[1, 0]),
            k2_density=float(self.K[1, 1]),
            reference=reference,
        )


def calibrate_lever_arms(strain, density, shifts, sigmas=None,
                         mode_names=None) -> CalibrationResult:
    """Fit the (m, 2) lever-arm matrix from reference measurements.

    Parameters
    ----------
    strain : (n,) known strain of each reference state, in the strain
        unit you intend to use everywhere (e.g. percent biaxial).
    density : (n,) known carrier density of each reference state, in
        your density unit.  Zero is a perfectly good known value: a
        strain-stage sweep enters as (strain_j, 0.0).
    shifts : (n, m) measured peak shifts of the m modes at each
        reference state, RELATIVE TO THE PRISTINE (zero strain, zero
        density) frequency, in cm^-1.  A single mode may be passed as
        (n,) and is treated as (n, 1).
    sigmas : optional (n, m), (m,) or scalar one-standard-deviation
        shift uncertainties (from the peak fitter).  With sigmas the
        parameter covariance is the exact known-noise result and a
        per-mode chi2 model check is returned; without them the
        covariance is estimated from the residuals, which requires
        n >= 3.
    mode_names : optional length-m labels.

    Returns a :class:`CalibrationResult`; see its docstring.

    Raises
    ------
    ValueError : on shape mismatches, non-finite input, non-positive
        sigmas, fewer than 2 reference points, a rank-deficient
        reference design (collinear reference states -- the arms are
        not identifiable), or sigmas absent with n < 3.
    """
    eps = np.asarray(strain, dtype=float).ravel()
    rho = np.asarray(density, dtype=float).ravel()
    y = np.asarray(shifts, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or eps.shape != rho.shape or y.shape[0] != eps.size:
        raise ValueError(
            "need strain (n,), density (n,) and shifts (n, m) with a "
            f"shared n; got {eps.shape}, {rho.shape}, {y.shape}"
        )
    n, m = y.shape
    for name, a in (("strain", eps), ("density", rho), ("shifts", y)):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} contains non-finite values")
    if n < 2:
        raise ValueError("need at least 2 reference points")

    X = np.column_stack([eps, rho])                      # (n, 2)
    # identifiability: the reference states must span the plane
    sv = np.linalg.svd(X, compute_uv=False)
    scale = float(sv[0])
    if scale == 0.0 or sv[-1] <= 1e-12 * scale:
        raise ValueError(
            "reference states are collinear in the (strain, density) "
            "plane, so the strain and density lever arms are not "
            "separately identifiable; add a reference point off that "
            "line (e.g. a gated point to a strain-only sweep)"
        )
    condition_number = float(sv[0] / sv[-1])

    if sigmas is None:
        if n < 3:
            raise ValueError(
                "without shift sigmas the parameter uncertainties come "
                "from the residual variance, which needs n >= 3 "
                "reference points (n = 2 fits exactly and carries no "
                "internal error estimate); provide sigmas or more points"
            )
        sig = None
    else:
        sig = np.broadcast_to(np.asarray(sigmas, dtype=float),
                              (n, m)).copy()
        if not np.all(np.isfinite(sig)) or np.any(sig <= 0.0):
            raise ValueError("sigmas must be finite and positive")

    K = np.empty((m, 2))
    K_sigma = np.empty((m, 2))
    cov = np.empty((m, 2, 2))
    chi2 = np.empty(m) if sig is not None else None
    dof = n - 2
    for k in range(m):
        w = np.ones(n) if sig is None else 1.0 / sig[:, k]
        Xw = X * w[:, None]
        yw = y[:, k] * w
        A = Xw.T @ Xw                                     # X^T W X
        beta = np.linalg.solve(A, Xw.T @ yw)
        Ainv = np.linalg.inv(A)
        r = yw - Xw @ beta
        rss = float(r @ r)
        if sig is not None:
            C = Ainv                                      # known noise
            chi2[k] = rss
        else:
            C = Ainv * (rss / dof)                        # estimated
        K[k] = beta
        cov[k] = C
        K_sigma[k] = np.sqrt(np.diag(C))

    names = (list(mode_names) if mode_names is not None
             else [f"mode{i + 1}" for i in range(m)])
    if len(names) != m:
        raise ValueError(f"mode_names must have length {m}")
    return CalibrationResult(
        K=K, K_sigma=K_sigma, cov=cov, chi2=chi2,
        chi2_dof=(dof if sig is not None else None), n_points=n,
        condition_number=condition_number, mode_names=names,
    )
