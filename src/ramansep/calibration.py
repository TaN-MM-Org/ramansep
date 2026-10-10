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
independent least-squares solve (`numpy.linalg.lstsq` on the whitened
system) to 1e-10; on the orthogonal unit design {(1,0), (0,1)} with unit sigmas
the parameter covariance is exactly the identity; a seeded Monte-Carlo
run finds the empirical scatter of the fitted arms compatible with the
reported sigmas; degenerate reference sets raise; and a full round trip
-- calibrate on synthetic references, build the ModeCoefficients, and
run the package's own inversion on forward-generated maps -- returns
the input maps.

Unknown starting state (new in v0.13): with `fit_offset=True` each
mode's fit gets a free constant,

    dw_m(j) = c_m + k_m_strain * strain_j + k_m_density * density_j,

so the strain and density of the reference states need to be known
only RELATIVE to the starting state (a stage reading, a change of gate
charge), and the shifts may be measured from any fixed reference
frequency, or be absolute peak positions. A real calibration is
usually in exactly that situation: the flake starts with some built-in
strain and doping that nobody knows. Forcing such data through zero
biases the lever arms; the tests show the bias equals the closed-form
projection of the offset onto the design, and that the offset fit
removes it. The lever arms then come from the weighted-centered design
(Frisch-Waugh-Lovell theorem: R. Frisch and F. V. Waugh, Econometrica
1, 387 (1933); M. C. Lovell, J. Am. Stat. Assoc. 58, 993 (1963)), and
the tests check the reported covariance against that closed form. The
price is one more parameter per mode: the reference states must not
lie on ANY straight line in the (strain, density) plane (not only a
line through the origin), and the residual degrees of freedom are
n - 3. The offset itself is a nuisance parameter: it is the shift at
the calibration's own zero, not the pristine reference frequency, so
it is reported but not used by the map inversions; `cov` holds the
lever-arm covariance with the offset uncertainty already accounted
for, which is what `separation_with_calibration` and
`multimode_with_calibration` need.
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
        degrees of freedom (n - 2, or n - 3 with `fit_offset=True`),
        when sigmas were provided; the
        chi2/dof ratio near 1 says the linear model and your stated
        sigmas are mutually consistent.  None when sigmas were not
        provided (the residuals then SET the error scale and cannot
        also test it).
    n_points : number of reference measurements used.
    condition_number : of the (unweighted) reference design; large
        values mean the reference states barely span the plane and the
        arms, though identifiable, are poorly determined. With
        `fit_offset=True` it is that of the centered design (each column
        minus its mean), since only the spread about the mean state
        determines the arms then.
    mode_names : labels carried through to `coefficients`.
    offset, offset_sigma : (m,) fitted per-mode offsets c_m (the shift
        at zero strain and zero density in the calibration's own
        coordinates, in the units of `shifts`) and their 1-sigma
        uncertainties, when `fit_offset=True`; None otherwise (new in
        v0.13).
    full_cov : (m, 3, 3) per-mode covariance of (strain arm, density
        arm, offset) when `fit_offset=True`; None otherwise. `cov` is
        its upper-left 2 x 2 block.
    """

    K: np.ndarray
    K_sigma: np.ndarray
    cov: np.ndarray
    chi2: np.ndarray | None
    chi2_dof: int | None
    n_points: int
    condition_number: float
    mode_names: list
    offset: np.ndarray | None = None
    offset_sigma: np.ndarray | None = None
    full_cov: np.ndarray | None = None

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
                         mode_names=None, *,
                         fit_offset: bool = False) -> CalibrationResult:
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
        shift uncertainties (from the peak fitter); for a single mode
        passed as (n,), sigmas may also be (n,), one per point (new in
        v0.13).  With sigmas the
        parameter covariance is the exact known-noise result and a
        per-mode chi2 model check is returned; without them the
        covariance is estimated from the residuals, which requires
        n >= 3.
    mode_names : optional length-m labels.
    fit_offset : False (default, unchanged behavior) fits lines
        through the origin, which needs the shifts measured from the
        true pristine frequency and the strain and density known in
        absolute terms. True (new in v0.13) fits one free offset per
        mode as well, so strain and density need to be known only
        relative to the starting state and the shifts may be taken from
        any fixed reference (see the module docstring). Needs
        reference states that do not lie on one straight line, n >= 3
        with sigmas and n >= 4 without.

    Returns a :class:`CalibrationResult`; see its docstring.

    Raises
    ------
    ValueError : on shape mismatches, non-finite input, non-positive
        sigmas, fewer than 2 reference points, a rank-deficient
        reference design (collinear reference states -- the arms are
        not identifiable), or sigmas absent with n < 3; with
        `fit_offset=True`, reference states on any one straight line,
        fewer than 3 points, or sigmas absent with n < 4.
    """
    eps = np.asarray(strain, dtype=float).ravel()
    rho = np.asarray(density, dtype=float).ravel()
    y = np.asarray(shifts, dtype=float)
    single = y.ndim == 1
    if single:
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
    if fit_offset:
        return _calibrate_with_offset(eps, rho, y, sigmas, mode_names,
                                      single)

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
        sig = _broadcast_sigmas(sigmas, n, m, single)

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


def _broadcast_sigmas(sigmas, n, m, single):
    """Shift sigmas as an (n, m) array: scalar, (m,) per mode, (n, m)
    per point and mode, or (n,) per point for a single mode given as
    (n,). Refuses other shapes and non-positive or non-finite values."""
    s = np.asarray(sigmas, dtype=float)
    if single and s.ndim == 1 and s.size == n:
        s = s[:, None]
    try:
        sig = np.broadcast_to(s, (n, m)).copy()
    except ValueError:
        raise ValueError(
            f"sigmas of shape {s.shape} do not fit {n} reference points "
            f"and {m} modes; give a scalar, ({m},) per mode or ({n}, {m})"
            " per point and mode") from None
    if not np.all(np.isfinite(sig)) or np.any(sig <= 0.0):
        raise ValueError("sigmas must be finite and positive")
    return sig


def _calibrate_with_offset(eps, rho, y, sigmas, mode_names, single):
    """`calibrate_lever_arms(..., fit_offset=True)`: per mode, weighted
    least squares on the design [strain, density, 1]."""
    n, m = y.shape
    X = np.column_stack([eps, rho])
    Xc = X - X.mean(axis=0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    if sv[0] == 0.0 or sv[-1] <= 1e-12 * sv[0]:
        raise ValueError(
            "with fit_offset=True the reference states must not lie on "
            "one straight line in the (strain, density) plane (any "
            "line, not only one through the origin): along a line, a "
            "change of offset and a change of lever arms fit the data "
            "equally well; add a reference point off that line")
    if n < 3:                                    # pragma: no cover
        raise ValueError("fit_offset=True needs at least 3 reference "
                         "points")
    if sigmas is None:
        if n < 4:
            raise ValueError(
                "without shift sigmas, fit_offset=True needs n >= 4 "
                "reference points (3 parameters per mode plus one "
                "degree of freedom for the residual variance); provide "
                "sigmas or more points")
        sig = None
    else:
        sig = _broadcast_sigmas(sigmas, n, m, single)
    X1 = np.column_stack([eps, rho, np.ones(n)])
    dof = n - 3
    K = np.empty((m, 2))
    K_sigma = np.empty((m, 2))
    cov = np.empty((m, 2, 2))
    full = np.empty((m, 3, 3))
    off = np.empty(m)
    off_sigma = np.empty(m)
    chi2 = np.empty(m) if sig is not None else None
    for k in range(m):
        w = np.ones(n) if sig is None else 1.0 / sig[:, k]
        Xw = X1 * w[:, None]
        yw = y[:, k] * w
        A = Xw.T @ Xw
        beta = np.linalg.solve(A, Xw.T @ yw)
        r = yw - Xw @ beta
        rss = float(r @ r)
        C = np.linalg.inv(A)
        if sig is not None:
            chi2[k] = rss
        else:
            C = C * (rss / dof)
        K[k] = beta[:2]
        off[k] = beta[2]
        full[k] = C
        cov[k] = C[:2, :2]
        K_sigma[k] = np.sqrt(np.diag(C)[:2])
        off_sigma[k] = np.sqrt(C[2, 2])
    names = (list(mode_names) if mode_names is not None
             else [f"mode{i + 1}" for i in range(m)])
    if len(names) != m:
        raise ValueError(f"mode_names must have length {m}")
    return CalibrationResult(
        K=K, K_sigma=K_sigma, cov=cov, chi2=chi2,
        chi2_dof=(dof if sig is not None else None), n_points=n,
        condition_number=float(sv[0] / sv[-1]), mode_names=names,
        offset=off, offset_sigma=off_sigma, full_cov=full)
