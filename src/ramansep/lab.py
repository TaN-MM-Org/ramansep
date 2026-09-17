"""Plan a calibration before measuring it.

`calibrate_lever_arms` turns measured reference states into lever arms
with uncertainties -- after the measurement. This module answers the
questions that come before it: which reference states are worth
preparing, how good will the calibration be, and how many repeats does
a target uncertainty cost?

Because the calibration model is linear, none of this is approximate:
the covariance `calibrate_lever_arms` will report depends only on the
reference design (the (strain, density) points and the shift
uncertainties), so it can be computed exactly before any spectrum is
taken -- the same (X^T W X)^-1 matrix, from the design alone. The
tests assert that equality to machine precision.

The design tool uses the standard determinant criterion (D-optimal
design; F. Pukelsheim, Optimal Design of Experiments, SIAM (2006)):
each greedy pick is the candidate that most shrinks the joint
lever-arm uncertainty. Collinear candidate sets -- for example a
strain-only sweep -- are refused with the same exact rank argument
`calibrate_lever_arms` uses, because no subset of a line can identify
two lever arms.

Units are whatever you use everywhere else in the package (e.g.
percent biaxial strain, density in 1e13 cm^-2, shifts and their
sigmas in cm^-1); the planned uncertainties come out in the matching
lever-arm units.
"""
from __future__ import annotations

import numpy as np

__all__ = ["plan_calibration", "design_references",
           "repeats_for_sigma"]


def _design_matrix(strain, density):
    eps = np.asarray(strain, dtype=float).ravel()
    rho = np.asarray(density, dtype=float).ravel()
    if eps.shape != rho.shape or eps.size < 1:
        raise ValueError("strain and density must be matching (n,) "
                         "arrays")
    if not (np.all(np.isfinite(eps)) and np.all(np.isfinite(rho))):
        raise ValueError("reference states must be finite")
    return np.column_stack([eps, rho])


def _weights(sigmas, n):
    sig = np.broadcast_to(np.asarray(sigmas, dtype=float), (n,)).copy()
    if np.any(sig <= 0.0) or not np.all(np.isfinite(sig)):
        raise ValueError("sigmas must be finite and positive")
    return 1.0 / sig


def plan_calibration(strain, density, sigmas=1.0):
    """Predicted lever-arm uncertainties for a planned reference set.

    strain, density : (n,) the reference states you intend to prepare.
    sigmas : expected 1-sigma shift uncertainty per point (scalar or
        (n,), cm^-1), e.g. the center error your peak fitter typically
        reports.

    Returns dict(identifiable, condition_number, cov, K_sigma):
    `cov` is the exact 2x2 covariance `calibrate_lever_arms` will
    report for each mode measured with these sigmas, and `K_sigma` its
    diagonal square root, ordered (strain arm, density arm). A
    collinear design is reported as identifiable=False with cov None
    -- the same rank arithmetic on which the calibration refuses after
    the fact.
    """
    X = _design_matrix(strain, density)
    w = _weights(sigmas, X.shape[0])
    Xw = X * w[:, None]
    A = Xw.T @ Xw
    sv = np.linalg.svd(X, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    identifiable = bool(sv[0] > 0 and sv[-1] > 1e-12 * sv[0]
                        and X.shape[0] >= 2)
    if not identifiable:
        return {"identifiable": False, "condition_number": cond,
                "cov": None, "K_sigma": None}
    cov = np.linalg.inv(A)
    return {"identifiable": True, "condition_number": cond,
            "cov": cov, "K_sigma": np.sqrt(np.diag(cov))}


def design_references(strain_candidates, density_candidates, n_pick,
                      sigmas=1.0):
    """Pick the most informative reference states to prepare.

    From the candidate (strain, density) points -- the states your
    stage and gate can actually reach -- greedily choose `n_pick`,
    each pick being the one that most increases the determinant of the
    information matrix (equivalently: shrinks the joint lever-arm
    uncertainty fastest). Transparent and monotone, but a
    good-practice heuristic, not a proof of the globally best subset.

    Returns dict(indices, condition_number, cov, K_sigma) with the
    chosen candidate indices in pick order and the planned covariance
    of the chosen set. Refuses when even the full candidate list is
    collinear, with the same explanation the calibration itself gives.
    """
    X = _design_matrix(strain_candidates, density_candidates)
    n = X.shape[0]
    n_pick = int(n_pick)
    if not 2 <= n_pick <= n:
        raise ValueError(f"n_pick must be between 2 and {n}")
    w = _weights(sigmas, n)
    full = plan_calibration(strain_candidates, density_candidates,
                            1.0 / w)
    if not full["identifiable"]:
        raise ValueError(
            "all candidate states are collinear in the (strain, "
            "density) plane, so the strain and density lever arms are "
            "not separately identifiable; add a candidate off that "
            "line (e.g. a gated point to a strain-only sweep)")
    rows = X * w[:, None]
    # column-scaled (unit-free) greedy: identical column scaling
    # multiplies every candidate determinant by the same constant, so
    # the choices are unchanged, while the tiny start-up regularizer
    # stays meaningful in every direction
    scale = np.sqrt(np.mean(rows * rows, axis=0))
    scale[scale == 0.0] = 1.0
    rs = rows / scale
    eps = 1e-12 * float(np.max(np.sum(rs * rs, axis=1)))
    fs = eps * np.eye(2)
    chosen = []
    for _ in range(n_pick):
        best_j, best_det = -1, -np.inf
        for j in range(n):
            if j in chosen:
                continue
            det = float(np.linalg.slogdet(fs + np.outer(rs[j],
                                                        rs[j]))[1])
            if det > best_det:
                best_j, best_det = j, det
        fs = fs + np.outer(rs[best_j], rs[best_j])
        chosen.append(best_j)
    idx = list(chosen)
    sub = plan_calibration(
        np.asarray(strain_candidates, dtype=float).ravel()[idx],
        np.asarray(density_candidates, dtype=float).ravel()[idx],
        (1.0 / w)[idx])
    return {"indices": idx, "condition_number":
            sub["condition_number"], "cov": sub["cov"],
            "K_sigma": sub["K_sigma"]}


def repeats_for_sigma(target_K_sigma, strain, density, sigmas=1.0):
    """How many times must the reference set be repeated?

    Repeating the same reference design r times scales the covariance
    by exactly 1/r (r independent copies of every measurement), so the
    smallest integer r meeting a target lever-arm uncertainty follows
    in closed form -- no search. `target_K_sigma` is the largest
    acceptable 1-sigma uncertainty of any lever arm, in your lever-arm
    units.

    Returns (r, plan) where `plan` is the `plan_calibration` result
    for the repeated design. Refuses a non-identifiable base design.
    """
    t = float(target_K_sigma)
    if not (np.isfinite(t) and t > 0.0):
        raise ValueError("target_K_sigma must be positive")
    base = plan_calibration(strain, density, sigmas)
    if not base["identifiable"]:
        raise ValueError(
            "the reference design is collinear in the (strain, "
            "density) plane; no number of repeats can identify both "
            "lever arms -- add a reference state off the line")
    worst = float(np.max(base["K_sigma"]))
    r = max(1, int(np.ceil((worst / t) ** 2)))
    X = _design_matrix(strain, density)
    n = X.shape[0]
    sig = np.broadcast_to(np.asarray(sigmas, dtype=float), (n,))
    plan = plan_calibration(np.tile(X[:, 0], r), np.tile(X[:, 1], r),
                            np.tile(sig, r))
    return r, plan
