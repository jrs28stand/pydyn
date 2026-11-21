# This file is part of pydyn.
#
# It is a Python port of matlab/nonlinear-filters/{sequential_importance_particle_filter,
# auxiliary_particle_filter, conditional_particle_filter, gaussian_filter, gaussian_filter_bank,
# nonlinear_kalman_filter, iterate_law_of_motion, measurement_equations, resample,
# traditional_resampling, multivariate_smooth_resampling, univariate_smooth_resampling, neff,
# probability2, reduced_rank_cholesky, unscented_sigma_points, spherical_radial_sigma_points}.m,
# matlab/estimation/qr2.m and matlab/missing/mex/local_state_space_iterations/
# local_state_space_iteration_2.m (with mex/sources/local_state_space_iterations/
# local_state_space_iteration_2.cc) of Dynare 7.2 (Copyright (C) 2009-2025 Dynare Team,
# https://www.dynare.org, GPL-3.0-or-later), modified in 2026.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Dynare's nonlinear filters for second-order approximated models.

The state space is the reduced form of ``non_linear_dsge_likelihood.m``: the rows of the decision
rule restricted to the union of the state and observed variables (``dr.restrict_var_list``),
``mf0`` the positions of the state variables and ``mf1`` those of the observed variables in it.
Particles are columns (``n`` state variables x ``N`` particles), as in Dynare.

Every filter returns ``(LIK, lik)``: minus the log likelihood summed from period ``start`` (1-based)
and the log density of every period (``NaN``/``-inf`` where Dynare stops early). Random numbers come
from a stream (:mod:`pydyn.octave_rng`) with MATLAB's ``randn(r, c)``/``rand(r, c)`` calls, so the
filters consume them in Dynare's order; with :class:`~pydyn.octave_rng.OctaveRandomStream` they
reproduce Dynare under Octave exactly.

``resampling_method=smooth`` cannot be reproduced bit for bit: Dynare orthogonalises the particles
with ``eig(P'*(P/N))``, and MATLAB/Octave pick the symmetric or the general eigensolver (different
order and signs of the eigenvectors, hence a different use of the uniform numbers) according to
whether the computed product happens to be exactly symmetric, which depends on the rounding of the
BLAS. pydyn applies the same rule to its own product; given the same decomposition the result is
Dynare's.

Dynare's behaviour is reproduced, including what looks like slips in its code (listed in
``DYNARE_QUIRKS``); configurations where Dynare's code stops with an error raise
:class:`NonlinearFilterError` with the reason.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.special import gammaln

__all__ = ["ReducedForm", "ParticleOptions", "NonlinearFilterError", "local_state_space_iteration_2",
           "iterate_law_of_motion", "measurement_equations", "unscented_sigma_points",
           "spherical_radial_sigma_points", "reduced_rank_cholesky", "probability2", "neff", "resample",
           "traditional_resampling", "multivariate_smooth_resampling", "univariate_smooth_resampling",
           "sequential_importance_particle_filter", "auxiliary_particle_filter", "conditional_particle_filter",
           "gaussian_filter", "nonlinear_kalman_filter", "FILTERS", "DYNARE_QUIRKS"]

#: behaviour of Dynare 7.2 that pydyn reproduces although it looks unintended
DYNARE_QUIRKS = (
    "resampling=generic resamples when neff < resampling_threshold * (number of periods), not * (number "
    "of particles) (sis, apf, cpf, gf with Monte Carlo distribution approximation)",
    "sis with resampling=generic: in a period without resampling the particles are not moved forward "
    "(the state vectors stay those of the previous period)",
    "resampling_method=stratified: noise.^(1./(N:-1:1)) broadcasts to an N x N matrix, so the sorted "
    "uniforms are the cumulative products of the uniforms and nearly all particles get index 1",
    "apf: the first-stage Student-t weights use z.^2 where z is already a squared Mahalanobis distance",
    "gf and nlkf with sigma points: no resampling options are used; gf adds 1e-6 to every weight",
)


class NonlinearFilterError(Exception):
    """A configuration or state where Dynare's code stops with an error."""


@dataclass
class ParticleOptions:
    """``options_.particle`` (the fields the filters use), with Dynare's defaults."""
    filter_algorithm: str = "sis"
    number_of_particles: int = 5000
    resampling: str = "systematic"          # systematic | none | generic
    resampling_threshold: float = 0.5
    resampling_method: str = "kitagawa"     # kitagawa | stratified | smooth
    proposal_approximation: str = "unscented"      # unscented | cubature | montecarlo
    distribution_approximation: str = "unscented"  # unscented | cubature | montecarlo
    unscented_alpha: float = 1.0
    unscented_beta: float = 2.0
    unscented_kappa: float = 1.0
    pruning: bool = False                   # options_.particle.pruning
    cpf_weights: str = "amisanotristani"    # amisanotristani | murrayjonesparslow
    initialization: int = 1
    initial_state_prior_std: float = 0.1

    def must_resample(self, weights, sample_size):
        return ((self.resampling == "generic" and neff(weights) < self.resampling_threshold * sample_size)
                or self.resampling == "systematic")


@dataclass
class ReducedForm:
    """Second-order reduced form (rows: ``dr.restrict_var_list``, decision-rule order)."""
    ghx: np.ndarray
    ghu: np.ndarray
    ghxx: np.ndarray
    ghuu: np.ndarray
    ghxu: np.ndarray
    ghs2: np.ndarray
    steadystate: np.ndarray
    constant: np.ndarray
    state_variables_steady_state: np.ndarray
    mf0: np.ndarray                 # 0-based positions of the state variables
    mf1: np.ndarray                 # 0-based positions of the observed variables
    Q: np.ndarray
    H: np.ndarray
    StateVectorMean: np.ndarray = None
    StateVectorVariance: np.ndarray = None


# ====================================================================== state space
def _kron_cols(a, b):
    """Column-wise Kronecker products: column j is kron(a[:, j], b[:, j])."""
    return (a[:, None, :] * b[None, :, :]).reshape(a.shape[0] * b.shape[0], a.shape[1])


def local_state_space_iteration_2(yhat, epsilon, ghx, ghu, constant, ghxx, ghuu, ghxu, yhat_=None, ss=None):
    """Second-order transition (Dynare's local_state_space_iteration_2): with ``yhat_`` and ``ss`` the
    pruned version, returning ``(y, y_)`` where ``y_`` is the first-order (pruned) state plus ``ss``."""
    yq = yhat if yhat_ is None else yhat_
    y = (constant[:, None] + ghx @ yhat + ghu @ epsilon + 0.5 * (ghxx @ _kron_cols(yq, yq))
         + 0.5 * (ghuu @ _kron_cols(epsilon, epsilon)) + ghxu @ _kron_cols(yq, epsilon))
    if yhat_ is None:
        return y
    return y, ghx @ yhat_ + ghu @ epsilon + ss[:, None]


def iterate_law_of_motion(StateVectors, epsilon, rf: ReducedForm, pruning=False, StateVectors_=None):
    """Particles at t+1 (all restricted variables, levels); with ``pruning`` also the pruned states."""
    ssv = rf.state_variables_steady_state[:, None]
    yhat = StateVectors - ssv
    if yhat.shape[1] == 1 and epsilon.shape[1] > 1:
        yhat = np.repeat(yhat, epsilon.shape[1], axis=1)
    if pruning:
        if StateVectors_ is None:
            raise NonlinearFilterError("iterate_law_of_motion: with pruning, the components of the augmented "
                                       "pruned state space are an required input")
        yhat_ = StateVectors_ - ssv
        if yhat_.shape[1] == 1 and epsilon.shape[1] > 1:
            yhat_ = np.repeat(yhat_, epsilon.shape[1], axis=1)
        return local_state_space_iteration_2(yhat, epsilon, rf.ghx, rf.ghu, rf.constant, rf.ghxx, rf.ghuu,
                                             rf.ghxu, yhat_, rf.steadystate)
    return local_state_space_iteration_2(yhat, epsilon, rf.ghx, rf.ghu, rf.constant, rf.ghxx, rf.ghuu,
                                         rf.ghxu), None


def measurement_equations(StateVectors, rf: ReducedForm):
    q = rf.Q.shape[0]
    y, _ = iterate_law_of_motion(StateVectors, np.zeros((q, StateVectors.shape[1])), rf)
    return y[rf.mf1, :]


# ====================================================================== numerical helpers
def unscented_sigma_points(n, alpha=1.0, beta=2.0, kappa=1.0):
    """Nodes ((2n+1) x n) and the weights W_m, W_c of the unscented transform."""
    lam = alpha ** 2 * (n + kappa) - n
    nodes = np.hstack([np.zeros((n, 1)), np.sqrt(n + lam) * np.hstack([np.eye(n), -np.eye(n)])]).T
    w0 = lam / (n + lam)
    temp = np.ones(2 * n) / (2 * (n + lam))
    W_m = np.concatenate([[w0], temp])
    W_c = np.concatenate([[w0 + (1 - alpha ** 2 + beta)], temp])
    return nodes, W_m, W_c


def spherical_radial_sigma_points(n):
    """Nodes (2n x n) and the (scalar) weight of the spherical-radial cubature rule."""
    return (np.sqrt(n) * np.hstack([np.eye(n), -np.eye(n)])).T, 1.0 / (2 * n)


def _chol_upper(X):
    """MATLAB's ``chol(X)`` (upper factor, from the upper triangle); LinAlgError if not PD."""
    X = np.atleast_2d(np.asarray(X, float))
    if not np.all(np.isfinite(X)):
        raise np.linalg.LinAlgError("not finite")
    return linalg.cholesky(X, lower=False, check_finite=False)


def _chol_lower_from_lower(X):
    """MATLAB's ``chol(X, 'lower')`` (from the lower triangle)."""
    X = np.atleast_2d(np.asarray(X, float))
    if not np.all(np.isfinite(X)):
        raise np.linalg.LinAlgError("not finite")
    return linalg.cholesky(X, lower=True, check_finite=False)


def reduced_rank_cholesky(X):
    """Dynare's reduced_rank_cholesky: ``T`` with ``T'T = X`` (upper Cholesky factor, or a reduced-rank
    factor from the eigen decomposition when X is only semi-definite). Returns NaN (1 x 1) when X is not
    positive semi-definite, as Dynare."""
    X = np.atleast_2d(np.asarray(X, float))
    try:
        return _chol_upper(X)
    except np.linalg.LinAlgError:
        pass
    if not np.all(np.isfinite(X)):
        return np.full((1, 1), np.nan)
    n = X.shape[0]
    D, U = np.linalg.eigh(X, UPLO="U")
    imax = np.argmax(np.abs(U), axis=0)
    neg = U[imax, np.arange(n)] < 0
    U[:, neg] = -U[:, neg]
    tol = np.sqrt(np.spacing(np.max(D)) * len(D) * 10)
    t = np.abs(D) > tol
    D = D[t]
    if np.any(D < 0):
        return np.full((1, 1), np.nan)
    return np.diag(np.sqrt(D)) @ U[:, t].T


def qr2_r(A):
    """R factor of ``qr2(A, 0)`` (economy QR with a non-negative diagonal)."""
    R = np.linalg.qr(A, mode="r")
    neg = np.diag(R) < 0
    R[neg, :] = -R[neg, :]
    return R


def _is_triangular(S):
    return np.array_equal(S, np.tril(S)) or np.array_equal(S, np.triu(S))


def _mldivide(S, B):
    S = np.atleast_2d(S)
    if S.shape == (1, 1):
        return B / S[0, 0]
    if np.array_equal(S, np.tril(S)):
        return linalg.solve_triangular(S, B, lower=True, check_finite=False)
    if np.array_equal(S, np.triu(S)):
        return linalg.solve_triangular(S, B, lower=False, check_finite=False)
    return np.linalg.solve(S, B)


def probability2(mu, S, X):
    """Gaussian densities N(mu, S S') of the columns of X (Dynare's probability2, with its 1e-99)."""
    X = np.atleast_2d(X)
    dim = X.shape[0]
    S = np.atleast_2d(np.asarray(S, float))
    mu = np.asarray(mu, float).reshape(-1, 1) if np.ndim(mu) == 1 else np.asarray(mu, float)
    foo = _mldivide(S, X - mu)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return np.exp(-0.5 * np.sum(foo * foo, axis=0)) / np.abs((2 * np.pi) ** (dim / 2) * np.prod(np.diag(S))) + 1e-99


def neff(w):
    return 1.0 / np.dot(w, w)


def _rcond(H):
    H = np.atleast_2d(H)
    if not np.all(np.isfinite(H)):
        return 0.0
    try:
        return 1.0 / np.linalg.cond(H, 1)
    except np.linalg.LinAlgError:
        return 0.0


# ====================================================================== resampling
def traditional_resampling(particles, weights, noise):
    """Kitagawa's systematic resampling (scalar ``noise``) or Dynare's "stratified" variant (one
    uniform per particle). ``particles=None`` returns the 0-based indices."""
    weights = np.asarray(weights, float).ravel()
    noise = np.asarray(noise, float).ravel()
    N = len(weights)
    c = np.cumsum(weights)
    if noise.size == 1:
        randvec = (np.arange(N) + noise[0]) / N
        indx = np.searchsorted(c, randvec, side="left")   # first j with randvec(i) <= c(j)
    elif noise.size == N:
        # Dynare: randvec = fliplr(cumprod(noise.^(1./(N:-1:1)))) with noise a column: the power
        # broadcasts to N x N and the first N elements are the cumulative products of noise.
        randvec = np.cumprod(noise)
        indx = np.searchsorted(c, randvec, side="left")   # sum(randvec(i) > c), 0-based
    else:
        raise NonlinearFilterError("particle::resampling: Unknown method! The size of the second argument is wrong.")
    if np.any(indx >= N):
        raise NonlinearFilterError("resampling: index out of bound (the cumulated weights do not reach the "
                                   "resampling threshold; Dynare stops here too)")
    if particles is None:
        return indx
    return particles[indx, :]


def univariate_smooth_resampling(weights, particles, number_of_new_particles, u):
    M = len(particles)
    lt = np.concatenate([[0.5 * weights[0]], 0.5 * (weights[:-1] + weights[1:]), [0.5 * weights[-1]]])
    lb = np.cumsum(lt)
    uj = (np.arange(number_of_new_particles) + u) / number_of_new_particles
    rj = np.searchsorted(lb, uj, side="left")          # 1-based index of the last lambda_bar < u_j
    if np.any(rj > M):
        raise NonlinearFilterError("univariate_smooth_resampling: index out of bound")
    out = np.empty(number_of_new_particles)
    first, last = rj == 0, rj == M
    mid = ~(first | last)
    out[first] = particles[0]
    out[last] = particles[M - 1]
    r = rj[mid]
    ustar = (uj[mid] - lb[r - 1]) / lt[r]
    out[mid] = (particles[r] - particles[r - 1]) * ustar + particles[r - 1]
    return out


def multivariate_smooth_resampling(particles, weights, rng, symmetric=None):
    """``multivariate_smooth_resampling.m``; ``symmetric`` forces the choice of eigensolver (default:
    the symmetric one if the product is exactly symmetric, as MATLAB/Octave)."""
    N = len(weights)
    s = particles.shape[1]
    A = particles.T @ (particles * (1.0 / N))
    if (np.array_equal(A, A.T) if symmetric is None else symmetric):
        # MATLAB/Octave's eig uses the symmetric solver (ascending eigenvalues) only for an exactly
        # symmetric matrix, the general one otherwise; which one Dynare gets depends on the rounding of
        # this product (see the module docstring)
        D, P = linalg.eigh(A, driver="ev")
    else:
        D, P = np.linalg.eig(A)
    if np.iscomplexobj(D) or np.iscomplexobj(P):
        if np.max(np.abs(np.imag(D))) > 0 or np.max(np.abs(np.imag(P))) > 0:
            raise NonlinearFilterError("multivariate_smooth_resampling: complex eigenvalues")
        D, P = np.real(D), np.real(P)
    vectors = P * np.sqrt(D)[None, :]
    orth = (particles @ vectors) / D[None, :]
    new = np.zeros((N, s))
    for j in range(s):
        order = np.argsort(orth[:, j], kind="stable")
        new[:, j] = univariate_smooth_resampling(weights[order], orth[order, j], N, rng.rand(1, 1)[0, 0])
    return new @ vectors.T


def resample(particles, weights, opts: ParticleOptions, rng):
    """``resample.m``: particles are rows (N x s); ``particles=None`` returns indices (0-based)."""
    weights = np.asarray(weights, float).ravel()
    m = opts.resampling_method
    if m == "kitagawa":
        return traditional_resampling(particles, weights, rng.rand(1, 1))
    if m == "stratified":
        return traditional_resampling(particles, weights, rng.rand(len(weights), 1))
    if m == "smooth":
        if particles is None:
            raise NonlinearFilterError("Particle = 0 is incompatible with the resampling_method=smooth method!")
        return multivariate_smooth_resampling(particles, weights, rng)
    raise NonlinearFilterError("Unknown sampling method!")


# ====================================================================== particle filters
def _H_or_zero(rf):
    H = np.atleast_2d(np.asarray(rf.H, float)) if rf.H is not None and np.size(rf.H) else np.zeros((1, 1))
    return H


def _lower_chol(X, what):
    try:
        return _chol_upper(X).T
    except np.linalg.LinAlgError:
        raise NonlinearFilterError(f"chol: {what} is not positive definite (Dynare stops with an error)") from None


def sequential_importance_particle_filter(rf: ReducedForm, Y, start, opts: ParticleOptions, rng):
    """``sequential_importance_particle_filter.m`` (``filter_algorithm = sis``)."""
    start = 1 if start is None else int(start)
    T = Y.shape[1]
    mf0, mf1 = rf.mf0, rf.mf1
    n, p, q = len(mf0), len(mf1), rf.Q.shape[0]
    N = int(opts.number_of_particles)
    H = _H_or_zero(rf)
    lik = np.full(T, np.nan)
    if _rcond(H) < 1e-12:
        return np.nan, lik
    with np.errstate(divide="ignore"):
        const_lik = np.log(2 * np.pi) * p + np.log(np.linalg.det(H))
    S0 = _lower_chol(rf.StateVectorVariance, "the initial state variance")
    Qc = _lower_chol(rf.Q, "Q")
    weights = np.ones(N) / N
    StateVectors = S0 @ rng.randn(S0.shape[1], N) + rf.StateVectorMean[:, None]
    pruning = bool(opts.pruning)
    StateVectors_ = StateVectors.copy() if pruning else None
    for t in range(T):
        eps = Qc @ rng.randn(q, N)
        tmp, tmp_ = iterate_law_of_motion(StateVectors, eps, rf, pruning, StateVectors_)
        PE = Y[:, t:t + 1] - tmp[mf1, :]
        lnw = -0.5 * (const_lik + np.sum(PE * _solve(H, PE), axis=0))
        dfac = np.max(lnw)
        wtilde = weights * np.exp(lnw - dfac)
        lik[t] = np.log(np.sum(wtilde)) + dfac
        weights = wtilde / np.sum(wtilde)
        if opts.must_resample(weights, T):
            if pruning:
                temp = resample(np.hstack([tmp[mf0, :].T, tmp_[mf0, :].T]), weights, opts, rng)
                StateVectors, StateVectors_ = temp[:, :n].T, temp[:, n:].T
            else:
                StateVectors = resample(tmp[mf0, :].T, weights, opts, rng).T
            weights = np.ones(N) / N
        elif opts.resampling == "none":
            StateVectors = tmp[mf0, :]
            if pruning:
                StateVectors_ = tmp_[mf0, :]
    return -np.sum(lik[start - 1:]), lik


def _solve(H, B):
    if H.shape == (1, 1):
        return B / H[0, 0]
    return np.linalg.solve(H, B)


def auxiliary_particle_filter(rf: ReducedForm, Y, start, opts: ParticleOptions, rng):
    """``auxiliary_particle_filter.m`` (``filter_algorithm = apf``)."""
    start = 1 if start is None else int(start)
    if opts.resampling_method == "smooth":
        raise NonlinearFilterError("auxiliary_particle_filter: resampling_method=smooth is not supported.")
    T = Y.shape[1]
    mf0, mf1 = rf.mf0, rf.mf1
    n, p, q = len(mf0), len(mf1), rf.Q.shape[0]
    N = int(opts.number_of_particles)
    S0 = _lower_chol(rf.StateVectorVariance, "the initial state variance")
    Qc = _lower_chol(rf.Q, "Q")
    lik = np.full(T, np.nan)
    H = _H_or_zero(rf)
    if _rcond(H) < 1e-12:
        return np.nan, lik
    with np.errstate(divide="ignore"):
        const_lik = np.log(2 * np.pi) * p + np.log(np.linalg.det(H))
    weights = np.ones(N) / N
    StateVectors = S0 @ rng.randn(S0.shape[1], N) + rf.StateVectorMean[:, None]
    pruning = bool(opts.pruning)
    StateVectors_ = StateVectors.copy() if pruning else None
    ddl = 3.0
    cst = np.exp(gammaln((ddl + 1) / 2) - gammaln(ddl / 2))
    for t in range(T):
        tmp, _ = iterate_law_of_motion(StateVectors, np.zeros((q, N)), rf, pruning, StateVectors_)
        PE = Y[:, t:t + 1] - tmp[mf1, :]
        z = np.sum(PE * _solve(H, PE), axis=0)
        tau_tilde = weights * (cst / (np.sqrt(ddl * np.pi) * (1 + (z ** 2) / ddl) ** ((ddl + 1) / 2)) + 1e-99)
        tau_tilde = tau_tilde / np.sum(tau_tilde)
        indx = resample(None, tau_tilde, opts, rng)
        weights_stage_1 = weights[indx] / tau_tilde[indx]
        eps = Qc @ rng.randn(q, N)
        tmp, tmp_ = iterate_law_of_motion(StateVectors[:, indx], eps, rf, pruning,
                                          StateVectors_[:, indx] if pruning else None)
        if pruning:
            StateVectors_ = tmp_[mf0, :]
        StateVectors = tmp[mf0, :]
        PE = Y[:, t:t + 1] - tmp[mf1, :]
        with np.errstate(over="ignore"):
            weights_stage_2 = weights_stage_1 * (np.exp(-0.5 * (const_lik + np.sum(PE * _solve(H, PE), axis=0))) + 1e-99)
        lik[t] = np.log(np.mean(weights_stage_2))
        weights = weights_stage_2 / np.sum(weights_stage_2)
        if opts.must_resample(weights, T):
            if pruning:
                temp = resample(np.hstack([StateVectors.T, StateVectors_.T]), weights, opts, rng)
                StateVectors, StateVectors_ = temp[:, :n].T, temp[:, n:].T
            else:
                StateVectors = resample(StateVectors.T, weights, opts, rng).T
            weights = np.ones(N) / N
    return -np.sum(lik[start - 1:]), lik


def _batched_lower_chol(A):
    """Lower Cholesky factors of a stack of matrices, from their lower triangles; None if one fails."""
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return None


def _batched_probability2_zero_mean(S, E):
    """probability2(0, S_i, E_i) for stacks S (N x p x p, lower triangular) and E (N x p)."""
    p = E.shape[1]
    foo = linalg.solve_triangular(S, E[:, :, None], lower=True, check_finite=False)[:, :, 0] \
        if S.ndim == 2 else np.stack([linalg.solve_triangular(S[i], E[i], lower=True, check_finite=False)
                                      for i in range(S.shape[0])])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return np.exp(-0.5 * np.sum(foo * foo, axis=1)) / np.abs((2 * np.pi) ** (p / 2) *
                                                                  np.prod(np.diagonal(S, axis1=-2, axis2=-1), axis=-1)) + 1e-99


def _batched_lower_solve(S, B):
    """Solve S_i x_i = b_i for lower triangular S_i (stack) and vectors/matrices B_i."""
    out = np.empty_like(B)
    for i in range(S.shape[0]):
        out[i] = linalg.solve_triangular(S[i], B[i], lower=True, check_finite=False)
    return out


def conditional_particle_filter(rf: ReducedForm, Y, start, opts: ParticleOptions, rng, model_pruning=False):
    """``conditional_particle_filter.m`` (``filter_algorithm = cpf``), vectorised over particles; the
    random numbers are drawn in Dynare's order (particle by particle)."""
    if opts.pruning:
        raise NonlinearFilterError("conditional_particle_filter: pruning is not yet implemented.")
    if model_pruning:
        raise NonlinearFilterError("iterate_law_of_motion: with pruning, the components of the augmented pruned "
                                   "state space are an required input (Dynare's cpf passes options_.pruning "
                                   "without the pruned states)")
    start = 1 if start is None else int(start)
    T = Y.shape[1]
    N = int(opts.number_of_particles)
    mf0, mf1 = rf.mf0, rf.mf1
    ns, p, q = len(mf0), len(mf1), rf.Q.shape[0]
    H = _H_or_zero(rf)
    Hc = _lower_chol(H, "H")
    S0 = _lower_chol(rf.StateVectorVariance, "the initial state variance")
    Qc = _lower_chol(rf.Q, "Q")
    prop = opts.proposal_approximation
    if prop == "montecarlo":
        if N % 10:
            raise NonlinearFilterError("cpf with proposal_approximation=montecarlo needs number_of_particles "
                                       "divisible by 10 (randn(number_of_particles/10, ...) in Dynare)")
        npts = N // 10
        weights = 1.0 / npts
        weights_c = weights
    elif prop == "cubature":
        nodes, weights = spherical_radial_sigma_points(q)
        weights_c = weights
        npts = nodes.shape[0]
    elif prop == "unscented":
        nodes, weights, weights_c = unscented_sigma_points(q, opts.unscented_alpha, opts.unscented_beta,
                                                           opts.unscented_kappa)
        npts = nodes.shape[0]
    else:
        raise NonlinearFilterError("Estimation: This approximation for the proposal is not implemented or unknown!")
    mjp = opts.cpf_weights == "murrayjonesparslow"
    if mjp and prop != "unscented":
        raise NotImplementedError("cpf_weights=murrayjonesparslow with proposal_approximation=cubature or "
                                  "montecarlo: Dynare 7.2 uses an (points x points) matrix as the predicted state "
                                  "variance there (dState*dState' with dState points x states); not reproduced")
    lik = np.full(T, np.nan)
    StateParticles = S0 @ rng.randn(S0.shape[1], N) + rf.StateVectorMean[:, None]
    SampleWeights = np.ones(N) / N
    sigma_branch = prop in ("cubature", "montecarlo")
    for t in range(T):
        y = Y[:, t]
        if prop == "montecarlo":
            draws = rng.randn(npts * q + ns, N)           # per particle: nodes, then the proposal draw
            nodes_i = draws[:npts * q, :].T.reshape(N, q, npts).transpose(0, 2, 1)   # (N, npts, q), column-major
            eps = np.einsum("ab,ikb->iak", Qc, nodes_i)    # (N, q, npts)
            prop_draw = draws[npts * q:, :]
        else:
            eps = np.broadcast_to((Qc @ nodes.T)[None], (N, q, npts))
            prop_draw = None
        X = np.repeat(StateParticles, npts, axis=1)       # particle-major columns
        E = eps.transpose(1, 0, 2).reshape(q, N * npts)
        tmp, _ = iterate_law_of_motion(X, E, rf)
        tS = tmp[mf0, :].reshape(ns, N, npts).transpose(1, 0, 2)    # (N, ns, npts)
        tO = tmp[mf1, :].reshape(p, N, npts).transpose(1, 0, 2)     # (N, p, npts)
        if np.ndim(weights) == 0:
            PSM = np.sum(tS * weights, axis=2)
            POM = np.sum(tO * weights, axis=2)
        else:
            PSM = tS @ weights
            POM = tO @ weights
        Err = y[None, :] - POM                                       # (N, p)
        if sigma_branch:
            sw = np.sqrt(weights)
            dState = (tS - PSM[:, :, None]).transpose(0, 2, 1) * sw    # (N, npts, ns)
            dObs = (tO - POM[:, :, None]).transpose(0, 2, 1) * sw      # (N, npts, p)
            top = np.concatenate([dObs, dState], axis=2)
            bottom = np.broadcast_to(np.hstack([Hc, np.zeros((p, ns))])[None], (N, p, p + ns))
            big = np.concatenate([top, bottom], axis=1)
            R = np.linalg.qr(big, mode="r")
            d = np.diagonal(R, axis1=1, axis2=2)
            R = R * np.where(d < 0, -1.0, 1.0)[:, :, None]
            mat = R.transpose(0, 2, 1)
            POVSR = mat[:, :p, :p]
            COSSR = mat[:, p:p + ns, :p]
            SVVSR = mat[:, p:p + ns, p:p + ns]
            # StateVectorMean = PSM + (COSSR / POVSR) * Err
            gain = np.linalg.solve(POVSR.transpose(0, 2, 1), COSSR.transpose(0, 2, 1)).transpose(0, 2, 1)
            SVM = PSM + np.einsum("ijk,ik->ij", gain, Err)
            W = SampleWeights * _batched_probability2_zero_mean(POVSR, Err)
        else:
            wc = np.asarray(weights_c)
            dState = tS - PSM[:, :, None]
            dObs = tO - POM[:, :, None]
            PSV = np.einsum("iak,k,ibk->iab", dState, wc, dState)
            POV = np.einsum("iak,k,ibk->iab", dObs, wc, dObs) + H[None]
            PSOC = np.einsum("iak,k,ibk->iab", dState, wc, dObs)
            K = np.linalg.solve(POV.transpose(0, 2, 1), PSOC.transpose(0, 2, 1)).transpose(0, 2, 1)
            SVM = PSM + np.einsum("ijk,ik->ij", K, Err)
            SVV = PSV - K @ POV @ K.transpose(0, 2, 1)
            SVV = 0.5 * (SVV + SVV.transpose(0, 2, 1))
            SVVSR = _batched_lower_chol(SVV)
            if SVVSR is None:
                lik[t] = -np.inf
                return -np.inf, lik
            if not mjp:
                POVc = _batched_lower_chol(_upper_to_sym(POV))      # chol(POV)' reads the upper triangle
                if POVc is None:
                    raise NonlinearFilterError("chol: the predicted observation variance is not positive definite")
                W = SampleWeights * _batched_probability2_zero_mean(POVc, Err)
        if prop_draw is None:
            prop_draw = rng.randn(ns, N)
        Prop = np.einsum("iab,bi->ai", SVVSR, prop_draw) + SVM.T     # (ns, N)
        if mjp:
            PSVs = 0.5 * (PSV + PSV.transpose(0, 2, 1))
            PSVc = _batched_lower_chol(PSVs)
            if PSVc is None:
                lik[t] = -np.inf
                return -np.inf, lik
            Prior = _batched_probability2_zero_mean(PSVc, Prop.T - PSM)
            Post = _batched_probability2_zero_mean(SVVSR, Prop.T - SVM)
            Like = probability2(y, Hc, measurement_equations(Prop, rf))
            W = SampleWeights * Like * (Prior / Post)
        StateParticles = Prop
        SampleWeights = W
        s = np.sum(SampleWeights)
        lik[t] = np.log(s)
        SampleWeights = SampleWeights / s
        if opts.must_resample(SampleWeights, T):
            StateParticles = resample(StateParticles.T, SampleWeights, opts, rng).T
            SampleWeights = np.ones(N) / N
    return -np.sum(lik[start - 1:]), lik


def _upper_to_sym(A):
    """Symmetric matrices made of the upper triangles of a stack (what MATLAB's chol(A) reads)."""
    return np.triu(A) + np.triu(A, 1).transpose(0, 2, 1)


# ====================================================================== Gaussian filters
def _proposal_nodes(opts: ParticleOptions, dim, rng, N):
    prop = opts.proposal_approximation
    if prop == "montecarlo":
        nodes = rng.randn(N, dim)
        return nodes, 1.0 / N, 1.0 / N
    if prop == "cubature":
        nodes, w = spherical_radial_sigma_points(dim)
        return nodes, w, w
    if prop == "unscented":
        return unscented_sigma_points(dim, opts.unscented_alpha, opts.unscented_beta, opts.unscented_kappa)
    raise NonlinearFilterError("This approximation for the proposal is not implemented or unknown!")


def _weighted_mean(X, w):
    return X @ w if np.ndim(w) else np.sum(X * w, axis=1)


def _sigma_propagate(rf, mean, S, Qc, nodes):
    n, q = len(rf.mf0), rf.Q.shape[0]
    if S.shape != (n, n) or Qc.shape != (q, q):
        raise NonlinearFilterError(f"vertical dimensions mismatch ({S.shape[0]}x{S.shape[1]} vs "
                                   f"{q}x{n}): the state (or shock) covariance matrix has reduced rank "
                                   "(Dynare stops with this error)")
    xbar = np.concatenate([mean, np.zeros(q)])
    sqr_Px = linalg.block_diag(S, Qc)
    sp = xbar[:, None] + sqr_Px @ nodes.T
    tmp, _ = iterate_law_of_motion(sp[:n, :], sp[n:, :], rf)
    return tmp


def gaussian_filter_bank(rf, obs, mean, S, Qc, Hc, H, opts, rng):
    """One step of ``gaussian_filter_bank.m``: predicted state mean and variance square root, updated
    state mean and variance square root."""
    mf0, mf1 = rf.mf0, rf.mf1
    n, p, q = len(mf0), len(mf1), rf.Q.shape[0]
    nodes, w, wc = _proposal_nodes(opts, n + q, rng, int(opts.number_of_particles))
    tmp = _sigma_propagate(rf, mean, S, Qc, nodes)
    PSM = _weighted_mean(tmp[mf0, :], w)
    POM = _weighted_mean(tmp[mf1, :], w)
    if opts.proposal_approximation in ("cubature", "montecarlo"):
        dState = (tmp[mf0, :] - PSM[:, None]).T * np.sqrt(w)
        dObs = (tmp[mf1, :] - POM[:, None]).T * np.sqrt(w)
        PSVSR = _lower_chol(dState.T @ dState, "the predicted state variance")
        if Hc.shape != (p, p):
            raise NonlinearFilterError("dimensions mismatch: H has reduced rank")
        big = np.vstack([np.hstack([dObs, dState]), np.hstack([Hc, np.zeros((p, n))])])
        mat = qr2_r(big).T
        POVSR = mat[:p, :p]
        COSSR = mat[p:p + n, :p]
        SVVSR = mat[p:p + n, p:p + n]
        PE = obs - POM
        SVM = PSM + np.linalg.solve(POVSR.T, COSSR.T).T @ PE
    else:
        dState = tmp[mf0, :] - PSM[:, None]
        dObs = tmp[mf1, :] - POM[:, None]
        PSV = (dState * wc) @ dState.T
        POV = (dObs * wc) @ dObs.T + H
        PSOC = (dState * wc) @ dObs.T
        PSVSR = _lower_chol(PSV, "the predicted state variance")
        PE = obs - POM
        K = np.linalg.solve(POV.T, PSOC.T).T
        SVM = PSM + K @ PE
        SVV = PSV - K @ POV @ K.T
        SVV = 0.5 * (SVV + SVV.T)
        SVVSR = reduced_rank_cholesky(SVV).T
    return PSM, PSVSR, SVM, SVVSR


def gaussian_filter(rf: ReducedForm, Y, start, opts: ParticleOptions, rng):
    """``gaussian_filter.m`` (``filter_algorithm = gf``): Gaussian proposal from a sigma-point (or Monte
    Carlo) Kalman step, distribution of the particles approximated by sigma points or draws."""
    if opts.pruning:
        raise NonlinearFilterError("gaussian_filter: pruning is not yet implemented.")
    start = 1 if start is None else int(start)
    T = Y.shape[1]
    mf0 = rf.mf0
    n = len(mf0)
    N = int(opts.number_of_particles)
    da = opts.distribution_approximation
    if da == "cubature":
        nodes2, weights2 = spherical_radial_sigma_points(n)
    elif da == "unscented":
        nodes2, weights2, _ = unscented_sigma_points(n, opts.unscented_alpha, opts.unscented_beta, opts.unscented_kappa)
    elif da != "montecarlo":
        raise NonlinearFilterError("This approximation for the distribution is unknown!")
    if rf.H is None or np.size(rf.H) == 0:
        H, Hc = np.zeros((1, 1)), np.zeros((1, 1))
    else:
        H = np.atleast_2d(np.asarray(rf.H, float))
        if np.any(H != np.diag(np.diag(H))):
            raise NotImplementedError("gaussian_filter with correlated measurement errors: Dynare takes sqrt(H) "
                                      "element by element (complex numbers); not reproduced")
        Hc = reduced_rank_cholesky(H).T
    mean = np.asarray(rf.StateVectorMean, float).copy()
    S = reduced_rank_cholesky(rf.StateVectorVariance).T
    Qc = reduced_rank_cholesky(rf.Q).T
    lik = np.full(T, np.nan)
    sqrtH = np.sqrt(H)
    for t in range(T):
        obs = Y[:, t]
        PSM, PSVSR, mean, S = gaussian_filter_bank(rf, obs, mean, S, Qc, Hc, H, opts, rng)
        if S.shape != (n, n) or not np.all(np.isfinite(S)):
            raise NonlinearFilterError("gaussian_filter: the updated state variance has reduced rank or is not "
                                       "positive semi-definite (Dynare stops with a dimension error)")
        if da == "montecarlo":
            SP = S @ rng.randn(S.shape[1], N) + mean[:, None]
        else:
            SP = mean[:, None] + S @ nodes2.T
        proposal = probability2(mean, S, SP)
        prior = probability2(PSM, PSVSR, SP)
        like = probability2(obs, sqrtH, measurement_equations(SP, rf))
        IW = like * prior / proposal
        SW = IW / N if da == "montecarlo" else weights2 * IW
        SW = SW + 1e-6
        s = np.sum(SW)
        lik[t] = np.log(s)
        SW = SW / s
        if da == "montecarlo" and opts.must_resample(SW, T):
            SP = resample(SP.T, SW, opts, rng).T
            SW = np.ones(N) / N
        mean = SP @ SW
        temp = SP - mean[:, None]
        S = reduced_rank_cholesky((temp * SW) @ temp.T).T
        if not np.all(np.isfinite(S)):
            raise NonlinearFilterError("reduced_rank_cholesky:: Input matrix is not semidefinite positive!")
    return -np.sum(lik[start - 1:]), lik


def nonlinear_kalman_filter(rf: ReducedForm, Y, start, opts: ParticleOptions, rng):
    """``nonlinear_kalman_filter.m`` (``filter_algorithm = nlkf``): sigma-point (or Monte Carlo) Kalman
    filter; the density of period t is the sigma-point mixture of N(y_t; y_i, H)."""
    if opts.pruning:
        raise NonlinearFilterError("nonlinear_kalman_filter: pruning is not yet implemented.")
    start = 1 if start is None else int(start)
    mf0, mf1 = rf.mf0, rf.mf1
    T = Y.shape[1]
    n, p, q = len(mf0), len(mf1), rf.Q.shape[0]
    nodes, w, wc = _proposal_nodes(opts, n + q, rng, int(opts.number_of_particles))
    H = _H_or_zero(rf)
    Hc = _lower_chol(H, "H")
    Qc = _lower_chol(rf.Q, "Q")
    mean = np.asarray(rf.StateVectorMean, float).copy()
    S = _lower_chol(rf.StateVectorVariance, "the initial state variance")
    lik = np.full(T, np.nan)
    for t in range(T):
        tmp = _sigma_propagate(rf, mean, S, Qc, nodes)
        PSM = _weighted_mean(tmp[mf0, :], w)
        POM = _weighted_mean(tmp[mf1, :], w)
        if opts.proposal_approximation in ("cubature", "montecarlo"):
            dState = (tmp[mf0, :] - PSM[:, None]).T * np.sqrt(w)
            dObs = (tmp[mf1, :] - POM[:, None]).T * np.sqrt(w)
            big = np.vstack([np.hstack([dObs, dState]), np.hstack([Hc, np.zeros((p, n))])])
            mat = qr2_r(big).T
            POVSR = mat[:p, :p]
            COSSR = mat[p:p + n, :p]
            S = mat[p:p + n, p:p + n]
            mean = PSM + np.linalg.solve(POVSR.T, COSSR.T).T @ (Y[:, t] - POM)
        else:
            dState = tmp[mf0, :] - PSM[:, None]
            dObs = tmp[mf1, :] - POM[:, None]
            PSV = (dState * wc) @ dState.T
            POV = (dObs * wc) @ dObs.T + H
            PSOC = (dState * wc) @ dObs.T
            K = np.linalg.solve(POV.T, PSOC.T).T
            mean = PSM + K @ (Y[:, t] - POM)
            SVV = PSV - K @ POV @ K.T
            try:
                S = _chol_lower_from_lower(SVV)
                _chol_lower_from_lower(POV)
            except np.linalg.LinAlgError:
                lik[t] = -np.inf
                return -np.inf, lik
        s = np.sum(probability2(Y[:, t], Hc, tmp[mf1, :]) * w)
        if s < 0:                     # log of a negative number: Dynare's LIK is complex (info 46)
            return complex(np.nan, np.pi), lik
        with np.errstate(divide="ignore"):
            lik[t] = np.log(s)
    return -np.sum(lik[start - 1:]), lik


#: Dynare's filter_algorithm names
FILTERS = {"sis": sequential_importance_particle_filter, "apf": auxiliary_particle_filter,
           "cpf": conditional_particle_filter, "gf": gaussian_filter, "nlkf": nonlinear_kalman_filter}


def warn_mex_pruning(q, n):
    if q > n:
        warnings.warn("pruning with more shocks than state variables: Dynare's MEX "
                      "local_state_space_iteration_2 computes the pruned first-order part wrongly in this case "
                      "(its loops assume q <= n); pydyn uses the formula of Dynare's MATLAB version", stacklevel=3)
