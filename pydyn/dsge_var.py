# This file is part of pydyn.
#
# It is a Python port of parts of Dynare 7.2 (Copyright (C) 2006-2026 Dynare Team,
# https://www.dynare.org), modified in 2026: matlab/estimation/dsge_var_likelihood.m,
# matlab/estimation/var_sample_moments.m, matlab/estimation/check_dsge_var_model.m and the DSGE-VAR
# set-up of matlab/estimation/dynare_estimation_1.m and dynare_estimation_init.m.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""DSGE-VAR (Del Negro and Schorfheide, 2004): Dynare's ``dsge_var`` and ``dsge_varlag`` options.

The likelihood of a VAR(``dsge_varlag``) with a constant whose prior is centred on the DSGE model,
with weight ``dsge_prior_weight`` (a model parameter the preprocessor adds; calibrated with
``dsge_var=VALUE``, estimated with ``dsge_var`` and an ``estimated_params`` line). As in Dynare:

* the sample moments use ``dsge_varlag`` observations before ``first_obs`` as initial conditions, and
  the sample size ``T`` in the formulas is ``nobs + dsge_varlag`` (Dynare's ``dataset_.nobs`` after
  ``makedataset`` added the initial conditions), although the VAR is fitted on ``nobs`` observations;
* measurement errors, correlated shocks, observation trends, missing observations and ``presample``
  are refused; the VAR always has a constant (``dynare_estimation_init.m`` sets ``noconstant = 0``).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def weight_value(model, params):
    if "dsge_prior_weight" not in model._ip:
        raise ValueError("dsge_var: the model has no dsge_prior_weight parameter (the preprocessor adds it "
                         "when the estimation command has the dsge_var option)")
    return float(params[model._ip["dsge_prior_weight"]])


def sample_moments(data, nlag):
    """``var_sample_moments.m`` with a constant: ``(mYY, mYX, mXX)`` from ``data`` (T_all, n), whose
    first ``nlag`` rows are the initial conditions."""
    data = np.asarray(data, float)
    Y = data[nlag:]
    N, n = Y.shape
    X = np.ones((N, n * nlag + 1))
    for lag in range(1, nlag + 1):
        X[:, (lag - 1) * n:lag * n] = data[nlag - lag:nlag - lag + N]
    return Y.T @ Y, Y.T @ X, X.T @ X


def minus_loglik(T, R, Q, mf, constant, lam, nobs, nlag, moments, qz_criterium, complex_threshold=1e-15):
    """Minus the DSGE-VAR log likelihood (``dsge_var_likelihood.m`` without the prior). Returns
    ``(lik, code, penalty)``: ``code`` is Dynare's info(1) when the likelihood is not defined
    (51: weight below (k + n) / T, 52 or 53: a covariance matrix is not positive definite), else 0."""
    from .kalman import lyapunov_symm
    mYY, mYX, mXX = moments
    n = len(mf)
    k = n * nlag + 1
    TT = nobs
    if lam < (k + n) / TT:
        return np.inf, 51, abs(TT * lam - (k + n))
    c = np.asarray(constant, float)
    cc = np.outer(c, c)
    tmp0 = lyapunov_symm(T, R @ Q @ R.T, qz_criterium, complex_threshold)
    acov = [tmp0[np.ix_(mf, mf)] + cc]
    for _ in range(nlag):
        tmp0 = T @ tmp0
        acov.append(tmp0[np.ix_(mf, mf)] + cc)
    GYX = np.zeros((n, k))
    for i in range(nlag):
        GYX[:, i * n:(i + 1) * n] = acov[i + 1]
    GYX[:, -1] = c
    GXX = np.kron(np.eye(nlag), acov[0])
    for i in range(1, nlag):
        GXX = GXX + np.kron(np.diag(np.ones(nlag - i), i), acov[i]) + np.kron(np.diag(np.ones(nlag - i), -i), acov[i].T)
    GXX = np.block([[GXX, np.kron(np.ones((nlag, 1)), c[:, None])], [np.kron(np.ones((1, nlag)), c[None, :]), np.ones((1, 1))]])
    GYY = acov[0]
    iGXX = np.linalg.inv(GXX)
    PHI_star = iGXX @ GYX.T
    SIGMA_u_star = GYY - GYX @ PHI_star
    ok, pen = _ispd(SIGMA_u_star)
    if not ok:
        return np.inf, 53, pen
    if np.isfinite(lam):
        t0 = lam * TT * acov[0] + mYY
        t1 = lam * TT * GYX + mYX
        t2 = np.linalg.inv(lam * TT * GXX + mXX)
        S = t0 - t1 @ t2 @ t1.T
        ok, pen = _ispd(S)
        if not ok:
            return np.inf, 52, pen
        S = S / (TT * (1 + lam))
        j = np.arange(1, n + 1)
        prodlng1 = np.sum(gammaln(0.5 * ((1 + lam) * TT - k + 1 - j)))
        prodlng2 = np.sum(gammaln(0.5 * (lam * TT - k + 1 - j)))
        ld = lambda A: np.log(np.linalg.det(A))  # noqa: E731
        lik = (0.5 * n * ld(lam * TT * GXX + mXX)
               + 0.5 * ((lam + 1) * TT - k) * ld((lam + 1) * TT * S)
               - 0.5 * n * ld(lam * TT * GXX)
               - 0.5 * (lam * TT - k) * ld(lam * TT * SIGMA_u_star)
               + 0.5 * n * TT * np.log(2 * np.pi)
               - 0.5 * np.log(2) * n * ((lam + 1) * TT - k)
               + 0.5 * np.log(2) * n * (lam * TT - k)
               - prodlng1 + prodlng2)
    else:
        iS = np.linalg.inv(SIGMA_u_star)
        lik = TT * (np.log(np.linalg.det(SIGMA_u_star)) + n * np.log(2 * np.pi)
                    + np.trace(iS @ (mYY - (mYX @ PHI_star).T - mYX @ PHI_star + PHI_star.T @ mXX @ PHI_star) / TT))
        lik = 0.5 * lik
    if np.isinf(lik):
        return np.inf, 50, 0.1
    if np.isnan(lik):
        return np.inf, 45, 0.1
    if np.iscomplexobj(lik) and np.imag(lik) != 0:
        return np.inf, 46, 0.1
    return float(np.real(lik)), 0, 0.0


def _ispd(A):
    """``ispd.m``: (is positive definite, penalty = sum of the absolute negative eigenvalues)."""
    try:
        np.linalg.cholesky(A)
        return True, 0.0
    except np.linalg.LinAlgError:
        ev = np.linalg.eigvalsh((A + A.T) / 2)
        return False, float(np.sum(np.abs(ev[ev < 0])))


def check_model(est):
    """``check_dsge_var_model.m`` and the other refusals of Dynare."""
    m, o = est.model, est.options
    if getattr(est, "_groups", {}).get("vn") or getattr(est, "_groups", {}).get("cn") or m.measurement_errors or getattr(m, "me_cov", None):
        raise ValueError("Estimation::DsgeVarLikelihood: Measurement errors are not allowed!")
    S = m.Sigma_e
    if getattr(est, "_groups", {}).get("cx") or (S.shape[0] > 1 and np.any(np.tril(S, -1) != 0)):
        raise ValueError("Estimation::DsgeVarLikelihood: Structural innovations cannot be correlated using "
                         "Dynare's interface! Introduce the correlations in the model block instead.")
    if getattr(est, "_trend_expr", None) is not None:
        raise ValueError("Estimation::DsgeVarLikelihood: Linear trend is not yet implemented!")
    if int(o["presample"]) != 0:
        raise ValueError("DSGE-VAR does not support the presample option.")
    if est.missing:
        raise ValueError("Estimation::DsgeVarLikelihood: I cannot estimate a DSGE-VAR model with missing observations!")
    if o["prefilter"]:
        raise NotImplementedError("DSGE-VAR with prefilter is not supported")
    if int(o["lik_init"]) != 1:
        raise NotImplementedError("DSGE-VAR needs a stationary model (lik_init=1)")
