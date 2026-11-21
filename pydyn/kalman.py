# This file is part of pydyn.
#
# It follows, and in places is a Python port of, matlab/kalman/likelihood/*.m (standard, missing-observation, univariate and diffuse Kalman filters) and matlab/kalman/DsgeSmoother.m from Dynare 7.2
# (Copyright (C) 2003-2026 Dynare Team, https://www.dynare.org), modified in 2026.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Kalman filters for the likelihood of a linear Gaussian state space, after Dynare 7.2.

State space (deviations from steady state)::

    a_{t+1} = T a_t + R eps_{t+1},   eps ~ N(0, Q)
    y_t     = a_t[z] + eta_t,        eta ~ N(0, H)

``z`` is an index vector (the observed variables are elements of the state vector). All filters
return *minus* the log likelihood, like Dynare, and ``inf`` when the forecast-error covariance is
singular. Ported from ``kalman_filter.m``, ``kalman_filter_ss.m``,
``missing_observations_kalman_filter.m``, ``kalman_filter_d.m``,
``missing_observations_kalman_filter_d.m``, ``lyapunov_symm.m`` and ``compute_Pinf_Pstar.m``.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

LOG2PI = np.log(2 * np.pi)


def _norm1(A):
    return np.abs(A).sum(axis=0).max()


def _inv_rcond(F):
    """Inverse and reciprocal condition number in the 1-norm (the quantity MATLAB's ``rcond``
    estimates). Returns (None, 0.0) for a singular or non-finite matrix."""
    if not np.all(np.isfinite(F)):
        return None, 0.0
    if F.shape[0] == 1:
        if F[0, 0] == 0:
            return None, 0.0
        return 1.0 / F, 1.0
    try:
        iF = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return None, 0.0
    if not np.all(np.isfinite(iF)):
        return None, 0.0
    return iF, 1.0 / (_norm1(F) * _norm1(iF))


def _rcond(F):
    return _inv_rcond(F)[1]


def _schur_unit_first(a, threshold):
    """Real Schur form with the eigenvalues of modulus above ``threshold`` ordered first."""
    T, U, k = linalg.schur(a, output="real", sort=lambda re, im: np.hypot(re, im) > threshold)
    return U, T, int(k)


def _lyap_backsub(ST, B, nk, exact_zero=False, complex_threshold=1e-15):
    """Solve X = S X S' + B on the trailing block of a quasi-triangular ``ST`` (rows/columns
    ``nk:``), by backward substitution. Returns a full-size matrix that is zero outside the block."""
    n = ST.shape[0]
    X = np.zeros((n, n))
    i = n - 1                                   # 0-based index of the current column
    while i >= nk + 1:
        sub = ST[i, i - 1]
        real_block = (sub == 0) if exact_zero else (abs(sub) < complex_threshold)
        r = slice(nk, i + 1)
        if real_block:
            if i == n - 1:
                c = np.zeros(i + 1 - nk)
            else:
                c = ST[r, :] @ (X[:, i + 1:] @ ST[i, i + 1:]) + ST[i, i] * (ST[r, i + 1:] @ X[i + 1:, i])
            q = np.eye(i + 1 - nk) - ST[r, r] * ST[i, i]
            X[r, i] = np.linalg.solve(q, B[r, i] + c)
            X[i, nk:i] = X[nk:i, i]
            i -= 1
        else:
            if i == n - 1:
                c = np.zeros(i + 1 - nk)
                c1 = np.zeros(i + 1 - nk)
            else:
                c = (ST[r, :] @ (X[:, i + 1:] @ ST[i, i + 1:]) + ST[i, i] * (ST[r, i + 1:] @ X[i + 1:, i])
                     + ST[i, i - 1] * (ST[r, i + 1:] @ X[i + 1:, i - 1]))
                c1 = (ST[r, :] @ (X[:, i + 1:] @ ST[i - 1, i + 1:])
                      + ST[i - 1, i - 1] * (ST[r, i + 1:] @ X[i + 1:, i - 1])
                      + ST[i - 1, i] * (ST[r, i + 1:] @ X[i + 1:, i]))
            m = i + 1 - nk
            q = np.block([[np.eye(m) - ST[r, r] * ST[i, i], -ST[r, r] * ST[i, i - 1]],
                          [-ST[r, r] * ST[i - 1, i], np.eye(m) - ST[r, r] * ST[i - 1, i - 1]]])
            z = np.linalg.solve(q, np.concatenate([B[r, i] + c, B[r, i - 1] + c1]))
            X[r, i] = z[:m]
            X[r, i - 1] = z[m:]
            X[i, nk:i] = X[nk:i, i]
            X[i - 1, nk:i - 1] = X[nk:i - 1, i - 1]
            i -= 2
    if i == nk:
        c = ST[nk, :] @ (X[:, nk + 1:] @ ST[nk, nk + 1:]) + ST[nk, nk] * (ST[nk, nk + 1:] @ X[nk + 1:, nk])
        X[nk, nk] = (B[nk, nk] + c) / (1 - ST[nk, nk] * ST[nk, nk])
    return X


def lyapunov_symm(a, b, qz_criterium=1 - 1e-6, complex_threshold=1e-15):
    """Solve ``x = a x a' + b`` (``lyapunov_symm.m``, default method).

    Eigenvalues of ``a`` with modulus above ``2 - qz_criterium`` are treated as unit roots and the
    equation is solved on the complementary Schur subspace only, as Dynare does.
    """
    a = np.asarray(a, float)
    if a.shape[0] == 1:
        return b / (1 - a * a)
    U, T, k = _schur_unit_first(a, 2 - qz_criterium)
    n = a.shape[0] - k
    if n == 0:
        return np.zeros_like(a)
    U2 = U[:, k:]
    x = _lyap_backsub(T[k:, k:], U2.T @ b @ U2, 0, complex_threshold=complex_threshold)
    return U2 @ x @ U2.T


def compute_Pinf_Pstar(mf, T, R, Q, qz_criterium=1 + 1e-6, restrict_columns=None):
    """Initial covariance matrices of the diffuse filter (``compute_Pinf_Pstar.m``).

    ``mf`` indexes the observed variables in the state vector, ``restrict_columns`` the states
    that enter the transition matrix with a lag (0-based)."""
    T = np.asarray(T, float)
    npf = T.shape[0]
    if not np.any(Q):
        return np.zeros_like(T), np.zeros_like(T)
    if restrict_columns is not None:
        indx = np.asarray(restrict_columns, int)
    else:
        indx = np.flatnonzero(np.max(np.abs(T), axis=0) >= 1e-10)
    indx0 = np.setdiff1d(np.arange(npf), indx)
    np0 = len(indx0)
    T0 = T[np.ix_(indx0, indx)]
    R0 = R[indx0, :]
    T = T[np.ix_(indx, indx)]
    R = R[indx, :]
    n = T.shape[0]
    QT, ST, nk = _schur_unit_first(T, 2 - qz_criterium)
    R1 = QT.T @ R
    B = R1 @ Q @ R1.T
    Pstar = _lyap_backsub(ST, B, nk, exact_zero=True)
    order = np.concatenate([indx0, indx])
    obs = np.isin(order, mf)
    if np0:
        Pstar = linalg.block_diag(np.zeros((np0, np0)), Pstar)
        ST = np.hstack([np.zeros((np0 + n, np0)), np.vstack([T0 @ QT, ST])])
        R1 = np.vstack([R0, R1])
        STinf = ST[np0:np0 + nk, np0:np0 + nk]
        iSTinf = np.linalg.inv(STinf) if nk else np.zeros((0, 0))
        ST0 = ST.copy()
        ST0[:, :np0 + nk] = 0
        ST00 = ST[:np0, np0:np0 + nk]
        ST0[:np0, np0 + nk:] = ST[:np0, np0 + nk:] - ST00 @ (iSTinf @ ST[np0:np0 + nk, np0 + nk:])
        R10 = R1.copy()
        R10[:np0, :] = R1[:np0, :] - ST00 @ (iSTinf @ R1[np0:np0 + nk, :])
        ST0[np0:np0 + nk, :] = 0
        R10[np0:np0 + nk, :] = 0
        Pstar = ST0 @ Pstar @ ST0.T + R10 @ Q @ R10.T
    Pinf = np.zeros((n, n))
    Pinf[:nk, :nk] = np.eye(nk)
    if np0:
        STtriu = STinf - np.eye(nk)
        STinf0 = ST00 @ (np.eye(nk) - iSTinf @ STtriu)
        Pinf = linalg.block_diag(np.zeros((np0, np0)), Pinf)
        QT = linalg.block_diag(np.eye(np0), QT)
        QTinf = QT.copy()
        QTinf[:np0, np0:np0 + nk] = STinf0
        tmp = np.empty_like(QTinf)
        tmp[order, :] = QTinf
        QTinf = tmp
        STinf1 = np.hstack([np.zeros((np0 + n, np0)),
                            np.vstack([STinf0, np.eye(nk), np.zeros((n - nk, nk))]),
                            np.zeros((np0 + n, n - nk))])
        for k in range(nk):
            # sic: Dynare indexes the rows with a logical mask over [indx0; indx]
            if np.linalg.norm(QTinf[obs, :] @ ST[order, k + np0]) < 1e-8:
                Pinf[k + np0, k + np0] = 0
        Pinf = STinf1 @ Pinf @ STinf1.T
        tmp = np.empty_like(QT)
        tmp[order, :] = QT
        QT = tmp
    else:
        for k in range(nk):
            if np.linalg.norm(QT[np.asarray(mf, int), :] @ ST[:, k]) < 1e-8:
                Pinf[k, k] = 0
    return QT @ Pstar @ QT.T, QT @ Pinf @ QT.T


def _forecast_error_inverse(F, kalman_tol, rescale):
    """Conditioning tests of kalman_filter.m. Returns (log det F, inverse of F), or None when F
    is treated as singular. ``rescale`` forces the computations on the correlation matrix."""
    n = F.shape[0]
    if not rescale:
        if n == 1:
            f = F[0, 0]
            if f != 0 and np.isfinite(f):         # rcond of a non-zero scalar is one
                return np.log(f), np.array([[1.0 / f]])
        elif n == 2:
            a, b, c, d = F[0, 0], F[0, 1], F[1, 0], F[1, 1]
            det = a * d - b * c
            if det != 0 and np.isfinite(det):
                n1 = max(abs(a) + abs(c), abs(b) + abs(d))
                if det * det / (n1 * n1) >= kalman_tol * abs(det):     # rcond = |det| / ||F||_1^2
                    return np.log(det), np.array([[d, -b], [-c, a]]) / det
        else:
            iF, rc = _inv_rcond(F)
            if rc >= kalman_tol:
                return np.log(np.linalg.det(F)), iF
    d = np.diag(F)
    if np.any(d < kalman_tol) or not np.all(np.isfinite(d)):
        return None
    sig = np.sqrt(d)
    ss = np.outer(sig, sig)
    iFs, rc = _inv_rcond(F / ss)
    if rc < kalman_tol:
        return None
    return np.log(np.linalg.det(F / ss)) + 2 * np.sum(np.log(sig)), iFs / ss


def kalman_filter(Y, a, P, T, QQ, H, z, kalman_tol=1e-10, riccati_tol=1e-6, presample=0,
                  rescale=False, diffuse_periods=0, data_index=None, no_more_missing=0,
                  return_lik=False):
    """Standard Kalman filter with a switch to the steady-state gain (``kalman_filter.m`` and
    ``missing_observations_kalman_filter.m``).

    ``Y`` is (n_obs_variables, periods) with the constant already removed; ``a``, ``P`` are the
    predicted state and covariance for the first period; ``QQ = R Q R'``; ``H`` is the
    measurement-error covariance (matrix). ``data_index`` (list of index arrays of the observed
    series in each period) selects the missing-observations filter; ``no_more_missing`` is the
    0-based first period from which no observation is missing.
    Returns minus the log likelihood summed after the presample (``inf`` if singular).
    ``QQ`` may be a (n, n, periods + 1) array (``heteroskedastic_filter``): period ``t`` (0-based) then
    uses ``QQ[:, :, t + 1]`` and the filter never switches to the steady-state gain, as in Dynare.
    """
    pp, smpl = Y.shape
    z = np.asarray(z, int)
    Tt = T.T
    lik = np.zeros(smpl)
    qvec = np.ndim(QQ) == 3
    QQv = QQ
    notsteady, F_singular = True, True
    oldK = np.inf
    t = 0
    missing = data_index is not None
    if missing:
        P = np.tril(P) + np.tril(P, -1).T
    K = iF = log_dF = None
    zz = np.ix_(z, z)
    while notsteady and t < smpl:
        if qvec:
            QQ = QQv[:, :, t + 1]
        if missing:
            d = data_index[t]
            if len(d) == 0:
                a = T @ a
                P = T @ P @ Tt + QQ
                t += 1
                continue
            zt = z[d]
            v = Y[d, t] - a[zt]
            F = P[np.ix_(zt, zt)] + H[np.ix_(d, d)]
        else:
            zt = z
            v = Y[:, t] - a[zt]
            F = P[zz] + H
        res = _forecast_error_inverse(F, kalman_tol, rescale)
        if res is None:
            return (np.inf, lik) if return_lik else np.inf
        F_singular = False
        log_dF, iF = res
        lik[t] = log_dF + v @ iF @ v + len(zt) * LOG2PI
        K = P[:, zt] @ iF
        P = T @ (P - K @ P[zt, :]) @ Tt + QQ
        a = T @ (a + K @ v)
        if (not missing or t >= no_more_missing) and not qvec:
            notsteady = np.max(np.abs(K.ravel() - oldK)) > riccati_tol
            oldK = K.ravel()
        t += 1
    if F_singular:
        raise FloatingPointError("the variance of the forecast error remains singular until the end of the sample")
    lik[:t] *= 0.5
    if t < smpl:                                  # steady-state filter for the remaining periods
        for s in range(t, smpl):
            v = Y[:, s] - a[z]
            lik[s] = v @ iF @ v
            a = T @ (a + K @ v)
        lik[t:] = 0.5 * (lik[t:] + log_dF + pp * LOG2PI)
    start = presample - diffuse_periods if presample > diffuse_periods else 0
    LIK = float(np.sum(lik[start:]))
    return (LIK, lik) if return_lik else LIK


def kalman_filter_d(Y, a, Pinf, Pstar, T, QQ, H, z, kalman_tol=1e-10, diffuse_kalman_tol=1e-6,
                    presample=0, data_index=None):
    """Diffuse stage of the filter (``kalman_filter_d.m`` / ``missing_observations_kalman_filter_d.m``).

    Returns ``(dLIK, dlik, a, Pstar)``: minus the log likelihood of the diffuse periods (after the
    presample), the per-period values, and the predicted state and covariance for the first
    non-diffuse period. ``dLIK`` is ``inf`` if a singularity stops the filter and ``nan`` if the
    diffuse stage never ends.
    """
    pp, smpl = Y.shape
    z = np.asarray(z, int)
    Tt = T.T
    dlik = np.zeros(smpl)
    missing = data_index is not None
    t = s = 0

    def diffuse_rank(M):
        if M.size == 0:
            return 0
        return int(np.sum(np.linalg.svd(M, compute_uv=False) > diffuse_kalman_tol))

    qvec = np.ndim(QQ) == 3
    QQv = QQ
    while t < smpl and diffuse_rank(Pinf if missing else Pinf[np.ix_(z, z)]):
        s = t + 1
        if qvec:
            QQ = QQv[:, :, t + 1]
        if missing:
            d = data_index[t]
            if len(d) == 0:
                a = T @ a
                Pstar = T @ Pstar @ Tt + QQ
                Pinf = T @ Pinf @ Tt
                t += 1
                continue
            zt, Ht = z[d], H[np.ix_(d, d)]
            v = Y[d, t] - a[zt]
        else:
            zt, Ht = z, H
            v = Y[:, t] - a[zt]
        const = len(zt) * LOG2PI
        Finf = Pinf[np.ix_(zt, zt)]
        if _rcond(Finf) < diffuse_kalman_tol:
            if not np.all(np.abs(Finf) < diffuse_kalman_tol):
                return np.inf, dlik[:s], a, Pstar
            Fstar = Pstar[np.ix_(zt, zt)] + Ht
            if _rcond(Fstar) < kalman_tol:
                return np.inf, dlik[:s], a, Pstar
            iFstar = np.linalg.inv(Fstar)
            Kstar = Pstar[:, zt] @ iFstar
            dlik[t] = np.log(np.linalg.det(Fstar)) + v @ iFstar @ v + const
            Pinf = T @ Pinf @ Tt
            Pstar = T @ (Pstar - Pstar[:, zt] @ Kstar.T) @ Tt + QQ
            a = T @ (a + Kstar @ v)
        else:
            dlik[t] = np.log(np.linalg.det(Finf)) + const
            iFinf = np.linalg.inv(Finf)
            Kinf = Pinf[:, zt] @ iFinf
            Fstar = Pstar[np.ix_(zt, zt)] + Ht
            Kstar = (Pstar[:, zt] - Kinf @ Fstar) @ iFinf
            Pstar = T @ (Pstar - Pstar[:, zt] @ Kinf.T - Pinf[:, zt] @ Kstar.T) @ Tt + QQ
            Pinf = T @ (Pinf - Pinf[:, zt] @ Kinf.T) @ Tt
            a = T @ (a + Kinf @ v)
        t += 1
    if t >= smpl:
        return np.nan, dlik, a, Pstar
    dlik = 0.5 * dlik[:s]
    return float(np.sum(dlik[presample:])), dlik, a, Pstar


def univariate_kalman_filter(Y, a, P, T, QQ, Hdiag, Z, kalman_tol=1e-10, riccati_tol=1e-6, presample=0,
                             diffuse_periods=0, data_index=None, no_more_missing=0, return_lik=False):
    """Univariate (sequential) Kalman filter, ``univariate_kalman_filter.m`` and
    ``univariate_kalman_filter_ss.m``. ``Z`` is the (n_obs, n_states) measurement matrix and
    ``Hdiag`` the vector of measurement-error variances (the errors must be uncorrelated).
    Observations whose forecast-error variance is below ``kalman_tol`` are skipped.
    Returns minus the log likelihood (``nan`` if a variance is negative)."""
    pp, smpl = Y.shape
    mm = T.shape[0]
    Tt = T.T
    lik = np.zeros((smpl, pp))
    K = np.full((mm, pp), np.nan)
    oldK = np.inf
    notsteady = True
    full = np.arange(pp)
    t = 0
    qvec = np.ndim(QQ) == 3
    QQv = QQ
    while notsteady and t < smpl:
        if qvec:
            QQ = QQv[:, :, t + 1]
        d = full if data_index is None else data_index[t]
        P = (P + P.T) / 2
        for i, di in enumerate(d):
            Zi = Z[di]
            err = Y[di, t] - Zi @ a
            PZ = P @ Zi
            Fi = Zi @ PZ + Hdiag[di]
            if Fi > kalman_tol:
                Ki = PZ / Fi
                if t >= no_more_missing:
                    K[:, i] = Ki
                lik[t, i] = np.log(Fi) + err * err / Fi + LOG2PI
                a = a + Ki * err
                P = P - np.outer(PZ, Ki)
                P = (P + P.T) / 2
            elif Fi < 0:
                return (np.nan, lik) if return_lik else np.nan
        a = T @ a
        P = T @ P @ Tt + QQ
        if t >= no_more_missing and not qvec:
            diff = np.abs(K.ravel() - oldK)
            notsteady = bool(np.nanmax(diff) > riccati_tol) if not np.all(np.isnan(diff)) else False
            oldK = K.ravel().copy()
        t += 1
    if t < smpl:                                  # the covariance matrix has converged
        for s in range(t, smpl):
            PP = P
            for i in range(pp):
                Zi = Z[i]
                err = Y[i, s] - Zi @ a
                PPZ = PP @ Zi
                Fi = Zi @ PPZ + Hdiag[i]
                if Fi > kalman_tol:
                    Ki = PPZ / Fi
                    a = a + Ki * err
                    PP = PP - np.outer(PPZ, Ki)
                    lik[s, i] = np.log(Fi) + err * err / Fi + LOG2PI
                elif Fi < 0:
                    return (np.nan, lik) if return_lik else np.nan
            a = T @ a
    lik *= 0.5
    start = presample - diffuse_periods if presample > diffuse_periods else 0
    LIK = float(np.sum(lik[start:]))
    return (LIK, lik.sum(axis=1)) if return_lik else LIK


def univariate_kalman_filter_d(Y, a, Pinf, Pstar, T, QQ, Hdiag, Z, kalman_tol=1e-10,
                               diffuse_kalman_tol=1e-6, presample=0, data_index=None):
    """Diffuse stage of the univariate filter (``univariate_kalman_filter_d.m``).
    Returns ``(dLIK, dlik, a, Pstar)`` like :func:`kalman_filter_d`."""
    pp, smpl = Y.shape
    Tt = T.T
    dlik = np.zeros(smpl)
    full = np.arange(pp)

    def rank(M):
        return int(np.sum(np.linalg.svd(M, compute_uv=False) > diffuse_kalman_tol)) if M.size else 0

    new_rank = rank(Pinf)
    t = s = 0
    qvec = np.ndim(QQ) == 3
    QQv = QQ
    while new_rank and t < smpl:
        s = t + 1
        if qvec:
            QQ = QQv[:, :, t + 1]
        d = full if data_index is None else data_index[t]
        for di in d:
            Zi = Z[di]
            err = Y[di, t] - Zi @ a
            Fstar = Zi @ Pstar @ Zi + Hdiag[di]
            Finf = Zi @ Pinf @ Zi
            Kstar = Pstar @ Zi
            if Finf > diffuse_kalman_tol and new_rank:
                Kinf = Pinf @ Zi
                Kinf_Finf = Kinf / Finf
                a = a + Kinf_Finf * err
                Pstar = (Pstar + np.outer(Kinf, Kinf_Finf) * (Fstar / Finf) - np.outer(Kstar, Kinf_Finf)
                         - np.outer(Kinf_Finf, Kstar))
                Pinf = Pinf - np.outer(Kinf, Kinf_Finf)
                dlik[t] += np.log(Finf) + LOG2PI
            elif Fstar > kalman_tol:
                dlik[t] += np.log(Fstar) + err * err / Fstar + LOG2PI
                a = a + Kstar * (err / Fstar)
                Pstar = Pstar - np.outer(Kstar, Kstar) / Fstar
            elif Fstar < 0 or Finf < 0:
                return np.nan, dlik[:s], a, Pstar
        a = T @ a
        Pstar = T @ Pstar @ Tt + QQ
        Pinf = T @ Pinf @ Tt
        if new_rank:
            new_rank = rank(Z @ Pinf @ Z.T)
        t += 1
    if t >= smpl:
        return np.nan, dlik, a, Pstar
    dlik = 0.5 * dlik[:s]
    return float(np.sum(dlik[presample:])), dlik, a, Pstar


def kalman_smoother(Y, a, P, T, R, Q, H, z, data_index=None):
    """Kalman filter and fixed-interval smoother for the stationary case.

    Returns a dict with smoothed states ``alpha`` (n_states, periods), smoothed structural shocks
    ``eps`` (n_shocks, periods), smoothed measurement errors ``eta``, updated states ``a_upd``
    (E[a_t | y_1..t]) and the one-step-ahead predictions ``a_pred``. Recursions of Durbin and
    Koopman (2012, section 4.5), with Dynare's timing: the shock dated t enters the state dated t.
    """
    pp, smpl = Y.shape
    mm = T.shape[0]
    z = np.asarray(z, int)
    QQ = R @ Q @ R.T
    a_pred = np.zeros((mm, smpl + 1))
    a_upd = np.zeros((mm, smpl))
    P_pred = np.zeros((mm, mm, smpl + 1))
    vs, iFs, Ks, idx = [], [], [], []
    a_pred[:, 0], P_pred[:, :, 0] = a, P
    for t in range(smpl):
        d = np.arange(pp) if data_index is None else np.asarray(data_index[t], int)
        zt = z[d]
        a, P = a_pred[:, t], P_pred[:, :, t]
        if len(d):
            v = Y[d, t] - a[zt]
            iF = np.linalg.inv(P[np.ix_(zt, zt)] + H[np.ix_(d, d)])
            K = P[:, zt] @ iF
            au = a + K @ v
            Pu = P - K @ P[zt, :]
        else:
            v, iF, K, au, Pu = np.zeros(0), np.zeros((0, 0)), np.zeros((mm, 0)), a, P
        vs.append(v); iFs.append(iF); Ks.append(K); idx.append((d, zt))
        a_upd[:, t] = au
        a_pred[:, t + 1] = T @ au
        P_pred[:, :, t + 1] = T @ Pu @ T.T + QQ
    alpha = np.zeros((mm, smpl))
    eps = np.zeros((Q.shape[0], smpl))
    eta = np.full((pp, smpl), np.nan)            # NaN where the observation is missing, as Dynare
    r = np.zeros(mm)
    QRt = Q @ R.T
    for t in range(smpl - 1, -1, -1):
        d, zt = idx[t]
        L = T - (T @ Ks[t]) @ np.eye(mm)[zt, :] if len(d) else T
        u = iFs[t] @ vs[t]
        rz = np.zeros(mm)
        if len(d):
            rz[zt] = u
            eta[d, t] = H[np.ix_(d, d)] @ (u - (T @ Ks[t]).T @ r)
        r = rz + L.T @ r                         # r_{t-1}
        eps[:, t] = QRt @ r                      # shocks dated t enter a_t: E[eps_t | Y] = Q R' r_{t-1}
        alpha[:, t] = a_pred[:, t] + P_pred[:, :, t] @ r
    return {"alpha": alpha, "eps": eps, "eta": eta, "a_upd": a_upd, "a_pred": a_pred[:, :smpl]}
