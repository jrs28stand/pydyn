# This file is part of pydyn.
#
# It is a Python port of parts of Dynare 7.2's matlab/+identification (analysis.m, checks.m, checks_via_subsets.m,
# get_jacobians.m, get_minimal_state_representation.m, run.m, cosn.m, vnorm.m) and of the first-order part of
# matlab/+pruned_SS/pruned_state_space_system.m, Copyright (C) 2010-2026 Dynare Team, GPL-3.0-or-later.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""The numerical core of Dynare's ``identification`` statement at first order (:mod:`pydyn.params_derivs` supplies the
derivatives of the perturbation solution with respect to the parameters).

For one parameter point :func:`analysis` computes the Jacobians of the moments (Iskrev), of the spectrum (Qu and
Tkachenko), of the minimal system (Komunjer and Ng), of the reduced form and of the dynamic model, and the rank,
condition, collinearity and "weak identification" results that Dynare stores in ``ide_moments_point`` and friends
(:func:`checks`, :func:`checks_via_subsets`).  Matrices have Dynare's layout; indices inside the returned structures are
Dynare's (1-based, as the MAT-file holds them) where they are stored (``ind_dMOMENTS``...), and 0-based inside the code.

Not ported (refused with NotImplementedError): orders 2 and 3, ``analytic_derivation_mode`` -1 and -2 (finite
differences), unit roots (the diffuse filter), the brute-force collinearity search of ``advanced=1``, prior Monte Carlo.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
from scipy import linalg

EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------------------------------- small helpers
def vech_dyn(M):
    """``dyn_vech``: the upper triangle of a square matrix, column after column (rows 1..col of column col)."""
    M = np.asarray(M)
    n = M.shape[0]
    return np.concatenate([M[:c + 1, c] for c in range(n)]) if n else np.zeros(0)


def vec(M):
    return np.asarray(M).reshape(-1, order="F")


def rank(X, tol=None):
    """Octave's ``rank(X, tol)``: number of singular values above ``tol`` (default ``max(size) eps(max sv)``)."""
    X = np.asarray(X)
    if X.size == 0:
        return 0
    s = linalg.svdvals(X)
    if tol is None:
        tol = max(X.shape) * np.spacing(s.max())
    return int(np.sum(s > tol))


def cond(X):
    X = np.asarray(X)
    if X.size == 0:
        return 0.0
    s = linalg.svdvals(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")


def vnorm(A):
    """``identification.vnorm``: 2-norm of the columns of a matrix with more than one row."""
    return np.sqrt(np.sum(np.abs(np.asarray(A, float)) ** 2, axis=0))


def vnorm_matlab(A):
    """``identification.vnorm(A)`` exactly: the norm runs along the first non-singleton dimension, so a single-row matrix
    gives one number (the norm of the row), not the norms of its columns."""
    A = np.asarray(A, float)
    if A.shape[0] == 1 and A.shape[1] > 1:
        return np.sqrt(np.sum(np.abs(A) ** 2))
    return vnorm(A)


# Octave's X\y (a rank-deficient least-squares problem) calls LAPACK dgelsd with rcond = -1, i.e. dlamch("E") = eps/2
def cosn(H):
    """Multiple cosine of the first column of ``H`` with the span of the others (``cosn.m``)."""
    H = np.asarray(H, float)
    y, X = H[:, 0], H[:, 1:]
    if X.shape[1] == 0:
        b = np.zeros(0)
    else:
        b = linalg.lstsq(X, y, cond=EPS / 2, lapack_driver="gelsd")[0]
        if np.any(np.isnan(b)) or np.any(np.isinf(b)):
            b = np.zeros_like(b)
    yhat = X @ b if X.shape[1] else np.zeros_like(y)
    if rank(yhat.reshape(-1, 1)):
        return float(abs(y @ yhat / np.sqrt((y @ y) * (yhat @ yhat))))
    return 0.0


def mldivide(A, B):
    """Octave's ``A\\B`` for a square ``A``: an LU solve (LAPACK dgetrf/dgetrs), also where the matrix is singular to
    machine precision (Octave then only warns); the minimum-norm least-squares solution (dgelsd) only where dgetrf meets an
    exactly zero pivot."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.size == 0:
        return np.zeros((A.shape[1],) + B.shape[1:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lu, piv = linalg.lu_factor(A, check_finite=False)
    if np.any(np.diag(lu) == 0) or not np.all(np.isfinite(lu)):
        return linalg.lstsq(A, B, cond=EPS / 2, lapack_driver="gelsd")[0]
    return linalg.lu_solve((lu, piv), B)


def minv(A):
    """Octave's ``inv``: a singular matrix gives Inf (with a warning), not an error."""
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.full(np.shape(A), np.inf)


def _find(mask):
    return np.flatnonzero(np.asarray(mask).ravel())


# ---------------------------------------------------------------------------------------------------- checks.m
def checks(X, test_flag, tol_rank, tol_sv, param_nbr=None):
    """``identification.checks``: returns a dict with cond, rank, ind0 (1 x p), indno (ixno x p), ino, Mco, Pco, jweak,
    jweak_pair (Dynare's names)."""
    X = np.asarray(X, float)
    if X.size == 0:
        return _empty_checks(param_nbr)
    if tol_rank is None or isinstance(tol_rank, str):
        tol_rank = max(X.shape) * np.spacing(np.linalg.norm(X, 2)) if X.size else 0.0
    if tol_sv is None:
        tol_sv = 1e-3
    if param_nbr is None:
        param_nbr = X.shape[1]
    indno = [np.zeros(param_nbr)]
    rankrequired = X.shape[1]
    if test_flag == 2:
        Xpar, Xrest = X[:, :param_nbr], X[:, param_nbr:]
    else:
        Xpar, Xrest = X, np.zeros((X.shape[0], 0))
    if Xpar.shape[0] > 1:
        ind1 = _find(vnorm(Xpar) >= EPS)
    else:
        ind1 = _find(np.abs(Xpar) >= EPS)
    nonzero = (lambda ind: Xpar[np.ix_(ind, ind)]) if test_flag == 3 else (lambda ind: Xpar[:, ind])
    Xn = nonzero(ind1)
    M = Xn if Xrest.shape[1] == 0 else np.hstack([Xn, Xrest])
    ee1 = linalg.svd(M, full_matrices=M.shape[0] < M.shape[1], lapack_driver="gesvd")[2].T
    condX = cond(M)
    rankX = rank(X, tol_rank)
    icheck = False
    if param_nbr > 0 and rankX < rankrequired:
        ee0 = np.arange(rankX, M.shape[1])
        ind11 = np.ones(len(ind1))
        for j in ee0:
            if np.sum(np.abs(ee1[:, j]) > tol_sv) == 1:
                icheck = True
                temp = (np.abs(ee1[:, j]) <= tol_sv).astype(float)
                ind11 = ind11 * (temp[:len(temp) - Xrest.shape[1]] if test_flag == 2 else temp)
        ind1 = ind1[_find(ind11)]
    if icheck:
        Xn = nonzero(ind1)
        M = Xn if Xrest.shape[1] == 0 else np.hstack([Xn, Xrest])
        ee1 = linalg.svd(M, full_matrices=M.shape[0] < M.shape[1], lapack_driver="gesvd")[2].T
        condX = cond(M)
        rankX = rank(X, tol_rank)
    ind0 = np.zeros(param_nbr)
    ind0[ind1] = 1
    if test_flag in (0, 3):
        # a Gram matrix: tildaX is its correlation-like normalisation; Octave's Mco(ind1,1) = ... makes a vector that
        # ends at the last index of ind1 (test_flag 0), Mco = NaN(param_nbr,1) before for test_flag 3
        Mco = np.full(param_nbr, np.nan) if test_flag == 3 else np.zeros(int(ind1.max()) + 1 if len(ind1) else 0)
        Xs = X[np.ix_(ind1, ind1)]
        deltaX = np.sqrt(np.diag(Xs))
        tildaX = Xs / np.outer(deltaX, deltaX)
        Mco[ind1] = 1 - 1 / np.diag(minv(tildaX))
        Pco = np.full((param_nbr, param_nbr), np.nan)
        Pco[np.ix_(ind1, ind1)] = minv(Xs)
        sd = np.sqrt(np.diag(Pco))
        with np.errstate(invalid="ignore", divide="ignore"):
            Pco = np.abs(Pco / np.outer(sd, sd))
    else:
        Mco = np.full(param_nbr, np.nan)
        cols = list(range(Xn.shape[1]))
        for ii in cols:
            others = [k for k in cols if k != ii]
            Mco[ind1[ii]] = cosn(np.hstack([Xn[:, [ii]], Xn[:, others]] + ([Xrest] if Xrest.shape[1] else [])))
    ixno = 0
    with np.errstate(invalid="ignore"):
        low = np.nanmin(1 - Mco) if np.any(~np.isnan(Mco)) else np.inf
    if param_nbr > 0 and (rankX < rankrequired or low < tol_rank):
        if len(ind1) < param_nbr:
            ixno += 1
            indno = [np.array([0.0 if k in set(ind1.tolist()) else 1.0 for k in range(param_nbr)])]
        ee0 = np.arange(rankX, M.shape[1])
        for j in ee0:
            ixno += 1
            row = np.zeros(param_nbr)
            temp = (np.abs(ee1[:, j]) > tol_sv).astype(float)
            if test_flag == 2:
                temp = temp[:len(temp) - Xrest.shape[1]]
            row[ind1] = temp
            while len(indno) < ixno:
                indno.append(np.zeros(param_nbr))
            indno[ixno - 1] = row
    indno = np.array(indno)
    jweak = np.zeros(param_nbr)
    jweak_pair = np.zeros((param_nbr, param_nbr))
    if test_flag != 0:
        Pco = np.full((param_nbr, param_nbr), np.nan)
        cols = list(range(Xn.shape[1]))
        for ii in cols:
            Pco[ind1[ii], ind1[ii]] = 1
            for jj in range(ii + 1, len(cols)):
                v = cosn(np.hstack([Xn[:, [ii]], Xn[:, [jj]]] + ([Xrest] if Xrest.shape[1] else [])))
                Pco[ind1[ii], ind1[jj]] = v
                Pco[ind1[jj], ind1[ii]] = v
        for j in range(param_nbr):
            if Mco[j] > 1 - tol_rank:
                jweak[j] = 1
                with np.errstate(invalid="ignore"):
                    for jx in _find(Pco[j, j + 1:] > 1 - tol_rank):
                        jweak_pair[j, jx + j + 1] = 1
                        jweak_pair[jx + j + 1, j] = 1
    return {"cond": condX, "rank": float(rankX), "ind0": ind0, "indno": indno, "ino": float(ixno),
            "Mco": Mco.reshape(-1, 1), "Pco": Pco, "jweak": jweak, "jweak_pair": vech_dyn(jweak_pair)}


# ---------------------------------------------------------------------------------------------------- minimal system
def _check_minimality(a, b, c):
    n = a.shape[0]
    CC, OO = b, c
    if n >= 2:
        for k in range(1, n):
            ak = np.linalg.matrix_power(a, k)
            CC = np.hstack([CC, ak @ b])
            OO = np.vstack([OO, c @ ak])
    return rank(CC) == n and rank(OO) == n


def _controllability_staircase(a, b, c, tol=None):
    ra = a.shape[0]
    cb = b.shape[1]
    ptjn1 = np.eye(ra)
    ajn1, bjn1 = a, b
    deltajn1 = 0
    k = np.zeros(ra)
    if tol is None:
        tol = ra * np.linalg.norm(a, 1) * EPS
    for jj in range(ra):
        uj, _, _ = linalg.svd(bjn1, lapack_driver="gesvd")
        rsj = bjn1.shape[0]
        uj = uj[:, ::-1]                              # uj*p with p the column-reversing permutation
        bb = uj.T @ bjn1
        roj = rank(bb, tol)
        sigmaj = bb.shape[0] - roj
        k[jj] = roj
        if roj == 0 or sigmaj == 0:
            break
        abxy = uj.T @ ajn1 @ uj
        aj = abxy[:sigmaj, :sigmaj]
        bj = abxy[:sigmaj, sigmaj:sigmaj + roj]
        ajn1, bjn1 = aj, bj
        ruj, cuj = uj.shape
        ptjn1 = ptjn1 @ np.block([[uj, np.zeros((ruj, deltajn1))], [np.zeros((deltajn1, cuj)), np.eye(deltajn1)]])
        deltajn1 += roj
    t = ptjn1.T
    return t @ a @ t.T, t @ b, c @ t.T, t, k


def _observability_staircase(a, b, c, tol):
    aa, bb, cc, t, k = _controllability_staircase(a.T, c.T, b.T, tol)
    return aa.T, cc.T, bb.T, t, k


def get_minimal_state_representation(SYS, derivs_flag=False, realsmall=1e-7):
    """``identification.get_minimal_state_representation``: returns ``(CheckCO, minns, minSYS)`` (``minSYS`` None if
    the reduction does not give a minimal system).  ``SYS`` is a dict with A, B, C, D and (derivs_flag) dA, dB, dC, dD
    (3-d arrays, parameters last)."""
    nspred, exo_nbr = SYS["B"].shape
    varobs_nbr = SYS["C"].shape[0]
    CheckCO = _check_minimality(SYS["A"], SYS["B"], SYS["C"])
    if CheckCO:
        return True, nspred, SYS
    a, b, c = SYS["A"], SYS["B"], SYS["C"]
    ns = b.shape[0]
    am, bm, cm, U, k = _controllability_staircase(a, b, c, realsmall)
    nu = ns - int(np.sum(k))
    am, bm, cm = am[nu:ns, nu:ns], bm[nu:ns, :], cm[:, nu:ns]
    ns = ns - nu
    if ns:
        am, bm, cm, _, k = _observability_staircase(am, bm, cm, realsmall)
        nu = ns - int(np.sum(k))
        am, bm, cm = am[nu:ns, nu:ns], bm[nu:ns, :], cm[:, nu:ns]
    minns = am.shape[0]
    if not _check_minimality(am, bm, cm):
        return False, None, None
    out = {"A": am, "B": bm, "C": cm, "D": SYS["D"]}
    if derivs_flag:
        tot = SYS["dA"].shape[2]
        out["dA"] = np.zeros((minns, minns, tot))
        out["dB"] = np.zeros((minns, exo_nbr, tot))
        out["dC"] = np.zeros((varobs_nbr, minns, tot))
        for jp in range(tot):
            out["dA"][:, :, jp] = (U @ SYS["dA"][:, :, jp] @ U.T)[:minns, :minns]
            out["dB"][:, :, jp] = (U @ SYS["dB"][:, :, jp])[:minns, :]
            out["dC"][:, :, jp] = (SYS["dC"][:, :, jp] @ U.T)[:, :minns]
        out["dD"] = SYS["dD"]
    return True, minns, out


# ---------------------------------------------------------------------------------------------------- pruned state space (order 1)
def _lyap(A, Q):
    """X = A X A' + Q."""
    return linalg.solve_discrete_lyapunov(A, Q)


def pruned_first_order(ghx, ghu, Sigma, Yss, indy, nlags, useautocorr, nstatic, nspred, derivs=None, qz_unit=1 - 1e-6):
    """First-order part of ``pruned_SS.pruned_state_space_system``: the state space A, B, C, D of the predetermined
    variables and observables, the covariances/autocovariances (correlations with ``useautocorr``) of the observables
    and, with ``derivs`` (a dict with dghx, dghu, dSigma_e, dYss, stderr+corr parameter count ``nsc``), their Jacobians
    in all parameters (order: stderr, corr, model)."""
    n = ghx.shape[0]
    indx = nstatic + np.arange(nspred)
    A, B = ghx[indx, :], ghu[indx, :]
    C, D = ghx[indy, :], ghu[indy, :]
    ev = np.abs(np.linalg.eigvals(A)) if A.size else np.zeros(0)
    if ev.size and ev.max() >= qz_unit:
        raise NotImplementedError("identification: unit roots (diffuse filter) are not supported")
    Varinov = Sigma
    Var_z = _lyap(A, B @ Varinov @ B.T) if A.size else np.zeros((0, 0))
    ny = len(indy)
    out = {"indx": indx, "indy": indy, "A": A, "B": B, "C": C, "D": D, "Varinov": Varinov}
    Var_y = C @ Var_z @ C.T + D @ Varinov @ D.T
    Var_y[np.abs(Var_y) < 1e-12] = 0
    sdy = np.sqrt(np.diag(Var_y))
    sy = np.outer(sdy, sdy)
    tmp = A @ Var_z @ C.T + B @ Varinov @ D.T
    Var_yi = np.zeros((ny, ny, nlags))
    Ai = np.eye(A.shape[0])
    for i in range(nlags):
        Var_yi[:, :, i] = C @ Ai @ tmp
        Ai = Ai @ A
    out.update(Var_y=Var_y, Var_yi=Var_yi, E_y=Yss[indy].copy())
    if useautocorr:
        out["Corr_yi"] = Var_yi / sy[:, :, None]
    if derivs is None:
        return out
    dA, dB = derivs["dghx"][indx, :, :], derivs["dghu"][indx, :, :]
    dC, dD = derivs["dghx"][indy, :, :], derivs["dghu"][indy, :, :]
    dS = derivs["dSigma_e"]
    nsc = derivs["nsc"]
    tot = dA.shape[2]
    dVar_z = np.zeros((A.shape[0], A.shape[0], tot))
    for j in range(tot):
        if j < nsc:
            Q = B @ dS[:, :, j] @ B.T
        else:
            dOm = dB[:, :, j] @ Varinov @ B.T + B @ Varinov @ dB[:, :, j].T
            Q = dA[:, :, j] @ Var_z @ A.T + A @ Var_z @ dA[:, :, j].T + dOm
        dVar_z[:, :, j] = _lyap(A, Q)
    dVar_y = np.zeros((ny, ny, tot))
    for j in range(tot):
        t = (dC[:, :, j] @ Var_z @ C.T + C @ dVar_z[:, :, j] @ C.T + C @ Var_z @ dC[:, :, j].T
             + dD[:, :, j] @ Varinov @ D.T + D @ dS[:, :, j] @ D.T + D @ Varinov @ dD[:, :, j].T)
        t[np.abs(t) < 1e-12] = 0
        dVar_y[:, :, j] = t
    dVar_yi = np.zeros((ny, ny, nlags, tot))
    dCorr_yi = np.zeros((ny, ny, nlags, tot)) if useautocorr else None
    for j in range(tot):
        Ai = np.eye(A.shape[0])
        dAi = np.zeros_like(A)
        dtmp = (dA[:, :, j] @ Var_z @ C.T + A @ dVar_z[:, :, j] @ C.T + A @ Var_z @ dC[:, :, j].T
                + dB[:, :, j] @ Varinov @ D.T + B @ dS[:, :, j] @ D.T + B @ Varinov @ dD[:, :, j].T)
        for i in range(nlags):
            dVar_yi[:, :, i, j] = dC[:, :, j] @ Ai @ tmp + C @ dAi @ tmp + C @ Ai @ dtmp
            if useautocorr:
                dsy = 0.5 / sdy * np.diag(dVar_y[:, :, j])
                dsy = np.outer(dsy, sdy) + np.outer(sdy, dsy)
                dCorr_yi[:, :, i, j] = (dVar_yi[:, :, i, j] * sy - dsy * Var_yi[:, :, i]) / (sy * sy)
            dAi = dAi @ A + Ai @ dA[:, :, j]
            Ai = Ai @ A
    dE_y = np.zeros((ny, tot))
    dE_y[:, nsc:] = derivs["dYss"][indy, :]
    out.update(dA=dA, dB=dB, dC=dC, dD=dD, dVarinov=dS, dVar_y=dVar_y, dVar_yi=dVar_yi, dE_y=dE_y)
    if useautocorr:
        out["dCorr_yi"] = dCorr_yi
    return out


def duplication(p):
    """``pruned_SS.duplication``: duplication matrix and its Moore-Penrose inverse (the elimination matrix)."""
    a = np.tril(np.ones((p, p)))
    flat = a.ravel(order="F")
    ii = np.flatnonzero(flat)
    flat[ii] = np.arange(1, len(ii) + 1)
    a = flat.reshape((p, p), order="F")
    a = a + np.tril(a, -1).T
    j = a.ravel(order="F").astype(int) - 1
    Dp = np.zeros((p * p, p * (p + 1) // 2))
    Dp[np.arange(p * p), j] = 1
    return Dp, np.linalg.solve(Dp.T @ Dp, Dp.T)


# ---------------------------------------------------------------------------------------------------- get_jacobians.m
class IdentificationWarning(UserWarning):
    pass


def get_jacobians(model, ghx, ghu, ys, pp, Sigma, derivs, indvobs, nstatic, nspred, nsc, opts):
    """``identification.get_jacobians`` at order 1.  ``derivs`` is :func:`pydyn.params_derivs.perturbation_params_derivs`'s
    result; ``indvobs`` the 0-based positions of the observables in decision-rule order.  Returns a dict."""
    from .perturbation import dr_ordering
    order_var = derivs["order_var"]
    n, nexo = model.n_endo, model.n_exo
    nlags, useautocorr, grid_nbr = int(opts["ar"]), opts["useautocorr"], int(opts["grid_nbr"])
    totparam = derivs["dSigma_e"].shape[2]
    nm = derivs["dYss"].shape[1]
    obs_nbr = len(indvobs)
    Yss = ys[order_var]
    lli = model.lead_lag_incidence
    cols = [p * n + i for p in range(3) for i in range(n) if lli[i, p]] + list(3 * n + np.arange(nexo))
    J = model.dynamic_jac(np.tile(ys, 3), model.exo_steady(), pp, ys)
    g1 = J[:, cols]
    dg1 = derivs["dg1"][:, cols, :]
    r = {}
    r["DYNAMIC"] = np.concatenate([Yss, vec(g1)])
    r["dDYNAMIC"] = np.vstack([derivs["dYss"], dg1.reshape(dg1.shape[0] * dg1.shape[1], nm, order="F")])
    ghxx = derivs["ghx"]
    r["REDUCEDFORM"] = np.concatenate([Yss, vec(ghxx), vech_dyn(ghu @ Sigma @ ghu.T)])
    dRF = np.zeros((n * nspred + n * (n + 1) // 2, totparam))
    for j in range(totparam):
        dRF[:, j] = np.concatenate([vec(derivs["dghx"][:, :, j]), vech_dyn(derivs["dOm"][:, :, j])])
    r["dREDUCEDFORM"] = np.vstack([np.hstack([np.zeros((n, nsc)), derivs["dYss"]]), dRF])
    pr = pruned_first_order(ghxx, ghu, Sigma, Yss, indvobs, nlags, useautocorr, nstatic, nspred,
                            dict(derivs, nsc=nsc))
    r["pruned"] = pr
    r["MEAN"], r["dMEAN"] = pr["E_y"], pr["dE_y"]
    if not opts["no_identification_moments"]:
        E_yyi = pr["Corr_yi"] if useautocorr else pr["Var_yi"]
        dE_yyi = pr["dCorr_yi"] if useautocorr else pr["dVar_yi"]
        mom = [pr["E_y"], vech_dyn(pr["Var_y"])] + [vec(E_yyi[:, :, i]) for i in range(nlags)]
        r["MOMENTS"] = np.concatenate(mom)
        nv = obs_nbr * (obs_nbr + 1) // 2
        d = np.zeros((obs_nbr + nv + nlags * obs_nbr ** 2, totparam))
        d[:obs_nbr] = pr["dE_y"]
        for jp in range(totparam):
            d[obs_nbr:obs_nbr + nv, jp] = vech_dyn(pr["dVar_y"][:, :, jp])
            for i in range(nlags):
                a = obs_nbr + nv + i * obs_nbr ** 2
                d[a:a + obs_nbr ** 2, jp] = vec(dE_yyi[:, :, i, jp])
        r["dMOMENTS"] = d
    else:
        r["MOMENTS"], r["dMOMENTS"] = np.zeros(0), np.zeros((0, totparam))
    if not opts["no_identification_spectrum"]:
        nf = grid_nbr // 2 + 1
        freqs = np.arange(nf) * (np.pi / (grid_nbr / 2))
        A, B, C, D = pr["A"], pr["B"], pr["C"], pr["D"]
        IA = np.eye(A.shape[0])
        S = np.zeros((totparam, totparam), complex)
        Sig = pr["Varinov"]
        dSig = pr["dVarinov"]
        w = np.full(nf, 2.0)
        w[0] = 1.0
        chunk = max(1, int(2e6 // max(1, obs_nbr ** 2 * totparam)))
        for a in range(0, nf, chunk):
            z = np.exp(1j * freqs[a:a + chunk])
            inv = np.linalg.inv(z[:, None, None] * IA - A)                # (F, nx, nx)
            T = D + C @ inv @ B                                           # transfer function (F, obs, nexo)
            Th = np.conj(np.swapaxes(T, 1, 2))
            dOm = np.zeros((len(z), obs_nbr ** 2, totparam), complex)
            for j in range(totparam):
                if j < nsc:
                    t = T @ dSig[:, :, j] @ Th
                else:
                    dinv = -inv @ (-pr["dA"][:, :, j]) @ inv
                    dT = pr["dD"][:, :, j] + pr["dC"][:, :, j] @ inv @ B + C @ dinv @ B + C @ inv @ pr["dB"][:, :, j]
                    t = dT @ Sig @ Th + T @ dSig[:, :, j] @ Th + T @ Sig @ np.conj(np.swapaxes(dT, 1, 2))
                dOm[:, :, j] = (1 / (2 * np.pi)) * np.swapaxes(t, 1, 2).reshape(len(z), -1)   # vec(), column-major
            wc = w[a:a + chunk]
            S += np.einsum("f,fpi,fpj->ij", wc, np.conj(dOm), dOm)
        S = np.real(2 * np.pi * S / (2 * nf - 1))
        r["dSPECTRUM_NO_MEAN"] = S
        r["dSPECTRUM"] = S + pr["dE_y"].T @ pr["dE_y"]
    else:
        r["dSPECTRUM_NO_MEAN"] = r["dSPECTRUM"] = np.zeros((0, 0))
    r["dMINIMAL"] = None
    r["minimal_message"] = ""
    if not opts["no_identification_minimal"]:
        if obs_nbr < nexo:
            r["minimal_message"] = ("WARNING: Komunjer and Ng (2011) failed:\n       There are more shocks and measurement "
                                    "errors than observables, this is not implemented (yet).\n       Skip identification "
                                    "analysis based on minimal state space system.\n")
        else:
            SYS = {"A": ghxx[pr["indx"], :],
                   "dA": derivs["dghx"][pr["indx"], :, :], "B": ghu[pr["indx"], :], "dB": derivs["dghu"][pr["indx"], :, :],
                   "C": ghxx[indvobs, :], "dC": derivs["dghx"][indvobs, :, :], "D": ghu[indvobs, :],
                   "dD": derivs["dghu"][indvobs, :, :]}
            ok, minnx, mS = get_minimal_state_representation(SYS, True)
            if not ok:
                r["minimal_message"] = ("WARNING: Komunjer and Ng (2011) failed:\n         Conditions for minimality are not "
                                        "fullfilled:\n         Skip identification analysis based on minimal state space "
                                        "system.\n")
            else:
                tot = totparam
                dminA = mS["dA"].reshape(minnx * minnx, tot, order="F")
                dminB = mS["dB"].reshape(minnx * nexo, tot, order="F")
                dminC = mS["dC"].reshape(obs_nbr * minnx, tot, order="F")
                dminD = mS["dD"].reshape(obs_nbr * nexo, tot, order="F")
                dvechSig = derivs["dSigma_e"].reshape(nexo * nexo, tot, order="F")
                ind = np.flatnonzero(np.tril(np.ones((nexo, nexo))).ravel(order="F"))
                dvechSig = dvechSig[ind, :]
                Inx, Inu = np.eye(minnx), np.eye(nexo)
                Enu = duplication(nexo)[1]
                mA, mB, mC, mD = mS["A"], mS["B"], mS["C"], mS["D"]
                nv = nexo * (nexo + 1) // 2
                DL = np.vstack([dminA, dminB, dminC, dminD, dvechSig])
                DT = np.vstack([np.kron(mA.T, Inx) - np.kron(Inx, mA), np.kron(mB.T, Inx), -np.kron(Inx, mC),
                                np.zeros((obs_nbr * nexo, minnx ** 2)), np.zeros((nv, minnx ** 2))])
                DU = np.vstack([np.zeros((minnx ** 2, nexo ** 2)), np.kron(Inu, mB), np.zeros((obs_nbr * minnx, nexo ** 2)),
                                np.kron(Inu, mD), -2 * Enu @ np.kron(Sigma, Inu)])
                dM = np.hstack([DL, DT, DU])
                r["dMINIMAL"] = np.vstack([np.hstack([pr["dE_y"], np.zeros((obs_nbr, minnx ** 2 + nexo ** 2))]), dM])
    return r


# ---------------------------------------------------------------------------------------------------- asymptotic Hessian
class _SingularF(Exception):
    """kalman_filter.m returns an infinite likelihood where the prediction error covariance is badly conditioned; the
    caller then fails (dsge_likelihood indexes a cell with it) and identification.analysis falls back to simulation."""


def univariate_asymptotic_hessian(Y, T, R, Q, Z, DT, DYss, DOm, Pstar, kalman_tol=1e-10, riccati_tol=1e-6, presample=0):
    """The ``asymptotic`` Hessian of ``univariate_kalman_filter.m`` (no measurement error), with
    ``univariate_computeDLIK``/``univariate_computeDstate`` and ``univariate_kalman_filter_ss``: the filter dsge_likelihood.m
    switches to where the prediction error covariance is singular.  Same arguments and result as
    :func:`asymptotic_hessian`."""
    mm, k = T.shape[0], DT.shape[2]
    Z = np.asarray(Z, int)
    pp = len(Z)
    QQ = R @ Q @ R.T
    nobs = Y.shape[1]
    a = np.zeros(mm)
    P = Pstar.copy()
    DP = np.zeros((mm, mm, k))
    for i in range(k):
        rhs = DT[:, :, i] @ Pstar @ T.T + T @ Pstar @ DT[:, :, i].T + DOm[:, :, i]
        DP[:, :, i] = _lyap(T, rhs)
    Da = np.zeros((mm, k))
    Hess = np.zeros((k, k))
    DDK = np.zeros((mm, pp, k))
    DDF = np.zeros((pp, k))
    K = np.full((mm, pp), np.nan)
    oldK = np.full(mm * pp, np.inf)

    def dstate(a, P, Da, DP, notsteady):
        Da1 = np.zeros_like(Da)
        DP1 = np.zeros_like(DP)
        for j in range(k):
            Da1[:, j] = T @ Da[:, j] + DT[:, :, j] @ a
            if notsteady:
                DP1[:, :, j] = T @ DP[:, :, j] @ T.T + DT[:, :, j] @ P @ T.T + T @ P @ DT[:, :, j].T
            else:
                DP1 = DP
        if notsteady:
            DP1 = DP1 + DOm
        return Da1, DP1

    notsteady, t = True, 0
    while notsteady and t < nobs:
        P = (P + P.T) / 2
        for i in range(pp):
            z = Z[i]
            pe = Y[i, t] - a[z]
            PZ = P[:, z].copy()
            Fi = PZ[z]
            if Fi > kalman_tol:
                Ki = PZ / Fi
                K[:, i] = Ki
                Dv = -Da[z, :] - DYss[z, :]
                DF = DP[z, z, :].copy()
                DK = DP[:, z, :] / Fi - np.outer(PZ, DF) / Fi ** 2
                DDK[:, i, :] = DK
                DDF[i, :] = DF
                Hesst = np.outer(DF, DF) / Fi ** 2 + 2 * np.outer(Dv, Dv) / Fi
                Da = Da + DK * pe + np.outer(Ki, Dv)
                DP1 = np.zeros_like(DP)
                for j in range(k):
                    DP1[:, :, j] = DP[:, :, j] - np.outer(DP[:, z, j], Ki) - np.outer(PZ, DK[:, j])
                DP = DP1
                if t + 1 > presample:
                    Hess = Hess + Hesst
                a = a + Ki * pe
                P = P - np.outer(PZ, Ki)
                P = (P + P.T) / 2
            elif Fi < 0:
                raise NotImplementedError("identification: negative prediction error variance in the univariate filter")
        Da, DP = dstate(a, P, Da, DP, True)
        a = T @ a
        P = T @ P @ T.T + QQ
        Kv = K.ravel(order="F")
        diff = np.abs(Kv - oldK)
        notsteady = bool(np.any(~np.isnan(diff)) and np.nanmax(diff) > riccati_tol)
        oldK = Kv.copy()
        t += 1
    Hess = -Hess / 2
    if t < nobs:
        H2 = np.zeros((k, k))
        for tt in range(t, nobs):
            PP = P.copy()
            for i in range(pp):
                z = Z[i]
                pe = Y[i, tt] - a[z]
                PPZ = PP[:, z].copy()
                Fi = PPZ[z]
                if Fi > kalman_tol:
                    Ki = PPZ / Fi
                    a = a + Ki * pe
                    PP = PP - np.outer(PPZ, Ki)
                    Dv = -Da[z, :] - DYss[z, :]
                    DK, DF = DDK[:, i, :], DDF[i, :]
                    H2 = H2 + np.outer(DF, DF) / Fi ** 2 + 2 * np.outer(Dv, Dv) / Fi
                    Da = Da + DK * pe + np.outer(Ki, Dv)
            Da, _ = dstate(a, P, Da, DP, False)
            a = T @ a
        Hess = Hess + (-H2 / 2)
    return Hess


def _rcond(F):
    """Octave's rcond (reciprocal condition number in the 1-norm)."""
    try:
        return 1.0 / (np.linalg.norm(F, 1) * np.linalg.norm(np.linalg.inv(F), 1))
    except np.linalg.LinAlgError:
        return 0.0


def asymptotic_hessian(Y, T, R, Q, Z, DT, DYss, DOm, Pstar, kalman_tol=1e-10, riccati_tol=1e-6, presample=0):
    """The ``asymptotic`` Hessian of ``kalman_filter.m`` (``analytic_Hessian='asymptotic'``) as ``dsge_likelihood.m`` returns it
    (``Hess = LIK1{3}``, the negative of the information matrix): multivariate filter with analytic score and
    ``Hesst = 2 Dv' iF Dv + vec(DF)' (iF kron iF) vec(DF)`` each period, then the steady-state filter once the gain has
    converged.  ``Y`` [pp x T] (data minus the steady state of the observables), ``Z`` 0-based positions of the observables
    in the restricted state vector, ``DT``/``DOm`` [mm x mm x k], ``DYss`` [mm x k] (columns of all k parameters)."""
    mm, k = T.shape[0], DT.shape[2]
    pp = len(Z)
    Z = np.asarray(Z, int)
    QQ = R @ Q @ R.T
    nobs = Y.shape[1]
    a = np.zeros(mm)
    P = Pstar.copy()
    # DP: derivatives of Pstar (lik_init=1): the first nsc are stderr/corr (no DT), the model ones have DT
    DP = np.zeros((mm, mm, k))
    for i in range(k):
        rhs = DT[:, :, i] @ Pstar @ T.T + T @ Pstar @ DT[:, :, i].T + DOm[:, :, i]
        DP[:, :, i] = _lyap(T, rhs)
    Da = np.zeros((mm, k))
    Hess = np.zeros((k, k))
    DK = DF = None
    notsteady, oldK, t = True, np.full(1, np.inf), 0
    K = iF = None

    def step(Da, v, tmp, K, iF, DK, DF):
        Dv = -Da[Z, :] - DYss[Z, :]
        Dtmp = np.empty_like(Da)
        for ii in range(k):
            Dtmp[:, ii] = Da[:, ii] + DK[:, :, ii] @ v + K @ Dv[:, ii]
            Da[:, ii] = DT[:, :, ii] @ tmp + T @ Dtmp[:, ii]
        vecDF = DF.reshape(pp * pp, k, order="F")
        Hesst = 2 * Dv.T @ iF @ Dv + vecDF.T @ np.kron(iF, iF) @ vecDF
        return Da, Hesst

    while notsteady and t < nobs:
        v = Y[:, t] - a[Z]
        F = P[np.ix_(Z, Z)]
        rescale = False
        if _rcond(F) < kalman_tol:
            sig = np.sqrt(np.diag(F))
            if np.any(np.diag(F) < kalman_tol) or _rcond(F / np.outer(sig, sig)) < kalman_tol:
                raise _SingularF()
            rescale = True
        iF = np.linalg.inv(F / np.outer(sig, sig)) / np.outer(sig, sig) if rescale else np.linalg.inv(F)
        K = P[:, Z] @ iF
        Ptmp = T @ (P - K @ P[Z, :]) @ T.T + QQ
        tmp = a + K @ v
        # computeDKalman
        tmpP = P - K @ P[Z, :]
        DF = np.zeros((pp, pp, k))
        DK = np.zeros((mm, pp, k))
        DP1 = np.zeros_like(DP)
        for ii in range(k):
            DF[:, :, ii] = DP[np.ix_(Z, Z)][:, :, ii]
            DiF = -iF @ DF[:, :, ii] @ iF
            DK[:, :, ii] = DP[:, Z, ii] @ iF + P[:, Z] @ DiF
            Dtmp = DP[:, :, ii] - DK[:, :, ii] @ P[Z, :] - K @ DP[Z, :, ii]
            DP1[:, :, ii] = DT[:, :, ii] @ tmpP @ T.T + T @ Dtmp @ T.T + T @ tmpP @ DT[:, :, ii].T + DOm[:, :, ii]
        DP = DP1
        Da, Hesst = step(Da, v, tmp, K, iF, DK, DF)
        if t + 1 > presample:
            Hess = Hess + Hesst
        a = T @ tmp
        P = Ptmp
        notsteady = np.max(np.abs(K.ravel(order="F") - oldK)) > riccati_tol
        oldK = K.ravel(order="F")
        t += 1
    Hess = -Hess / 2
    if t < nobs:
        H2 = np.zeros((k, k))
        for tt in range(t, nobs):
            v = Y[:, tt] - a[Z]
            tmp = a + K @ v
            Da, Hesst = step(Da, v, tmp, K, iF, DK, DF)
            H2 = H2 + Hesst
            a = T @ tmp
        Hess = Hess + (-H2 / 2)
    return Hess


# ---------------------------------------------------------------------------------------------------- options
_FLAG_OPTIONS = ("no_identification_strength", "no_identification_reducedform", "no_identification_moments",
                 "no_identification_minimal", "no_identification_spectrum")
_COSMETIC = {"nodisplay", "nograph", "graph_format", "tex", "TeX", "prior_trunc", "gsa_sample_file", "load_ident_files",
             "diffuse_filter", "lik_init", "schur_vec_tol", "kalman_algo", "kalman_tol", "diffuse_kalman_tol",
             "use_univariate_filters_if_singularity_is_detected", "heteroskedastic_filter", "analytic_derivation",
             "periods", "replic", "prior_range"}
_KNOWN = {"parameter_set", "ar", "useautocorr", "normalize_jacobians", "tol_rank", "tol_deriv", "tol_sv",
          "checks_via_subsets", "max_dim_subsets_groups", "max_dim_cova_group", "grid_nbr", "advanced", "prior_mc", "order",
          "analytic_derivation_mode", *_FLAG_OPTIONS} | _COSMETIC


def identification_options(raw, totparam_nbr, prior_exist, default_mode=0):
    """``options_ident`` as run.m completes it (defaults of ``set_default_option``)."""
    raw = dict(raw or {})
    unknown = sorted(set(raw) - _KNOWN)
    if unknown:
        raise NotImplementedError("identification option(s) not supported: " + ", ".join(unknown))
    o = {"gsa_sample_file": 0, "parameter_set": "prior_mean", "load_ident_files": 0, "useautocorr": 0, "ar": 1,
         "prior_mc": 1, "prior_range": 0, "periods": 300, "replic": 100, "advanced": 0, "normalize_jacobians": 1,
         "grid_nbr": 5000, "tol_rank": "robust", "tol_deriv": 1e-8, "tol_sv": 1e-3, "schur_vec_tol": 1e-11}
    for k, v in raw.items():
        o[k] = v
    for k in ("ar", "grid_nbr", "prior_mc", "advanced", "useautocorr", "normalize_jacobians", "load_ident_files"):
        o[k] = int(o[k])
    if o["grid_nbr"] % 2:
        o["grid_nbr"] += 1
    for f in _FLAG_OPTIONS:
        o[f] = 1 if f in raw else 0
    if not o["no_identification_strength"] or o["advanced"] == 1:
        o["no_identification_reducedform"] = 0
        o["no_identification_moments"] = 0
    if o["gsa_sample_file"] or o["load_ident_files"]:
        raise NotImplementedError("identification: gsa_sample_file and load_ident_files are not supported")
    o.setdefault("lik_init", 1)
    if raw.get("diffuse_filter"):
        raise NotImplementedError("identification: the diffuse filter (unit roots) is not supported")
    o.setdefault("analytic_derivation", 1)
    o.setdefault("order", 1)
    o["order"] = int(o["order"])
    o.setdefault("analytic_derivation_mode", default_mode)
    o["analytic_derivation_mode"] = int(o["analytic_derivation_mode"])
    if not prior_exist:
        o["prior_mc"], o["prior_range"] = 1, 0
    if totparam_nbr < 2:
        o["advanced"] = 0
    o["max_dim_cova_group"] = min(int(o.get("max_dim_cova_group", min(2, totparam_nbr - 1))), totparam_nbr - 1)
    o.setdefault("checks_via_subsets", 0)
    o["checks_via_subsets"] = int(o["checks_via_subsets"])
    o["max_dim_subsets_groups"] = int(o.get("max_dim_subsets_groups", min(4, totparam_nbr - 1)))
    return o


# ---------------------------------------------------------------------------------------------------- the analysis
def _rows_above(M, tol):
    """``find(max(abs(M'),[],1) > tol)``: rows of M with an element above tol (0-based)."""
    if M.size == 0:
        return np.zeros(0, int)
    return np.flatnonzero(np.max(np.abs(M), axis=1) > tol)


def _empty_checks(param_nbr):
    """``checks`` of an empty Jacobian (no parameter to study): what Octave returns."""
    return {"cond": 0.0, "rank": 0.0, "ind0": np.zeros((1, 0)), "indno": np.zeros((1, 0)), "ino": 0.0,
            "Mco": np.zeros((0, 1)), "Pco": np.zeros((0, 0)), "jweak": np.zeros((1, 0)), "jweak_pair": np.zeros((1, 0))}


def _norm_rows(M):
    if M.size == 0:
        return np.zeros((0, 0))
    return np.repeat(np.max(np.abs(M), axis=1)[:, None], M.shape[1], axis=1)


def _svd_SV(M, totparam_nbr):
    """``[~,S,V] = svd(M,0)`` padded as analysis.m does."""
    U, s, Vh = linalg.svd(M, full_matrices=M.shape[0] < M.shape[1], lapack_driver="gesvd")
    V = Vh.T
    S = np.concatenate([s, np.zeros(M.shape[1] - M.shape[0])]) if M.shape[0] < M.shape[1] else s
    if totparam_nbr > 8:
        sel = list(range(4)) + list(range(len(S) - 4, len(S)))
        return S[sel], V[:, sel]
    return S, V


def _remove_problematic(idx, problparset):
    """``RemoveProblematicParameterSets``: drop the subsets that contain an already known problematic set."""
    idx = [list(r) for r in idx]
    if not idx:
        return idx
    iset = len(idx[0])
    for iii in range(1, iset):
        sets = problparset[iii - 1]
        if sets is None or len(sets) == 0:
            continue
        if iii == 1:
            sets = [[v] for v in sets]
        for st in sets:
            ss = set(st)
            idx = [r for r in idx if sum(1 for x in r if x in ss) != iii]
    return idx


def bruteforce(J, max_dim_cova_group, tol_deriv):
    """``identification.bruteforce``: for every parameter the group of up to ``max_dim_cova_group`` other parameters whose
    Jacobian columns best reproduce its column (multiple cosine, largest over all groups of that size). Returns the cell
    ``pars`` (totparam x max_dim, 1-based indices, NaN for no collinearity, [] if not reached) and ``cosnJ``."""
    tot = J.shape[1]
    cosnJ = np.zeros((tot, max_dim_cova_group))
    pars = np.empty((tot, max_dim_cova_group), dtype=object)
    for r in range(tot):
        for c in range(max_dim_cova_group):
            pars[r, c] = np.zeros((0, 0))
    for ll in range(1, max_dim_cova_group + 1):
        for ii in range(tot):
            tmp = [k for k in range(tot) if k != ii]
            combos = list(itertools.combinations(tmp, ll))
            c2 = np.array([cosn(np.hstack([J[:, [ii]], J[:, list(cm)]])) for cm in combos])
            cosnJ[ii, ll - 1] = np.max(c2)
            if cosnJ[ii, ll - 1] > tol_deriv:
                if ll > 1 and (cosnJ[ii, ll - 1] - cosnJ[ii, ll - 2]) < tol_deriv:
                    pars[ii, ll - 1] = np.concatenate([np.ravel(pars[ii, ll - 2]), [np.nan]]).reshape(1, -1)
                    cosnJ[ii, ll - 1] = cosnJ[ii, ll - 2]
                else:
                    k = int(np.flatnonzero(c2 == np.max(c2))[0])
                    pars[ii, ll - 1] = (np.array(combos[k], float) + 1).reshape(1, -1)
            else:
                pars[ii, ll - 1] = np.full((1, ll), np.nan)
    return pars, cosnJ


def _grow(a, n):
    """MATLAB/Octave growth by assignment: a larger array with zeros in the new rows."""
    if a.shape[0] >= n:
        return a
    return np.vstack([a, np.zeros((n - a.shape[0], a.shape[1]))])


class IdentificationStop(Exception):
    """The model does not solve at the point tested (Dynare prints a message and returns)."""


class Identification:
    """One ``identification`` statement on a :class:`pydyn.Model`: the parameter set (``estimated_params`` or all stderr and
    model parameters), the solution and the Jacobians at a point (:meth:`analysis`) and the statement itself
    (:meth:`run_point`)."""

    def __init__(self, model, raw_options, estimation=None, rng=None, no_priors=False):
        self.rng = rng
        from .estimation import Estimation
        self.model = model
        self.raw = dict(raw_options or {})
        self.varobs = list(model.varobs)
        if not self.varobs:
            raise NotImplementedError("identification: the model has no varobs statement (Dynare stops there too)")
        self.est = estimation
        if estimation is None and not no_priors and any(st.get("statementName") == "estimated_params"
                                                        for st in model.statements):
            self.est = _IdentEstimation(model, order=1)
        self.prior_exist = self.est is not None
        n_exo = model.n_exo
        if self.prior_exist:
            g = self.est._groups
            if g["vn"] or g["cn"]:
                raise NotImplementedError("Identification does not (yet) support measurement errors. Instead, define them "
                                          "explicitly as varexo and provide measurement equations in the model definition.")
            self.indpmodel = np.array([k for _, k, _ in g["p"]], int)
            self.indpstderr = np.array([k for _, k, _ in g["vx"]], int)
            self.indpcorr = np.array([k for _, k, _ in g["cx"]], int).reshape(-1, 2)
            self.names = list(self.est.names)
        else:
            self.indpmodel = np.arange(len(model.param_names))
            self.indpstderr = np.arange(n_exo)
            self.indpcorr = np.zeros((0, 2), int)
            self.names = [f"SE_{e}" for e in model.exo_names] + list(model.param_names)
        self.nsc = len(self.indpstderr) + len(self.indpcorr)
        self.totparam_nbr = self.nsc + len(self.indpmodel)
        self.modparam_nbr = len(self.indpmodel)
        self.stderrparam_nbr = len(self.indpstderr)
        self.corrparam_nbr = len(self.indpcorr)
        from .perturbation import dr_ordering
        order_var, state_var, cnt = dr_ordering(model.lead_lag_incidence)
        self.order_var = order_var
        inv = np.empty(model.n_endo, int)
        inv[order_var] = np.arange(model.n_endo)
        self.indvobs = inv[[model._ie[v] for v in self.varobs]]
        self.nstatic, self.nspred = cnt["nstatic"], cnt["npred"] + cnt["nboth"]
        self.opts = identification_options(self.raw, self.totparam_nbr, self.prior_exist)
        self._pers = {}
        self._set_derivation_mode(default_mode_set="analytic_derivation_mode" in self.raw)
        if self.prior_exist and not self.est.prior.is_bayesian:
            self.opts["prior_mc"] = 1                 # only bounds in estimated_params: one single point
        if self.opts["order"] != 1:
            raise NotImplementedError("identification: orders 2 and 3 are not ported (order=1 only)")
        if self.opts["analytic_derivation_mode"] not in (0, 1, -2):
            raise NotImplementedError("identification: analytic_derivation_mode -1 (finite differences of all Jacobians) "
                                      "is not ported; the modes 0, 1 and -2 are")

    def _set_derivation_mode(self, default_mode_set):
        """dynare_estimation_init.m: where the steady-state file changes parameter values (or leaves some NaN) at the
        estimated parameters varied by 1 %, Dynare re-sets ``analytic_derivation_mode`` to -2 (numerical derivatives of
        the steady state and the dynamic Jacobian) unless the statement gives the mode itself."""
        from .steady import steady_state, SteadyStateError
        o = self.opts
        if not o["analytic_derivation"] or o["order"] != 1:
            return
        m = self.model
        if self.prior_exist:
            if len(self.indpmodel) == 0:
                return
            x0 = np.array(self.est.x0, float)
            pvec = np.array(self.est.calibration(x0)["params"], float)
            pvec[self.indpmodel] *= 1.01
        else:
            pvec = np.array(m.params, float) * 1.01
        try:
            _, pp = steady_state(m, pvec)
        except (SteadyStateError, NotImplementedError):
            return
        pp = np.asarray(pp, float)
        with np.errstate(invalid="ignore"):
            diff = pp - pvec
        if np.any(diff != 0):                                 # NaN differences count, as find() does
            if not default_mode_set:
                o["analytic_derivation_mode"] = -2
                self.messages_init = ["The steadystate file changed the values of parameters or left some NaN: the "
                                      "derivatives of the Jacobian and steady state are computed numerically "
                                      "(analytic_derivation_mode = -2)"]

    # ---------------------------------------------------------------------------------------- parameter vectors
    def calibration(self):
        """``get_all_parameters``: the model's own values of the parameters studied, in Dynare's order."""
        m = self.model
        if not self.prior_exist:
            return np.concatenate([np.sqrt(np.diag(np.asarray(m.Sigma_e, float))), np.asarray(m.params, float)])
        e = self.est
        sd = np.sqrt(np.diag(np.asarray(m.Sigma_e, float)))
        out = [sd[k] for k in self.indpstderr] + [e._corr0[a, b] for a, b in self.indpcorr] \
              + [m.params[k] for k in self.indpmodel]
        return np.array(out, float)

    def point(self):
        """The parameter vector and its label for the statement's ``parameter_set`` (run.m)."""
        ps = self.opts["parameter_set"]
        if not self.prior_exist:
            return self.calibration(), "Current_params"
        e = self.est
        if not e.prior.is_bayesian:
            return np.array(e.x0, float), "ML_Starting_value"
        if ps == "calibration":
            return self.calibration(), "calibration"
        if ps == "prior_mean":
            return np.array(e.prior.p1, float), "prior_mean"
        raise NotImplementedError(f"identification: parameter_set={ps} is not ported (calibration and prior_mean are)")

    # ---------------------------------------------------------------------------------------- solution
    def solve(self, theta):
        """Steady state, first-order solution and parameter derivatives at ``theta`` (``compute_decision_rules`` +
        ``get_perturbation_params_derivs``); raises :class:`IdentificationStop` where the model does not solve."""
        from .params_derivs import perturbation_params_derivs, ParamsDerivsError
        from .perturbation import BlanchardKahnError
        from .steady import steady_state, SteadyStateError
        m = self.model
        theta = np.asarray(theta, float)
        if self.prior_exist:
            cal = self.est.calibration(theta)
            pvec, Sigma = cal["params"], cal["Sigma_e"]
        else:
            pvec, Sigma = np.array(theta[m.n_exo:], float), np.array(m.Sigma_e, float)
        old_sigma = m.Sigma_e
        try:
            m.Sigma_e = Sigma
            try:
                ys, pp = steady_state(m, pvec)
                pp = np.asarray(pp, float)
                if not (np.all(np.isfinite(ys)) and np.all(np.isfinite(pp))):
                    raise SteadyStateError("steady state or parameters are not finite")
                numerical = self.opts["analytic_derivation_mode"] == -2
                if not numerical and np.any((pp != pvec) & ~(np.isnan(pp) & np.isnan(pvec))):
                    raise NotImplementedError("identification: the steady-state file changes parameter values although "
                                              "analytic_derivation_mode is 0 or 1 (Dynare's analytic derivatives are then "
                                              "wrong); use analytic_derivation_mode=-2")
                d = perturbation_params_derivs(m, self.indpmodel, self.indpstderr, self.indpcorr, ys=ys, params=pp,
                                               numerical=numerical)
            except (SteadyStateError, BlanchardKahnError) as exc:
                raise IdentificationStop(str(exc)) from None
            except ParamsDerivsError as exc:
                raise NotImplementedError(f"identification: {exc}") from None
        finally:
            m.Sigma_e = old_sigma
        return {"ys": ys, "pp": pp, "Sigma": Sigma, "derivs": d}

    # ---------------------------------------------------------------------------------------- analysis.m
    def analysis(self, theta, init=True):
        """``identification.analysis`` at one point: dicts ide_moments, ide_spectrum, ide_minimal, ide_hess,
        ide_reducedform, ide_dynamic and the error indicators."""
        o = self.opts
        tot, modn = self.totparam_nbr, self.modparam_nbr
        tol_deriv, tol_rank, tol_sv = o["tol_deriv"], o["tol_rank"], o["tol_sv"]
        nlags = o["ar"]
        sol = self.solve(theta)
        ghx, ghu = sol["derivs"]["ghx"], sol["derivs"]["ghu"]
        J = get_jacobians(self.model, ghx, ghu, sol["ys"], sol["pp"], sol["Sigma"], sol["derivs"], self.indvobs,
                          self.nstatic, self.nspred, self.nsc, o)
        err = {"identification_strength": 0, "identification_reducedform": 0, "identification_moments": 0,
               "identification_minimal": 0, "identification_spectrum": 0}
        messages = [J["minimal_message"]] if J["minimal_message"] else []
        dMINIMAL = J["dMINIMAL"]
        if dMINIMAL is None:
            err["identification_minimal"] = 1
        MOMENTS, dMOMENTS = J["MOMENTS"], J["dMOMENTS"]
        dSPECTRUM = J["dSPECTRUM"]
        ind_dMOMENTS = ind_dSPECTRUM = ind_dMINIMAL = ind_dREDUCEDFORM = None
        if not init:
            # persistent indices of the first call (analysis.m): in a Monte Carlo loop the thresholds could give
            # vectors of different length
            ind_dMOMENTS, ind_dREDUCEDFORM, ind_dDYNAMIC = (self._pers.get(k) for k in
                                                            ("moments", "reducedform", "dynamic"))
            if not o["no_identification_moments"] and ind_dMOMENTS is None:
                raise ValueError("identification: no stored indices of the first analysis")
        if init:
            if not o["no_identification_moments"]:
                if np.any(np.isnan(MOMENTS)):
                    raise ValueError("There are NaN's in the theoretical moments.")
                ind_dMOMENTS = _rows_above(dMOMENTS, tol_deriv)
            if not o["no_identification_spectrum"]:
                ind_dSPECTRUM = _rows_above(dSPECTRUM, tol_deriv)
            if not o["no_identification_minimal"] and dMINIMAL is not None:
                ind_dMINIMAL = _rows_above(dMINIMAL, tol_deriv)
            if err["identification_moments"] and err["identification_minimal"] and err["identification_spectrum"]:
                raise ValueError("identification_analyis: Stationarity condition(s) failed")
            if not o["no_identification_moments"] and not err["identification_moments"]:
                opts_local = dict(o)
                while len(ind_dMOMENTS) < tot and nlags < 10:
                    messages.append("The number of moments with non-zero derivative is smaller than the number of parameters\n"
                                    f"Try increasing ar = {nlags + 1}")
                    nlags += 1
                    opts_local.update(no_identification_minimal=1, no_identification_spectrum=1, ar=nlags)
                    Jl = get_jacobians(self.model, ghx, ghu, sol["ys"], sol["pp"], sol["Sigma"], sol["derivs"], self.indvobs,
                                       self.nstatic, self.nspred, self.nsc, opts_local)
                    MOMENTS, dMOMENTS = Jl["MOMENTS"], Jl["dMOMENTS"]
                    ind_dMOMENTS = _rows_above(dMOMENTS, tol_deriv)
                if len(ind_dMOMENTS) < tot:
                    messages.append("WARNING: Order condition for dMOMENTS failed: There are not enough moments and too many "
                                    "parameters.\n         Skip identification analysis based on moments.\n"
                                    "         Skip identification strength analysis.")
                    err["identification_moments"] = 1
                    err["identification_strength"] = 1
            if not o["no_identification_minimal"] and not err["identification_minimal"]:
                if len(ind_dMINIMAL) < dMINIMAL.shape[1]:
                    messages.append("WARNING: Order condition for dMINIMAL failed: There are too many parameters or too few "
                                    "observable variables.\n         Skip identification analysis based on minimal state "
                                    "space system.")
                    err["identification_minimal"] = 1
            if err["identification_moments"] and err["identification_minimal"] and err["identification_spectrum"]:
                raise ValueError("identification_analyis: Order condition(s) failed")
        dREDUCEDFORM, dDYNAMIC = J["dREDUCEDFORM"], J["dDYNAMIC"]
        if init:
            if not o["no_identification_reducedform"] and not err["identification_reducedform"]:
                ind_dREDUCEDFORM = _rows_above(dREDUCEDFORM, tol_deriv)
            ind_dDYNAMIC = _rows_above(dDYNAMIC, tol_deriv)
        DYNAMIC = J["DYNAMIC"][ind_dDYNAMIC]
        si_dDYNAMIC = dDYNAMIC[ind_dDYNAMIC, :]
        REDUCEDFORM, si_dREDUCEDFORM = J["REDUCEDFORM"], None
        if not o["no_identification_reducedform"] and not err["identification_reducedform"]:
            REDUCEDFORM = REDUCEDFORM[ind_dREDUCEDFORM]
            si_dREDUCEDFORM = dREDUCEDFORM[ind_dREDUCEDFORM, :]
        ide_hess, ide_dynamic, ide_moments, ide_reducedform = {}, {}, {}, {}
        ide_spectrum, ide_minimal = {}, {}
        if not o["no_identification_moments"] and not err["identification_moments"]:
            MOMENTS = MOMENTS[ind_dMOMENTS]
            si_dMOMENTS = dMOMENTS[ind_dMOMENTS, :]
            if not o["no_identification_strength"] and not err["identification_strength"] and init:
                st = self._strength(theta, sol, J, si_dMOMENTS, si_dREDUCEDFORM, si_dDYNAMIC, ind_dMOMENTS, ind_dREDUCEDFORM,
                                    ind_dDYNAMIC, nlags)
                ide_hess.update(st["ide_hess"])
                ind_dREDUCEDFORM, si_dREDUCEDFORM = st["ind_dREDUCEDFORM"], st["si_dREDUCEDFORM"]
                ind_dDYNAMIC, si_dDYNAMIC = st["ind_dDYNAMIC"], st["si_dDYNAMIC"]
                ide_dynamic["si_dDYNAMICnorm"] = st["si_dDYNAMICnorm"]
                ide_moments["si_dMOMENTSnorm"] = st["si_dMOMENTSnorm"]
                ide_reducedform["si_dREDUCEDFORMnorm"] = st["si_dREDUCEDFORMnorm"]
        if init:
            self._pers = {"moments": ind_dMOMENTS, "reducedform": ind_dREDUCEDFORM, "dynamic": ind_dDYNAMIC}
        norm = o["normalize_jacobians"]
        norm_dDYNAMIC = _norm_rows(si_dDYNAMIC) if norm else 1
        ide_dynamic.update(ind_dDYNAMIC=ind_dDYNAMIC + 1, norm_dDYNAMIC=norm_dDYNAMIC, si_dDYNAMIC=si_dDYNAMIC,
                           dDYNAMIC=dDYNAMIC, DYNAMIC=DYNAMIC)
        if not o["no_identification_reducedform"] and not err["identification_reducedform"]:
            norm_dRF = _norm_rows(si_dREDUCEDFORM) if norm else 1
            ide_reducedform.update(ind_dREDUCEDFORM=ind_dREDUCEDFORM + 1, norm_dREDUCEDFORM=norm_dRF,
                                   si_dREDUCEDFORM=si_dREDUCEDFORM, dREDUCEDFORM=dREDUCEDFORM, REDUCEDFORM=REDUCEDFORM)
        if not o["no_identification_moments"] and not err["identification_moments"]:
            norm_dMOM = _norm_rows(si_dMOMENTS) if norm else 1
            ide_moments.update(ind_dMOMENTS=ind_dMOMENTS + 1, norm_dMOMENTS=norm_dMOM, si_dMOMENTS=si_dMOMENTS,
                               dMOMENTS=dMOMENTS, MOMENTS=MOMENTS)
            if o["advanced"]:
                # no normalisation: the OLS in bruteforce is very sensitive to norm_dMOMENTS
                ide_moments["pars"], ide_moments["cosndMOMENTS"] = bruteforce(dMOMENTS[ind_dMOMENTS, :],
                                                                              o["max_dim_cova_group"], tol_deriv)
            S, V = _svd_SV(dMOMENTS[ind_dMOMENTS, :], tot)
            ide_moments["S"], ide_moments["V"] = S, V
        if not o["no_identification_minimal"] and not err["identification_minimal"]:
            if norm:
                ind_dMINIMAL = _rows_above(dMINIMAL, tol_deriv)
                norm_dMIN = _norm_rows(dMINIMAL[ind_dMINIMAL, :])
            else:
                norm_dMIN = 1
            ide_minimal.update(ind_dMINIMAL=ind_dMINIMAL + 1, norm_dMINIMAL=norm_dMIN, dMINIMAL=dMINIMAL)
        if not o["no_identification_spectrum"] and not err["identification_spectrum"]:
            if norm:
                ind_dSPECTRUM = _rows_above(dSPECTRUM, tol_deriv)
                tilda = np.zeros(dSPECTRUM.shape)
                delta = np.sqrt(np.diag(dSPECTRUM[np.ix_(ind_dSPECTRUM, ind_dSPECTRUM)]))
                tilda[np.ix_(ind_dSPECTRUM, ind_dSPECTRUM)] = dSPECTRUM[np.ix_(ind_dSPECTRUM, ind_dSPECTRUM)] \
                    / np.outer(delta, delta)
                norm_dSPEC = _norm_rows(dSPECTRUM[ind_dSPECTRUM, :])
            else:
                tilda, norm_dSPEC = dSPECTRUM, 1
            ide_spectrum.update(ind_dSPECTRUM=ind_dSPECTRUM + 1, norm_dSPECTRUM=norm_dSPEC, tilda_dSPECTRUM=tilda,
                                dSPECTRUM=dSPECTRUM, dSPECTRUM_NO_MEAN=J["dSPECTRUM_NO_MEAN"])
        nd = lambda a: a / norm_dDYNAMIC if norm else a  # noqa: E731
        if o["checks_via_subsets"]:
            self._checks_via_subsets(dDYNAMIC, ide_dynamic, ide_reducedform, ide_moments, ide_spectrum, ide_minimal, err)
            ide_minimal["minimal_state_space"] = 0.0 if err["identification_minimal"] else 1.0
        else:
            ide_dynamic.update(checks(nd(dDYNAMIC[ind_dDYNAMIC, :]), 1, tol_rank, tol_sv, modn))
            if not o["no_identification_reducedform"] and not err["identification_reducedform"]:
                ide_reducedform.update(checks(dREDUCEDFORM[ind_dREDUCEDFORM, :] / ide_reducedform["norm_dREDUCEDFORM"], 1,
                                              tol_rank, tol_sv, tot))
            if not o["no_identification_moments"] and not err["identification_moments"]:
                ide_moments.update(checks(dMOMENTS[ind_dMOMENTS, :] / ide_moments["norm_dMOMENTS"], 1, tol_rank, tol_sv, tot))
            if not o["no_identification_minimal"]:
                if not err["identification_minimal"]:
                    ide_minimal.update(checks(dMINIMAL[ind_dMINIMAL, :] / ide_minimal["norm_dMINIMAL"], 2, tol_rank, tol_sv,
                                              tot))
                    ide_minimal["minimal_state_space"] = 1.0
                else:
                    ide_minimal["minimal_state_space"] = 0.0
            if not o["no_identification_spectrum"] and not err["identification_spectrum"]:
                ide_spectrum.update(checks(ide_spectrum["tilda_dSPECTRUM"], 3, tol_rank, tol_sv, tot))
        return {"ide_moments": ide_moments, "ide_spectrum": ide_spectrum, "ide_minimal": ide_minimal, "ide_hess": ide_hess,
                "ide_reducedform": ide_reducedform, "ide_dynamic": ide_dynamic, "error_indicator": err,
                "messages": messages, "ar": nlags}

    # ---------------------------------------------------------------------------------------- strength
    def _simulate_data(self, sol, periods):
        """``stoch_simul(M_, options_, oo_, varobs)`` with ``options_.periods = periods + 100``: the random draws and the
        observables of the last ``periods`` periods."""
        from .simulate import simulate
        if self.rng is None:
            raise NotImplementedError("identification: the identification strength measure draws random numbers; no "
                                      "random stream was given")
        m = self.model
        d = sol["derivs"]
        n_sim = periods + 100
        Sigma = sol["Sigma"]
        active = np.flatnonzero(np.diag(Sigma) != 0)
        shocks = np.zeros((n_sim, m.n_exo))
        if len(active):
            chol = np.linalg.cholesky(Sigma[np.ix_(active, active)]).T
            z = self.rng.randn((len(active), n_sim))
            shocks[:, active] = z.T @ chol
        from .perturbation import dr_ordering
        order_var, state_var, _ = dr_ordering(m.lead_lag_incidence)
        dr = {"ys": sol["ys"], "order_var": order_var, "state_var": state_var, "ghx": d["ghx"], "ghu": d["ghu"], "order": 1}
        y = simulate(dr, shocks, order=1, pruning=False, y0=sol["ys"])
        return y[100:, [m._ie[v] for v in self.varobs]]

    def _simulated_moment_uncertainty(self, sol, indx, periods, replic, nlags):
        """``identification.simulated_moment_uncertainty``: covariance matrix of the selected moments over ``replic``
        simulations of ``periods`` periods (``options_.drop`` = 100 dropped, the draws are Dynare's)."""
        from .simulate import simulate
        m = self.model
        d = sol["derivs"]
        Sigma = sol["Sigma"]
        active = np.flatnonzero(np.diag(Sigma) != 0)
        chol = np.linalg.cholesky(Sigma[np.ix_(active, active)]).T
        from .perturbation import dr_ordering
        order_var, state_var, _ = dr_ordering(m.lead_lag_incidence)
        dr = {"ys": sol["ys"], "order_var": order_var, "state_var": state_var, "ghx": d["ghx"], "ghu": d["ghu"], "order": 1}
        ivar = [m._ie[v] for v in self.varobs]
        mm = np.zeros((len(indx), replic))
        for j in range(replic):
            shocks = np.zeros((periods, m.n_exo))
            shocks[:, active] = self.rng.randn((len(active), periods)).T @ chol
            y = simulate(dr, shocks, order=1, pruning=False, y0=sol["ys"])[100:, ivar]
            N = y.shape[0]
            mean = y.mean(axis=0)
            y = y - mean
            var = y.T @ y / N
            s = np.sqrt(np.mean(y * y, axis=0))
            zero = np.abs(s) < 1e-10
            var[zero, :] = 0
            var[:, zero] = 0
            dum = [mean, vech_dyn(var)]
            sdv = np.sqrt(np.diag(var))
            for i in range(1, nlags + 1):
                a0, a1 = y[nlags:, :], y[nlags - i:N - i, :]
                ac = a0.T @ a1 / ((N - nlags) * np.outer(np.std(a0, axis=0, ddof=1), np.std(a1, axis=0, ddof=1)))
                ac[zero, :] = np.nan
                ac[:, zero] = np.nan
                dum.append(vec(ac * np.outer(sdv, sdv)))
            mm[:, j] = np.concatenate(dum)[indx]
        return np.cov(mm)

    def _try_asymptotic_hessian(self, theta, sol, J, data):
        """The ``try`` branch of analysis.m: ``dsge_likelihood`` with ``analytic_Hessian='asymptotic'`` on the simulated data.
        Returns the (negated) asymptotic Hessian or None where Dynare's code fails (the caller then simulates)."""
        from .perturbation import dr_ordering
        from .estimation import _ispd
        m = self.model
        o = self.opts
        if not self.prior_exist:
            return None                   # check_bounds_and_definiteness_estimation: empty bounds, an error in Octave
        est = self.est
        if not est.prior.is_bayesian:
            return None                   # prior_bounds.m: "unknown distribution shape (type 0)" for an ML estimation
        lb, ub = est.prior.bounds(float(o.get("prior_trunc", 1e-10)))
        theta = np.asarray(theta, float)
        if np.any(theta < lb) or np.any(theta > ub):
            return None
        Sigma = sol["Sigma"]
        if est._groups["cx"] and not _ispd(Sigma[np.ix_(est._Q_check, est._Q_check)]):
            return None
        d = sol["derivs"]
        n = m.n_endo
        iv = np.asarray(est._restrict_dr if hasattr(est, "_restrict_dr") else self._restrict_dr(), int)
        nst, nsp = self.nstatic, self.nspred
        Tfull = np.zeros((n, n))
        Tfull[:, nst:nst + nsp] = d["ghx"]
        T = Tfull[np.ix_(iv, iv)]
        R = d["ghu"][iv, :]
        Z = np.searchsorted(iv, self.indvobs)
        tot = self.totparam_nbr
        DTf = np.zeros((n, n, tot))
        DTf[:, nst:nst + nsp, :] = d["dghx"]
        DT = DTf[np.ix_(iv, iv)]
        DOm = d["dOm"][np.ix_(iv, iv)]
        DYss = np.hstack([np.zeros((len(iv), self.nsc)), d["dYss"][iv, :]])
        Pstar = _lyap(T, R @ Sigma @ R.T)
        Yss_obs = sol["ys"][[m._ie[v] for v in self.varobs]]
        Y = data.T - Yss_obs[:, None]
        try:
            Hess = asymptotic_hessian(Y, T, R, Sigma, Z, DT, DYss, DOm, Pstar)
        except _SingularF:
            # kalman_filter.m returns an infinite likelihood: dsge_likelihood switches to the univariate filter
            if not self.raw.get("use_univariate_filters_if_singularity_is_detected", 1):
                raise NotImplementedError("identification: singular prediction error covariance with "
                                          "use_univariate_filters_if_singularity_is_detected=0 (not ported)") from None
            Hess = univariate_asymptotic_hessian(Y, T, R, Sigma, Z, DT, DYss, DOm, Pstar)
        return -Hess

    def _restrict_dr(self):
        return np.union1d(self.indvobs, self.nstatic + np.arange(self.nspred))

    def _strength(self, theta, sol, J, si_dMOMENTS, si_dREDUCEDFORM, si_dDYNAMIC, ind_dMOMENTS, ind_dREDUCEDFORM, ind_dDYNAMIC,
                  nlags):
        """The identification strength block of analysis.m (``ide_hess``, the normalised Jacobians)."""
        o = self.opts
        tot, sc, modn = self.totparam_nbr, self.nsc, self.modparam_nbr
        tol_rank, tol_deriv = o["tol_rank"], o["tol_deriv"]
        params = np.asarray(theta, float)
        if self.prior_exist:
            normaliz_prior_std = np.array(self.est.prior.p2, float)
        else:
            normaliz_prior_std = np.full(tot, np.nan)
        data = self._simulate_data(sol, o["periods"])
        AHess = self._try_asymptotic_hessian(theta, sol, J, data)
        ide_hess = {}
        if AHess is not None and not isinstance(tol_rank, str) and np.min(np.linalg.eigvalsh((AHess + AHess.T) / 2)) < -tol_rank:
            AHess = None                  # error('Analytic Hessian is not positive semi-definite!') -> catch
        uncert = np.full(tot, np.nan)
        if AHess is not None:
            flag_score = 1
            ide_hess["AHess"] = AHess
            deltaM = np.sqrt(np.diag(AHess))
            iflag = np.any(deltaM * deltaM == 0)
            tildaM = AHess / np.outer(deltaM, deltaM)
            use = AHess if (iflag or rank(AHess) > rank(tildaM)) else tildaM
            ch = checks(use, 0, tol_rank, o["tol_sv"], tot)
            ide_hess.update({k: ch[k] for k in ("cond", "rank", "ind0", "indno", "ino", "Mco", "Pco")})
            indok = np.flatnonzero(np.max(ch["indno"], axis=0) == 0)
            uncert[indok] = np.sqrt(np.diag(minv(AHess[np.ix_(indok, indok)])))
            ind1 = np.flatnonzero(ch["ind0"])
            sol_ = mldivide(AHess[np.ix_(ind1, ind1)], si_dMOMENTS[:, ind1].T)
            cmm = si_dMOMENTS[:, ind1] @ sol_
            temp1 = mldivide(AHess[np.ix_(ind1, ind1)], si_dREDUCEDFORM[:, ind1].T)
            diag_chh = np.sum(si_dREDUCEDFORM[:, ind1].T * temp1, axis=0)
            ind1 = ind1[ind1 >= sc]
            sd_ = si_dDYNAMIC[:, ind1 - sc]
            cdynamic = sd_ @ mldivide(AHess[np.ix_(ind1, ind1)], sd_.T)
        else:
            flag_score = 0
            replic = max(o["replic"], len(ind_dMOMENTS) * 3)
            cmm = self._simulated_moment_uncertainty(sol, ind_dMOMENTS, o["periods"], replic, nlags)
            sd = np.sqrt(np.diag(cmm))
            cc = cmm / np.outer(sd, sd)
            DD, VV = np.linalg.eigh((cc + cc.T) / 2)
            idd = np.flatnonzero(DD > tol_deriv)
            siTMP = si_dMOMENTS / sd[:, None]
            MIM = (siTMP.T @ VV[:, idd]) @ np.diag(1 / DD[idd]) @ (VV[:, idd].T @ siTMP)
            ide_hess["AHess"] = MIM
            deltaM = np.sqrt(np.diag(MIM))
            iflag = np.any(deltaM * deltaM == 0)
            tildaM = MIM / np.outer(deltaM, deltaM)
            use = MIM if (iflag or rank(MIM) > rank(tildaM)) else tildaM
            ch = checks(use, 0, tol_rank, o["tol_sv"], tot)
            ide_hess.update({k: ch[k] for k in ("cond", "rank", "ind0", "indno", "ino", "Mco", "Pco")})
            indok = np.flatnonzero(np.max(ch["indno"], axis=0) == 0)
            ind1 = np.flatnonzero(ch["ind0"])
            temp1 = mldivide(MIM[np.ix_(ind1, ind1)], si_dREDUCEDFORM[:, ind1].T)
            diag_chh = np.sum(si_dREDUCEDFORM[:, ind1].T * temp1, axis=0)
            ind1 = ind1[ind1 >= sc]
            sd_ = si_dDYNAMIC[:, ind1 - sc]
            cdynamic = sd_ @ mldivide(MIM[np.ix_(ind1, ind1)], sd_.T)
            if len(indok):
                uncert[indok] = np.sqrt(np.diag(minv(tildaM[np.ix_(indok, indok)]))) / deltaM[indok]
        strength = np.full(tot, np.nan)
        strength_prior = np.full(tot, np.nan)
        strength[indok] = 1.0 / (uncert[indok] / np.abs(params[indok]))
        strength_prior[indok] = 1.0 / (uncert[indok] / normaliz_prior_std[indok])
        sens_zero = np.flatnonzero(np.isinf(deltaM))
        deltaM_prior = deltaM * np.abs(normaliz_prior_std)
        deltaM = deltaM * np.abs(params)
        quant = si_dMOMENTS / np.sqrt(np.diag(cmm))[:, None]
        si_dMOMENTSnorm = np.abs(quant) * normaliz_prior_std if quant.shape[0] == 1 else vnorm(quant) * normaliz_prior_std
        iy = np.flatnonzero(diag_chh)
        ind_dREDUCEDFORM = ind_dREDUCEDFORM[iy]
        si_dREDUCEDFORM = si_dREDUCEDFORM[iy, :]
        if len(iy):
            quant = si_dREDUCEDFORM / np.sqrt(diag_chh[iy])[:, None]
            si_dREDUCEDFORMnorm = np.abs(quant) * normaliz_prior_std if quant.shape[0] == 1 else \
                vnorm(quant) * normaliz_prior_std
        else:
            si_dREDUCEDFORMnorm = np.zeros((0, 0))
        diag_cd = np.diag(cdynamic)
        iy = np.flatnonzero(diag_cd)
        ind_dDYNAMIC = ind_dDYNAMIC[iy]
        si_dDYNAMIC = si_dDYNAMIC[iy, :]
        if len(iy):
            quant = si_dDYNAMIC / np.sqrt(diag_cd[iy])[:, None]
            nps = normaliz_prior_std[sc:]
            si_dDYNAMICnorm = np.abs(quant) * nps if quant.shape[0] == 1 else vnorm(quant) * nps
        else:
            si_dDYNAMICnorm = np.zeros((0, 0))
        ide_hess.update(ide_strength_dMOMENTS=strength, ide_strength_dMOMENTS_prior=strength_prior, deltaM=deltaM,
                        deltaM_prior=deltaM_prior, sensitivity_zero_pos=(sens_zero + 1.0 if len(sens_zero) else np.zeros((0, 1))),
                        identified_parameter_indices=(indok + 1.0 if len(indok) else np.zeros((1, 0))), flag_score=flag_score)
        return {"ide_hess": ide_hess, "ind_dREDUCEDFORM": ind_dREDUCEDFORM, "si_dREDUCEDFORM": si_dREDUCEDFORM,
                "ind_dDYNAMIC": ind_dDYNAMIC, "si_dDYNAMIC": si_dDYNAMIC, "si_dDYNAMICnorm": si_dDYNAMICnorm,
                "si_dMOMENTSnorm": si_dMOMENTSnorm, "si_dREDUCEDFORMnorm": si_dREDUCEDFORMnorm}

    # ---------------------------------------------------------------------------------------- prior draws
    def _prior(self):
        """``dprior(bayestopt_, options_.prior_trunc, options_ident.prior_range)``: the prior, its truncation bounds and
        the prior used for sampling (uniform on the supports for ``prior_range``)."""
        import copy
        e = self.est
        lb, ub = e.prior.bounds(float(self.raw.get("prior_trunc", 1e-10)))
        pr = e.prior
        if self.opts["prior_range"]:
            pr = copy.copy(pr)
            pr.shape = np.full(len(pr.shape), 5, int)
        return pr, lb, ub

    def prior_draw(self):
        """``Prior.draw()``: uniform and normal priors replay Octave's streams (``rand``/``randn``); the others cannot
        (Octave's ``gamrnd``/``betarnd`` use ``randg``, which Dynare does not seed) and come from a fixed numpy generator."""
        from .samplers import prior_draw
        if self.rng is None:
            raise NotImplementedError("identification: prior draws need a random stream")
        pr, lb, ub = self._prior()
        if not hasattr(self, "_gen"):
            self._gen = np.random.default_rng(0)
        if np.any(np.isin(pr.shape, (1, 2, 4, 6, 8))):
            self._not_replayed = True
        return prior_draw(pr, lb, ub, self.rng, self._gen)

    # ---------------------------------------------------------------------------------------- the statement
    def run(self):
        """``identification.run`` for the statement: the point analysis (with up to 50 prior draws where the model does not
        solve there) and, for ``prior_mc > 1``, the Monte Carlo analysis over prior draws.  Returns None where the model does
        not solve; otherwise a dict with ``label``, ``point`` (the variables of ``<name>_identif.mat`` and of the file named
        by the label), ``extra`` (variables appended to ``<name>_identif.mat``), ``sto`` (the ``<name>_identif_<k>.mat``
        files, as a list of dicts), ``clear_sto`` (delete earlier ones) and ``messages``."""
        o = self.opts
        theta, label = self.point()
        messages = []
        res = None
        try:
            res = self.analysis(theta, init=True)
        except IdentificationStop as exc:
            messages.append(f"The model does not solve for {label}: {exc}")
            if self.prior_exist and self.est.prior.is_bayesian:
                for _ in range(50):
                    theta = self.prior_draw()
                    try:
                        res = self.analysis(theta, init=True)
                        break
                    except IdentificationStop:
                        continue
                if res is not None:
                    label = "Random_prior_params"
        if res is None:
            return None
        messages += res["messages"]
        res["ide_hess"]["params"] = np.asarray(theta, float)
        store = {k: v for k, v in self.opts.items()}
        point = {"ide_moments_point": res["ide_moments"], "ide_spectrum_point": res["ide_spectrum"],
                 "ide_minimal_point": res["ide_minimal"], "ide_hess_point": res["ide_hess"],
                 "ide_reducedform_point": res["ide_reducedform"], "ide_dynamic_point": res["ide_dynamic"],
                 "store_options_ident": store}
        out = {"label": label, "point": point, "extra": {}, "sto": [], "clear_sto": False, "messages": messages}
        if o["prior_mc"] > 1:
            self._monte_carlo(out, res)
        return out

    def run_point(self):
        """Compatibility: ``(label, variables, messages)`` of :meth:`run` for a single point."""
        out = self.run()
        if out is None:
            return None
        return out["label"], out["point"], out["messages"]

    def _monte_carlo(self, out, point_res):
        o = self.opts
        S = o["prior_mc"]
        tot, modn = self.totparam_nbr, self.modparam_nbr
        if o["checks_via_subsets"]:
            raise NotImplementedError("identification: prior_mc > 1 with checks_via_subsets=1 (Dynare's own Monte Carlo "
                                      "loop stops there, as checks_via_subsets does not return cond, ind0, ...)")
        if o["no_identification_reducedform"]:
            raise NotImplementedError("identification: prior_mc > 1 with no_identification_reducedform (Dynare's "
                                      "Monte Carlo loop needs the reduced form criterion)")
        if o["gsa_sample_file"]:
            raise NotImplementedError("identification: external prior samples are not supported")
        opts_ident = dict(o)
        opts_mc = dict(o, advanced=0)
        no_rf, no_mom = opts_mc["no_identification_reducedform"], opts_mc["no_identification_moments"]
        no_spec, no_min = opts_mc["no_identification_spectrum"], opts_mc["no_identification_minimal"]
        # Dynare sets options_ident.checks_via_subsets = 0 after copying options_MC: the later reruns use it, the draws not
        opts_rerun = dict(opts_ident, checks_via_subsets=0)
        iteration = run_index = 0
        pdraws = np.zeros((S, tot))
        sto = {}
        files = []
        max_runs = None
        n_attempts = 0
        max_attempts = max(1000, 200 * S)
        self.opts = opts_mc
        try:
            while iteration < S:
                n_attempts += 1
                if n_attempts > max_attempts:
                    raise ValueError("identification: no admissible parameter draw (the model does not solve) in "
                                     f"{max_attempts} attempts")
                theta = self.prior_draw()
                try:
                    res = self.analysis(theta, init=False)
                except IdentificationStop:
                    continue
                ide_dyn, ide_rf, ide_mom = res["ide_dynamic"], res["ide_reducedform"], res["ide_moments"]
                ide_spec, ide_min, err = res["ide_spectrum"], res["ide_minimal"], res["error_indicator"]
                if iteration == 0:
                    out["clear_sto"] = True
                    rows_rf = ide_rf["si_dREDUCEDFORM"].shape[0]
                    max_runs = min(S, int(np.ceil(1e7 / (rows_rf * tot) / 8)))
                    sto["si_dDYNAMIC"] = np.zeros((ide_dyn["si_dDYNAMIC"].shape[0], modn, max_runs))
                    sto["DYNAMIC"] = np.zeros((len(ide_dyn["DYNAMIC"]), S))
                    sto["IDE_DYNAMIC"] = self._ide_store(ide_dyn, S, modn, ind="ind_dDYNAMIC")
                    sto["si_dREDUCEDFORM"] = np.zeros((rows_rf, tot, max_runs))
                    sto["REDUCEDFORM"] = np.zeros((len(ide_rf["REDUCEDFORM"]), S))
                    sto["IDE_REDUCEDFORM"] = self._ide_store(ide_rf, S, tot, ind="ind_dREDUCEDFORM")
                    if not no_mom:
                        if not ide_mom:
                            raise NotImplementedError("identification: no moment criterion at the first draw")
                        sto["si_dMOMENTS"] = np.zeros((ide_mom["si_dMOMENTS"].shape[0], tot, max_runs))
                        sto["MOMENTS"] = np.zeros((len(ide_mom["MOMENTS"]), S))
                        sto["IDE_MOMENTS"] = self._ide_store(ide_mom, S, tot, ind="ind_dMOMENTS")
                        k8 = min(8, tot)
                        sto["IDE_MOMENTS"]["S"] = np.zeros((S, k8))
                        sto["IDE_MOMENTS"]["V"] = np.zeros((S, tot, k8))
                    if not no_spec:
                        if err["identification_spectrum"] or not ide_spec:
                            raise NotImplementedError("identification: no spectrum criterion at the first draw")
                        sto["dSPECTRUM"] = np.zeros((ide_spec["dSPECTRUM"].shape[0], ide_spec["dSPECTRUM"].shape[1],
                                                     max_runs))
                    if not no_min:
                        if err["identification_minimal"] or not ide_min or not ide_min.get("minimal_state_space"):
                            sto["dMINIMAL"] = None    # Dynare: STO_dMINIMAL = {} (saved as an empty cell)
                        else:
                            sto["dMINIMAL"] = np.zeros((ide_min["dMINIMAL"].shape[0], ide_min["dMINIMAL"].shape[1],
                                                        max_runs))
                iteration += 1
                run_index += 1
                pdraws[iteration - 1, :] = theta
                it = iteration - 1
                sto["DYNAMIC"][:, it] = ide_dyn["DYNAMIC"]
                sto["si_dDYNAMIC"][:, :, run_index - 1] = ide_dyn["si_dDYNAMIC"]
                self._ide_put(sto["IDE_DYNAMIC"], it, ide_dyn)
                sto["REDUCEDFORM"][:, it] = ide_rf["REDUCEDFORM"]
                sto["si_dREDUCEDFORM"][:, :, run_index - 1] = ide_rf["si_dREDUCEDFORM"]
                self._ide_put(sto["IDE_REDUCEDFORM"], it, ide_rf)
                if not no_mom:
                    sto["MOMENTS"][:, it] = ide_mom["MOMENTS"]
                    sto["si_dMOMENTS"][:, :, run_index - 1] = ide_mom["si_dMOMENTS"]
                    self._ide_put(sto["IDE_MOMENTS"], it, ide_mom)
                    sto["IDE_MOMENTS"]["S"][it, :] = np.ravel(ide_mom["S"])
                    sto["IDE_MOMENTS"]["V"][it, :, :] = ide_mom["V"]
                if not no_spec:
                    sto["dSPECTRUM"][:, :, run_index - 1] = ide_spec["dSPECTRUM"]
                if not no_min and not err["identification_minimal"]:
                    if sto["dMINIMAL"] is None:
                        raise NotImplementedError("identification: the minimal system is available at a later draw only "
                                                  "(Dynare's Monte Carlo loop fails there)")
                    sto["dMINIMAL"][:, :, run_index - 1] = ide_min["dMINIMAL"]
                if run_index == max_runs or iteration == S:
                    n = run_index
                    f = {"STO_si_dDYNAMIC": sto["si_dDYNAMIC"][:, :, :n].copy(),
                         "STO_si_dREDUCEDFORM": sto["si_dREDUCEDFORM"][:, :, :n].copy()}
                    if not no_mom:
                        f["STO_si_dMOMENTS"] = sto["si_dMOMENTS"][:, :, :n].copy()
                    if not no_spec:
                        f["STO_dSPECTRUM"] = sto["dSPECTRUM"][:, :, :n].copy()
                    if not no_min:
                        f["STO_dMINIMAL"] = (sto["dMINIMAL"][:, :, :n].copy() if sto["dMINIMAL"] is not None
                                             else np.empty((0, 0), dtype=object))
                    files.append(f)
                    for k in ("si_dDYNAMIC", "si_dREDUCEDFORM", "si_dMOMENTS", "dSPECTRUM", "dMINIMAL"):
                        if sto.get(k) is not None:
                            sto[k][:] = 0
                    run_index = 0
        finally:
            self.opts = opts_ident
        out["sto"] = files
        # --- normalisation by the standard deviation over the sample
        def std0(a, axis):
            return np.std(a, axis=axis, ddof=1) if a.shape[axis] > 1 else np.zeros(a.shape[1 - axis])
        norm_dyn = std0(sto["DYNAMIC"], 1)
        norm_rf = std0(sto["REDUCEDFORM"], 1)
        norm_mom = std0(sto["MOMENTS"], 1) if not no_mom else None
        normaliz1 = std0(pdraws, 0)
        nsc = self.nsc
        iter_ = 0
        res_dyn = res_rf = res_mom = None
        with np.errstate(all="ignore"):
            for f in files:
                runs = [f["STO_si_dDYNAMIC"].shape[2], f["STO_si_dREDUCEDFORM"].shape[2]]
                if not no_mom:
                    runs.append(f["STO_si_dMOMENTS"].shape[2])
                if not no_spec:
                    runs.append(f["STO_dSPECTRUM"].shape[2])
                if not no_min and f["STO_dMINIMAL"].ndim == 3:
                    runs.append(f["STO_dMINIMAL"].shape[2])
                nr = max(runs)
                # Dynare re-creates these NaN arrays for every file; assignments beyond the rows grow them with zeros
                res_dyn = np.full((nr, modn), np.nan)
                res_rf = np.full((nr, tot), np.nan)
                res_mom = np.full((nr, tot), np.nan) if not no_mom else None
                for irun in range(nr):
                    iter_ += 1
                    res_dyn = _grow(res_dyn, iter_)
                    res_rf = _grow(res_rf, iter_)
                    A = f["STO_si_dDYNAMIC"][:, :, irun] if irun < runs[0] else None
                    if A is not None:
                        res_dyn[iter_ - 1, :] = vnorm_matlab(A / norm_dyn[:, None]) * normaliz1[nsc:]
                    if irun < runs[1]:
                        res_rf[iter_ - 1, :] = vnorm_matlab(f["STO_si_dREDUCEDFORM"][:, :, irun] / norm_rf[:, None]) * normaliz1
                    if not no_mom:
                        res_mom = _grow(res_mom, iter_)
                        if irun < f["STO_si_dMOMENTS"].shape[2]:
                            res_mom[iter_ - 1, :] = vnorm_matlab(f["STO_si_dMOMENTS"][:, :, irun] / norm_mom[:, None]) * normaliz1
        sto["IDE_DYNAMIC"]["si_dDYNAMICnorm"] = res_dyn
        extra = {"pdraws": pdraws, "IDE_DYNAMIC": sto["IDE_DYNAMIC"], "STO_DYNAMIC": sto["DYNAMIC"]}
        sto["IDE_REDUCEDFORM"]["si_dREDUCEDFORMnorm"] = res_rf
        extra.update(IDE_REDUCEDFORM=sto["IDE_REDUCEDFORM"], STO_REDUCEDFORM=sto["REDUCEDFORM"])
        if not no_mom:
            sto["IDE_MOMENTS"]["si_dMOMENTSnorm"] = res_mom
            extra.update(IDE_MOMENTS=sto["IDE_MOMENTS"], STO_MOMENTS=sto["MOMENTS"])
        out["extra"].update(extra)
        if getattr(self, "_not_replayed", False):
            out["messages"].append("prior draws of gamma, beta, inverse gamma and Weibull priors come from numpy's generator: "
                                   "Octave's gamrnd/betarnd use randg, which Dynare does not seed, so they cannot be replayed")
        # --- advanced: the draws with the highest and the lowest condition number, or the rank deficient ones
        if o["advanced"]:
            ino = np.ravel(sto["IDE_MOMENTS"]["ino"]) if not no_mom else np.zeros(S)
            jcrit = np.flatnonzero(ino)
            if len(jcrit) < S:
                if len(jcrit) == 0:
                    cond = np.ravel(sto["IDE_MOMENTS"]["cond"])
                    jmax, jmin = int(np.argmax(cond)), int(np.argmin(cond))
                    self.opts = opts_rerun
                    try:
                        for tag, j in (("max", jmax), ("min", jmin)):
                            r = self.analysis(pdraws[j, :], init=True)
                            out["extra"].update({f"ide_hess_{tag}": r["ide_hess"], f"ide_moments_{tag}": r["ide_moments"],
                                                 f"ide_spectrum_{tag}": r["ide_spectrum"],
                                                 f"ide_minimal_{tag}": r["ide_minimal"],
                                                 f"ide_reducedform_{tag}": r["ide_reducedform"],
                                                 f"ide_dynamic_{tag}": r["ide_dynamic"], f"j{tag}": float(j + 1)})
                    finally:
                        self.opts = opts_ident
                else:
                    raise NotImplementedError("identification: advanced=1 with rank deficient draws in the Monte Carlo "
                                              "sample (struct arrays of the analyses) is not ported")

    @staticmethod
    def _ide_store(ide, S, ncol, ind):
        d = {ind: ide[ind], "ino": np.zeros((S, ncol if ind == "ind_dDYNAMIC" else 1)), "ind0": np.zeros((S, ncol)), "jweak": np.zeros((S, ncol)),
             "jweak_pair": np.zeros((S, ncol * (ncol + 1) // 2)), "cond": np.zeros((S, 1)), "Mco": np.zeros((S, ncol))}
        return d

    @staticmethod
    def _ide_put(store, it, ide):
        store["cond"][it, 0] = ide["cond"]
        store["ino"][it, 0] = ide["ino"]
        store["ind0"][it, :] = np.ravel(ide["ind0"])
        store["jweak"][it, :] = np.ravel(ide["jweak"])
        store["jweak_pair"][it, :] = np.ravel(ide["jweak_pair"])
        store["Mco"][it, :] = np.ravel(ide["Mco"])

    def _checks_via_subsets(self, dDYNAMIC, ide_dynamic, ide_reducedform, ide_moments, ide_spectrum, ide_minimal, err):
        """``identification.checks_via_subsets``: ranks of the Jacobians and, by brute force, the parameter subsets (up to
        ``max_dim_subsets_groups`` elements) whose columns are collinear (``problpars``)."""
        o = self.opts
        tot, modn = self.totparam_nbr, self.modparam_nbr
        tol_rank = o["tol_rank"]
        maxdim = o["max_dim_subsets_groups"]
        rk = (lambda X: rank(X)) if isinstance(tol_rank, str) else (lambda X: rank(X, tol_rank))   # noqa: E731
        # spectrum: Dynare compares abs(dSPECTRUM(j,j)) with tol_rank without the 'robust' test (a char vector then)
        small = (lambda x: bool(np.all(x < np.array([ord(c) for c in tol_rank])))) if isinstance(tol_rank, str) \
            else (lambda x: x < tol_rank)    # noqa: E731
        use = {"dynamic": True, "reducedform": not o["no_identification_reducedform"] and not err["identification_reducedform"],
               "moments": not o["no_identification_moments"] and not err["identification_moments"],
               "spectrum": not o["no_identification_spectrum"] and not err["identification_spectrum"],
               "minimal": not o["no_identification_minimal"] and not err["identification_minimal"]}
        mats, need, fixed = {}, {}, 0
        ind = ide_dynamic["ind_dDYNAMIC"].astype(int) - 1
        D = dDYNAMIC.copy()
        D[ind, :] = D[ind, :] / ide_dynamic["norm_dDYNAMIC"]
        if tot > modn:
            D = np.hstack([np.zeros((D.shape[0], tot - modn)), D])
        mats["dynamic"], need["dynamic"] = D, tot
        if use["reducedform"]:
            D = ide_reducedform["dREDUCEDFORM"].copy()
            ind = ide_reducedform["ind_dREDUCEDFORM"].astype(int) - 1
            D[ind, :] = D[ind, :] / ide_reducedform["norm_dREDUCEDFORM"]
            mats["reducedform"], need["reducedform"] = D, tot
        if use["moments"]:
            D = ide_moments["dMOMENTS"].copy()
            ind = ide_moments["ind_dMOMENTS"].astype(int) - 1
            D[ind, :] = D[ind, :] / ide_moments["norm_dMOMENTS"]
            mats["moments"], need["moments"] = D, tot
        if use["spectrum"]:
            mats["spectrum"], need["spectrum"] = ide_spectrum["tilda_dSPECTRUM"], tot
        if use["minimal"]:
            D = ide_minimal["dMINIMAL"].copy()
            ind = ide_minimal["ind_dMINIMAL"].astype(int) - 1
            D[ind, :] = D[ind, :] / ide_minimal["norm_dMINIMAL"]
            mats["minimal"] = D
            fixed = D.shape[1] - tot
            need["minimal"] = tot + fixed
        ides = {"dynamic": ide_dynamic, "reducedform": ide_reducedform, "moments": ide_moments, "spectrum": ide_spectrum,
                "minimal": ide_minimal}
        probl = {k: [None] * maxdim for k in mats}
        for k in probl:
            probl[k][0] = []
        active, indparam, ranks = {}, {}, {}
        for k, D in mats.items():
            ranks[k] = rk(D)
            ides[k]["rank"] = float(ranks[k])
            active[k] = ranks[k] != need[k]
            indparam[k] = list(range(1, tot + 1)) if active[k] else []
        for j in range(tot):
            for k, D in mats.items():
                if not active[k]:
                    continue
                if k == "spectrum":
                    if small(abs(D[j, j])):
                        probl[k][0].append(j + 1)
                elif k == "minimal":
                    par, rest = D[:, :tot], D[:, tot:]
                    if rk(np.hstack([par[:, [j]], rest])) == fixed:
                        probl[k][0].append(j + 1)
                else:
                    if rk(D[:, [j]]) == 0:
                        probl[k][0].append(j + 1)
        for k in mats:
            if not active[k]:
                continue
            if len(probl[k][0]) == need[k] - ranks[k]:
                active[k] = False
            else:
                indparam[k] = [x for x in indparam[k] if x not in set(probl[k][0])]
        indtot = sorted(set().union(*[set(v) for v in indparam.values()]))
        for j in range(2, min(len(indtot), maxdim) + 1):
            if not any(active.values()):
                indexj = []
            else:
                indexj = [list(c) for c in itertools.combinations(indtot, j)]
            for k, D in mats.items():
                if not active[k]:
                    continue
                idx = _remove_problematic(indexj, probl[k])
                rj = np.zeros(len(idx))
                for r, row in enumerate(idx):
                    cols = [c - 1 for c in row]
                    if k == "spectrum":
                        rj[r] = rk(D[np.ix_(cols, cols)])
                    elif k == "minimal":
                        rj[r] = rk(np.hstack([D[:, :tot][:, cols], D[:, tot:]]))
                    else:
                        rj[r] = rk(D[:, cols])
                thr = j + (fixed if k == "minimal" else 0)
                probl[k][j - 1] = [row for row, rv in zip(idx, rj) if rv < thr]
        probl["dynamic"][0] = [x for x in probl["dynamic"][0] if x > tot - modn]
        for k in ides:
            if k not in mats:
                # a criterion that is switched off keeps empty cells (Dynare initialises problpars for all of them)
                empty = np.empty((1, maxdim), dtype=object)
                for i in range(maxdim):
                    empty[0, i] = np.zeros((0, 0))
                ides[k]["problpars"] = empty
                continue
            cell = np.empty((1, maxdim), dtype=object)
            for i in range(maxdim):
                v = probl[k][i]
                if i == 0:
                    cell[0, i] = np.array(v, float).reshape(-1, 1) if len(v) else np.zeros((0, 0))
                elif v is None:
                    cell[0, i] = np.zeros((0, 0))
                else:
                    cell[0, i] = np.array(v, float).reshape(-1, i + 1) if len(v) else np.zeros((0, i + 1))
            ides[k]["problpars"] = cell


from .estimation import Estimation as _Estimation  # noqa: E402


class _IdentEstimation(_Estimation):
    """The parameter bookkeeping of :class:`pydyn.estimation.Estimation` (estimated_params, priors, calibration) without
    data: ``identification`` needs no datafile."""

    def _setup_data(self, data):
        self.data = None
        self.rawdata = None
        self.nobs = 0
        self.missing = False
        self._data_index, self._no_more_missing = None, 0
