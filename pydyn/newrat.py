# This file is part of pydyn.
#
# It is a Python port of matlab/optimization/{newrat, mr_hessian, mr_gstep}.m of Dynare 7.2 (Copyright (C) 2004-2026
# Dynare Team, https://www.dynare.org), modified in 2026. The line search and the BFGS
# update are those of pydyn.estimation (ports of csminit1.m and bfgsi1.m).
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Dynare's default optimiser ``mode_compute=5`` (newrat): Newton-like iterations with an
outer-product-of-gradients Hessian built from the contributions of every observation to the likelihood
(``mr_hessian.m``), line searches of csminwel, univariate gradient steps (``mr_gstep.m``) and Dynare's safeguards.

The objective ``fcn(x)`` returns ``(fval, code, penalty, contributions)``: minus the log posterior kernel, Dynare's
``info(1)`` and ``info(4)`` (where ``code`` is non-zero the value used is the current penalty plus ``penalty``,
``penalty_objective_function.m``) and the vector ``[-log prior; minus the log density of every observation]``
(``DLIK`` of ``dsge_likelihood.m``; ``None`` where the objective is not defined). :func:`estimation_objective` builds it
for an :class:`~pydyn.estimation.Estimation`.
"""
from __future__ import annotations

import numpy as np

__all__ = ["newrat", "mr_hessian", "mr_gstep", "estimation_objective", "default_hess_info"]

_EPS = np.finfo(float).eps


def estimation_objective(est, bounds=None, counter=None):
    """``dsge_likelihood`` for newrat: ``x -> (fval, code, penalty, contributions)`` with ``contributions`` =
    ``[-log prior; minus the log density of each observation after the presample]``."""
    from .estimation import _Rejected

    lb, ub = (est.lb, est.ub) if bounds is None else bounds
    nonlinear = hasattr(est, "run_filter")        # pydyn.nonlinear_estimation (particle filters)

    def f(theta):
        if counter is not None:
            counter[0] += 1
        theta = np.asarray(theta, float)
        if np.any(theta < lb):
            k = theta < lb
            return np.inf, 41, float(np.sum((lb[k] - theta[k]) ** 2)), None
        if np.any(theta > ub):
            k = theta > ub
            return np.inf, 42, float(np.sum((theta[k] - ub[k]) ** 2)), None
        try:
            with np.errstate(all="ignore"):
                LIK, lik = est._minus_loglik(theta, return_lik=True)
        except (_Rejected, FloatingPointError, np.linalg.LinAlgError) as exc:
            return np.inf, getattr(exc, "code", 45) or 45, getattr(exc, "penalty", 0.1), None
        lp = est.log_prior(theta)
        if not np.isfinite(lp):
            return np.inf, 40, 0.1, None
        fval = float(LIK) - lp
        if est.options.get("endogenous_prior"):   # dsge_likelihood: in fval, not in DLIK
            try:
                fval -= float(est._endogenous_prior(theta))
            except Exception as exc:              # noqa: BLE001 - as any other failure of the objective
                return np.inf, getattr(exc, "code", 45) or 45, getattr(exc, "penalty", 0.1), None
        if nonlinear:                             # non_linear_dsge_likelihood.m returns DLIK = []
            return fval, 0, 0.0, None
        return fval, 0, 0.0, np.concatenate([[-lp], np.asarray(lik, float).ravel()])
    return f


def default_hess_info(n, gradient_epsilon=1e-6, gstep=(1e-2, 1.0), tol_gstep=np.nan, tol_gstep_rel=np.nan, crit=1e-5,
                      robust=False):
    """``hess_info`` as ``dynare_minimize_objective.m`` sets it for newrat."""
    return dict(gstep=gstep, htol=crit if np.isnan(tol_gstep) else tol_gstep, htol_rel=tol_gstep_rel,
                h1=gradient_epsilon * np.ones(n), robust=robust)


def _octave_spd(A):
    return A.shape[0] == A.shape[1] and np.array_equal(A, A.T) and np.all(np.diag(A) > 0)


def _factor(A):
    """Octave's choice of factorisation for ``A \\ b`` and ``inv(A)``: Cholesky for a symmetric matrix with a positive
    diagonal when it succeeds, else LU; and whether Octave calls the matrix singular (``rcond + 1 == 1``)."""
    from scipy.linalg import lapack
    anorm = float(np.max(np.sum(np.abs(A), axis=0))) if A.size else 0.0
    if _octave_spd(A):
        c, info = lapack.dpotrf(A, lower=0)
        if info == 0:
            rc, _ = lapack.dpocon(c, anorm)
            return ("chol", c), (rc + 1.0 == 1.0 or np.isnan(rc))
    lu, piv, info = lapack.dgetrf(A)
    if info > 0:
        return ("lu", (lu, piv)), True
    rc, _ = lapack.dgecon(lu, anorm)
    return ("lu", (lu, piv)), (rc + 1.0 == 1.0 or np.isnan(rc))


def _mldivide(A, B):
    """MATLAB's ``A \\ B`` as Octave computes it: Cholesky (``dpotrf``/``dpotrs``) for a symmetric matrix with a positive
    diagonal when it is positive definite, LU otherwise; a matrix singular to machine precision gets a warning and the
    minimum-norm least-squares solution (``dgelsd``), as Octave's ``mldivide`` does."""
    from scipy.linalg import lapack
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    (kind, fac), singular = _factor(A)
    if singular:
        import warnings
        warnings.warn("matrix singular to machine precision")
        return np.linalg.lstsq(A, B, rcond=-1)[0]
    if kind == "chol":
        x, _ = lapack.dpotrs(fac, B, lower=0)
        return x
    x, _ = lapack.dgetrs(fac[0], fac[1], B)
    return x


def _inv(A):
    """Octave's ``inv``: through the Cholesky factor (``dpotri``) for a symmetric positive definite matrix, LU otherwise;
    a matrix singular to machine precision gives a matrix of Inf (with a warning), as in Octave."""
    from scipy.linalg import lapack
    A = np.asarray(A, float)
    if A.size == 0:
        return A.copy()
    (kind, fac), singular = _factor(A)
    if singular:
        import warnings
        warnings.warn("matrix singular to machine precision")
        return np.full(A.shape, np.inf)
    if kind == "chol":
        inv, _ = lapack.dpotri(fac, lower=0)
        return np.triu(inv) + np.triu(inv, 1).T
    inv, _ = lapack.dgetri(fac[0], fac[1])
    return inv


def _pen(fcn, x, penalty):
    fval, code, pen, contrib = fcn(x)
    if code:
        return penalty + pen, False, None
    return fval, True, contrib


def _check_bounds(x, bounds, tiny):
    x = np.array(x, float)
    k = x >= bounds[:, 1]
    x[k] = bounds[k, 1] - tiny
    k = x <= bounds[:, 0]
    x[k] = bounds[k, 0] + tiny
    return x


def _csminit(fcn, x0, penalty, f0, g0, H0, log):
    from .estimation import _csminit as csminit1
    return csminit1(lambda x: _pen(fcn, x, penalty)[0], np.asarray(x0, float), f0, np.asarray(g0, float), False,
                    np.asarray(H0, float), log)


def mr_hessian(x, fcn, penalty, hflag, htol0, hess_info, bounds, prior_std):
    """``mr_hessian.m``: numerical gradient with adaptive steps and the outer-product-of-gradients Hessian.

    Returns ``(hessian_mat, gg, htol1, ihh, hh_mat0, hh1, hess_info, fcount)`` with ``hessian_mat`` flattened in
    MATLAB's (column-major) order, or ``None`` (and ``ihh``, ``hh_mat0``, ``hh1``) when the objective gives no
    contributions at ``x``."""
    hess_info = dict(hess_info)
    hess_info["h1"] = np.array(hess_info["h1"], float)
    x = np.asarray(x, float)
    n = x.size
    fcount = 1
    f0, _, ff0 = _pen(fcn, x, penalty)
    h2 = bounds[:, 1] - bounds[:, 0]
    hmax = np.minimum(bounds[:, 1] - x, x - bounds[:, 0])
    opg = ff0 is not None
    hess_info["h1"] = np.minimum(hess_info["h1"], 0.9 * hmax)
    if htol0 < hess_info["htol"]:
        hess_info["htol"] = htol0
    htol1 = None
    if not np.isnan(hess_info["htol_rel"]):
        htol1 = hess_info["htol"]
        hess_info["htol"] = abs(hess_info["htol_rel"] * f0)
    xh1 = x.copy()
    f1, f_1 = np.zeros(n), np.zeros(n)
    if opg:
        ggh = np.zeros((ff0.size, n))
        ff1 = ff_1 = np.zeros(ff0.size)
    hhtol = hess_info["htol"] * np.ones(n)
    h1 = hess_info["h1"]
    i = 0
    while i < n:
        i += 1
        k = i - 1
        hess_info["htol"] = hhtol[k]
        htol = hess_info["htol"]
        h10 = h1[k]
        hcheck = 0
        xh1[k] = x[k] + h1[k]
        fx, _, ffx = _pen(fcn, xh1, penalty)
        fcount += 1
        dx = [fx - f0]
        ic = 0
        icount = 0
        h0 = [h1[k]]
        istoobig = [False]
        istoosmall = [False]
        give_it_up = False
        it = 0                                    # 0-based index of the last entry of dx / h0
        while ((abs(dx[it]) < 0.5 * htol or abs(dx[it]) > 3 * htol) and icount < 10 and ic == 0
               and hmax[k] > 2 * 1e-10 and not give_it_up):
            icount += 1
            istoobig[it] = False
            istoosmall[it] = False
            if abs(dx[it]) < 0.5 * htol:
                istoosmall[it] = True
                if h1[k] == 0.9 * hmax[k]:
                    give_it_up = True
                htmp = 0.9 * htol / abs(dx[it]) * h1[k] if abs(dx[it]) != 0 else 2.1 * h1[k]
                htmp = min(htmp, 0.9 * hmax[k])
                big = [h for h, b in zip(h0, istoobig) if b]
                if any(h != 0 for h in big):
                    htmp = 0.5 * (min(big) + max(h for h, s in zip(h0, istoosmall) if s))
                h1[k] = max(htmp, 1e-10)
                xh1[k] = x[k] + h1[k]
                fx, _, ffx = _pen(fcn, xh1, penalty)
                fcount += 1
            if abs(dx[it]) > 3 * htol:
                istoobig[it] = True
                htmp = htol / abs(dx[it]) * h1[k]
                small = [h for h, s in zip(h0, istoosmall) if s]
                if any(h != 0 for h in small):
                    htmp = 0.5 * (min(h for h, b in zip(h0, istoobig) if b) + max(small))
                h1[k] = max(htmp, 1e-10)
                xh1[k] = x[k] + h1[k]
                fx, _, ffx = _pen(fcn, xh1, penalty)
                fcount += 1
                itr = 0
                while (fx - f0) == 0 and itr < 50:
                    h1[k] = h1[k] * 2
                    xh1[k] = x[k] + h1[k]
                    fx, _, ffx = _pen(fcn, xh1, penalty)
                    fcount += 1
                    ic = 1
                    itr += 1
            it += 1
            dx.append(fx - f0)
            h0.append(h1[k])
            istoobig.append(False)
            istoosmall.append(False)
            if h1[k] < 1e-12 * min(1, h2[k]) and h1[k] < 0.9 * hmax[k]:
                ic = 1
                hcheck = 1
        if icount == 10 or h1[k] == 1e-10:
            istoobig[it] = abs(dx[it]) > 3 * htol
            istoosmall[it] = abs(dx[it]) < 0.5 * htol
            if any(istoobig) and (istoosmall[it] or istoobig[it]):
                cand = [(d, j) for j, (d, b) in enumerate(zip(dx, istoobig)) if b]
                ddx = min(d for d, _ in cand)
                ij = [d for d, _ in cand].index(ddx)
                fx = f0 + ddx
                h1[k] = h0[ij]                    # Dynare indexes h0 with the position among the too-big steps
        f1[k] = fx
        if opg:
            ff1 = (np.ones(ff0.size) * fx / ff0.size) if (ffx is None or np.any(np.isnan(ffx))) else ffx
        xh1[k] = x[k] - h1[k]
        fx, _, ffx = _pen(fcn, xh1, penalty)
        fcount += 1
        f_1[k] = fx
        if opg:
            ff_1 = (np.ones(ff0.size) * fx / ff0.size) if (ffx is None or np.any(np.isnan(ffx))) else ffx
            ggh[:, k] = (ff1 - ff_1) / (2 * h1[k])
        xh1[k] = x[k]
        if hcheck and hess_info["htol"] < 1:
            hess_info["htol"] = min(1, max(min(abs(np.array(dx))) * 2, hess_info["htol"] * 10))
            h1[k] = h10
            hhtol[k] = hess_info["htol"]
            i -= 1
    hess_info["h1"] = h1
    gg = (f1 - f_1) / (2 * h1)
    if opg:
        if hflag == 2:
            raise NotImplementedError("mr_hessian with hflag=2 (mixed second-order Hessian) is not ported")
        hessian_mat = None
        if hflag == 1:
            hm = np.zeros(n * n)
            for i in range(n):
                dum = (f1[i] + f_1[i] - 2 * f0) / (h1[i] * h1[i])
                hm[i * n + i] = dum
                if dum <= _EPS:
                    hm[i * n + i] = max(_EPS, gg[i] ** 2)
            hessian_mat = hm
        gga = ggh * (2 * h1)[None, :]
        hh_mat = gga.T @ gga
        hh_mat0 = ggh.T @ ggh
        A = np.diag(2 * h1)
        ihh = A.T @ _mldivide(hh_mat, A)
        if hflag > 0 and np.min(np.linalg.eigvals(hessian_mat.reshape(n, n, order="F")).real) > 0:
            hh = hessian_mat.reshape(n, n, order="F")
            hh0 = A @ hh @ A.T
            sd0 = np.sqrt(np.diag(hh0))
            sd = np.sqrt(np.diag(hh_mat))
            hh_mat = hh_mat / np.outer(sd, sd) * np.outer(sd0, sd0)
            ihh = A.T @ _mldivide(hh_mat, A)
            sd = np.sqrt(np.diag(ihh))
            sdh = np.sqrt(1 / np.diag(hh))
            sd0 = np.minimum(np.asarray(prior_std, float), sd)
            sd0 = 10 ** (0.5 * (np.log10(sd0) + np.log10(sdh)))
            inv_A = np.linalg.inv(A)
            ihh = ihh / np.outer(sd, sd) * np.outer(sd0, sd0)
            igg = inv_A.T @ ihh @ inv_A
            hh_mat0 = inv_A.T @ np.linalg.inv(igg) @ inv_A
        if hflag < 2:
            hessian_mat = hh_mat0.ravel(order="F")
        if np.any(np.isnan(hessian_mat)):
            hh_mat0 = np.eye(n)
            ihh = hh_mat0
            hessian_mat = hh_mat0.ravel(order="F")
        hh1 = h1.copy()
    else:
        hessian_mat = ihh = hh_mat0 = hh1 = None
    if np.isnan(hess_info["htol_rel"]):
        htol1 = hhtol
    return hessian_mat, gg, htol1, ihh, hh_mat0, hh1, hess_info, fcount


def mr_gstep(h1, x, bounds, fcn, penalty, htol0, gradient_epsilon, robust, log):
    """``mr_gstep.m``: a line search along every coordinate in turn. Returns ``(f0, x, ig, fcount)``."""
    x = np.array(x, float)
    n = x.size
    h1 = gradient_epsilon * np.ones(n) if h1 is None else np.asarray(h1, float)
    htol = 1e-6 * np.ones(n) if htol0 is None else np.asarray(htol0, float) * np.ones(n)
    fcount = 1
    f0 = _pen(fcn, x, penalty)[0]
    xh1 = x.copy()
    ig = np.zeros(n, int)
    for i in range(n):
        xh1[i] = x[i] + h1[i]
        f1 = _pen(fcn, xh1, penalty)[0]
        xh1[i] = x[i] - h1[i]
        f_1 = _pen(fcn, xh1, penalty)[0]
        fcount += 2
        gg = np.zeros(n)
        hh = np.zeros(n)
        gg[i] = (f1 - f_1) / (2 * h1[i])
        hh[i] = 1 / max(1e-9, abs((f1 + f_1 - 2 * f0) / (h1[i] * h1[i])))
        if gg[i] * (hh[i] * gg[i]) / 2 > htol[i]:
            ff, xx, fc, retcode = _csminit(fcn, x, penalty, f0, gg, np.diag(hh), log)
            fcount += fc
            if retcode and robust:
                if abs(x[i]) < 1e-6:
                    xa = np.linspace(x[i] / 2, np.sign(x[i]) * 1e-6 * 3 / 2, 7)
                else:
                    xa = np.linspace(x[i] / 2, x[i] * 3 / 2, 7)
                fa = np.full(7, np.nan)
                for k in range(7):
                    xh1[i] = xa[k]
                    fa[k] = _pen(fcn, xh1, penalty)[0]
                    fcount += 1
                b = np.linalg.lstsq(np.column_stack([np.ones(7), xa, xa * xa / 2]), fa, rcond=None)[0]
                gg[i] = x[i] * b[2] + b[1]
                hh[i] = 1 / b[2]
                ff2, xx2, fc, _ = _csminit(fcn, x, penalty, f0, gg, np.diag(hh), log)
                fcount += fc
                if ff2 < ff:
                    ff, xx = ff2, xx2
                if np.min(fa) < ff:
                    im = int(np.argmin(fa))
                    ff = fa[im]
                    xx = np.array(xx, float)
                    xx[i] = xa[im]
            ig[i] = 1
            if robust:
                if not np.array_equal(xx, _check_bounds(xx, bounds, 1e-10)):
                    xx = _check_bounds(xx, bounds, 1e-10)
                    xx[i] = min(xx[i] + h1[i], 0.5 * (xx[i] + x[i])) if xx[i] < x[i] else \
                        max(xx[i] - h1[i], 0.5 * (xx[i] + x[i]))
                    ff, ok, _ = _pen(fcn, xx, penalty)
                    fcount += 1
                    if not ok:
                        log("last step exited with bad status!")
                    elif ff < f0:
                        f0, x = ff, xx
                else:
                    if abs(f0 - ff) < abs(gg[i] * (hh[i] * gg[i]) / 2 / 100) or abs(x[i] - xx[i]) < 1e-10:
                        ff1, xx1, fc, _ = _csminit(fcn, x, penalty, f0, -gg, np.diag(hh), log)
                        fcount += fc
                        if not np.array_equal(xx1, _check_bounds(xx1, bounds, 1e-10)):
                            xx1 = _check_bounds(xx1, bounds, 1e-10)
                            xx1[i] = min(xx1[i] + h1[i], 0.5 * (xx1[i] + x[i])) if xx1[i] < x[i] else \
                                max(xx1[i] - h1[i], 0.5 * (xx1[i] + x[i]))
                            ff1, ok, _ = _pen(fcn, xx1, penalty)
                            fcount += 1
                            if not ok:
                                log("last step exited with bad status!")
                        if ff1 < ff:
                            ff, xx = ff1, xx1
                    f0, x = ff, np.array(xx, float)
            else:
                f0 = ff
                x = _check_bounds(xx, bounds, 1e-10)
        xh1 = x.copy()
    return f0, x, ig, fcount


def newrat(fcn, x, bounds, ftol0=1e-5, nit=1000, flagg=0, hess_info=None, prior_std=None, gradient_epsilon=1e-6,
           verbose=False):
    """``newrat.m`` without analytic derivatives. ``bounds``: (n, 2) array (lower, upper), the bounds of the mode
    search. Returns a dict: ``x``, ``f``, ``hh`` (outer-product Hessian), ``gg`` (gradient), ``igg``, ``hess_info``,
    ``iterations``, ``fcount``, ``exitflag``, ``message`` and ``fvals`` (the value after every iteration)."""
    from .estimation import _bfgsi
    log = print if verbose else (lambda *_: None)
    x = np.array(x, float)
    nx = x.size
    bounds = np.asarray(bounds, float)
    prior_std = np.full(nx, np.nan) if prior_std is None else np.asarray(prior_std, float)
    hess_info = default_hess_info(nx, gradient_epsilon) if hess_info is None else dict(hess_info)
    penalty = 1e8
    fcount = 0
    icount = 0
    xparam1 = x.copy()
    htol_base = max(1e-7, hess_info["htol"])
    flagit = 0
    ftol = ftol0
    gtol = 1e-3
    htol = htol_base
    out = dict(x=x, f=None, hh=None, gg=None, igg=None, hess_info=hess_info, iterations=0, fcount=0)
    fval0, ok, _ = _pen(fcn, x, penalty)
    fcount += 1
    if not ok:
        out.update(igg=np.full((nx, nx), np.nan), exitflag=-2, message="Bad initial parameter.", f=fval0, fcount=fcount)
        return out
    fval0 = [fval0]
    fval = fval0[0]
    opg = True
    penalty = fval0[0]
    dum, gg, htol0, igg, hhg, h1, hess_info, fc = mr_hessian(x, fcn, penalty, flagit, htol, hess_info, bounds, prior_std)
    fcount += fc
    if dum is None:
        opg = False
        igg = 1e-4 * np.eye(nx)
        hh = hhg = None
    else:
        hh0 = dum.reshape(nx, nx, order="F")
        hh = hhg
        if np.min(np.linalg.eigvals(hh0).real) < 0:
            hh0 = hhg
    if np.max(htol0) > htol:
        log("Numerical noise in the likelihood: tolerance has to be relaxed")
    H = igg
    gs = [np.array(gg)]
    xs = [x.copy()]
    check = 0
    igrad = 1
    inx = np.eye(nx)
    jit = 0
    ig = np.ones(nx, int)
    ggx = np.zeros(nx)
    message = ""
    exitflag = -3
    while np.linalg.norm(gg) > gtol and check == 0 and jit < nit:
        jit += 1
        icount += 1
        penalty = fval0[icount - 1]
        log(f"Iteration {icount}")
        fval, x0, fc, _ = _csminit(fcn, xparam1, penalty, fval0[icount - 1], gg, H, log)
        fcount += fc
        if igrad:
            fval1, x01, fc, _ = _csminit(fcn, x0, penalty, fval, gg, inx, log)
            fcount += fc
            if (fval - fval1) > 1:
                log("Gradient step!!")
            else:
                igrad = 0
            fval, x0 = fval1, x01
        ig_pos = np.flatnonzero(ig)
        if ig_pos.size < nx:
            ggx = ggx * 0
            ggx[ig_pos] = gg[ig_pos]
            hhx = hh if not opg else dum.reshape(nx, nx, order="F")
            iggx = np.eye(nx)
            iggx[np.ix_(ig_pos, ig_pos)] = _inv(hhx[np.ix_(ig_pos, ig_pos)])
            _, x0, fc, _ = _csminit(fcn, x0, penalty, fval, ggx, iggx, log)
            fcount += fc
        x0 = np.asarray(x0, float)
        if not np.array_equal(x0, _check_bounds(x0, bounds, _EPS)):
            x0 = _check_bounds(x0, bounds, _EPS)
            fvala, ok, _ = _pen(fcn, x0, penalty)
            fcount += 1
            if ok:
                penalty = fvala
            else:
                log("last step exited with bad status!")
        fvala, x0, ig, fc = mr_gstep(h1, x0, bounds, fcn, penalty, htol0, gradient_epsilon, hess_info["robust"], log)
        fcount += fc
        if not np.array_equal(x0, _check_bounds(x0, bounds, _EPS)):
            x0 = _check_bounds(x0, bounds, _EPS)
            fvala, ok, _ = _pen(fcn, x0, penalty)
            fcount += 1
            if ok:
                penalty = fvala
            else:
                log("last step exited with bad status!")
        fval = fvala
        if (fval0[icount - 1] - fval) < ftol and flagit == 0:
            ihh = np.diag(1 / np.diag(hhg))
            fval2, x0, fc, _ = _csminit(fcn, x0, penalty, fval, gg, ihh, log)
            fcount += fc
            x0 = _check_bounds(x0, bounds, _EPS)
            fval = fval2
        if (fval0[icount - 1] - fval) < ftol and flagit == 0:
            ihh0 = inx * 1e-4
            fval3, x0, fc, _ = _csminit(fcn, x0, penalty, fval, gg, ihh0, log)
            fcount += fc
            x0 = _check_bounds(x0, bounds, _EPS)
            fval = fval3
        xparam1 = np.array(x0, float)
        xs.append(xparam1.copy())
        fval0.append(fval)
        if (fval0[icount - 1] - fval) < ftol:
            message = "No further improvement is possible!"
            log(message)
            check = 1
            if flagg > 0:
                dum, gg, htol0, igg, hhg, h1, hess_info, fc = mr_hessian(xparam1, fcn, penalty, flagg, htol_base,
                                                                         hess_info, bounds, prior_std)
                fcount += fc
                hh = dum.reshape(nx, nx, order="F") if flagg == 2 else hhg
            gs.append(np.array(gg))
        else:
            htol = htol_base
            if np.linalg.norm(xs[icount - 1] - xparam1) > 1e-12:
                dum, gg, htol0, igg, hhg, h1, hess_info, fc = mr_hessian(xparam1, fcn, penalty, flagit, htol,
                                                                         hess_info, bounds, prior_std)
                fcount += fc
                if dum is None:
                    opg = False
                if np.max(htol0) > htol:
                    log("Numerical noise in the likelihood: tolerance has to be relaxed")
                if not opg:
                    H = _bfgsi(H, gg - gs[icount - 1], xparam1 - xs[icount - 1], log)
                    hh = _inv(H)
                    hhg = hh
                else:
                    hh0 = dum.reshape(nx, nx, order="F")
                    hh = hhg
                    H = igg
            gs.append(np.array(gg))
    if jit == nit:
        message = (message + " " if message else "") + "Maximum number of iterations reached."
        exitflag = 0
    if np.linalg.norm(gg) <= gtol:
        message = (message + " " if message else "") + f"Estimation ended: Gradient norm < {gtol:g}"
        exitflag = -1
    if check == 1:
        message = (message + " " if message else "") + "Estimation successful."
        exitflag = 1
    out.update(x=xparam1, f=float(fval), hh=hh, gg=np.array(gg), igg=igg, hess_info=hess_info, iterations=jit,
               fcount=fcount, exitflag=exitflag, message=message, fvals=np.array(fval0))
    return out


def run_mode(est, x0, opt, counter=None, verbose=False):
    """``mode_compute=5`` for :meth:`Estimation.mode <pydyn.estimation.Estimation.mode>`, as
    ``dynare_minimize_objective.m`` calls newrat: tolerance ``TolFun`` (default 1e-5), ``MaxIter`` (1000),
    ``TolGstep``/``TolGstepRel`` (step-size tolerances of the numerical gradient; default: ``TolFun``), ``robust``,
    the bounds of the mode search and the prior standard deviations. ``Hessian`` (``newrat.hess``): 1 (default; the
    Hessian at the mode is then computed by ``hessian.m`` as for the other optimisers); 0 and 2 (the outer product of
    gradients with univariate filters) are not ported. ``verbosity`` and ``SaveFiles`` have no effect."""
    import warnings
    unknown = set(opt) - {"MaxIter", "Hessian", "robust", "TolFun", "TolGstep", "TolGstepRel", "verbosity", "SaveFiles"}
    if unknown:
        warnings.warn(f"newrat: Unknown option ({', '.join(sorted(unknown))})!")
    if int(opt.get("Hessian", 1)) != 1:
        raise NotImplementedError("newrat: Hessian=0 or 2 (outer-product Hessian from univariate filters) is not ported")
    crit = float(opt.get("TolFun", 1e-5))
    gstep_crit = float(opt.get("TolGstep", np.nan))
    gstep_rel = float(opt.get("TolGstepRel", np.nan))
    o = est.options
    hess_info = dict(gstep=o["gstep"], htol=crit if np.isnan(gstep_crit) else gstep_crit, htol_rel=gstep_rel,
                     h1=float(o["gradient_epsilon"]) * np.ones(est.n), robust=bool(opt.get("robust", False)))
    fcn = estimation_objective(est, counter=counter)
    res = newrat(fcn, x0, np.column_stack([est.lb, est.ub]), ftol0=crit, nit=int(opt.get("MaxIter", 1000)), flagg=0,
                 hess_info=hess_info, prior_std=est.prior.p2, gradient_epsilon=float(o["gradient_epsilon"]),
                 verbose=verbose)
    if res["exitflag"] == -2:
        from .estimation import EstimationError
        raise EstimationError(f"newrat: bad initial parameter ({est.last_info})")
    return res
