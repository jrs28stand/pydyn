# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Derivatives of the first-order perturbation solution with respect to the parameters
(Dynare 7.2's ``+identification/get_perturbation_params_derivs.m``, orders 1, ``analytic_derivation_mode`` 0 and 1).

Dynare reads the derivatives of the model with respect to the parameters from files written by the
preprocessor (``static_params_derivs.m``, ``dynamic_params_derivs.m``).  pydyn evaluates the same exact derivatives
differently: the generated model functions run on complex numbers and the derivative with respect to a parameter is the
imaginary part of the function at ``p + i h`` divided by ``h`` (complex-step differentiation, ``h = 1e-30``: no
subtraction, hence exact to rounding).  Total derivatives (those through the steady state) are obtained in one evaluation
by shifting the steady state along its own derivative, ``ys + i h dys``.  Models whose equations use a function that has
no complex extension (``abs``, ``sign``, ``max``, ``min``, comparisons, ``normcdf``, ``erf``...) are refused.

The solution is differentiated in declaration order and mapped to the decision-rule order at the end: with
``(f0 + fp G) G + fp ... `` the first-order condition  ``fp G G + f0 G + fm = 0``  gives

    (f0 + fp G) dG + fp dG G = -(dfm + df0 G + dfp G G)             (a Sylvester equation in the state columns of dG)
    dH = -(f0 + fp G)^-1 (dfu + (df0 + dfp G + fp dG) H)

which is what Dynare's ``sylvester3`` and ``dghu`` formulas compute.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from .perturbation import dr_ordering, first_order

H_STEP = 1e-30


class ParamsDerivsError(RuntimeError):
    pass


_NOT_ANALYTIC = {"abs", "sign", "max", "min", "_lt", "_gt", "_le", "_ge", "_eq", "_ne", "erf", "erfc", "normcdf", "round",
                 "floor", "ceil", "mod"}


def _check_analytic(model):
    """Refuse models whose generated functions use an operator the complex step would silently get wrong
    (``abs`` of a complex number is its modulus) or cannot evaluate, or an external function."""
    bad = set()
    for fn in ("static_resid", "dynamic_resid", "dynamic_jac"):
        names = set(model._fn[fn].__code__.co_names)
        bad |= names & _NOT_ANALYTIC
    if getattr(model, "_ext_names", None):
        bad |= {"external function " + str(n) for n in model._ext_names}
    if bad:
        raise ParamsDerivsError("analytic parameter derivatives are refused: the model uses " + ", ".join(sorted(bad))
                                + ", which has no complex-step derivative here")


def _complex_eval(model, fn, y, x, params, ss):
    """Generated function ``fn`` on complex arguments: values as a complex array."""
    try:
        with np.errstate(all="raise"):
            return np.array(model._fn[fn](y, x, params, ss), dtype=complex)
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise ParamsDerivsError(f"the model has an operator without a complex-step derivative ({exc}); "
                                "analytic parameter derivatives are refused") from None


def steady_state_derivatives(model, ys, params, qz_criterium=1 + 1e-6):
    """``dys`` ([endo x param] derivative of the steady state, declaration order) by the implicit function theorem
    (Ratto and Iskrev 2012, eq. 5), and the derivative of the static residuals ``rp`` ([endo x param])."""
    n, npar = model.n_endo, len(model.param_names)
    x = np.asarray(model.exo_steady(), float)
    ysc = np.asarray(ys, complex)
    rp = np.zeros((n, npar))
    for j in range(npar):
        pc = np.asarray(params, complex).copy()
        pc[j] += 1j * H_STEP
        rp[:, j] = _complex_eval(model, "static_resid", ysc, x.astype(complex), pc, None).imag / H_STEP
    g1 = model.static_jac(np.asarray(ys, float), x, np.asarray(params, float))
    dys = None
    try:
        with np.errstate(all="ignore"):
            if np.linalg.cond(g1) < 1 / np.finfo(float).eps:
                # one right-hand side at a time: where a parameter does not enter, the solution stays exactly zero (a blocked
                # triangular solve with several right-hand sides can leave rounding noise there; identification's
                # ``find(diag_chh)`` tests exact zeros)
                lu = linalg.lu_factor(g1)
                dys = -np.column_stack([linalg.lu_solve(lu, rp[:, j]) for j in range(rp.shape[1])])
    except np.linalg.LinAlgError:
        dys = None
    if dys is None or not np.all(np.isfinite(dys)):
        # singular static Jacobian (unit roots): the stationary part only, as get_perturbation_params_derivs.m does
        Ts, Us, k = linalg.schur(g1, output="real", sort=lambda re, im: np.hypot(re, im) < qz_criterium - 1)
        Ts = Ts[k:, k:]
        Us = Us[:, k:]
        dys = -Us @ np.linalg.solve(Ts, Us.T @ rp)
    return dys, rp


def numerical_dys_dJ(model, indpmodel, params, x, dynatol_x=1e-5):
    """Two-sided finite differences (``fjaco.m``: step ``dynatol.x * max(|p|, 1)``, divided by ``(p+h)-(p-h)``) of the steady
    state and of the dynamic Jacobian with respect to the selected model parameters, the steady state (with the steady-state
    file's parameter changes) recomputed at each point.  Returns ``dys`` [endo x all parameters, only the selected columns
    filled] and ``dJ`` [selected x endo x (3 endo + exo)], columns of the Jacobian in declaration order."""
    from .steady import steady_state
    n = model.n_endo
    rows, cols = model.jac_rows, model.jac_cols
    indpmodel = np.asarray(indpmodel, int).ravel()
    ncol = 3 * n + model.n_exo
    dys = np.zeros((n, len(model.param_names)))
    dJ = np.zeros((len(indpmodel), n, ncol))
    for jj, jp in enumerate(indpmodel):
        x0 = params[jp]
        h = dynatol_x * max(abs(x0), 1.0)
        xh1, xh0 = x0 + h, x0 - h
        hh = xh1 - xh0
        vals = []
        for xv in (xh1, xh0):
            pv = params.copy()
            pv[jp] = xv
            ysv, ppv = steady_state(model, pv)
            ysv, ppv = np.asarray(ysv, float), np.asarray(ppv, float)
            Jv = model.dynamic_jac(np.tile(ysv, 3), x, ppv, ysv)
            vals.append((ysv, Jv))
        dys[:, jp] = (vals[0][0] - vals[1][0]) / hh
        dJ[jj][rows, cols] = ((vals[0][1] - vals[1][1]) / hh)[rows, cols]
    return dys, dJ


def _kron_solution(model, G, H, fm, f0, fp, fu, dJ, order_var, state_var, nm, nexo, n):
    """``analytic_derivation_mode = 1`` of get_perturbation_params_derivs.m (Iskrev 2010, appendix A): the derivatives of
    the Kalman transition matrix and of ghu by the implicit function theorem with Kronecker products, in decision-rule
    order, returned as the declaration-order ``dG`` [nm x n x n] and ``dH`` [nm x n x nexo] used below."""
    ov = np.asarray(order_var, int)
    I = np.eye(n)
    KA = G[np.ix_(ov, ov)]
    GAM0, GAM1 = f0[:, ov], -fp[:, ov]
    dGAM0 = np.stack([dJ[j][:, n:2 * n][:, ov].ravel(order="F") for j in range(nm)], axis=1)
    dGAM1 = np.stack([-dJ[j][:, 2 * n:3 * n][:, ov].ravel(order="F") for j in range(nm)], axis=1)
    dGAM2 = np.stack([-dJ[j][:, :n][:, ov].ravel(order="F") for j in range(nm)], axis=1)
    GAM3 = -fu
    dGAM3 = np.stack([-dJ[j][:, 3 * n:].ravel(order="F") for j in range(nm)], axis=1)
    dF = np.kron(I, GAM0) - np.kron(KA.T, GAM1) - np.kron(I, GAM1 @ KA)
    Fp = np.kron(KA.T, I) @ dGAM0 - np.kron(KA.T @ KA.T, I) @ dGAM1 - dGAM2
    dKA = -np.linalg.solve(dF, Fp)
    MM = GAM0 - GAM1 @ KA
    invMM = np.linalg.solve(MM, I)
    dghu = (-np.kron((invMM @ GAM3).T, invMM) @ (dGAM0 - np.kron(KA.T, I) @ dGAM1 - np.kron(I, GAM1) @ dKA)
            + np.kron(np.eye(nexo), invMM) @ dGAM3)
    dG = np.zeros((nm, n, n))
    dH = np.zeros((nm, n, nexo))
    inv = np.empty(n, int)
    inv[ov] = np.arange(n)
    for j in range(nm):
        dKj = dKA[:, j].reshape((n, n), order="F")          # DR rows, DR columns
        dG[j] = dKj[np.ix_(inv, inv)]                       # declaration order
        dH[j] = dghu[:, j].reshape((n, nexo), order="F")[inv]
    return dG, dH


def perturbation_params_derivs(model, indpmodel, indpstderr=(), indpcorr=(), ys=None, params=None, order=1,
                               qz_criterium=1 + 1e-6, numerical=False, dynatol_x=1e-5, kron=False):
    """Derivatives of the first-order solution with respect to selected parameters, in Dynare's layout.

    ``indpmodel``: 0-based positions in ``model.param_names``; ``indpstderr``: 0-based positions of shocks (declaration
    order) whose standard deviation is a parameter; ``indpcorr``: pairs (i, j) of 0-based shock positions.  The
    parameter order of the totals is stderr, corr, model.  Returns a dict with Dynare's names:
    ``dYss`` [endo x modparam], ``dSigma_e`` [exo x exo x totparam], ``dCorrelation_matrix``, ``dghx``
    [endo x nspred x totparam], ``dghu`` [endo x exo x totparam] (zero pages for the stderr and corr parameters, as Dynare), ``dOm`` [endo x endo x totparam] (DR order),
    and ``dg1`` [endo x (3 endo + exo) x modparam] (total derivative of the dynamic Jacobian, columns
    [y(-1) y y(+1) exo] in declaration order, rows equations)."""
    if order != 1:
        raise NotImplementedError("get_perturbation_params_derivs: orders 2 and 3 are not ported")
    if model.n_exo_det:
        raise ParamsDerivsError("get_perturbation_params_derivs: not compatible with deterministic exogenous variables, "
                                "please declare as endogenous.")
    _check_analytic(model)
    from .steady import steady_state
    if ys is None:
        ys, params = steady_state(model, params)
    params = np.asarray(model.params if params is None else params, float)
    ys = np.asarray(ys, float)
    n, nexo = model.n_endo, model.n_exo
    indpmodel = np.asarray(indpmodel, int).ravel()
    indpstderr = np.asarray(indpstderr, int).ravel()
    indpcorr = np.asarray(indpcorr, int).reshape(-1, 2)
    nm, ns, nc = len(indpmodel), len(indpstderr), len(indpcorr)
    ntot = nm + ns + nc
    sol = first_order(model, ys=ys, params=params, loglinear=False, qz_criterium=qz_criterium)
    G, H = sol.G, sol.H
    order_var, state_var, cnt = dr_ordering(model.lead_lag_incidence)
    x = np.asarray(model.exo_steady(), float)
    y3 = np.tile(ys, 3)
    J = model.dynamic_jac(y3, x, params, ys)
    fm, f0, fp, fu = J[:, :n], J[:, n:2 * n], J[:, 2 * n:3 * n], J[:, 3 * n:]

    rows, cols = model.jac_rows, model.jac_cols
    dJ = np.zeros((nm, n, J.shape[1]))
    if numerical:
        # analytic_derivation_mode = -2: two-sided finite differences (fjaco.m, step dynatol.x * max(|x|, 1)) of the steady
        # state and of the dynamic Jacobian, the steady state (with the steady-state file's parameter changes) recomputed
        # at each point; the solution derivatives below are analytic as in mode 0
        dys, dJ = numerical_dys_dJ(model, indpmodel, params, x, dynatol_x)
    else:
        dys, _ = steady_state_derivatives(model, ys, params, qz_criterium)
    # total derivative of the dynamic Jacobian: one complex-step evaluation per parameter
    for jj, jp in enumerate(indpmodel if not numerical else ()):
        pc = params.astype(complex)
        pc[jp] += 1j * H_STEP
        d = dys[:, jp]
        v = _complex_eval(model, "dynamic_jac", (y3 + 1j * H_STEP * np.tile(d, 3)), x.astype(complex), pc,
                          ys + 1j * H_STEP * d)
        if model._det_view is not None:
            v = v[model._det_view["keep"]["dynamic_jac"]]
        dJ[jj][rows, cols] = v.imag / H_STEP
    kron_sol = None
    if kron and n * n <= 6400:
        kron_sol = _kron_solution(model, G, H, fm, f0, fp, fu, dJ, order_var, state_var, nm, nexo, n)
    # Sylvester equation in the state columns of G
    st = np.asarray(state_var, int)
    ncs = len(st)
    A = f0 + fp @ G
    Gss = G[np.ix_(st, st)]
    dG = np.zeros((nm, n, n))
    dH = np.zeros((nm, n, nexo))
    if ncs:
        K = np.kron(np.eye(ncs), A) + np.kron(Gss.T, fp)
        lu = linalg.lu_factor(K)
    for jj in range(nm):
        dfm, df0, dfp, dfu = dJ[jj][:, :n], dJ[jj][:, n:2 * n], dJ[jj][:, 2 * n:3 * n], dJ[jj][:, 3 * n:]
        if ncs:
            R = -(dfm + df0 @ G + dfp @ G @ G)[:, st]
            X = linalg.lu_solve(lu, R.reshape(-1, order="F")).reshape((n, ncs), order="F")
            dG[jj][:, st] = X
        dA = df0 + dfp @ G + fp @ dG[jj]
        dH[jj] = -np.linalg.solve(A, dfu + dA @ H)
    if kron_sol is not None:
        dG, dH = kron_sol
    Sigma = np.asarray(model.Sigma_e, float)
    sd = np.sqrt(np.diag(Sigma))
    with np.errstate(all="ignore"):
        corr = Sigma / np.outer(sd, sd)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    dSigma = np.zeros((nexo, nexo, ntot))
    dCorr = np.zeros((nexo, nexo, ntot))
    isdiag = not np.any(Sigma - np.diag(np.diag(Sigma)))
    for jp, i in enumerate(indpstderr):
        dSigma[i, i, jp] = 2 * np.sqrt(Sigma[i, i])
        if not isdiag:
            for kk in range(nexo):
                if kk != i:
                    dSigma[i, kk, jp] = corr[i, kk] * np.sqrt(Sigma[kk, kk])
                    dSigma[kk, i, jp] = dSigma[i, kk, jp]
    for jp, (i, k) in enumerate(indpcorr):
        dSigma[i, k, ns + jp] = np.sqrt(Sigma[i, i]) * np.sqrt(Sigma[k, k])
        dSigma[k, i, ns + jp] = dSigma[i, k, ns + jp]
        dCorr[i, k, ns + jp] = dCorr[k, i, ns + jp] = 1.0
    ghu = H[order_var]
    ghx = G[np.ix_(order_var, st)]
    dghx = np.zeros((n, ncs, nm))
    dghu = np.zeros((n, nexo, nm))
    dOm = np.zeros((n, n, ntot))
    for jj in range(nm):
        dghx[:, :, jj] = dG[jj][np.ix_(order_var, st)]
        dghu[:, :, jj] = dH[jj][order_var]
        dOm[:, :, ns + nc + jj] = dghu[:, :, jj] @ Sigma @ ghu.T + ghu @ Sigma @ dghu[:, :, jj].T
    for jp in range(ns + nc):
        dOm[:, :, jp] = ghu @ dSigma[:, :, jp] @ ghu.T
    return {"indpmodel": indpmodel, "indpstderr": indpstderr, "indpcorr": indpcorr,
            "dYss": dys[np.ix_(order_var, indpmodel)], "dSigma_e": dSigma, "dCorrelation_matrix": dCorr,
            "dghx": np.concatenate([np.zeros((n, ncs, ns + nc)), dghx], axis=2),
            "dghu": np.concatenate([np.zeros((n, nexo, ns + nc)), dghu], axis=2), "dOm": dOm, "dg1": np.moveaxis(dJ, 0, 2), "ghx": ghx, "ghu": ghu,
            "order_var": order_var, "state_var": st}
