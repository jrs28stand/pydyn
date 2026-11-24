# This file is part of pydyn.
#
# It is a Python port of parts of Dynare 7.2 (Copyright (C) 2010-2025 Dynare Team, https://www.dynare.org), modified in
# 2026: matlab/ep/extended_path.m, extended_path_initialization.m, extended_path_shocks.m, extended_path_core.m,
# extended_path_homotopy.m, setup_stochastic_perfect_foresight_model_solver.m.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""The deterministic extended path (``extended_path(order=0)``).

In each period ``t`` of the sample the model is solved as a perfect-foresight problem over ``ep.periods`` periods ahead,
starting from the previous period's solution, with the realised shock in the first period and the exogenous variables at their
steady state afterwards; the first period of that solution is the outcome of period ``t`` (Fair and Taylor, 1983).  The
stacked problems are solved with pydyn's perfect-foresight solver (:class:`pydyn.perfect_foresight._Sim`: Newton's method with a
sparse LU factorisation; with ``lmmcp`` the LMMCP solver).  Dynare solves them with ``dynare_solve`` and ``options_.ep.solve_algo``
(trust region by default); the solutions agree up to the tolerance ``options_.dynatol.f`` of the residuals, not iterate by iterate.

The stochastic extended path (``order > 0``: integration over the future shocks with a tree of histories) is not implemented.
"""
from __future__ import annotations

import importlib

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from . import steady as _steady

_pf = importlib.import_module(__package__ + ".perfect_foresight")


class EPError(Exception):
    """An error Dynare's extended_path raises (the message is Dynare's)."""


class EPResult:
    """``endo``: ``(maximum_lag + samplesize) x n_endo`` (``ts.data``); ``exo``: the rows of ``oo_.exo_simul``; ``status``."""

    def __init__(self, endo, exo, status, shocks):
        self.endo, self.exo, self.status, self.shocks = endo, exo, status, shocks


def draw_shocks(rng, Sigma_e, samplesize):
    """``extended_path_shocks.m`` (gaussian): ``shocks(:, positive) = transpose(transpose(Omega)*randn(k, samplesize))``."""
    Sigma_e = np.asarray(Sigma_e, float)
    nx = Sigma_e.shape[0]
    pos = np.flatnonzero(np.diag(Sigma_e) > 0)
    shocks = np.zeros((samplesize, nx))
    if pos.size:
        Omega = np.linalg.cholesky(Sigma_e[np.ix_(pos, pos)]).T            # chol(Sigma, 'upper')
        z = rng.randn((pos.size, samplesize))
        shocks[:, pos] = (Omega.T @ z).T
    return shocks


def _sparse_solve(A, b):
    """``A \\ b`` for a sparse square matrix; NaN where the matrix is singular."""
    try:
        with np.errstate(all="ignore"):
            x = spla.splu(sparse.csc_matrix(A)).solve(b)
        return x
    except RuntimeError:
        return np.full(len(b), np.nan)


def _dogleg(r, b, d, delta):
    """Subfunction dogleg of trust_region.m with a sparse ``r``."""
    x = _sparse_solve(r, b)
    qnorm = np.linalg.norm(d * x)
    if qnorm <= delta:
        return x
    s = (r.T @ b) / d
    gnorm = np.linalg.norm(s)
    if gnorm > 0:
        s = (s / gnorm) / d
        temp0 = np.linalg.norm(r @ s)
        sgnorm = gnorm / (temp0 * temp0)
        if sgnorm < delta:
            bnorm = np.linalg.norm(b)
            temp1 = delta / qnorm
            temp2 = sgnorm / delta
            temp0 = bnorm * bnorm * temp2 / (gnorm * qnorm)
            temp0 = temp0 - temp1 * temp2 * temp2 + np.sqrt((temp0 - temp1) ** 2 + (1.0 - temp1 * temp1) * (1.0 - temp2 * temp2))
            alpha = temp1 * (1.0 - temp2 * temp2) / temp0
        else:
            alpha = 0.0
    else:
        alpha = delta / qnorm
        sgnorm = 0.0
    if alpha > 0:
        return alpha * x + (1.0 - alpha) * min(sgnorm, delta) * s
    return min(sgnorm, delta) * s


def _finite(*arrays):
    return all(np.all(np.isfinite(a.data if sparse.issparse(a) else a)) for a in arrays)


def trust_region(fun, x, tolf, tolx, maxiter, factor=1.0):
    """matlab/optimization/trust_region.m (``solve_algo = 9``) on the whole system, ``fun(x)`` returning the residuals and a
    sparse Jacobian.  The same algorithm as ``pydyn.steady._trust_region`` (dense).  Returns ``(x, errorflag)``."""
    radiusfactor, actualreductionthreshold, relativereductionthreshold = 0.5, 0.001, 0.1
    eps = _steady._EPS
    x = np.array(x, float)
    errorflag, it, info, delta, ncsucc, ncslow = True, 1, 0, 0.0, 0, 0
    xinit = x.copy()
    try:
        fval, fjac = fun(x)
    except (ArithmeticError, ValueError):
        return x, True
    fjac = sparse.csc_matrix(fjac)
    fnorm = np.linalg.norm(fval)
    if not _finite(fval, fjac):
        return x, True
    if fnorm < tolf:
        return x, False
    fjacnorm_ = None
    xnorm = 0.0
    while it <= maxiter and not info:
        fjacnorm = np.sqrt(np.asarray(fjac.multiply(fjac).sum(axis=0)).ravel())
        if it == 1:
            fjacnorm_ = fjacnorm.copy()
            fjacnorm_[fjacnorm < eps] = 1.0
            xnorm = np.linalg.norm(fjacnorm_ * x)
            delta = xnorm * factor if xnorm > 0 else factor
        else:
            fjacnorm_ = np.maximum(0.1 * fjacnorm_, fjacnorm)
            xnorm = np.linalg.norm(fjacnorm_ * x)
        p = _dogleg(fjac, fval, fjacnorm_, delta)
        pnorm = np.linalg.norm(fjacnorm_ * p)
        xx = x - p
        x0 = x.copy()
        ww = fval - fjac @ p
        try:
            fval1 = fun(xx)[0]
        except (ArithmeticError, ValueError):
            delta *= radiusfactor
            it += 1
            continue
        if not _finite(fval1):
            delta *= radiusfactor
            it += 1
            continue
        fnorm1 = np.linalg.norm(fval1)
        if fnorm1 < tolf:
            x = xx
            errorflag = False
            info = 1
            continue
        actualreduction = 1.0 - (fnorm1 / fnorm) ** 2 if fnorm1 < fnorm else -1.0
        tt = np.linalg.norm(ww)
        predictedreduction = 1.0 - (tt / fnorm) ** 2 if tt < fnorm else 0.0
        ratio = actualreduction / predictedreduction if predictedreduction > 0 else 0.0
        if it == 1:
            delta = min(delta, pnorm)
        if ratio < relativereductionthreshold:
            ncsucc = 0
            delta *= radiusfactor
        else:
            ncsucc += 1
            if abs(ratio - 1.0) < relativereductionthreshold:
                delta = pnorm / radiusfactor
            elif ratio >= radiusfactor or ncsucc > 1:
                delta = max(delta, pnorm / radiusfactor)
        if ratio > 1e-4:
            x = xx
            fval = fval1
            xnorm = np.linalg.norm(fjacnorm_ * x)
            fnorm = fnorm1
        ncslow += 1
        if actualreduction >= actualreductionthreshold:
            ncslow = 0
        it += 1
        if it == maxiter:
            info, x = 2, xinit
            continue
        if delta < tolx * xnorm:
            info, x, errorflag = 3, xinit, True
            continue
        if tolx > 0 and max(0.1 * delta, pnorm) <= 10 * np.spacing(xnorm) * xnorm:
            info, x, errorflag = 5, xinit, True
            continue
        if ncslow == 15:
            info, x, errorflag = 4, xinit, True
            continue
        fjac0 = fjac
        try:
            fjac = sparse.csc_matrix(fun(x)[1])
        except (ArithmeticError, ValueError):
            x, fjac = x0, fjac0
            delta *= radiusfactor
            continue
        if not _finite(fjac):
            x, fjac = x0, fjac0
            delta *= radiusfactor
    return x, errorflag


def core(sim, endo, exo, ss, xss, periods):
    """``perfect_foresight_solver_core`` as called by extended_path_core.m: returns ``(endo, success)``."""
    o, m = sim.o, sim.model
    lag, lead = int(m.info["maximum_lag"]), int(m.info["maximum_lead"])
    stack = int(o["stack_solve_algo"])
    if stack == 0 or sim.mcp is not None or not (m.info["maximum_endo_lag"] > 0 and m.info["maximum_endo_lead"] > 0) \
            or o["linear_approximation"] or o.get("linear"):
        return sim.core(endo, exo, ss, xss, periods)[:2]
    if stack != 7 or int(o["solve_algo"]) != 9:
        raise NotImplementedError(f"ep.stack_solve_algo = {stack} with ep.solve_algo = {o['solve_algo']} (implemented: "
                                  "stack_solve_algo 0, and 7 with solve_algo 9 or 10)")
    n = m.n_endo
    problem = sim.problem(ss)
    y0, yT = _pf._boundaries(m, endo, periods)
    X = exo[lag:lag + periods]
    tolf, tolx = o["dynatol.f"], o["dynatol.x"]

    def fun(z):
        res, A = problem(z.reshape(periods, n), y0, yT, X)
        return res, A

    z = endo[lag:lag + periods].reshape(-1).copy()
    with np.errstate(all="ignore"):
        f0, _ = fun(z)
        if np.all(np.isfinite(f0)) and np.max(np.abs(f0)) < tolf:
            err = False
        else:
            z, err = trust_region(fun, z, tolf, tolx, int(o["simul.maxit"]), o.get("trust_region_initial_step_bound_factor", 1.0))
    out = endo.copy()
    out[lag:lag + periods] = z.reshape(periods, n)
    return out, bool((not err) and np.all(np.isfinite(z)))


def homotopy(sim, endo, exo, ss, xss, periods, method=2, log=None):
    """``extended_path_homotopy.m`` for ``order == 0``: ``endo`` is ``(rows x n)``, ``exo`` ``(rows x n_exo)``.  Returns
    ``(info_convergence, endo)``."""
    endo0 = endo.copy()
    noconv = True
    if method in (1, 2):
        iteration, weight, maxiter = 0, 0.1, 100
        increase_flag, increase_factor, decrease_factor = False, 1.2, 1.1
        state = [False] * 5
        oldweight = weight
        while noconv:
            iteration += 1
            e = endo.copy()
            e[0] = ss + weight * (endo0[0] - ss)
            x = weight * exo + (1 - weight) * xss[None, :]
            new, success, *_ = sim.core(e, x, ss, xss, periods)
            flag = bool(success)
            state = [flag] + state[:-1]
            if flag:
                if weight == 1:
                    noconv = False
                    break
                if all(state):
                    increase_factor = 1 + (increase_factor - 1) * 1.1
                    state = [False] * 5
                oldweight = weight
                weight = min(weight * increase_factor, 1.0)
                increase_flag = True
                endo = new
            else:
                if increase_flag:
                    weight = oldweight + (weight - oldweight) / 100
                else:
                    weight = min(weight / decrease_factor, 1.0)
            if iteration > maxiter:
                break
            if weight < 1e-9:
                break
        info = not noconv
    if method == 3 or (method == 2 and noconv):
        if method == 2:
            endo = endo0
        weights = np.arange(1001) / 1000.0
        noconv, index = True, 0
        while noconv:
            weight = weights[index]
            x = weight * exo + (1 - weight) * xss[None, :]
            new, success = core(sim, endo.copy(), x, ss, xss, periods)
            if success:
                if weight == 1:
                    noconv = False
                    continue
                index += 1
                endo = new
            else:
                break
        info = not noconv
    return info, endo


def run(model, sim, ep, samplesize, y0, shocks, ss, xss, no_homotopy=False, log=None):
    """The loop of ``extended_path.m`` for ``order == 0``.  ``sim`` is a ``perfect_foresight._Sim`` whose options hold the
    EP settings; ``ep`` the dict of ``options_.ep`` (``periods``); ``y0`` the endogenous variables of period 0; ``shocks`` the
    ``samplesize x n_exo`` innovations.  Returns an :class:`EPResult`."""
    n, nx = model.n_endo, model.n_exo
    lag, lead = int(model.info["maximum_lag"]), int(model.info["maximum_lead"])
    periods = int(ep["periods"])
    rows = lag + periods + lead
    exo_full = np.tile(np.asarray(xss, float)[None, :], (rows, 1))
    path = np.full((samplesize + lag, n), np.nan)
    path[0] = y0
    status = True
    guess_tail = None                                    # endogenousvariablespaths of the previous period (rows x n)
    t = 1
    while t <= samplesize:
        t += 1
        exo = exo_full.copy()
        exo[1] = shocks[t - 2]
        if t > 2:
            init = np.vstack([guess_tail[1:], np.asarray(ss, float)[None, :]])
        else:
            init = np.vstack([y0[None, :], np.tile(np.asarray(ss, float)[None, :], (periods + lead, 1))])
        # extended_path_core.m
        endo, success = core(sim, init.copy(), exo, ss, xss, periods)
        if not success and not no_homotopy:
            success, endo = homotopy(sim, endo if endo is not None else init, exo, ss, xss, periods, 2, log)
        if not success:
            if log:
                log(f"extended_path: No convergence of the (stochastic) perfect foresight solver (in period {t})!")
            status = False
            break
        path[t - 1] = endo[1]
        guess_tail = endo
        # the next period starts from the realised value of this one
        y0 = endo[1]
    if status is False:
        pass
    exo_sim = np.vstack([np.tile(np.asarray(xss, float)[None, :], (lag, 1)), shocks])
    return EPResult(path, exo_sim, status, shocks)
