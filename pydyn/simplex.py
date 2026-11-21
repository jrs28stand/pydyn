# This file is part of pydyn. It is a Python port of matlab/optimization/simplex_optimization_routine.m
# and of the mode_compute=8 branch of matlab/optimization/dynare_minimize_objective.m of Dynare 7.2
# (Copyright (C) 2010-2025 Dynare Team, GPL-3.0-or-later), modified in 2026.
# Released under the GNU General Public License, version 3 or later; see LICENSE. No warranty.
"""Dynare's Nelder-Mead simplex (``mode_compute = 8``), with its default options (no randomised
coefficients, optimised expansion step, restarts of the simplex)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["simplex_optimization_routine", "SimplexResult", "simplex_options"]


@dataclass
class SimplexResult:
    x: np.ndarray
    fval: float
    exitflag: int
    iterations: int
    func_count: int
    message: str


def simplex_options(optim=None) -> dict:
    """``options_.simplex`` with the ``optim`` keys of mode_compute=8 (MaxIter, TolFun, TolX,
    MaxFunEvals, MaxFunEvalFactor, InitialSimplexSize)."""
    o = {"tolerance_x": 1e-4, "tolerance_f": 1e-4, "maxiter": 10000, "maxfcallfactor": 500, "maxfcall": None,
         "delta_factor": 0.05}
    names = {"MaxIter": "maxiter", "TolFun": "tolerance_f", "TolX": "tolerance_x", "MaxFunEvals": "maxfcall",
             "MaxFunEvalFactor": "maxfcallfactor", "InitialSimplexSize": "delta_factor"}
    for k, v in dict(optim or {}).items():
        if k in names:
            o[names[k]] = v
        elif k != "verbosity":
            raise NotImplementedError(f"simplex: option {k} is not supported")
    return o


def _initialization(fun, point, point_score, delta, zero_delta, check_delta):
    """simplex_initialization: vertices point and point(j)*(1+delta(j)) (zero_delta if point(j) is 0);
    a vertex with a penalty is pulled back (delta/1.1) until it is valid."""
    fc = 0
    n = len(point)
    v = np.zeros((n, n + 1))
    v[:, 0] = point
    fv = np.zeros(n + 1)
    fv[0] = point_score
    delta = np.full(n, float(delta)) if np.ndim(delta) == 0 else np.array(delta, float)
    for j in range(n):
        y = point.copy()
        y[j] = (1 + delta[j]) * y[j] if y[j] != 0 else zero_delta
        v[:, j + 1] = y
        fv[j + 1], ok = fun(y)
        fc += 1
        if check_delta:
            while not ok:
                if y[j] != 0:
                    delta[j] = delta[j] / 1.1
                else:
                    zero_delta = zero_delta / 1.1
                y = point.copy()
                y[j] = (1 + delta[j]) * y[j] if y[j] != 0 else zero_delta
                v[:, j + 1] = y
                fv[j + 1], ok = fun(y)
                fc += 1
    idx = np.argsort(fv, kind="stable")
    return v[:, idx], fv[idx], delta, fc


def simplex_optimization_routine(fun, x, options=None) -> SimplexResult:
    """Minimise ``fun`` (returning ``(value, no_penalty)``; values may be ``inf``) from ``x``."""
    o = simplex_options() if options is None else options
    x = np.array(x, float)
    n = len(x)
    max_func_calls = o["maxfcallfactor"] * n if o["maxfcall"] is None else o["maxfcall"]
    x_tol, f_tol, max_iterations = o["tolerance_x"], o["tolerance_f"], o["maxiter"]
    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
    optimize_expansion_parameter = True
    delta = o["delta_factor"]
    DELTA = delta
    zero_delta = delta / 200
    max_no_improvements = n * 10
    iter_count_total = 0
    func_count_total = 0
    initial_score, ok = fun(x)
    func_count_total += 1
    if not ok:
        raise ValueError("simplex_optimization_routine:: Initial condition is wrong!")
    v, fv, delta, fc = _initialization(fun, x, initial_score, delta, zero_delta, True)
    func_count_total += fc
    func_count = n + 1
    iter_count = 1
    no_improvements = 0
    simplex_iterations = 1
    max_simplex_algo_iterations = 3
    simplex_algo_iterations = 1
    best_point, best_point_score = v[:, 0].copy(), fv[0]
    convergence = tooslow = False
    iter_no_improvement_break = 0
    max_no_improvement_break = 1
    critF = critX = 1.0

    def f(z):
        nonlocal func_count_total
        func_count_total += 1
        return fun(z)[0]

    while func_count < max_func_calls and iter_count < max_iterations and simplex_algo_iterations <= max_simplex_algo_iterations:
        if critF <= max(f_tol, 10 * np.spacing(abs(fv[0]))) and critX <= max(x_tol, 10 * np.spacing(abs(np.max(v[:, 0])))):
            convergence = True
        if critX <= x_tol ** 2 and critF > 1:
            tooslow = True
        xbar = np.mean(v[:, :n], axis=1)
        xr = xbar + rho * (xbar - v[:, -1])
        fxr = f(xr)
        func_count += 1
        if fxr < fv[0]:
            xe = xbar + rho * chi * (xbar - v[:, -1])
            fxe = f(xe)
            func_count += 1
            if fxe < fxr:
                if optimize_expansion_parameter:
                    xee = xbar + rho * chi * 1.01 * (xbar - v[:, -1])
                    fxee = f(xee)
                    func_count += 1
                    if fxee < fxe:
                        weight = rho * chi * 1.02
                        fold, xold = fxee, xee
                        while True:
                            weight = 1.02 * weight
                            xeee = xbar + weight * (xbar - v[:, -1])
                            fxeee = f(xeee)
                            func_count += 1
                            if fxeee < fold and -(fxeee - fold) > f_tol * 10 * fold:
                                fold, xold = fxeee, xeee
                            else:
                                break
                    else:
                        weight = rho * chi
                        fold, xold = fxee, xee
                        while True:
                            weight = weight / 1.02
                            xeee = xbar + weight * (xbar - v[:, -1])
                            fxeee = f(xeee)
                            func_count += 1
                            if fxeee < fold and -(fxeee - fold) > f_tol * 10 * fold:
                                fold, xold = fxeee, xeee
                            else:
                                break
                    xe, fxe = xold, fold
                v[:, -1], fv[-1] = xe, fxe
            else:
                v[:, -1], fv[-1] = xr, fxr
        else:
            move = None
            if fxr < fv[n - 1]:
                v[:, -1], fv[-1] = xr, fxr
            else:
                if fxr < fv[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * v[:, -1]
                    fxc = f(xc)
                    func_count += 1
                    if fxc <= fxr:
                        v[:, -1], fv[-1] = xc, fxc
                    else:
                        move = "shrink"
                else:
                    xcc = (1 - psi) * xbar + psi * v[:, -1]
                    fxcc = f(xcc)
                    func_count += 1
                    if fxcc < fv[-1]:
                        v[:, -1], fv[-1] = xcc, fxcc
                    else:
                        move = "shrink"
                if move == "shrink":
                    for j in range(1, n + 1):
                        v[:, j] = v[:, 0] + sigma * (v[:, j] - v[:, 0])
                        fv[j] = f(v[:, j].copy())
                    func_count += n
        idx = np.argsort(fv, kind="stable")
        fv, v = fv[idx], v[:, idx]
        with np.errstate(invalid="ignore"):
            critF = np.max(np.abs(fv[0] - fv[1:]))
            critX = np.max(np.abs(v[:, 1:] - v[:, [0]]))
        iter_count += 1
        iter_count_total += 1
        simplex_iterations += 1
        if abs(best_point_score - fv[0]) < f_tol:
            no_improvements += 1
        else:
            no_improvements = 0
        best_point, best_point_score = v[:, 0].copy(), fv[0]
        if no_improvements > max_no_improvements:
            if simplex_algo_iterations <= max_simplex_algo_iterations:
                delta = delta * 1.05
                v, fv, delta, fc = _initialization(fun, best_point, best_point_score, delta, zero_delta, True)
                func_count_total += fc
                no_improvements = 0
                func_count += n
                iter_count += 1
                iter_count_total += 1
                iter_no_improvement_break += 1
                simplex_iterations += 1
        if (func_count == max_func_calls or iter_count == max_iterations
                or iter_no_improvement_break == max_no_improvement_break or convergence or tooslow):
            v, fv, delta, fc = _initialization(fun, best_point, best_point_score, DELTA, zero_delta, True)
            func_count_total += fc
            if not (func_count == max_func_calls or iter_count == max_iterations) \
                    and iter_no_improvement_break == max_no_improvement_break:
                iter_no_improvement_break = 0
                if simplex_algo_iterations == max_simplex_algo_iterations:
                    max_no_improvements = np.inf
                    continue
            if simplex_algo_iterations < max_simplex_algo_iterations:
                delta = delta * 1.05
                v, fv, delta, fc = _initialization(fun, best_point, best_point_score, delta, zero_delta, True)
                func_count_total += fc
                func_count = n
                iter_count = 1
                convergence = False
                no_improvements = 0
                simplex_iterations += 1
                simplex_algo_iterations += 1
            else:
                break
    exitflag = 1
    msgs = []
    if func_count >= max_func_calls:
        msgs.append("Maximum number of objective function calls has been exceeded!")
        exitflag = 0
    if iter_count >= max_iterations:
        msgs.append("Maximum number of iterations has been exceeded!")
        exitflag = 0
    return SimplexResult(x=v[:, 0].copy(), fval=float(fv[0]), exitflag=exitflag, iterations=iter_count_total,
                         func_count=func_count_total, message=" ".join(msgs) or "The function converged to a solution x.")
