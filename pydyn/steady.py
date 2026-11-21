# This file is part of pydyn.
#
# The numerical solvers below (dynare_solve, _trust_region, _trust_region_f08, _dogleg, the random
# restarts) are a Python port of parts of Dynare 7.2 (Copyright (C) 2001-2026 Dynare Team,
# https://www.dynare.org), modified in 2026: matlab/optimization/dynare_solve.m,
# matlab/optimization/trust_region.m, mex/sources/block_trust_region/{trust_region.f08,mexFunction.f08,
# dulmage_mendelsohn.f08}, all GPL-3.0-or-later.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Steady-state computation."""
from __future__ import annotations

import os

import numpy as np
from scipy import optimize

from . import matlab
from .expr import FAST_ERRORS, FAST_NAMESPACE, NAMESPACE, Translator, safe_name


class SteadyStateError(ValueError):
    pass


#: compile the steady_state_model block once per model instead of translating and eval-ing every
#: assignment at each call (PYDYN_SS_COMPILE=0 restores the interpreted version)
COMPILE_BLOCK = os.environ.get("PYDYN_SS_COMPILE", "1") != "0"
#: numerical steady states: start Newton from the previous solution of the same model when no
#: guess is given, falling back to the search from initval. Off by default because with several
#: steady states the result could then depend on the order of a parameter loop.
WARM_START = os.environ.get("PYDYN_SS_WARMSTART", "0") == "1"


def _resolver(model):
    # the arrays have names no variable of the block can have: a block-local variable called x, y
    # or params must not hide them
    def resolve(name, shift):
        if name in model._ie:
            return f"_Y_[{model._ie[name]}]"
        if name in model._ix:
            return f"_X_[{model._ix[name]}]"
        if name in model._ip:
            return f"_P_[{model._ip[name]}]"
        return None
    return resolve


def _local(name):
    """Python name of a block-local variable (never one of the names the generated code uses)."""
    name = safe_name(name)
    return name + "_" if name in ("_Y_", "_X_", "_P_", "_ssm") or name in NAMESPACE else name


def _compiled_block(model):
    """The steady_state_model block as one function ``f(y, x, params)`` working in place, bound
    to Python-float and to numpy functions (see expr.FAST_NAMESPACE)."""
    key = id(model.steady_state_model)
    cached = model.__dict__.get("_ssm_compiled")
    if cached is not None and cached[0] == key:
        return cached[1]
    if getattr(model, "matlab_functions", None):
        raise NotImplementedError("blocks that call MATLAB helper functions are interpreted")
    tr = Translator(_resolver(model))
    lines, assigned = ["def _ssm(_Y_, _X_, _P_):"], set()
    for lhs, rhs in model.steady_state_model:
        if not isinstance(lhs, str) or not lhs.isidentifier():
            raise NotImplementedError(f"multiple-output steady-state assignment: {lhs}")
        if lhs in model._ie:
            target = f"_Y_[{model._ie[lhs]}]"
        elif lhs in model._ip:
            target = f"_P_[{model._ip[lhs]}]"
        else:
            target = safe_name(lhs)
            assigned.add(target)
        lines.append(f"    {target} = {tr(rhs)}")
    code = compile("\n".join(lines) + "\n", f"<pydyn steady_state_model:{model.json_dir}>", "exec")
    fns = []
    for base in (FAST_NAMESPACE, NAMESPACE):
        ns = dict(base)
        ns.update(model.functions)
        exec(code, ns)  # noqa: S102 - generated from the model
        fns.append(ns["_ssm"])
    # a name assigned in the block is local to the generated function for all of it; the
    # interpreted version is kept for blocks that read such a name before assigning it
    model._ssm_compiled = (key, tuple(fns))
    return model._ssm_compiled[1]


def _from_steady_state_model(model, params, x=None):
    """Evaluate a ``steady_state_model`` block. May update parameters (returns a copy)."""
    params = np.array(params, float)
    # as in Dynare, variables the block does not set keep their initval (default 0)
    y = np.array([model.initval.get(v, 0.0) for v in model.endo_names])
    x = model.exo_steady() if x is None else x
    if COMPILE_BLOCK:
        try:
            fast, slow = _compiled_block(model)
        except (SyntaxError, NotImplementedError):      # several outputs, MATLAB helpers: interpreted
            fast = slow = None
        if fast is not None:
            try:
                yl, pl = y.tolist(), params.tolist()
                fast(yl, np.asarray(x, float).tolist(), pl)
                return np.array(yl, dtype=float), np.array(pl, dtype=float)
            except UnboundLocalError:
                return _from_steady_state_model_interpreted(model, params, x, y)
            except FAST_ERRORS:
                pass
            try:
                y2, p2 = y.copy(), params.copy()
                slow(y2, x, p2)
                return y2, p2
            except UnboundLocalError:
                pass
    return _from_steady_state_model_interpreted(model, params, x, y)


def _from_steady_state_model_interpreted(model, params, x, y):
    tr = Translator(_resolver(model))
    ns = dict(NAMESPACE)
    ns.update(model.functions)
    ns.update(_Y_=y, _X_=x, _P_=params)

    def store(lhs, val):
        if lhs in model._ie:
            y[model._ie[lhs]] = val
        elif lhs in model._ip:
            params[model._ip[lhs]] = val
        else:
            ns[safe_name(lhs)] = val

    for lhs, rhs in model.steady_state_model:
        try:
            if isinstance(lhs, (list, tuple)):          # [a, b] = f(...)
                with matlab.nargout(len(lhs)):
                    vals = eval(tr(rhs), ns)  # noqa: S307 - model expression
                if not isinstance(vals, (list, tuple)) or len(vals) < len(lhs):
                    raise SteadyStateError(f"steady_state_model: {rhs[:60]} does not return {len(lhs)} values")
                for name, val in zip(lhs, vals):
                    store(name, val)
                continue
            if not isinstance(lhs, str) or not lhs.isidentifier():
                raise NotImplementedError(f"multiple-output steady-state assignment: {lhs}")
            val = eval(tr(rhs), ns)  # noqa: S307 - model expression
        except matlab.Unsupported as exc:
            raise SteadyStateError(f"steady_state_model: a MATLAB function called in {rhs[:60]!r} cannot be "
                                   f"interpreted ({exc}); pass a Python function in Model(..., functions=...)") from None
        except matlab.MatlabError as exc:
            raise SteadyStateError(f"steady_state_model: MATLAB error in {rhs[:60]!r}: {exc}") from None
        store(lhs, val)
    return y, params


def steady_state_file_fn(model):
    """The model's ``<name>_steadystate.m`` as a steady-state function (interpreted or translated by
    ``pydyn.matlab``), or None if there is no such file."""
    if model.steady_state_file is None:
        return None
    fn = getattr(model, "_steady_state_file_fn", None)
    if fn is None or fn.path != model.steady_state_file:
        fn = model._steady_state_file_fn = matlab.SteadyStateFile(model.steady_state_file)
    return fn


def fill_auxiliary(model, ys, x, params):
    """Set auxiliary variables (added by the preprocessor for leads/lags > 1 etc.) from their
    defining equations, which are linear in the auxiliary variable itself."""
    aux = [d["endo_index"] - 1 for d in model.raw.get("aux_vars", [])]
    ys = ys.copy()
    for _ in range(2 if aux else 0):
        for i in aux:
            ys[i] = 0.0 if not np.isfinite(ys[i]) else ys[i]
            r = model.static_resid(ys, x, params)[i]
            ys[i] -= r / model.static_jac(ys, x, params)[i, i]
    return ys


#: Dynare's default tolerance on the steady-state residuals (options_.solve_tolf = eps^(1/3))
DYNARE_TOLF = float(np.finfo(float).eps ** (1.0 / 3.0))

#: seed of the random restarts of the numerical steady-state search (see _random_starts). Dynare
#: draws from MATLAB's/Octave's generator, so the restart points are not Dynare's.
RESTART_SEED = 0
#: at most this many solver runs from random restart points (Dynare runs its solver once, from the
#: first draw that gives finite residuals and Jacobian)
RESTART_SOLVES = 10
#: residuals below this at the initial guess: the guess is returned as it is (the solver is not run)
EXACT = 1e-14
#: Dynare's default steady-state algorithm (options_.solve_algo)
DYNARE_SOLVE_ALGO = 4


def steady_state(model, params=None, guess=None, tol=None, check=None, exo=None, maxit=None, solve_algo=None,
                 tolx=None):
    """Return ``(ys, params)``.

    Uses a Python steady-state function, the ``steady_state_model`` block or a steady-state ``.m``
    file when the model has one, otherwise solves the static model numerically from ``initval``
    (or ``guess``). The residuals are then checked against ``tol``: by default the ``tolf`` given
    to ``steady`` in the .mod file, else Dynare's default (about 6e-6). A residual above 1e-8 that
    passes is reported in a warning. ``check=False`` (or ``steady(nocheck)``) skips the test.

    Numerical search: a guess whose residuals are below 1e-14 is returned unchanged; when the
    residuals or the Jacobian at the guess are not finite, the search restarts from random guesses
    as Dynare's ``dynare_solve`` does (with pydyn's own random stream, seed ``RESTART_SEED``: a
    model with several steady states can end at another one than Dynare; a warning says when this
    happened). ``maxit`` (default: ``steady(maxit=...)`` of the .mod file, else unlimited as before)
    bounds the solver's iterations. ``solve_algo`` (default: the .mod file's): 4, 9, 13 and 14 run
    pydyn's port of Dynare's algorithm (with ``tolx``, default ``steady(tolx=...)`` or eps^(2/3), and
    Dynare's maxit default 50); for the other values pydyn's own search is used, with a warning unless
    Newton's method from the same guess reaches the same root. Without ``solve_algo`` pydyn's own
    search runs (it polishes the root to 1e-14, where Dynare's default solve_algo=4 stops at ``tol``).
    """
    from . import options as _opt
    so = model.options.get("steady", {})
    if tol is None:
        tol = float(so.get("solve_tolf", DYNARE_TOLF))
    if check is None:
        check = not so.get("steadystate.nocheck", False)
    if maxit is None and so.get("steady.maxit") is not None:
        maxit = int(so["steady.maxit"])
    if solve_algo is None and so.get("solve_algo") is not None:
        solve_algo = int(so["solve_algo"])
    ported = solve_algo is not None and int(solve_algo) in PORTED_SOLVE_ALGOS
    if tolx is None and ported and so.get("solve_tolx") is not None:
        tolx = float(so["solve_tolx"])
    _opt.warn_unused(model, "steady", {"solve_tolf", "steadystate.nocheck", "homotopy_mode", "homotopy_steps",
                                       "steady.maxit", "solve_algo"} | ({"solve_tolx"} if ported else set()),
                     "steady_state")
    params = model.params if params is None else np.asarray(params, float)
    x = model.exo_steady() if exo is None else np.asarray(exo, float)
    with np.errstate(all="ignore"):
        ys, params = _steady_state(model, params, guess, x, tol=tol, maxit=maxit, solve_algo=solve_algo, tolx=tolx)
    if check:
        res = np.max(np.abs(model.static_resid(ys, x, params)))
        if not res < tol:
            raise SteadyStateError(f"steady state not found: max |residual| = {res:.3e}")
        if res > 1e-8:
            _opt.warn_once(model, f"steady state accepted with max |residual| = {res:.1e} "
                                  f"(tolerance {tol:.1e}, as in Dynare); results are only that accurate")
    return ys, params


def _warn_once(model, message):
    import warnings
    seen = model.__dict__.setdefault("_warned", set())
    if message not in seen:
        seen.add(message)
        warnings.warn(message, UserWarning, stacklevel=4)


def _finite_at(model, y, x, params):
    """dynare_solve.m's test of a starting point: residuals and Jacobian finite (pydyn's model
    functions return NaN where MATLAB would return a complex number)."""
    if not np.all(np.isfinite(y)):
        return False
    r = model.static_resid(y, x, params)
    if not np.all(np.isfinite(r)):
        return False
    return bool(np.all(np.isfinite(model.static_jac(y, x, params))))


def _random_draws(ok, y0):
    """The randomised initial guesses of Dynare's dynare_solve.m (options_.solve_randomize_initial_guess,
    on by default): the entries ``idx`` are redrawn, first as rand*10, then randn*10, then -rand*10,
    up to 10*len(idx)+1 times each, keeping the draws ``y`` for which ``ok(y)`` (residuals and
    Jacobian finite). As in dynare_solve.m, ``idx`` is ``find(x)`` (the non-zero entries of the
    guess) unless the whole guess is zero, then every entry. Draws come from numpy's generator with
    seed RESTART_SEED, not from Dynare's stream."""
    y0 = np.array(y0, float)
    nz = np.flatnonzero(np.where(np.isnan(y0), 1.0, y0))     # NaN counts as non-zero, as in MATLAB's find
    idx = nz if nz.size else np.arange(len(y0))
    k = len(idx)
    rng = np.random.default_rng(RESTART_SEED)
    for draw in (lambda: rng.random(k) * 10, lambda: rng.standard_normal(k) * 10, lambda: -rng.random(k) * 10):
        for _ in range(10 * k + 1):
            y = y0.copy()
            y[idx] = draw()
            with np.errstate(all="ignore"):
                good = ok(y)
            if good:
                yield y


def _random_starts(model, y0, x, params):
    return _random_draws(lambda y: _finite_at(model, y, x, params), y0)


def _solve_from(model, y0, x, params, maxit):
    """scipy's hybrid Powell method from ``y0``, then Newton steps to polish the root. With
    ``maxit`` the hybrid method gets maxit+1 function evaluations (about maxit iterations) and the
    polish only runs on a point whose residuals are already below Dynare's tolerance."""
    kw = {} if maxit is None else {"options": {"maxfev": int(maxit) + 1}}
    sol = optimize.root(lambda y: model.static_resid(y, x, params),
                        y0, jac=lambda y: model.static_jac(y, x, params),
                        method="hybr", tol=1e-13, **kw)
    ys = sol.x
    if maxit is not None:
        r = model.static_resid(ys, x, params)
        if not (np.all(np.isfinite(r)) and np.max(np.abs(r)) < DYNARE_TOLF):
            return ys
    for _ in range(50):
        r = model.static_resid(ys, x, params)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(ys)):
            return None
        if np.max(np.abs(r)) < EXACT:
            break
        try:
            ys = ys - np.linalg.solve(model.static_jac(ys, x, params), r)
        except np.linalg.LinAlgError:
            ys = ys - np.linalg.lstsq(model.static_jac(ys, x, params), r, rcond=None)[0]
    return ys


def _converged(model, ys, x, params, tol):
    if ys is None or not np.all(np.isfinite(ys)):
        return False
    r = model.static_resid(ys, x, params)
    return bool(np.all(np.isfinite(r)) and np.max(np.abs(r)) < tol)


def _numerical(model, y0, x, params, tol, maxit, solve_algo=None, tolx=None):
    """The numerical search (Dynare: dynare_solve.m called by evaluate_steady_state.m). With a
    ``solve_algo`` pydyn has ported (PORTED_SOLVE_ALGOS) Dynare's algorithm is run as it is, with
    Dynare's defaults maxit=50 and tolx=eps^(2/3); otherwise pydyn's own search."""
    if solve_algo is not None and int(solve_algo) in PORTED_SOLVE_ALGOS:
        def fun(y):
            return model.static_resid(y, x, params), model.static_jac(y, x, params)
        ys, err, restarted = dynare_solve(fun, y0, tol, _EPS ** (2 / 3) if tolx is None else float(tolx),
                                          50 if maxit is None else int(maxit), int(solve_algo))
        if restarted:
            _warn_once(model, "steady state: the residuals or the Jacobian are not finite at the initial guess, so the "
                              "search restarted from random guesses, as Dynare's dynare_solve does; pydyn's random draws "
                              f"(seed {RESTART_SEED}) differ from Dynare's, so with several steady states the one found "
                              "can differ from Dynare's")
        if err:
            raise SteadyStateError(f"steady state not found (solve_algo={int(solve_algo)}, as Dynare's dynare_solve)")
        return ys
    if np.all(np.isfinite(y0)):
        r0 = model.static_resid(y0, x, params)
        if np.all(np.isfinite(r0)) and np.max(np.abs(r0), initial=0.0) < EXACT:
            return y0.copy()                                     # already a solution: never move away from it
    if _finite_at(model, y0, x, params):
        ys = _solve_from(model, y0, x, params, maxit)
        if ys is None:
            raise SteadyStateError("steady state not found: non-finite values during the search")
        return ys
    solves, last = 0, None
    for start in _random_starts(model, y0, x, params):
        if solves == 0:
            _warn_once(model, "steady state: the residuals or the Jacobian are not finite at the initial guess, so the "
                              "search restarted from random guesses, as Dynare's dynare_solve does; pydyn's random draws "
                              f"(seed {RESTART_SEED}) differ from Dynare's, so with several steady states the one found "
                              "can differ from Dynare's")
        solves += 1
        ys = _solve_from(model, start, x, params, maxit)
        if _converged(model, ys, x, params, tol):
            return ys
        last = ys if ys is not None else last
        if solves >= RESTART_SOLVES:
            break
    if solves == 0:
        raise SteadyStateError("steady state not found: the residuals or the Jacobian are not finite at the initial "
                               "guess nor at any of the random restart points")
    if last is None:
        raise SteadyStateError("steady state not found: non-finite values during the search")
    return last


def _newton_reaches(model, y0, ys, x, params, maxit=50):
    """Does plain Newton from ``y0`` converge to ``ys``? (Dynare's solve_algo values are Newton-type
    methods; from a guess where Newton converges they find the same root.)"""
    y = np.array(y0, float)
    for _ in range(maxit):
        r = model.static_resid(y, x, params)
        if not np.all(np.isfinite(r)):
            return False
        if np.max(np.abs(r)) < 1e-12:
            break
        try:
            y = y - np.linalg.solve(model.static_jac(y, x, params), r)
        except np.linalg.LinAlgError:
            return False
    else:
        return False
    return bool(np.all(np.abs(y - ys) <= 1e-8 * np.maximum(1.0, np.abs(ys))))


def _check_solve_algo(model, solve_algo, y0, ys, x, params):
    if solve_algo is None or int(solve_algo) in PORTED_SOLVE_ALGOS:
        return
    a = int(solve_algo)
    if a in (10, 11):
        _warn_once(model, f"steady state: solve_algo={a} (a complementarity problem) is not implemented: the bounds of "
                          "the complementarity conditions are ignored by pydyn's steady-state search")
        return
    if not (np.all(np.isfinite(y0)) and _newton_reaches(model, y0, ys, x, params)):
        _warn_once(model, f"steady state: pydyn does not have Dynare's solve_algo={a}; its own search was used, and "
                          "Newton's method from the same guess does not confirm the root, so with several steady states "
                          "the one found can differ from Dynare's")


def _steady_state(model, params, guess, x, tol=DYNARE_TOLF, maxit=None, solve_algo=None, tolx=None):
    # precedence: Python function given by the caller, steady_state_model block, steady-state .m file
    fn = model.steady_state_fn
    if fn is None and model.steady_state_model is None:
        fn = steady_state_file_fn(model)
    if fn is not None:
        try:
            out = fn(model, params.copy(), x)
        except matlab.Unsupported as exc:
            raise SteadyStateError(
                f"{model.steady_state_file.name} cannot be interpreted ({exc}): pass an equivalent "
                "function as Model(..., steady_state_fn=...)") from None
        if isinstance(out, tuple):
            ys, params = np.asarray(out[0], float), np.asarray(out[1], float)
        else:
            ys = np.asarray(out, float)
        n_orig = model.raw.get("orig_endo_nbr", model.n_endo)
        if ys.shape == (n_orig,) and n_orig < model.n_endo:
            ys = np.concatenate([ys, np.full(model.n_endo - n_orig, np.nan)])
        if ys.shape != (model.n_endo,):
            raise SteadyStateError(f"steady_state_fn returned {ys.shape}, expected ({n_orig},) or ({model.n_endo},)")
        ys = fill_auxiliary(model, ys, x, params)
    elif model.steady_state_model is not None:
        with matlab.steady_context(model, params):      # MATLAB helpers may read "global M_"
            ys, params = _from_steady_state_model(model, params, x)
        if np.any(~np.isfinite(ys)):
            bad = [v for v, ok in zip(model.endo_names, np.isfinite(ys)) if not ok]
            raise SteadyStateError(f"steady_state_model gives non-finite values for: {bad}")
    else:
        if guess is None and WARM_START:
            prev = model.__dict__.get("_ss_prev")
            if prev is not None and np.array_equal(prev[0], x):
                ys = _newton(model, prev[1], x, params)
                if ys is not None:
                    model._ss_prev = (np.array(x, float), ys.copy())
                    return ys, params
        if guess is None:
            y0 = np.array([model.initval.get(v, 0.0) for v in model.endo_names])
            y0 = fill_auxiliary(model, y0, x, params)      # as Dynare does before solving
        else:
            # evaluate_steady_state.m sets the auxiliary variables of any starting point
            y0 = fill_auxiliary(model, np.asarray(guess, float), x, params)
        ys = _numerical(model, y0, x, params, tol, maxit, solve_algo, tolx)
        _check_solve_algo(model, solve_algo, y0, ys, x, params)
        if WARM_START and guess is None:
            model._ss_prev = (np.array(x, float), ys.copy())
    return ys, params


def _newton(model, ys, x, params, maxit=8, tol=1e-14):
    """A few Newton steps from a nearby solution; None unless they converge."""
    for _ in range(maxit):
        r = model.static_resid(ys, x, params)
        if not np.all(np.isfinite(r)):
            return None
        if np.max(np.abs(r)) < tol:
            return ys
        try:
            ys = ys - np.linalg.solve(model.static_jac(ys, x, params), r)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(ys)):
            return None
    return None


# ----------------------------------------------------------------------------------- Dynare's solvers
_EPS = float(np.finfo(float).eps)


def _octave_solve(r, b):
    """Octave's dense ``r \\ b``: LU, or a least-squares solution (LAPACK gelsd) when the matrix is
    singular to machine precision (Octave's Matrix::solve falls back to lssolve)."""
    import warnings
    from scipy import linalg
    with warnings.catch_warnings():
        warnings.simplefilter("error", linalg.LinAlgWarning)
        try:
            return linalg.solve(r, b, check_finite=False)
        except (linalg.LinAlgError, linalg.LinAlgWarning, ValueError):
            pass
    with np.errstate(all="ignore"):
        try:
            return linalg.lstsq(r, b, lapack_driver="gelsd", check_finite=False)[0]
        except (linalg.LinAlgError, ValueError):
            return np.full(len(b), np.nan)


def _dogleg(r, b, d, delta):
    """Subfunction dogleg of trust_region.m."""
    x = _octave_solve(r, b)
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
    return all(np.all(np.isfinite(a)) for a in arrays)


def _trust_region(fun, x, j1, j2, tolf, tolx, maxiter, factor=1.0):
    """matlab/optimization/trust_region.m (solve_algo 4 and 9): returns (x, errorflag, info).
    ``fun(x)`` returns the full residual vector and Jacobian."""
    radiusfactor, actualreductionthreshold, relativereductionthreshold = 0.5, 0.001, 0.1
    x = np.array(x, float)
    errorflag, it, info, delta, ncsucc, ncslow = True, 1, 0, 0.0, 0, 0
    xinit = x.copy()
    ix = np.ix_(j1, j2)
    try:
        fv, fj = fun(x)
    except (ArithmeticError, ValueError):
        return x, True, 0
    fval, fjac = fv[j1], fj[ix]
    fnorm = np.linalg.norm(fval)
    if not _finite(fval, fjac):
        return x, True, 0
    if fnorm < tolf:
        return x, False, -1
    fjacnorm_ = None
    xnorm = 0.0
    while it <= maxiter and not info:
        fjacnorm = np.sqrt(np.sum(fjac * fjac, axis=0))
        if it == 1:
            fjacnorm_ = fjacnorm.copy()
            fjacnorm_[fjacnorm < _EPS] = 1.0
            xnorm = np.linalg.norm(fjacnorm_ * x[j2])
            delta = xnorm * factor if xnorm > 0 else factor
        else:
            fjacnorm_ = np.maximum(0.1 * fjacnorm_, fjacnorm)
            xnorm = np.linalg.norm(fjacnorm_ * x[j2])
        p = _dogleg(fjac, fval, fjacnorm_, delta)
        pnorm = np.linalg.norm(fjacnorm_ * p)
        xx = x.copy()
        x0 = x.copy()
        xx[j2] = x[j2] - p
        ww = fval - fjac @ p
        try:
            fval1 = fun(xx)[0][j1]
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
            xnorm = np.linalg.norm(fjacnorm_ * x[j2])
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
            fjac = fun(x)[1][ix]
        except (ArithmeticError, ValueError):
            x, fjac = x0, fjac0
            delta *= radiusfactor
            continue
        if not _finite(fjac):
            x, fjac = x0, fjac0
            delta *= radiusfactor
    return x, errorflag, info


def _trust_region_f08(fun, x, tolx, tolf, maxiter, factor):
    """trust_region_solve of mex/sources/block_trust_region/trust_region.f08 (solve_algo 13 and 14),
    on the system ``fun(x) -> (fvec, fjac)``; x is updated in place. Returns info."""
    maxslowiter = 15

    def dogleg(r, b, d, delta, gn):
        xn = np.linalg.norm(d * gn)
        if xn <= delta:
            return gn.copy()
        s = (r.T @ b) / d
        sn = np.linalg.norm(s)
        if sn > 0:
            s = s / sn / d
            t = r @ s
            tn = np.linalg.norm(t)
            snm = sn / tn ** 2
            if snm < delta:
                bn, dxn, snmd = np.linalg.norm(b), delta / xn, snm / delta
                tmp = (bn / sn) * (bn / xn) * snmd
                alpha = dxn * (1 - snmd ** 2) / (tmp - dxn * snmd ** 2 + np.sqrt((tmp - dxn) ** 2 + (1 - dxn ** 2) * (1 - snmd ** 2)))
            else:
                alpha = 0.0
        else:
            alpha = delta / xn
            snm = 0.0
        return alpha * gn + ((1 - alpha) * min(snm, delta)) * s

    def gauss_newton(r, b):
        from scipy import linalg
        try:
            lu = linalg.lu_factor(r, check_finite=False)
            if np.any(np.diag(lu[0]) == 0):
                raise linalg.LinAlgError("singular")
            return linalg.lu_solve(lu, b, check_finite=False)
        except (linalg.LinAlgError, ValueError):
            return linalg.lstsq(r, b, cond=-1.0, lapack_driver="gelsd", check_finite=False)[0]

    niter, ncsucc, ncslow, info = 1, 0, 0, 0
    fvec, fjac = fun(x)
    fn = np.linalg.norm(fvec)
    jcn = np.sqrt(np.sum(fjac * fjac, axis=0))
    recompute_gn, gn = True, None
    if np.any(np.isnan(fvec)) or np.any(np.isnan(fjac)):
        return 0
    if fn < tolf:
        return -1
    dg = None
    delta = 0.0
    while info == 0:
        dg = np.where(jcn != 0, jcn, 1.0) if niter == 1 else np.maximum(0.1 * dg, jcn)
        xnorm = np.linalg.norm(dg * x)
        if niter == 1:
            delta = (xnorm if xnorm > 0 else 1.0) * factor
        if recompute_gn:
            with np.errstate(all="ignore"):
                gn = gauss_newton(fjac, fvec)
        recompute_gn = False
        with np.errstate(all="ignore"):
            p = -dogleg(fjac, fvec, dg, delta, gn)
        pnorm = np.linalg.norm(dg * p)
        x2 = x + p
        fvec2 = fun(x2)[0]
        fn2 = np.linalg.norm(fvec2)
        if fn2 < tolf:
            x[:] = x2
            info = 1
            continue
        actred = 1 - (fn2 / fn) ** 2 if fn2 < fn else -1.0
        w = fvec + fjac @ p
        t = np.linalg.norm(w)
        prered = 1 - (t / fn) ** 2 if t < fn else 0.0
        ratio = actred / prered if prered > 0 else 0.0
        if niter == 1:
            delta = min(delta, pnorm)
        if ratio < 0.1:
            delta /= 2
            ncsucc = 0
        else:
            ncsucc += 1
            if abs(1 - ratio) <= 0.1:
                delta = 2 * pnorm
            elif ratio >= 0.5 or ncsucc > 1:
                delta = max(delta, 2 * pnorm)
        if ratio >= 1e-4:
            x0 = x.copy()
            x[:] = x2
            fvec, fjac = fun(x)
            recompute_gn = True
            fn = np.linalg.norm(fvec)
            jcn = np.sqrt(np.sum(fjac * fjac, axis=0))
            if np.any(np.isnan(fvec)) or np.any(np.isnan(fjac)):
                x[:] = x0
                fvec, fjac = fun(x)
                recompute_gn = True
                fn = np.linalg.norm(fvec)
                jcn = np.sqrt(np.sum(fjac * fjac, axis=0))
                delta /= 2
                ncslow += 1
                niter += 1
                continue
            xnorm = np.linalg.norm(dg * x)
        ncslow += 1
        if actred >= 0.001:
            ncslow = 0
        niter += 1
        if niter == maxiter:
            info = 2
            continue
        if delta <= tolx * xnorm:
            info = 3
            continue
        if tolx > 0 and max(0.1 * delta, pnorm) <= 10 * _EPS * xnorm:
            info = 5
            continue
        if ncslow == maxslowiter:
            info = 4
    return info


def _dm_blocks(jac):
    """Square diagonal blocks of the Dulmage-Mendelsohn (block upper triangular) decomposition of
    the structure of ``jac`` (MATLAB's dmperm), in the order Dynare solves them (the last block of
    the upper triangular form first, so that each block only depends on blocks already solved).
    None when the matrix is structurally singular (Dynare then has non-square blocks)."""
    from scipy.sparse import csgraph, csr_matrix
    n = jac.shape[0]
    S = csr_matrix(jac != 0)
    match = csgraph.maximum_bipartite_matching(S, perm_type="column")      # match[i]: column of row i
    if np.any(match < 0):
        return None
    rowof = np.empty(n, np.intp)
    rowof[match] = np.arange(n)
    coo = S.tocoo()
    G = csr_matrix((np.ones(coo.nnz), (coo.row, rowof[coo.col])), shape=(n, n))
    nb, labels = csgraph.connected_components(G, directed=True, connection="strong")
    # order the strongly connected components so that dependencies come first (iterative DFS)
    deps = [set() for _ in range(nb)]
    for i, k in zip(coo.row, rowof[coo.col]):
        if labels[i] != labels[k]:
            deps[labels[i]].add(labels[k])
    order, state = [], np.zeros(nb, int)
    for root in range(nb):
        if state[root]:
            continue
        stack = [(root, iter(sorted(deps[root])))]
        state[root] = 1
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                state[node] = 2
                order.append(node)
            elif not state[nxt]:
                state[nxt] = 1
                stack.append((nxt, iter(sorted(deps[nxt]))))
    blocks = []
    for b in order:
        rows = np.flatnonzero(labels == b)
        blocks.append((rows, match[rows]))
    return blocks


def dynare_solve(fun, x, tolf, tolx, maxit, solve_algo=4, factor=1.0, randomize=True):
    """matlab/optimization/dynare_solve.m for ``solve_algo`` 4 (Dulmage-Mendelsohn blocks, then the
    whole system, with trust_region.m), 9 (trust_region.m on the whole system), 13 and 14 (the
    block_trust_region MEX, with and without the block decomposition). ``fun(x)`` returns the
    residuals and the dense Jacobian. Returns ``(x, errorflag, restarted)``.

    As in Dynare: a guess whose residuals are below ``tolf`` is returned unchanged; when the
    residuals or the Jacobian are not finite at the guess, new guesses are drawn (see
    _random_starts; pydyn's draws are not Dynare's). A structurally singular Jacobian (for which
    MATLAB's dmperm returns non-square blocks) is solved without the block decomposition."""
    x = np.array(x, float)
    n = len(x)
    with np.errstate(all="ignore"):
        fvec, fjac = fun(x)
    restarted = False
    if _finite(fvec) and np.max(np.abs(fvec), initial=0.0) < tolf:
        return x, False, False
    if not _finite(fvec, fjac):
        if not randomize:
            return x, True, False
        start = next(_random_draws(lambda y: _finite(*fun(y)), x), None)
        if start is None:
            return x, True, False
        x, restarted = start, True
        with np.errstate(all="ignore"):
            fvec, fjac = fun(x)
    allidx = np.arange(n)
    with np.errstate(all="ignore"):
        if solve_algo == 9:
            x, err, _ = _trust_region(fun, x, allidx, allidx, tolf, tolx, maxit, factor)
            return x, err, restarted
        if solve_algo == 4:
            blocks = _dm_blocks(fjac)
            if blocks is not None:
                for rows, cols in blocks:
                    x, err, _ = _trust_region(fun, x, rows, cols, tolf, tolx, maxit, factor)
                    if err:
                        return x, True, restarted
                fvec = fun(x)[0]
                if not np.max(np.abs(fvec)) > tolf:
                    return x, False, restarted
            x, err, _ = _trust_region(fun, x, allidx, allidx, tolf, tolx, maxit, factor)
            return x, err, restarted
        if solve_algo in (13, 14):
            blocks = _dm_blocks(fjac) if solve_algo == 13 else None
            info = 0
            if blocks is not None:
                for rows, cols in blocks:
                    def sub(z, rows=rows, cols=cols):
                        y = x.copy()
                        y[cols] = z
                        fv, fj = fun(y)
                        return fv[rows], fj[np.ix_(rows, cols)]
                    z = x[cols].copy()
                    info = _trust_region_f08(sub, z, tolx, tolf, maxit, factor)
                    x[cols] = z
                fvec = fun(x)[0]
                if np.max(np.abs(fvec)) > tolf:
                    info = _trust_region_f08(fun, x, tolx, tolf, maxit, factor)
                elif len(blocks) > 1:
                    info = 1
            else:
                info = _trust_region_f08(fun, x, tolx, tolf, maxit, factor)
            return x, info not in (1, -1), restarted
    raise NotImplementedError(f"solve_algo={solve_algo} is not available in pydyn's port of dynare_solve")


#: values of solve_algo that pydyn runs as Dynare does
PORTED_SOLVE_ALGOS = (4, 9, 13, 14)
