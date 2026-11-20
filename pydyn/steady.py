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
    def resolve(name, shift):
        if name in model._ie:
            return f"y[{model._ie[name]}]"
        if name in model._ix:
            return f"x[{model._ix[name]}]"
        if name in model._ip:
            return f"params[{model._ip[name]}]"
        return None
    return resolve


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
    lines, assigned = ["def _ssm(y, x, params):"], set()
    for lhs, rhs in model.steady_state_model:
        if not isinstance(lhs, str) or not lhs.isidentifier():
            raise NotImplementedError(f"multiple-output steady-state assignment: {lhs}")
        if lhs in model._ie:
            target = f"y[{model._ie[lhs]}]"
        elif lhs in model._ip:
            target = f"params[{model._ip[lhs]}]"
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
    ns.update(y=y, x=x, params=params)
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


def steady_state(model, params=None, guess=None, tol=1e-10, check=True, exo=None):
    """Return ``(ys, params)``.

    Uses the ``steady_state_model`` block when the model has one, otherwise solves the
    static model numerically from ``initval`` (or ``guess``).
    """
    params = model.params if params is None else np.asarray(params, float)
    x = model.exo_steady() if exo is None else np.asarray(exo, float)
    with np.errstate(all="ignore"):
        ys, params = _steady_state(model, params, guess, x)
    if check:
        res = np.max(np.abs(model.static_resid(ys, x, params)))
        if not res < tol:
            raise SteadyStateError(f"steady state not found: max |residual| = {res:.3e}")
    return ys, params


def _steady_state(model, params, guess, x):
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
            y0 = np.asarray(guess, float)
        if not np.all(np.isfinite(y0)):
            raise SteadyStateError("initial guess for the steady state is not finite")
        sol = optimize.root(lambda y: model.static_resid(y, x, params),
                            y0, jac=lambda y: model.static_jac(y, x, params),
                            method="hybr", tol=1e-13)
        ys = sol.x
        # polish with Newton steps
        for _ in range(50):
            r = model.static_resid(ys, x, params)
            if not np.all(np.isfinite(r)) or not np.all(np.isfinite(ys)):
                raise SteadyStateError("steady state not found: non-finite values during the search")
            if np.max(np.abs(r)) < 1e-14:
                break
            try:
                ys = ys - np.linalg.solve(model.static_jac(ys, x, params), r)
            except np.linalg.LinAlgError:
                ys = ys - np.linalg.lstsq(model.static_jac(ys, x, params), r, rcond=None)[0]
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
