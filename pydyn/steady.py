"""Steady-state computation."""
from __future__ import annotations

import numpy as np
from scipy import optimize

from .expr import NAMESPACE, Translator, safe_name


class SteadyStateError(ValueError):
    pass


def _from_steady_state_model(model, params, x=None):
    """Evaluate a ``steady_state_model`` block. May update parameters (returns a copy)."""
    params = np.array(params, float)
    # as in Dynare, variables the block does not set keep their initval (default 0)
    y = np.array([model.initval.get(v, 0.0) for v in model.endo_names])
    x = model.exo_steady() if x is None else x

    def resolve(name, shift):
        if name in model._ie:
            return f"y[{model._ie[name]}]"
        if name in model._ix:
            return f"x[{model._ix[name]}]"
        if name in model._ip:
            return f"params[{model._ip[name]}]"
        return None

    tr = Translator(resolve)
    ns = dict(NAMESPACE)
    ns.update(model.functions)
    ns.update(y=y, x=x, params=params)
    for lhs, rhs in model.steady_state_model:
        if not isinstance(lhs, str) or not lhs.isidentifier():
            raise NotImplementedError(f"multiple-output steady-state assignment: {lhs}")
        val = eval(tr(rhs), ns)  # noqa: S307 - model expression
        if lhs in model._ie:
            y[model._ie[lhs]] = val
        elif lhs in model._ip:
            params[model._ip[lhs]] = val
        else:
            ns[safe_name(lhs)] = val
    return y, params


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
    if model.steady_state_fn is not None:
        out = model.steady_state_fn(model, params.copy(), x)
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
        if model.steady_state_file is not None and guess is None and not model.initval:
            raise SteadyStateError(
                f"{model.steady_state_file.name} cannot be run from Python: pass an equivalent "
                "function as Model(..., steady_state_fn=...)")
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
    return ys, params
