"""Deterministic (perfect foresight) simulation: Newton's method on the stacked system."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .steady import steady_state


class PerfectForesightError(Exception):
    pass


@dataclass
class PerfectForesightResult:
    endo: np.ndarray        # (periods + 2, n_endo): initial condition, the simulation, terminal condition
    exo: np.ndarray         # (periods + 2, n_exo)
    iterations: int
    max_residual: float


def exo_path(model, periods):
    """Exogenous path from initval/endval and the deterministic ``shocks`` blocks (levels)."""
    x0, xT = model.exo_steady(), model.exo_steady(terminal=True)
    x = np.tile(xT, (periods + 2, 1))
    x[0] = x0
    for j, p1, p2, val in model.det_shocks:
        if p2 > periods:
            raise PerfectForesightError("a shock is dated after the end of the simulation")
        x[p1:p2 + 1, j] = val
    return x


def perfect_foresight(model, periods=None, params=None, exo=None, y_init=None, y_term=None,
                      guess=None, tol=None, maxit=None) -> PerfectForesightResult:
    """Solve F(y[t-1], y[t], y[t+1], x[t]) = 0 for t = 1..periods.

    By default the path starts at the steady state for ``initval``, ends at the steady state for
    ``endval`` (or the initial one), and the exogenous path comes from the ``shocks`` blocks, as
    with ``perfect_foresight_setup`` followed by ``perfect_foresight_solver`` in Dynare.
    """
    from . import options as _opt
    so = model.options.get("perfect_foresight_solver", {})
    for key in ("lmmcp", "linear_approximation", "endogenous_terminal_period"):
        if so.get(key):
            raise NotImplementedError(f"perfect_foresight_solver({key}) is not implemented")
    if any("mcp" in (eq.get("tags") or {}) for eq in model.raw.get("model", []) if isinstance(eq, dict)):
        raise NotImplementedError("equations tagged [mcp=...] (complementarity conditions) are not implemented")
    tol = float(so.get("dynatol.f", 1e-9)) if tol is None else tol      # Dynare's default is 1e-5; ours is tighter
    maxit = int(so.get("simul.maxit", 50)) if maxit is None else maxit
    _opt.warn_unused(model, "perfect_foresight_solver", {"dynatol.f", "simul.maxit"}, "perfect_foresight")
    _opt.warn_unused(model, "perfect_foresight_setup", {"periods"}, "perfect_foresight")
    if periods is None:
        periods = model.options.get("perfect_foresight_setup", {}).get("periods")
        if periods is None:
            raise ValueError("periods not given and not set in the .mod file")
    n, T = model.n_endo, int(periods)
    x = exo_path(model, T) if exo is None else np.asarray(exo, float)
    if x.shape != (T + 2, model.n_exo):
        raise ValueError(f"exo must have shape ({T + 2}, {model.n_exo})")
    ys0, params = steady_state(model, params, exo=model.exo_steady())
    if model.endval:
        guess_T = np.array([model.endval.get(v, ys0[i]) for i, v in enumerate(model.endo_names)])
        ysT, params = steady_state(model, params, exo=model.exo_steady(terminal=True), guess=guess_T) \
            if model.steady_state_model is None and model.steady_state_fn is None \
            else steady_state(model, params, exo=model.exo_steady(terminal=True))
    else:
        ysT = ys0
    y0 = ys0 if y_init is None else np.asarray(y_init, float)
    yT = ysT if y_term is None else np.asarray(y_term, float)
    Y = np.tile(ysT, (T, 1)) if guess is None else np.asarray(guess, float).copy()

    def stacked(Y):
        full = np.vstack([y0, Y, yT])
        res = np.empty((T, n))
        rows, cols, vals = [], [], []
        for t in range(T):
            yy = full[t:t + 3].ravel()
            res[t] = model.dynamic_resid(yy, x[t + 1], params, ysT)
            J = model.dynamic_jac(yy, x[t + 1], params, ysT)[:, :3 * n]
            r, c = np.nonzero(J)
            gc = (t - 1) * n + c                      # column in the stacked unknown vector
            keep = (gc >= 0) & (gc < T * n)
            rows.append(t * n + r[keep]); cols.append(gc[keep]); vals.append(J[r[keep], c[keep]])
        A = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(T * n, T * n))
        return res.ravel(), A

    for it in range(1, maxit + 1):
        with np.errstate(all="ignore"):
            res, A = stacked(Y)
        err = np.max(np.abs(res))
        if not np.isfinite(err):
            raise PerfectForesightError("non-finite residuals; try a different initial guess")
        if err < tol:
            return PerfectForesightResult(endo=np.vstack([y0, Y, yT]), exo=x, iterations=it - 1, max_residual=err)
        Y = Y - spsolve(A, res).reshape(T, n)
    raise PerfectForesightError(f"no convergence in {maxit} iterations (max residual {err:.2e})")
