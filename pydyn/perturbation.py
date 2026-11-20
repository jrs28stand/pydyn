"""First-order perturbation solution."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

QZ_CRITERIUM = 1.0 + 1e-6  # Dynare's default: eigenvalues below this count as stable


class BlanchardKahnError(Exception):
    pass


@dataclass
class FirstOrderSolution:
    """y_t - ys = G (y_{t-1} - ys) + H u_t, variables in declaration order.

    ``ghx`` / ``ghu`` reproduce Dynare's ``oo_.dr.ghx`` / ``oo_.dr.ghu``: rows in
    decision-rule order (``order_var``), ``ghx`` columns restricted to the state variables.
    """
    ys: np.ndarray
    G: np.ndarray
    H: np.ndarray
    order_var: np.ndarray      # 0-based indices into declaration order
    state_var: np.ndarray      # 0-based declaration indices of states, in DR order
    eigenvalues: np.ndarray
    jacobian: np.ndarray
    params: np.ndarray = None

    def as_dr(self) -> dict:
        """Decision rule in the dict form used by ``pydyn.simulate`` / ``pydyn.irf``."""
        return {"ys": self.ys, "order_var": self.order_var, "state_var": self.state_var,
                "ghx": self.ghx, "ghu": self.ghu, "order": 1}

    @property
    def ghx(self):
        return self.G[np.ix_(self.order_var, self.state_var)]

    @property
    def ghu(self):
        return self.H[self.order_var]

    def irf(self, shock: np.ndarray, periods: int = 40) -> np.ndarray:
        """Deviations from steady state after a one-off shock vector at t=1; shape (periods, n)."""
        out = np.empty((periods, self.G.shape[0]))
        y = self.H @ np.asarray(shock, float)
        for t in range(periods):
            out[t] = y
            y = self.G @ y
        return out

    def simulate(self, shocks: np.ndarray, y0=None) -> np.ndarray:
        """Simulate deviations from steady state for a (T, n_exo) shock array."""
        y = np.zeros(self.G.shape[0]) if y0 is None else np.asarray(y0, float)
        out = np.empty((len(shocks), y.size))
        for t, u in enumerate(shocks):
            y = self.G @ y + self.H @ u
            out[t] = y
        return out


def dr_ordering(lead_lag_incidence: np.ndarray):
    """Dynare's decision-rule ordering: static, purely backward, mixed, purely forward."""
    lag = lead_lag_incidence[:, 0] > 0
    lead = lead_lag_incidence[:, 2] > 0
    idx = np.arange(len(lag))
    static, pred, both, fwrd = idx[~lag & ~lead], idx[lag & ~lead], idx[lag & lead], idx[~lag & lead]
    order_var = np.concatenate([static, pred, both, fwrd])
    return order_var, np.concatenate([pred, both]), dict(nstatic=len(static), npred=len(pred),
                                                         nboth=len(both), nfwrd=len(fwrd))


def solve_linear_re(fm, f0, fp, fu, qz_criterium=QZ_CRITERIUM):
    """Solve fp E y(+1) + f0 y + fm y(-1) + fu u = 0 for y = G y(-1) + H u."""
    n = f0.shape[0]
    I, Z0 = np.eye(n), np.zeros((n, n))
    # A x_{t+1} = B x_t with x_t = [y_{t-1}; y_t]
    A = np.block([[I, Z0], [f0, fp]])
    B = np.block([[Z0, I], [-fm, Z0]])
    stable = lambda alpha, beta: np.abs(alpha) < qz_criterium * np.abs(beta)  # noqa: E731
    S, T, alpha, beta, Q, Z = linalg.ordqz(B, A, sort=stable, output="real")
    with np.errstate(divide="ignore", invalid="ignore"):
        eig = alpha / beta
    n_stable = int(np.sum(np.abs(alpha) < qz_criterium * np.abs(beta)))
    if n_stable != n:
        kind = "indeterminacy" if n_stable > n else "no stable solution"
        raise BlanchardKahnError(
            f"Blanchard-Kahn conditions not satisfied ({kind}): {n_stable} stable roots for {n} variables")
    Z11, Z21 = Z[:n, :n], Z[n:, :n]
    if np.linalg.cond(Z11) > 1e12:
        raise BlanchardKahnError("rank condition not satisfied")
    G = np.linalg.solve(Z11.T, Z21.T).T
    H = -np.linalg.solve(fp @ G + f0, fu)
    return G, H, eig


def first_order(model, ys=None, params=None) -> FirstOrderSolution:
    from .steady import steady_state

    if ys is None:
        ys, params = steady_state(model, params)
    params = model.params if params is None else params
    n = model.n_endo
    J = model.dynamic_jac(np.tile(ys, 3), model.exo_steady(), params, ys)
    fm, f0, fp, fu = J[:, :n], J[:, n:2 * n], J[:, 2 * n:3 * n], J[:, 3 * n:]
    G, H, eig = solve_linear_re(fm, f0, fp, fu)
    order_var, state_var, _ = dr_ordering(model.lead_lag_incidence)
    # clean numerical noise in columns of non-state variables
    mask = np.ones(n, bool)
    mask[state_var] = False
    G[:, mask] = 0.0
    return FirstOrderSolution(ys=ys, G=G, H=H, order_var=order_var, state_var=state_var,
                              eigenvalues=eig, jacobian=J, params=np.array(params, float))
