"""Occasionally binding constraints: piecewise-linear solution (Guerrieri and Iacoviello, 2015).

Follows the algorithm of Dynare's ``occbin.solve_one_constraint`` / ``solve_two_constraints``:
for each period in which an unanticipated shock arrives, guess the periods in which each
constraint binds, build the implied time-varying decision rules by backward recursion from the
reference-regime rule, simulate, check the constraints, and update the guess until it is verified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .expr import NAMESPACE, Translator, tokenize
from .perturbation import first_order
from .steady import steady_state


class OccbinError(Exception):
    pass


@dataclass
class OccbinResult:
    piecewise: np.ndarray            # (periods, n_endo), levels
    linear: np.ndarray               # (periods, n_endo), levels, ignoring the constraints
    shocks: np.ndarray               # (n_shock_periods, n_exo)
    regime_history: list             # per shock period: [(regimes, starts) for each constraint], starts 1-based
    ys: np.ndarray
    iterations: list = field(default_factory=list)


def _split_comparison(expr: str):
    """Split 'lhs < rhs' at the top-level comparison operator; returns (lhs, op, rhs) or None."""
    depth, pos = 0, 0
    toks = tokenize(expr)
    # recover character offsets by re-scanning
    idx = 0
    for kind, val in toks:
        idx = expr.index(val, pos)
        if val == "(":
            depth += 1
        elif val == ")":
            depth -= 1
        elif depth == 0 and val in ("<", ">", "<=", ">="):
            return expr[:idx], val, expr[idx + len(val):]
        pos = idx + len(val)
    return None


class Occbin:
    """Piecewise-linear solver for a model with an ``occbin_constraints`` block."""

    def __init__(self, model, params=None):
        if not model.occbin_constraints:
            raise ValueError("model has no occbin_constraints block")
        self.model = m = model
        self.names = [c["name"] for c in m.occbin_constraints]
        self.k = len(self.names)
        self.switch = [m._ip[f"occbin_{nm}_bind"] for nm in self.names]

        base = np.array(m.params if params is None else params, float)
        base[self.switch] = 0.0
        self.ys, self.params = steady_state(m, base)
        self.ref = first_order(m, self.ys, self.params)
        self.P, self.Q = self.ref.G, self.ref.H
        self.n, self.n_exo = m.n_endo, m.n_exo
        self._regimes: dict[tuple, tuple] = {}
        self._compile_constraints()

    # ---------------------------------------------------------------- regime matrices
    def regime_matrices(self, regime: tuple):
        """(C, B, A, J, D): coefficients on y(-1), y, y(+1), u and the constant, for a regime."""
        if regime not in self._regimes:
            p = self.params.copy()
            for s, on in zip(self.switch, regime):
                p[s] = 1.0 if on else 0.0
            yy, x0, n = np.tile(self.ys, 3), self.model.exo_steady(), self.n
            J = self.model.dynamic_jac(yy, x0, p, self.ys)
            D = self.model.dynamic_resid(yy, x0, p, self.ys) if any(regime) else np.zeros(n)
            self._regimes[regime] = (J[:, :n], J[:, n:2 * n], J[:, 2 * n:3 * n], J[:, 3 * n:], D)
        return self._regimes[regime]

    # ---------------------------------------------------------------- constraints
    def _compile_constraints(self):
        m = self.model

        def resolve(name, shift):
            if name in m._ie:
                if shift != 0:
                    raise NotImplementedError("leads/lags in occbin constraints")
                return f"z[:, {m._ie[name]}]"
            if name in m._ip:
                return f"params[{m._ip[name]}]"
            return None

        tr = Translator(resolve, steady=lambda v: f"ss[{m._ie[v]}]", shiftable=set(m._ie))
        ns = dict(NAMESPACE)

        def make(expr):
            cond = eval("lambda z, params, ss: " + tr(expr), ns)  # noqa: S307 - model expression
            parts = _split_comparison(expr)
            if parts is None:
                err = lambda z, params, ss: np.asarray(cond(z, params, ss), float)  # noqa: E731
            else:
                err = eval(f"lambda z, params, ss: abs(({tr(parts[0])}) - ({tr(parts[2])})) + 0*z[:, 0]", ns)  # noqa: S307
            return cond, err

        self._cons = []
        for c in m.occbin_constraints:
            bind, ebind = make(c["bind"])
            if c.get("relax"):
                relax, erelax = make(c["relax"])
            else:
                relax, erelax = None, None
            self._cons.append((bind, ebind, relax, erelax))

    def check(self, z_levels):
        """Return boolean arrays (binding, relax) and error arrays, each of shape (T, k)."""
        T = z_levels.shape[0]
        binding, relax = np.zeros((T, self.k), bool), np.zeros((T, self.k), bool)
        eb, er = np.zeros((T, self.k)), np.zeros((T, self.k))
        for j, (bind, ebind, rel, erel) in enumerate(self._cons):
            binding[:, j] = np.broadcast_to(bind(z_levels, self.params, self.ys), (T,))
            eb[:, j] = ebind(z_levels, self.params, self.ys)
            if rel is None:
                relax[:, j], er[:, j] = ~binding[:, j], eb[:, j]
            else:
                relax[:, j] = np.broadcast_to(rel(z_levels, self.params, self.ys), (T,))
                er[:, j] = erel(z_levels, self.params, self.ys)
        return binding, relax, eb, er

    # ---------------------------------------------------------------- path for a given guess
    def rules(self, indicator: np.ndarray):
        """Time-varying rules y_t = T_t y_{t-1} + R_t u_t + C_t for t = 1..Tmax (list of tuples)."""
        any_bind = indicator.any(axis=1)
        Tmax = int(np.nonzero(any_bind)[0][-1]) + 1 if any_bind.any() else 0
        n, out = self.n, [None] * Tmax
        T_next, C_next = self.P, np.zeros(n)
        for i in range(Tmax - 1, -1, -1):
            C, B, A, J, D = self.regime_matrices(tuple(bool(b) for b in indicator[i]))
            sol = -np.linalg.solve(B + A @ T_next, np.column_stack([C, J, A @ C_next + D]))
            T_next, R, C_next = sol[:, :n], sol[:, n:n + self.n_exo], sol[:, -1]
            out[i] = (T_next, R, C_next)
        return out

    def path(self, indicator: np.ndarray, init: np.ndarray, shock: np.ndarray):
        """Deviations from steady state for len(indicator) periods given a regime guess."""
        rules = self.rules(indicator)
        N = indicator.shape[0]
        z = np.empty((N, self.n))
        y = init
        for t in range(N):
            if t < len(rules):
                T, R, C = rules[t]
                y = T @ y + C + (R @ shock if t == 0 else 0.0)
            else:
                y = self.P @ y + (self.Q @ shock if t == 0 else 0.0)
            z[t] = y
        return z, rules

    @staticmethod
    def map_regime(col):
        regime, start = [int(col[0])], [1]
        for i in range(1, len(col)):
            if int(col[i]) != regime[-1]:
                regime.append(int(col[i]))
                start.append(i + 1)
        return regime, start

    # ---------------------------------------------------------------- simulation
    def simulate(self, shocks, periods=100, check_ahead_periods=200, maxit=30, init=None,
                 curb_retrench=False) -> OccbinResult:
        """Piecewise-linear simulation under a sequence of unanticipated shocks.

        ``shocks`` has shape (n_shock_periods, n_exo); the simulation runs for ``periods`` periods.
        """
        shocks = np.atleast_2d(np.asarray(shocks, float))
        S = shocks.shape[0]
        periods = max(periods, S)
        N = max(check_ahead_periods, periods - S)          # look-ahead horizon (rows = N + 1)
        y0 = np.zeros(self.n) if init is None else np.asarray(init, float)
        init_orig = y0.copy()
        bi = np.zeros((N + 1, self.k), bool)
        piecewise = np.zeros((periods, self.n))
        history, iters, z = [], [], None

        for sp in range(S):
            changed, it, increased, seen = True, 0, False, []
            while changed and it < maxit:
                it += 1
                if bi[-1].any():
                    bi = np.vstack([bi, np.zeros((1, self.k), bool)])
                    increased = True
                used = bi.copy()
                seen.append(used)
                if sp == 0 or np.any(shocks[sp] != 0):
                    z, _ = self.path(used, y0, shocks[sp])
                    binding, relax, _, _ = self.check(z + self.ys)
                    changed = bool(np.any(binding & ~used) or np.any(relax & used))
                    if curb_retrench:
                        # slow the update down: release at most the last binding period per constraint
                        retrench = np.zeros_like(used)
                        for j in range(self.k):
                            rb = np.nonzero(relax[:, j] & used[:, j])[0]
                            r_all, b_all = np.nonzero(relax[:, j])[0], np.nonzero(used[:, j])[0]
                            if len(rb) and len(b_all) and r_all[-1] >= b_all[-1]:
                                retrench[rb[-1], j] = True
                        bi = (used | binding) & ~retrench
                    else:
                        bi = (used | binding) & ~(used & relax)
                    if it > 1 and changed and not increased and any(
                            g.shape == bi.shape and np.array_equal(g, bi) for g in seen[:-1]):
                        raise OccbinError(f"period {sp + 1}: regime guesses cycle without converging")
                else:
                    changed = False
                    z = np.vstack([z[1:], self.P @ z[-1]])
                    if z.shape[0] < bi.shape[0]:
                        z = np.vstack([z, self.P @ z[-1]])
            if changed:
                raise OccbinError(f"period {sp + 1}: no convergence in {maxit} iterations")
            if used[-1].any():
                raise OccbinError(f"period {sp + 1}: a constraint binds at the end of the horizon; "
                                  "increase check_ahead_periods")
            history.append([self.map_regime(used[:, j]) for j in range(self.k)])
            iters.append(it)
            y0 = z[0].copy()
            piecewise[sp] = y0
            bi = np.vstack([used[1:], np.zeros((1, self.k), bool)])
        piecewise[S:] = z[1:periods - S + 1]

        linear = np.empty((periods, self.n))
        y = init_orig
        for t in range(periods):
            y = self.P @ y + (self.Q @ shocks[t] if t < S else 0.0)
            linear[t] = y
        return OccbinResult(piecewise=piecewise + self.ys, linear=linear + self.ys, shocks=shocks,
                            regime_history=history, ys=self.ys, iterations=iters)

    def run(self, **overrides) -> OccbinResult:
        """Simulate as the .mod file specifies: its shocks(surprise) blocks and the options given
        to occbin_setup / occbin_solver (simul_periods, simul_check_ahead_periods, simul_maxit,
        simul_curb_retrench). Keyword arguments override them."""
        o = {**self.model.options.get("occbin_setup", {}), **self.model.options.get("occbin_solver", {})}
        kw = dict(periods=o.get("simul.periods", 100),
                  check_ahead_periods=o.get("simul.check_ahead_periods", 200),
                  maxit=o.get("simul.maxit", 30), curb_retrench=bool(o.get("simul.curb_retrench", False)))
        kw.update(overrides)
        return self.simulate(self.surprise_shocks(), **kw)

    def irf(self, impulse, periods=40, baseline_shocks=None, **kw):
        """Piecewise-linear impulse response: simulation with ``impulse`` added to the first
        period's shocks minus the simulation without it. Returns (periods, n_endo).

        Unlike a linear IRF this depends on the size and sign of the impulse and on the baseline.
        """
        base = np.zeros((1, self.n_exo)) if baseline_shocks is None else np.atleast_2d(np.asarray(baseline_shocks, float)).copy()
        hit = base.copy()
        hit[0] += np.asarray(impulse, float)
        a = self.simulate(hit, periods=periods, **kw)
        b = self.simulate(base, periods=periods, **kw)
        return a.piecewise - b.piecewise

    def surprise_shocks(self):
        """Shock sequence from the model's ``shocks(surprise)`` blocks, shape (max period, n_exo)."""
        m = self.model
        blocks = [st for st in m.options.get("shock_paths", []) if st.get("surprise")]
        if not blocks:
            raise ValueError("model has no shocks(surprise) block")
        last = max(v["period2"] for st in blocks for s in st["surprise_shocks"] for v in s["values"])
        out = np.zeros((last, m.n_exo))
        for st in blocks:
            for s in st["surprise_shocks"]:
                j = m._ix[s["var"]]
                for v in s["values"]:
                    out[v["period1"] - 1:v["period2"], j] = m.eval_expr(v["value"])
        return out
