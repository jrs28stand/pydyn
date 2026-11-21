# This file is part of pydyn. It is a Python port of matlab/+occbin/{solve_one_constraint, solve_two_constraints,
# mkdatap_anticipated_dyn, mkdatap_anticipated_2constraints_dyn, map_regime, mkdata}.m from Dynare 7.2 (public domain:
# original authors Luca Guerrieri and Matteo Iacoviello, adapted by the Dynare Team), modified in 2026. The option
# handling (set_default_options.m, setup.m, solver.m), the IRF routine (irf.m) and the constraint-error defaults
# (preprocessor ComputingTasks.cc, occbin_difference) follow Dynare Team files released under GPL-3.0-or-later.
# Released under the GNU General Public License, version 3 or later; see LICENSE. No warranty.
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
    """Failure of the OccBin solver. ``error_flag`` is Dynare's code: 310 (guesses alternate between
    two regimes and ``periodic_solution`` is off), 311 (no convergence within ``maxit``), 312 (a
    constraint still binds at the end of the check-ahead horizon: increase ``check_ahead_periods``),
    313 (infinite loop of regime guesses). ``period`` is the 1-based shock period that failed."""
    error_flag = 0

    def __init__(self, message="", error_flag=None, period=None):
        super().__init__(message)
        if error_flag is not None:
            self.error_flag = error_flag
        self.period = period


class OccbinPeriodicError(OccbinError):
    error_flag = 310


class OccbinMaxitError(OccbinError):
    error_flag = 311


class OccbinHorizonError(OccbinError):
    error_flag = 312


class OccbinLoopError(OccbinError):
    error_flag = 313


@dataclass
class OccbinResult:
    piecewise: np.ndarray            # (periods, n_endo), levels
    linear: np.ndarray               # (periods, n_endo), levels, ignoring the constraints
    shocks: np.ndarray               # (n_shock_periods, n_exo)
    regime_history: list             # per shock period: [(regimes, starts) for each constraint], starts 1-based
    ys: np.ndarray
    iterations: list = field(default_factory=list)
    exo_pos: tuple = ()              # 0-based positions of the columns of ``shocks`` among the exogenous variables
    error_flag: int = 0              # always 0: failures raise OccbinError
    periodic: list = field(default_factory=list)   # per shock period: True if a periodic solution was selected
    diagnostics: list = field(default_factory=list)  # per shock period: dict(max_err, violates, n_viol, selected), per iteration
    ss: list = field(default_factory=list)           # per shock period: (T, R, C) of y_t - ys = T (y_{t-1} - ys) + R u_t + C
                                                     # (Dynare's ss.T/R/C of occbin.solver; declaration order, R on all shocks)


def _split_comparison(expr: str):
    """Split 'lhs < rhs' at the top-level comparison operator; returns (lhs, op, rhs) or None."""
    depth, pos = 0, 0
    toks = tokenize(expr)
    # recover character offsets by re-scanning
    idx = 0
    for _kind, val in toks:
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
        self.ref = first_order(m, self.ys, self.params, loglinear=False)   # piecewise rules are in levels
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

        def fn(src):
            return eval("lambda z, params, ss: " + tr(src), ns)  # noqa: S307 - model expression

        def absdiff(expr):
            """abs(lhs - rhs) of a comparison, Dynare's default constraint error."""
            parts = _split_comparison(expr)
            if parts is None:
                f = fn(expr)
                return lambda z, params, ss: np.asarray(f(z, params, ss), float)
            return fn(f"abs(({parts[0]}) - ({parts[2]}))")

        self._cons = []
        for c in m.occbin_constraints:
            bind, relax = c["bind"], c.get("relax") or None
            eb_src, er_src = c.get("error_bind") or None, c.get("error_relax") or None
            # the defaults of ComputingTasks.cc (OccbinConstraintsStatement::writeOutput)
            ebind = fn(eb_src) if eb_src else absdiff(bind)
            if er_src:
                erelax = fn(er_src)
            elif relax:
                erelax = absdiff(relax)
            else:
                erelax = absdiff(bind)       # equals ebind unless error_bind was given
            self._cons.append((fn(bind), ebind, fn(relax) if relax else None, erelax))

    def check(self, z_levels):
        """Return boolean arrays (binding, relax) and error arrays, each of shape (T, k).

        The errors are Dynare's ``err.binding_constraint_j`` / ``err.relax_constraint_j``: the
        ``error_bind`` / ``error_relax`` expressions of the ``occbin_constraints`` block if given,
        otherwise the absolute difference between the two sides of the bind / relax condition."""
        T = z_levels.shape[0]
        binding, relax = np.zeros((T, self.k), bool), np.zeros((T, self.k), bool)
        eb, er = np.zeros((T, self.k)), np.zeros((T, self.k))
        args = (z_levels, self.params, self.ys)
        for j, (bind, ebind, rel, erel) in enumerate(self._cons):
            binding[:, j] = np.broadcast_to(np.asarray(bind(*args), float) != 0, (T,))
            eb[:, j] = np.broadcast_to(np.asarray(ebind(*args), float), (T,))
            relax[:, j] = ~binding[:, j] if rel is None else np.broadcast_to(np.asarray(rel(*args), float) != 0, (T,))
            er[:, j] = np.broadcast_to(np.asarray(erel(*args), float), (T,))
        return binding, relax, eb, er

    # ---------------------------------------------------------------- path for a given guess
    @staticmethod
    def last_regime_start(col) -> int:
        """1-based start of the last regime of a 0/1 column (``regime_start(end)`` in map_regime)."""
        ch = np.nonzero(col[1:] != col[:-1])[0]
        return int(ch[-1]) + 2 if len(ch) else 1

    def tmax(self, indicator) -> int:
        """Dynare's ``Tmax = max(regime_start(end)) - 1``: last period of the last binding regime."""
        return max(self.last_regime_start(indicator[:, j]) for j in range(self.k)) - 1

    def rules(self, indicator: np.ndarray, Tmax: int | None = None):
        """Time-varying rules y_t = T_t y_{t-1} + R_t u_t + C_t for t = 1..Tmax (list of tuples)."""
        if Tmax is None:
            Tmax = self.tmax(indicator)
        n, out = self.n, [None] * Tmax
        T_next, C_next = self.P, np.zeros(n)
        for i in range(Tmax - 1, -1, -1):
            reg = tuple(bool(b) for b in indicator[i])
            if i == Tmax - 1:
                # Dynare takes the regime of the last binding period from the guess for two
                # constraints (10, 11, else 01) and as "binding" for one constraint
                reg = (True,) if self.k == 1 else (reg if any(reg) else (False, True))
            C, B, A, J, D = self.regime_matrices(reg)
            sol = -np.linalg.solve(B + A @ T_next, np.column_stack([C, J, A @ C_next + D]))
            T_next, R, C_next = sol[:, :n], sol[:, n:n + self.n_exo], sol[:, -1]
            out[i] = (T_next, R, C_next)
        return out

    def path(self, indicator: np.ndarray, init: np.ndarray, shock: np.ndarray, Tmax: int | None = None):
        """Deviations from steady state for len(indicator) periods given a regime guess."""
        rules = self.rules(indicator, Tmax)
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

    def _apply_regime_guess(self, bi, per):
        """Dynare's ``init_regime`` guess for one shock period: fill the columns of ``bi`` from
        [(regime, start), ...] (one pair per constraint, starts 1-based), as in solve_*_constraint.m.
        Like MATLAB's indexed assignment, a start beyond the current rows grows the matrix."""
        bi = bi.copy()
        for j, (regime, start) in enumerate(per):
            bi[:, j] = bool(regime[-1])
            for ir in range(len(regime) - 1):
                a, b = int(start[ir]) - 1, int(start[ir + 1]) - 1
                if b > bi.shape[0]:
                    bi = np.vstack([bi, np.zeros((b - bi.shape[0], self.k), bool)])
                bi[a:b, j] = bool(regime[ir])
        return bi

    # ---------------------------------------------------------------- simulation
    def simulate(self, shocks, periods=100, check_ahead_periods=200, maxit=30, init=None,
                 curb_retrench=False, *, periodic_solution=False, periodic_solution_threshold=1,
                 periodic_solution_strict=True, init_regime=None, init_binding_indicator=None,
                 max_check_ahead_periods=np.inf, reset_check_ahead_periods_in_new_period=False,
                 reset_regime_in_new_period=False, exo_pos=None, algo_truncation=1) -> OccbinResult:
        """Piecewise-linear simulation under a sequence of unanticipated shocks (Dynare's
        ``occbin.solver`` with the ``simul`` options, defaults as in ``set_default_options.m``).

        ``shocks`` has shape (n_shock_periods, n_exo), or (n_shock_periods, len(exo_pos)) when
        ``exo_pos`` (0-based positions or names of the exogenous variables, Dynare's ``exo_pos``)
        is given. The simulation runs for ``periods`` periods. ``init`` is ``endo_init``:
        deviations from steady state in declaration order.

        ``init_regime`` (Dynare's ``init_regime``) is a list with one entry per leading shock period,
        each a list of ``(regime, regime_start)`` pairs, one per constraint, i.e. the format of
        ``OccbinResult.regime_history``. ``init_binding_indicator`` is a (rows, n_constraints)
        0/1 array: the initial guess for the first shock period.

        Failures raise OccbinError whose ``error_flag`` is Dynare's (310-313); see OccbinError.
        """
        k, n, n_exo, m = self.k, self.n, self.n_exo, self.model
        shocks_in = np.atleast_2d(np.asarray(shocks, float))
        if exo_pos is None:
            exo_idx = tuple(range(n_exo))
        else:
            exo_idx = tuple(m._ix[p] if isinstance(p, str) else int(p) for p in np.atleast_1d(np.asarray(exo_pos, dtype=object)))
        if shocks_in.shape[1] != len(exo_idx):
            raise ValueError(f"shocks have {shocks_in.shape[1]} columns but exo_pos selects {len(exo_idx)} shocks")
        S = shocks_in.shape[0]
        shocks = np.zeros((S, n_exo))
        shocks[:, list(exo_idx)] = shocks_in
        n_periods = max(periods, S)
        max_check = np.inf if max_check_ahead_periods is None else max_check_ahead_periods
        if check_ahead_periods > max_check:          # occbin.solver: check_ahead cannot exceed max
            check_ahead_periods = int(max_check)
        n0 = max(check_ahead_periods, n_periods - S)  # nperiods_0
        y0 = np.zeros(n) if init is None or np.size(init) == 0 else np.asarray(init, float).ravel().copy()
        init_orig = y0.copy()

        bi = np.zeros((0, k), bool) if init_binding_indicator is None else \
            np.asarray(init_binding_indicator, bool).reshape(-1, k)
        if bi.shape[0] > n0 + 1:
            raise ValueError("init_binding_indicator has more rows than check_ahead_periods + 1")
        if bi.shape[0] < n0 + 1:
            bi = np.vstack([bi, np.zeros((n0 + 1 - bi.shape[0], k), bool)])

        piecewise = np.zeros((n_periods, n))
        history, iters, periodic_flags, diag = [], [], [], []
        z = used = None
        ss_out, rules_cur = [], []

        def first_rule(rules):
            return rules[0] if rules else (self.P, self.Q, np.zeros(n))

        for sp in range(S):
            changed, it, increased = True, 0, False
            is_periodic = is_loop = False
            guess_it = init_regime is not None and sp < len(init_regime)
            hist = []
            max_err = np.full(maxit, np.nan)
            viol_exp = np.zeros(maxit, bool)
            n_viol = np.zeros(maxit)
            shock_here = sp == 0 or np.any(shocks[sp] != 0)
            chosen_periodic, selected = False, None

            while changed and it < maxit and not is_periodic and not is_loop:
                it += 1
                if bi[-1].any() and n0 < max_check:
                    bi = np.vstack([bi, np.zeros((1, k), bool)])
                    n0 += 1
                    increased = True
                elif n0 >= max_check:
                    bi = bi[:int(max_check) + 1].copy()
                    bi[-1] = False
                if bi.shape[0] < n0 + 1:
                    bi = np.vstack([bi, np.zeros((n0 + 1 - bi.shape[0], k), bool)])
                if it == 1 and guess_it:
                    bi = self._apply_regime_guess(bi, init_regime[sp])
                    n0 = bi.shape[0] - 1
                used = bi.copy()
                hist.append(used)

                if shock_here:
                    if it == 1 and reset_regime_in_new_period:
                        if reset_check_ahead_periods_in_new_period:
                            n0 = max(check_ahead_periods, n_periods - S)
                            bi = np.zeros((n0 + 1, k), bool)
                        else:
                            bi = np.zeros_like(bi)
                        used = bi.copy()
                        hist[-1] = used
                    z, rules_cur = self.path(used, y0, shocks[sp], self.tmax(used))
                    cur_trc = first_rule(rules_cur)
                    binding, relax, eb, er = self.check(z + self.ys)
                    L = bi.shape[0]
                    if np.isfinite(max_check) and max_check < L:
                        E, pad = int(max_check), L - int(max_check)
                    else:
                        E, pad = L, 0
                    new_viol = binding[:E] & ~bi[:E]
                    new_rel = relax[:E] & bi[:E]
                    if new_viol.any() or new_rel.any():
                        max_err[it - 1] = np.max(np.abs(np.concatenate([eb[:E][new_viol], er[:E][new_rel]])))
                        changed = True
                        viol_exp[it - 1] = new_viol.any()
                    else:
                        changed = False
                        max_err[it - 1] = 0.0
                    # enforce the unconstrained regime beyond max_check_ahead_periods
                    bind_new = np.vstack([binding[:E], np.zeros((pad, k), bool)])
                    rel_new = np.vstack([relax[:E], np.ones((pad, k), bool)])
                    if k == 1:   # solve_one_constraint: sum(bi - next guess), can be negative
                        n_viol[it - 1] = np.sum(bi.astype(int) - ((bi | bind_new) & ~(bi & rel_new)).astype(int))
                    else:        # solve_two_constraints: (bi - (bi|new)) & ~(bi&relax), i.e. newly binding periods
                        n_viol[it - 1] = max(int(np.sum(~bi[:, c] & bind_new[:, c])) for c in range(k))
                    if curb_retrench:
                        # slow the update down: release at most the last binding period per constraint
                        retrench = np.zeros_like(bi)
                        for j in range(k):
                            rb = np.nonzero(relax[:E, j] & bi[:E, j])[0]
                            r_all, b_all = np.nonzero(relax[:E, j])[0], np.nonzero(bi[:E, j])[0]
                            if len(rb) and r_all[-1] >= b_all[-1]:
                                retrench[rb[-1], j] = True
                        bi = (bi | bind_new) & ~retrench
                    else:
                        bi = (bi | bind_new) & ~(bi & rel_new)

                    if it > 1 and changed and not increased:
                        is_loop = any(h.shape == bi.shape and np.array_equal(h, bi) for h in hist[:it - 1])
                        prev = hist[it - 2]
                        same = prev.shape == bi.shape and np.array_equal(prev, bi)
                        if k == 1:
                            small = int(np.sum(used.astype(int) - bi.astype(int))) <= periodic_solution_threshold
                        else:
                            small = all(int(np.sum(used[:, c].astype(int) - bi[:, c].astype(int))) <= periodic_solution_threshold
                                        for c in range(k))
                        is_periodic = bool(same and small)
                        is_per_all = np.zeros(it - 1, bool)
                        is_per_all[it - 2] = is_periodic
                        strict = is_periodic
                        if is_loop and not is_periodic and not periodic_solution_strict:
                            cand = n_viol[:it][~viol_exp[:it]]
                            ok = np.any(cand <= periodic_solution_threshold) if k == 1 else \
                                np.any(cand < periodic_solution_threshold)
                            if ok:
                                is_periodic = True
                                is_per_all = np.zeros(it - 1, bool)
                                is_per_all[0] = True    # consider all guesses
                        if is_periodic and periodic_solution:
                            inx = list(range(int(np.argmax(is_per_all)), it))
                            inx1 = [i for i in inx if not viol_exp[i]]
                            if inx1:
                                inx = inx1
                            vals = max_err[inx] if (strict or not inx1) else n_viol[inx]
                            best = inx[int(np.nanargmin(vals))]
                            bi = hist[best].copy()
                            chosen_periodic, selected = True, best + 1
                            if best < it - 1:
                                used = bi.copy()
                                z, rules_cur = self.path(used, y0, shocks[sp], self.tmax(used))
                                cur_trc = first_rule(rules_cur)
                else:
                    changed = False
                    rules_cur = rules_cur[1:]          # Dynare: SS = SS(2:end), rule of the next period
                    cur_trc = first_rule(rules_cur)
                    z = np.vstack([z[1:], self.P @ z[-1]])
                    if z.shape[0] < bi.shape[0]:
                        z = np.vstack([z, self.P @ z[-1]])

            if changed:
                if maxit > algo_truncation:
                    if is_periodic:
                        if not periodic_solution:
                            raise OccbinPeriodicError(
                                f"period {sp + 1}: regime guesses alternate between two regimes (error 310); "
                                "use periodic_solution=True to pick the guess with the smallest error", period=sp + 1)
                    elif is_loop:
                        raise OccbinLoopError(f"period {sp + 1}: regime guesses cycle without converging (error 313)",
                                              period=sp + 1)
                    else:
                        raise OccbinMaxitError(f"period {sp + 1}: no convergence in {maxit} iterations (error 311)",
                                               period=sp + 1)
                else:
                    bi = hist[-1].copy()
            if used[-1].any():
                raise OccbinHorizonError(f"period {sp + 1}: a constraint binds at the end of the horizon; "
                                         "increase check_ahead_periods (error 312)", period=sp + 1)
            history.append([self.map_regime(used[:, j]) for j in range(k)])
            iters.append(it)
            ss_out.append(cur_trc)
            periodic_flags.append(bool(changed and chosen_periodic))
            diag.append(dict(max_err=max_err[:it].copy(), violates=viol_exp[:it].copy(), n_viol=n_viol[:it].copy(),
                             selected=selected))
            y0 = z[0].copy()
            piecewise[sp] = y0
            bi = np.vstack([bi[1:], np.zeros((1, k), bool)])
        piecewise[S:] = z[1:n_periods - S + 1]

        linear = np.empty((n_periods, n))
        y = init_orig
        for t in range(n_periods):
            y = self.P @ y + (self.Q @ shocks[t] if t < S else 0.0)
            linear[t] = y
        return OccbinResult(piecewise=piecewise + self.ys, linear=linear + self.ys, shocks=shocks_in,
                            regime_history=history, ys=self.ys, iterations=iters, exo_pos=exo_idx,
                            periodic=periodic_flags, diagnostics=diag, ss=ss_out)

    # option names of occbin_setup / occbin_solver (JSON keys) -> simulate() keywords
    _MOD_OPTIONS = {
        "simul.periods": ("periods", int),
        "simul.maxit": ("maxit", int),
        "simul.curb_retrench": ("curb_retrench", bool),
        "simul.check_ahead_periods": ("check_ahead_periods", int),
        "simul.max_check_ahead_periods": ("max_check_ahead_periods", int),
        "simul.periodic_solution": ("periodic_solution", bool),
        "simul.periodic_solution_threshold": ("periodic_solution_threshold", int),
        "simul.periodic_solution_strict": ("periodic_solution_strict", bool),
        "simul.reset_check_ahead_periods_in_new_period": ("reset_check_ahead_periods_in_new_period", bool),
        "simul.reset_regime_in_new_period": ("reset_regime_in_new_period", bool),
    }

    def mod_options(self) -> dict:
        """simulate() keywords set by the .mod file's occbin_setup / occbin_solver statements
        (``simul_debug`` is accepted and has no effect). Unset options keep Dynare's defaults."""
        o = {**self.model.options.get("occbin_setup", {}), **self.model.options.get("occbin_solver", {})}
        out = {}
        for key, (kw, typ) in self._MOD_OPTIONS.items():
            if key in o:
                v = o[key]
                out[kw] = (str(v).lower() == "true") if typ is bool and isinstance(v, str) else typ(v)
        return out

    def run(self, **overrides) -> OccbinResult:
        """Simulate as the .mod file specifies: its shocks(surprise) blocks and the options given
        to occbin_setup / occbin_solver (simul_periods, simul_maxit, simul_curb_retrench,
        simul_check_ahead_periods, simul_max_check_ahead_periods, simul_periodic_solution,
        simul_periodic_solution_threshold, simul_periodic_solution_strict,
        simul_reset_check_ahead_periods, simul_reset_regime_in_new_period). Keyword arguments
        override them."""
        kw = self.mod_options()
        kw.update(overrides)
        return self.simulate(self.surprise_shocks(), **kw)

    def irf(self, impulse, periods=40, baseline_shocks=None, **kw):
        """Piecewise-linear impulse response: simulation with ``impulse`` added to the first
        period's shocks minus the simulation without it. Returns (periods, n_endo).

        Unlike a linear IRF this depends on the size and sign of the impulse and on the baseline.
        See ``irf_occbin`` for Dynare's own definition (shock size, signs, horizon).
        """
        base = np.zeros((1, self.n_exo)) if baseline_shocks is None else np.atleast_2d(np.asarray(baseline_shocks, float)).copy()
        hit = base.copy()
        hit[0] += np.asarray(impulse, float)
        a = self.simulate(hit, periods=periods, **kw)
        b = self.simulate(base, periods=periods, **kw)
        return a.piecewise - b.piecewise

    def irf_occbin(self, periods=40, shocksize=None, exo_names=None, shocksigns=("pos", "neg"),
                   check_ahead_periods=30, maxit=30, init_regime=None, **kw):
        """OccBin IRFs as Dynare's ``occbin.irf`` computes them (``t0 = 0``, no smoother).

        For each shock in ``exo_names`` (default all) and each sign: one simulation with
        ``+/- shocksize`` in period 1 and zero shocks for ``periods + 1`` shock periods (Dynare
        passes ``options_.irf + 1`` rows of shocks), minus the simulation without the shock.
        ``shocksize`` defaults to the shock's standard deviation (0.01 if that is below 1e-9) and
        may be a scalar or one value per shock. ``check_ahead_periods`` (30) and ``maxit`` (30) are
        the ``irf`` options, which replace the ``simul`` ones; further keywords (``curb_retrench``,
        ``periodic_solution``...) go to ``simulate``. Returns {(exo_name, sign): {"piecewise": array,
        "linear": array}} with arrays of shape (periods + 1, n_endo) in deviations from steady state.
        """
        m = self.model
        names = list(m.exo_names if exo_names is None else exo_names)
        idx = [m._ix[nm] for nm in names]
        if shocksize is None:
            size = np.sqrt(np.diag(m.Sigma_e)[idx])
            size[size < 1e-9] = 0.01
        else:
            size = np.atleast_1d(np.asarray(shocksize, float))
            if size.size == 1:
                size = np.repeat(size, len(idx))
        sim = dict(periods=periods, check_ahead_periods=check_ahead_periods, maxit=maxit,
                   init_regime=init_regime, **kw)
        base_shocks = np.zeros((periods + 1, self.n_exo))
        base = self.simulate(base_shocks, **sim)
        out = {}
        for sign in shocksigns:
            for nm, j, sz in zip(names, idx, size):
                sh = base_shocks.copy()
                sh[0, j] += sz if "pos" in sign else -sz
                r = self.simulate(sh, **sim)
                out[(nm, sign)] = {"piecewise": r.piecewise - base.piecewise, "linear": r.linear - base.linear}
        return out

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
