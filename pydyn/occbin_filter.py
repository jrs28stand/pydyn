# This file is part of pydyn. It is a Python port, modified in 2026, of these files of Dynare 7.2:
#   matlab/+occbin/IVF_core.m, matlab/+occbin/match_function.m (public domain: Pablo Cuba-Borda, Luca
#     Guerrieri, Matteo Iacoviello and Molin Zhong, adapted for Dynare by the Dynare Team);
#   matlab/+occbin/map_regime.m (public domain: Luca Guerrieri and Matteo Iacoviello, adapted by the
#     Dynare Team);
#   matlab/+occbin/{IVF_posterior, kalman_update_engine, kalman_update_algo_1, kalman_update_algo_3,
#     DSGE_smoother, check_regimes, backward_map_regime, shock_decomposition,
#     squeeze_shock_decomposition, write_regimes_to_xls, set_default_options, setup, set_option}.m,
#   matlab/kalman/likelihood/missing_observations_kalman_filter.m (OccBin branch),
#   matlab/kalman/missing_DiffuseKalmanSmootherH3_Z.m and matlab/kalman/DsgeSmoother.m (OccBin
#     branches), the OccBin parts of matlab/estimation/dsge_likelihood.m
#     (Copyright (C) Dynare Team, GPL-3.0-or-later);
#   matlab/optimization/csolve.m (Copyright (C) 1993-2007 Christopher Sims, 2007-2023 Dynare Team,
#     GPL-3.0-or-later).
# Released under the GNU General Public License, version 3 or later; see LICENSE. No warranty.
"""Likelihood of models with occasionally binding constraints (OccBin), after Dynare 7.2.

Two filters, both built on the piecewise-linear solution of :class:`pydyn.occbin.Occbin`:

* the **inversion filter** (Cuba-Borda, Guerrieri, Iacoviello and Zhong 2019; Dynare's
  ``IVF_core``/``IVF_posterior``): as many shocks as observed variables and no measurement error;
  in each period the shocks that reproduce the observations are found through the piecewise-linear
  solution (Newton iterations of ``csolve``), and the likelihood is the shock density times the
  Jacobian of the shock-to-observable map;
* the **piecewise Kalman filter** (Giovannini, Pfeiffer and Ratto 2021; Dynare's
  ``missing_observations_kalman_filter`` with ``kalman_update_engine``/``kalman_update_algo_1``):
  the Kalman update of each period is iterated with the regime-dependent transition implied by the
  smoothed shocks until the regime is consistent.

Also: the OccBin smoother (:func:`occbin_smoother`, Dynare's ``occbin.DSGE_smoother``), the OccBin
shock decomposition (:func:`occbin_shock_decomposition`), the regime table of
``occbin_write_regimes`` (:func:`regime_table`) and plotting data (:func:`plot_data`).
:class:`pydyn.occbin_estimation.OccbinEstimation` wraps them for estimation.

Regimes are written as Dynare's ``regime_history`` entries: one ``(regime, regimestart)`` pair of
tuples per constraint, e.g. ``(((0, 1, 0), (1, 3, 5)),)`` for one constraint binding in periods 3 and 4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import options as _opt
from .kalman import _rcond
from .occbin import Occbin, OccbinError

LOG2PI = np.log(2 * np.pi)

# ====================================================================== options
#: set_default_options.m, flags 'likelihood', 'filter', 'smoother', 'simul' (the fields used here)
DEFAULTS = {
    # likelihood
    "likelihood.brute_force_regime_guess": True,
    "likelihood.brute_force_extra_regime_guess": True,
    "likelihood.curb_retrench": False,
    "likelihood.first_period_binding_regime_allowed": 1,
    "likelihood.first_period_occbin_update": 1,
    "likelihood.IF_likelihood": False,
    "likelihood.inversion_filter": False,
    "likelihood.IVF_shock_observable_mapping": None,
    "likelihood.loss_function_regime_guess": False,
    "likelihood.maxit": 30,
    "likelihood.max_number_of_iterations": 10,
    "likelihood.max_check_ahead_periods": np.inf,
    "likelihood.number_of_initial_periods_with_extra_regime_guess": 0,
    "likelihood.periods": 3,
    "likelihood.check_ahead_periods": 200,
    "likelihood.periodic_solution": False,
    "likelihood.restrict_state_space": True,
    "likelihood.use_updated_regime": True,
    # filter
    "filter.periodic_solution": True,
    "filter.use_relaxation": False,
    "filter.use_relaxation_tol_period": 1,
    "filter.particle.status": False,
    "filter.init_periods_using_particles": False,
    "filter.state_covariance": False,
    # smoother
    "smoother.curb_retrench": False,
    "smoother.debug": False,
    "smoother.fast": False,
    "smoother.first_period_occbin_update": 1,
    "smoother.inversion_filter": False,
    "smoother.linear_smoother": True,
    "smoother.maxit": 30,
    "smoother.max_check_ahead_periods": np.inf,
    "smoother.max_number_of_iterations": 10,
    "smoother.periods": 3,
    "smoother.check_ahead_periods": 200,
    "smoother.periodic_solution": False,
    "smoother.plot": True,
    # simul (fields the filters inherit from options_.occbin.simul)
    "simul.periodic_solution_threshold": 1,
    "simul.periodic_solution_strict": True,
    "simul.reset_check_ahead_periods_in_new_period": False,
    "simul.reset_regime_in_new_period": False,
    # occbin.solver (dynare_solve for the inversion filter)
    "solver.maxit": 10,
    "solver.solve_tolf": 1e-5,
    "solver.solve_tolx": 1e-10,
}

#: the occbin_setup options that occbin.setup.m copies into options_.occbin (Dynare 7.2): all
#: other occbin_setup options are parsed by the preprocessor and then dropped by Dynare
SETUP_COPIED = {
    "simul.periods", "simul.curb_retrench", "simul.maxit", "simul.check_ahead_periods", "simul.debug",
    "simul.periodic_solution",
    "smoother.periods", "smoother.curb_retrench", "smoother.maxit", "smoother.check_ahead_periods",
    "smoother.debug", "smoother.periodic_solution", "smoother.inversion_filter",
    "likelihood.periods", "likelihood.curb_retrench", "likelihood.maxit", "likelihood.check_ahead_periods",
    "likelihood.periodic_solution", "likelihood.max_number_of_iterations", "filter.use_relaxation",
    "likelihood.inversion_filter",
}

#: names of the manual (occbin_setup) -> keys used here
MANUAL_NAMES = {
    "likelihood_inversion_filter": "likelihood.inversion_filter",
    "likelihood_piecewise_kalman_filter": None,          # = likelihood_inversion_filter false
    "likelihood_max_kalman_iterations": "likelihood.max_number_of_iterations",
    "smoother_inversion_filter": "smoother.inversion_filter",
    "smoother_piecewise_kalman_filter": None,            # = smoother_inversion_filter false
    "filter_use_relaxation": "filter.use_relaxation",
}
for _k in ("periods", "maxit", "check_ahead_periods", "max_check_ahead_periods", "curb_retrench",
           "periodic_solution"):
    MANUAL_NAMES[f"likelihood_{_k}"] = f"likelihood.{_k}"
    MANUAL_NAMES[f"smoother_{_k}"] = f"smoother.{_k}"
for _k in ("first_period_occbin_update", "max_number_of_iterations", "plot", "debug"):
    MANUAL_NAMES[f"smoother_{_k}"] = f"smoother.{_k}"


def _key(name: str) -> str:
    if "." in name:
        return name
    if name in MANUAL_NAMES and MANUAL_NAMES[name]:
        return MANUAL_NAMES[name]
    for pre in ("likelihood_", "smoother_", "filter_", "simul_", "solver_"):
        if name.startswith(pre):
            return pre[:-1] + "." + name[len(pre):]
    raise KeyError(f"unknown OccBin option {name!r}")


def occbin_options(model, overrides=None, what="OccBin filtering") -> dict:
    """OccBin filter/smoother/likelihood options as Dynare 7.2 has them after the .mod file's
    ``occbin_setup`` statements: ``set_default_options.m`` defaults, then, for each ``occbin_setup``
    in order, the options that ``occbin.setup`` copies (cumulatively; a later statement does not
    reset earlier ones). The other ``occbin_setup`` options have no effect in Dynare 7.2 and are
    reported once with :func:`pydyn.options.warn_once`. ``overrides`` (dict, keys either dotted,
    ``'likelihood.maxit'``, or the manual's names, ``'likelihood_maxit'``) are applied last and
    are always honoured."""
    o = dict(DEFAULTS)
    dropped = set()
    for st in getattr(model, "statements", []):
        if st.get("statementName") != "occbin_setup":
            continue
        for k, v in st.get("options", {}).items():
            if k in SETUP_COPIED:
                o[k] = v
            elif not k.startswith("simul."):
                dropped.add(k)
    # occbin_solver writes options_.occbin.simul.* directly
    for st in getattr(model, "statements", []):
        if st.get("statementName") == "occbin_solver":
            for k, v in st.get("options", {}).items():
                if k.startswith("simul."):
                    o[k] = v
    if dropped:
        _opt.warn_once(model, f"{what}: these occbin_setup options have no effect (Dynare 7.2's occbin.setup "
                              f"does not copy them into options_.occbin, so they are ignored here too; pass them "
                              f"as overrides to use them): {', '.join(sorted(dropped))}")
    for k, v in (overrides or {}).items():
        if k in ("likelihood_piecewise_kalman_filter", "smoother_piecewise_kalman_filter"):
            o[k.split("_")[0] + ".inversion_filter"] = not v
            continue
        o[_key(k)] = v
    for k in ("likelihood.max_check_ahead_periods", "smoother.max_check_ahead_periods"):
        if o[k] is None:
            o[k] = np.inf
    return o


def _sim_kwargs(o: dict, prefix: str) -> dict:
    """occbin.solver options for the likelihood or the smoother (``set_occbin_options`` of
    dsge_likelihood.m, DSGE_smoother.m): simul options with the ``prefix`` ones on top."""
    return dict(periods=int(o[f"{prefix}.periods"]), maxit=int(o[f"{prefix}.maxit"]),
                check_ahead_periods=int(o[f"{prefix}.check_ahead_periods"]),
                max_check_ahead_periods=o[f"{prefix}.max_check_ahead_periods"],
                curb_retrench=bool(o[f"{prefix}.curb_retrench"]),
                periodic_solution=bool(o[f"{prefix}.periodic_solution"]),
                periodic_solution_threshold=int(o["simul.periodic_solution_threshold"]),
                periodic_solution_strict=bool(o["simul.periodic_solution_strict"]),
                reset_check_ahead_periods_in_new_period=bool(o["simul.reset_check_ahead_periods_in_new_period"]),
                reset_regime_in_new_period=bool(o["simul.reset_regime_in_new_period"]))


# ====================================================================== regimes
def base_regime(k: int):
    return tuple(((0,), (1,)) for _ in range(k))


def as_regime(entry):
    """``OccbinResult.regime_history`` entry ([(regimes, starts), ...]) -> hashable regime."""
    return tuple((tuple(int(x) for x in r), tuple(int(x) for x in s)) for r, s in entry)


def map_regime(ind):
    """map_regime.m: 0/1 vector -> (regime, regimestart)."""
    ind = np.atleast_1d(np.asarray(ind, bool))
    if ind.size == 0:
        return (0,), (1,)
    reg, start = [int(ind[0])], [1]
    for i in range(1, ind.size):
        if int(ind[i]) != reg[-1]:
            reg.append(int(ind[i]))
            start.append(i + 1)
    return tuple(reg), tuple(start)


def backward_map_regime(regime, start):
    """backward_map_regime.m: binding indicator of length ``start[-1]`` (the last regime's own
    period is left at 0, as in Dynare)."""
    n = int(start[-1])
    ind = np.zeros(n, bool)
    for i in range(len(regime) - 1):
        ind[int(start[i]) - 1:int(start[i + 1]) - 1] = bool(regime[i])
    return ind


def _shift_regime(reg):
    """Regime one period later (kalman_update_algo_1.m: regimes0(2:3) from regimes0(1))."""
    out = []
    for r, s in reg:
        ind = backward_map_regime(r, s)[1:]
        out.append(map_regime(ind))
    return tuple(out)


def _max_start(regimes) -> int:
    return max(max(s) for reg in regimes for _, s in reg)


def regime_indicator(regime_history, periods=None):
    """Binding indicator per period (first period of each regime entry), shape (len, k): whether each
    constraint binds in the period the entry refers to."""
    out = np.array([[r[0] for r, _ in reg] for reg in regime_history], int)
    return out if periods is None else out[:periods]


# ====================================================================== solver
class _Solver:
    """Dynare's occbin.solver for the filters: an :class:`Occbin` at one parameter vector."""

    def __init__(self, ob: Occbin):
        self.ob = ob
        self.k = ob.k
        self.base = base_regime(ob.k)

    def solve(self, shocks, init, sim: dict, init_regime=None, exo_pos=None, **kw):
        """Returns (error_flag, result); result.regime_history as hashable regimes."""
        args = dict(sim)
        args.update(kw)
        try:
            res = self.ob.simulate(shocks, init=init, init_regime=None if init_regime is None else
                                   [list(r) for r in init_regime], exo_pos=exo_pos, **args)
        except OccbinError as exc:
            return int(exc.error_flag), None
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return 311, None
        res.regimes = [as_regime(e) for e in res.regime_history]
        return 0, res

    def regime_rule(self, reg, sim: dict):
        """(T, R, C) of the first period of regime ``reg`` (check_regimes.m: occbin.solver with
        ``init_regime`` and ``maxit = 1``); the linear rule for the base regime."""
        if reg == self.base:
            ob = self.ob
            return ob.P, ob.Q, np.zeros(ob.n)
        nper = _max_start([reg])
        kw = dict(sim)
        kw.update(maxit=1, periods=nper)
        flag, res = self.solve(np.zeros((1, self.ob.n_exo)), None, kw, init_regime=[reg])
        if flag:
            # Dynare keeps the matrices computed before the error (312: binding at the horizon)
            ind = np.zeros((nper + 1, self.k), bool)
            for j, (r, s) in enumerate(reg):
                b = backward_map_regime(r, s)
                ind[:len(b), j] = b
            rules = self.ob.rules(ind, self.ob.tmax(ind))
            return rules[0] if rules else (self.ob.P, self.ob.Q, np.zeros(self.ob.n))
        return res.ss[0]


# ====================================================================== csolve
def csolve(fun, x, gradfun, crit, itmax, rng=None):
    """csolve.m (Sims): Newton steps with backtracking on sum(|f|). ``gradfun(x)`` returns (f, grad).
    Returns (x, rc), rc = 0 on success. The random search of csolve (when no progress is made) uses
    numpy's generator, so it cannot reproduce MATLAB's draws."""
    delta, alpha = 1e-6, 1e-3
    x = np.asarray(x, float).copy()
    nv = x.size
    tvec = delta * np.eye(nv)
    f0 = fun(x)
    af0 = np.sum(np.abs(f0))
    af00 = af0
    itct = 0
    rc = 0
    rng = rng or np.random.default_rng(0)
    while True:
        if itct > 3 and af00 - af0 < crit * max(1.0, af0) and itct % 2 == 1:
            randomize = True
        else:
            f0, grad = gradfun(x)
            grad = np.asarray(grad, float)
            randomize = False
            with np.errstate(all="ignore"):
                try:
                    rc_ = 1.0 / np.linalg.cond(grad, 1) if np.all(np.isfinite(grad)) else 0.0
                except np.linalg.LinAlgError:
                    rc_ = 0.0
            if not np.isfinite(rc_) or rc_ < 1e-12:
                grad = grad + tvec
            try:
                dx0 = -np.linalg.solve(grad, f0) if np.all(np.isfinite(grad)) else np.full(nv, np.nan)
            except np.linalg.LinAlgError:
                dx0 = -np.linalg.lstsq(grad, f0, rcond=None)[0]
        if randomize:
            dx0 = np.linalg.norm(x) / rng.standard_normal(nv)
        lam, lammin = 1.0, 1.0
        fmin, xmin, afmin = f0, x, af0
        dxsize = np.linalg.norm(dx0)
        factor, shrink, subdone = 0.6, True, False
        while not subdone:
            dx = lam * dx0
            f = fun(x + dx)
            af = np.sum(np.abs(f))
            if af < afmin:
                afmin, fmin, lammin, xmin = af, f, lam, x + dx
            if (lam > 0 and af0 - af < alpha * lam * af0) or (lam < 0 and af0 - af < 0):
                if not shrink:
                    factor = factor ** 0.6
                    shrink = True
                if abs(lam * (1 - factor)) * dxsize > 0.1 * delta:
                    lam = factor * lam
                elif lam > 0 and factor == 0.6:
                    lam = -0.3
                else:
                    subdone = True
                    rc = (2 if factor == 0.6 else 1) if lam > 0 else 3
            elif lam > 0 and af - af0 > (1 - alpha) * lam * af0:
                if shrink:
                    factor = factor ** 0.6
                    shrink = False
                lam = lam / factor
            else:
                subdone, rc = True, 0
            if not np.isfinite(lam) or not np.isfinite(dxsize):
                subdone, rc = True, 3
        itct += 1
        x, f0 = xmin, fmin
        af00, af0 = af0, afmin
        if itct >= itmax:
            return x, 4
        if af0 < crit:
            return x, 0


# ====================================================================== inversion filter
@dataclass
class InversionFilterResult:
    filtered_errs: np.ndarray        # (T, n_obs) shocks, columns as err_index
    resids: np.ndarray               # (T, n_obs) observation residuals at the solution
    Emat: np.ndarray                 # (T, n_endo, n_obs) response of the variables to the shocks
    stateval: np.ndarray             # (T, n_endo) deviations from steady state, declaration order
    regime_history: list             # per period: regime
    error_code: int = 0
    error_period: int = 0


def inversion_filter(ob: Occbin, data, obs_idx, err_index, options=None) -> InversionFilterResult:
    """IVF_core.m: shocks that reproduce ``data`` (T x n_obs, levels, NaN = missing) period by
    period through the piecewise-linear solution. ``obs_idx``: declaration indices of the observed
    variables; ``err_index``: exogenous indices of the shocks mapped to them (Dynare's
    ``IVF_shock_observable_mapping``, default the shocks with non-zero variance). The solution in
    each period is found with ``csolve`` (``solver.maxit`` iterations, tolerance ``solver.solve_tolf``
    on sum(|residual|)) from zero shocks, as Dynare's ``dynare_solve`` with ``solve_algo = 3``."""
    o = dict(DEFAULTS) if options is None else options
    data = np.atleast_2d(np.asarray(data, float))
    T, nobs = data.shape
    obs_idx = np.asarray(obs_idx, int)
    err_index = np.asarray(err_index, int)
    n = ob.n
    sim = _sim_kwargs(o, "likelihood")
    solver = _Solver(ob)
    tolf, maxit_solve = float(o["solver.solve_tolf"]), int(o["solver.maxit"])
    resids = np.zeros((T, nobs))
    stateval = np.zeros((T, n))
    Emat = np.zeros((T, n, nobs))
    errs = np.zeros((T, nobs))
    regimes = []
    init = np.zeros(n)

    def make_match(cur_obs, oi, ep, y0):
        cache = {}

        def match(err):
            key = tuple(np.asarray(err, float).ravel())
            if key in cache:
                return cache[key]
            flag, res = solver.solve(np.asarray(err, float).reshape(1, -1), y0, sim, exo_pos=ep)
            if flag:
                out = (np.full(len(oi), 100.0), np.full((len(oi), len(ep)), np.nan), None, None, None)
            else:
                Tr, Rr, Cr = res.ss[0]
                out = (res.piecewise[0, oi] - cur_obs, Rr[np.ix_(oi, ep)], res.piecewise[0] - ob.ys, Rr[:, ep], res)
            cache[key] = out
            return out
        return match

    for t in range(T):
        inan = ~np.isnan(data[t])
        cur = data[t, inan]
        oi, ep = obs_idx[inan], err_index[inan]
        match = make_match(cur, oi, ep, init.copy())
        x0 = np.zeros(int(inan.sum()))
        f0, g0 = match(x0)[:2]
        if np.all(np.isfinite(f0)) and np.max(np.abs(f0), initial=0.0) < tolf:
            x, rc = x0, 0                                  # dynare_solve: initial guess solves the problem
        else:
            x, rc = csolve(lambda e: match(e)[0], x0, lambda e: match(e)[:2], tolf, maxit_solve)
        if rc != 0:
            return InversionFilterResult(errs, resids, Emat, stateval, regimes, 304, t + 1)
        errs[t, inan] = x
        r_, _, state, E, res = match(x)
        if res is None:
            return InversionFilterResult(errs, resids, Emat, stateval, regimes, 304, t + 1)
        resids[t, inan] = r_
        stateval[t] = state
        Emat[t][:, inan] = E
        init = state
        regimes.append(res.regimes[0])
        if np.max(np.abs(x), initial=0.0) > 1e8:
            return InversionFilterResult(errs, resids, Emat, stateval, regimes, 306, t + 1)
        if np.max(np.abs(resids[t]), initial=0.0) > 1e-3:
            return InversionFilterResult(errs, resids, Emat, stateval, regimes, 303, t + 1)
    return InversionFilterResult(errs, resids, Emat, stateval, regimes)


def inversion_filter_loglik(ivf: InversionFilterResult, data, obs_idx, Sigma_e, err_index, presample=0):
    """IVF_posterior.m: (minus log likelihood, per-period contributions 0.5*likei) of an inversion
    filter run, or raises ValueError with Dynare's code (301 inf, 302 NaN, 303 residuals)."""
    data = np.atleast_2d(np.asarray(data, float))
    err_index = np.asarray(err_index, int)
    obs_idx = np.asarray(obs_idx, int)
    COV = np.asarray(Sigma_e, float)[np.ix_(err_index, err_index)]
    e = ivf.filtered_errs
    T = e.shape[0]
    likei = np.full(T, np.nan)
    with np.errstate(all="ignore"):
        if not np.isnan(data).any():
            ldc = np.log(np.linalg.det(COV))
            iC = np.linalg.inv(COV)
            for t in range(T):
                G = ivf.Emat[t][obs_idx, :]
                likei[t] = ldc + 2 * np.log(abs(np.linalg.det(G))) + e[t] @ iC @ e[t]
            likei += len(obs_idx) * LOG2PI
        else:
            for t in range(T):
                inan = ~np.isnan(data[t])
                C = COV[np.ix_(inan, inan)]
                G = ivf.Emat[t][np.ix_(obs_idx[inan], np.flatnonzero(inan))]
                likei[t] = (np.log(np.linalg.det(C)) + 2 * np.log(abs(np.linalg.det(G)))
                            + e[t, inan] @ np.linalg.solve(C, e[t, inan]) + inan.sum() * LOG2PI)
    like = 0.5 * np.sum(likei[presample:])
    if np.isinf(like):
        raise ValueError(301)
    if np.isnan(like):
        raise ValueError(302)
    if np.max(np.abs(ivf.resids), initial=0.0) > 1e-3:
        raise ValueError(303)
    return float(like), 0.5 * likei


# ====================================================================== piecewise Kalman filter
class _PKFContext:
    """What the piecewise Kalman filter needs besides the filter quantities: the solver, the state
    space (``r``: declaration indices of the filter states, in Dynare's decision-rule order), the
    selection matrix ``Z``, the covariance matrices and the options. ``prefix`` selects the
    ``likelihood.*`` or ``smoother.*`` occbin.solver options."""

    def __init__(self, ob: Occbin, r, Z, H, Q, options, prefix="likelihood"):
        self.ob, self.solver = ob, _Solver(ob)
        self.r = np.asarray(r, int)
        self.Z, self.H, self.Q = Z, H, Q
        self.o = options
        self.k = ob.k
        self.base = base_regime(ob.k)
        self.sim = _sim_kwargs(options, prefix)
        self.T0 = ob.P[np.ix_(self.r, self.r)]
        self.R0 = ob.Q[self.r, :]
        self.mm = len(self.r)
        self.nexo = ob.n_exo
        self.n = ob.n
        self.kalman_tol = float(options.get("kalman_tol", 1e-10))

    def rule(self, trc):
        T, R, C = trc
        r = self.r
        return T[np.ix_(r, r)], R[r, :], C[r]


def _rank_deficient(F):
    if F.size == 0:
        return False
    return np.linalg.matrix_rank(F) < F.shape[0]


class _Step:
    """The two-period problem of one PKF update (period t-1 already filtered, period t new):
    kalman_update_algo_1 (multivariate) or kalman_update_algo_3 (univariate)."""

    def __init__(self, ctx, a_prev, a1_1, P_prev, P1_1, di1, di2, y2, TT2, RR2, CC2, v1,
                 Fi1=None, Ki1=None, univariate=False):
        self.ctx = ctx
        self.a_prev, self.a1_1, self.P_prev, self.P1_1 = a_prev, a1_1, P_prev, P1_1
        self.di1, self.di2, self.y2, self.v1 = np.asarray(di1, int), np.asarray(di2, int), y2, v1
        self.TT2, self.RR2, self.CC2 = TT2, RR2, CC2
        self.Fi1, self.Ki1 = Fi1, Ki1
        self.univariate = univariate
        self.pre_error = 0
        if not univariate:
            Z, H, mm = ctx.Z, ctx.H, ctx.mm
            ZZ1 = Z[self.di1]
            F = ZZ1 @ P1_1 @ ZZ1.T + H[np.ix_(self.di1, self.di1)]
            if np.isnan(F).any():
                self.pre_error = 325
            elif _rank_deficient(F):
                self.pre_error = 326
            elif len(self.di1):
                sig = np.sqrt(np.diag(F))
                ss = np.outer(sig, sig)
                with np.errstate(all="ignore"):
                    self.iF1 = np.linalg.inv(F / ss) / ss
                self.L1 = np.eye(mm) - P1_1 @ ZZ1.T @ self.iF1 @ ZZ1
            else:
                self.iF1, self.L1 = np.zeros((0, 0)), np.eye(mm)

    def update(self, T2, R2, C2):
        return self._update_uni(T2, R2, C2) if self.univariate else self._update_multi(T2, R2, C2)

    def _update_multi(self, T2, R2, C2):
        """occbin_kalman_update0 of kalman_update_algo_1.m."""
        ctx = self.ctx
        Z, H, Q, mm = ctx.Z, ctx.H, ctx.Q, ctx.mm
        o = ctx.o
        rescale = bool(o.get("rescale_prediction_error_covariance", False))
        IFl = bool(o["likelihood.IF_likelihood"])
        di2 = self.di2
        out = dict(error_flag=0, lik=np.inf)
        QQ = R2 @ Q @ R2.T
        a1_2 = T2 @ self.a_prev + C2
        a_2 = a1_2.copy()
        P1_2 = T2 @ self.P_prev @ T2.T + QQ
        P_2 = P1_2.copy()
        v2 = np.zeros(Z.shape[0])
        if len(di2) == 0:
            lik, iF2 = 0.0, None
        else:
            ZZ = Z[di2]
            v2[di2] = self.y2[di2] - ZZ @ a_2
            F = ZZ @ P_2 @ ZZ.T + H[np.ix_(di2, di2)]
            if np.isnan(F).any():
                out["error_flag"] = 325
                return out
            if _rank_deficient(F):
                out["error_flag"] = 326
                return out
            with np.errstate(all="ignore"):
                if rescale:
                    sig = np.sqrt(np.diag(F))
                    ss = np.outer(sig, sig)
                    log_dF = np.log(np.linalg.det(F / ss)) + 2 * np.sum(np.log(sig))
                    iF2 = np.linalg.inv(F / ss) / ss
                else:
                    log_dF = np.log(np.linalg.det(F))
                    iF2 = np.linalg.inv(F)
            lik = np.inf
            if not IFl:
                lik = log_dF + v2[di2] @ iF2 @ v2[di2] + len(di2) * LOG2PI
            PZI = P_2 @ ZZ.T @ iF2
            a_2 = a_2 + PZI @ v2[di2]
            P_2 = P_2 - PZI @ ZZ @ P_2
        r2 = Z[di2].T @ iF2 @ v2[di2] if len(di2) else np.zeros(mm)
        alpha2 = a1_2 + P1_2 @ r2
        eta2 = Q @ R2.T @ r2
        if IFl and len(di2):
            ZZ = Z[di2]
            ish = np.any(ZZ @ R2 != 0, axis=0) & (eta2 != 0)
            G = ZZ @ R2[:, ish]
            if G.shape[0] == G.shape[1]:
                Qs = Q[np.ix_(ish, ish)]
                lik = (np.log(np.linalg.det(Qs)) + 2 * np.log(abs(np.linalg.det(G)))
                       + eta2[ish] @ np.linalg.solve(Qs, eta2[ish]) + len(di2) * LOG2PI)
            else:
                lik = np.inf
        r1 = self.L1.T @ (T2.T @ r2)
        if len(self.di1):
            r1 = r1 + Z[self.di1].T @ self.iF1 @ self.v1[self.di1]
        alpha1 = self.a1_1 + self.P1_1 @ r1
        out.update(a=np.column_stack([self.a_prev, a_2]), a1=np.column_stack([self.a1_1, a1_2]), P_upd=P_2,
                   P1_2=P1_2, v2=v2, alphahat=np.column_stack([alpha1, alpha2]), etahat2=eta2, lik=float(lik))
        return out

    def _update_uni(self, T2, R2, C2):
        """occbin_kalman_update of kalman_update_algo_3.m (observations processed one by one)."""
        ctx = self.ctx
        Z, H, Q, mm = ctx.Z, ctx.H, ctx.Q, ctx.mm
        tol = ctx.kalman_tol
        Hd = np.diag(H)
        di2 = self.di2
        pp = Z.shape[0]
        QQ = R2 @ Q @ R2.T
        a1_2 = T2 @ self.a_prev + C2
        a_2 = a1_2.copy()
        P1_2 = T2 @ self.P_prev @ T2.T + QQ
        P_2 = P1_2.copy()
        v2, Fi2, Ki2 = np.zeros(pp), np.zeros(pp), np.zeros((mm, pp))
        lik = 0.0
        if len(di2):
            vv = self.y2[di2] - Z[di2] @ a_2
            F = Z[di2] @ P_2 @ Z[di2].T + np.diag(Hd[di2])
            sig = np.sqrt(np.diag(F))
        for i in di2:
            Zi = Z[i]
            v2[i] = self.y2[i] - Zi @ a_2
            Fi2[i] = Zi @ P_2 @ Zi + Hd[i]
            Ki2[:, i] = P_2 @ Zi
            if Fi2[i] > tol:
                a_2 = a_2 + Ki2[:, i] * v2[i] / Fi2[i]
                P_2 = P_2 - np.outer(Ki2[:, i], Ki2[:, i]) / Fi2[i]
        if len(di2):
            with np.errstate(all="ignore"):
                ss = np.outer(sig, sig)
                log_dF = np.log(np.linalg.det(F / ss)) + 2 * np.sum(np.log(sig))
                iF = np.linalg.inv(F / ss) / ss
                lik = log_dF + vv @ iF @ vv + len(di2) * LOG2PI
        ri = np.zeros(mm)
        for i in di2[::-1]:
            if Fi2[i] > tol:
                ri = Z[i] * v2[i] / Fi2[i] + ri - Z[i] * (Ki2[:, i] @ ri) / Fi2[i]
        alpha2 = a1_2 + P1_2 @ ri
        eta2 = Q @ R2.T @ ri
        ri = T2.T @ ri
        for i in self.di1[::-1]:
            if self.Fi1[i] > tol:
                ri = Z[i] * self.v1[i] / self.Fi1[i] + ri - Z[i] * (self.Ki1[:, i] @ ri) / self.Fi1[i]
        alpha1 = self.a1_1 + self.P1_1 @ ri
        return dict(error_flag=0, a=np.column_stack([self.a_prev, a_2]), a1=np.column_stack([self.a1_1, a1_2]),
                    P_upd=P_2, P1_2=P1_2, v2=v2, Fi2=Fi2, Ki2=Ki2, alphahat=np.column_stack([alpha1, alpha2]),
                    etahat2=eta2, lik=float(lik))


@dataclass
class _Update:
    """Output of kalman_update_algo_1/3 and kalman_update_engine for one period."""
    error_flag: int
    lik: float
    regimes: tuple = ()             # regimes of periods t, t+1, t+2
    a: np.ndarray = None            # (mm, 2): updated state at t, predicted state at t+1 (piecewise)
    a1: np.ndarray = None           # (mm, 2): a1[:, 1] = prediction for t under the final regime
    P: tuple = None                 # (P_t|t, P_t+1|t)
    P1_2: np.ndarray = None
    v2: np.ndarray = None
    Fi2: np.ndarray = None
    Ki2: np.ndarray = None
    T: tuple = None                 # (T_t, T_t+1), filter state space
    R: tuple = None
    C: tuple = None
    etahat: np.ndarray = None
    alphahat: np.ndarray = None


def _algo(ctx: _PKFContext, step: _Step, regimes0=None, guess_regime=False) -> _Update:
    """kalman_update_algo_1.m (multivariate step) or kalman_update_algo_3.m (univariate step).
    ``regimes0``: None (start from the linear model) or a list of regimes (period t first)."""
    o = ctx.o
    base = ctx.base
    uni = step.univariate
    fail = _Update(error_flag=0, lik=np.inf, regimes=(base, base, base))
    if step.pre_error:
        fail.error_flag = step.pre_error
        return fail
    if regimes0 is not None:
        regimes0 = list(regimes0)
        if guess_regime:
            T2, R2, C2 = ctx.rule(ctx.solver.regime_rule(regimes0[0], ctx.sim))
            r1 = _shift_regime(regimes0[0])
            regimes0 = [regimes0[0], r1, _shift_regime(r1)]
        else:
            T2, R2, C2 = step.TT2, step.RR2, step.CC2
    else:
        T2, R2, C2 = ctx.T0, ctx.R0, np.zeros(ctx.mm)
        regimes0 = [base, base, base]
    up = step.update(T2, R2, C2)
    if up["error_flag"]:
        fail.error_flag = up["error_flag"]
        return fail
    r = ctx.r
    sim = dict(ctx.sim)
    shocks = np.zeros((3, ctx.nexo))
    shocks[0] = up["etahat2"]

    def init_of(alpha1):
        y0 = np.zeros(ctx.n)
        y0[r] = alpha1
        return y0

    init = init_of(up["alphahat"][:, 0])
    if guess_regime:
        flag, out = ctx.solver.solve(shocks, init, sim, init_regime=regimes0)
        if flag:
            flag, out = ctx.solver.solve(shocks, init, sim)
    else:
        flag, out = ctx.solver.solve(shocks, init, sim)
        if flag:
            flag, out = ctx.solver.solve(shocks, init, sim, init_regime=regimes0)
    if flag:
        fail.error_flag = flag
        return fail
    regs = out.regimes
    regime_hist = [regimes0[0]]
    lik_hist = [up["lik"]]
    niter = 1
    is_periodic = False
    maxiter = int(o["likelihood.max_number_of_iterations"])
    any_binding = any(any(x for x in rg) for reg in regs for rg, _ in reg)
    if any_binding or regs[0] != regimes0[0]:
        while regs[0] != regimes0[0] and not is_periodic and niter <= maxiter:
            niter += 1
            T2, R2, C2 = ctx.rule(out.ss[0])
            regime_hist.append(regs[0])
            up = step.update(T2, R2, C2)
            if up["error_flag"]:
                return _Update(error_flag=up["error_flag"], lik=up["lik"], regimes=tuple(regs))
            lik_hist.append(up["lik"])
            shocks[0] = up["etahat2"]
            init = init_of(up["alphahat"][:, 0])
            sim["periods"] = max(sim["periods"], _max_start(regs))
            flag, out2 = ctx.solver.solve(shocks, init, sim, init_regime=[regs[0]])
            if flag and not uni:          # algo_3 does not retry
                flag, out2 = ctx.solver.solve(shocks, init, sim)
            if flag:
                return _Update(error_flag=flag, lik=np.inf, regimes=tuple(regs))
            out = out2
            regimes0 = regs
            regs = out.regimes
            hits = [kk for kk in range(niter - 1) if regime_hist[kk] == regs[0]]
            if hits:
                is_periodic = True
                if o["filter.periodic_solution"]:
                    i0 = hits[0]
                    im = int(np.argmin(lik_hist[i0:]))
                    sim["periods"] = max(sim["periods"], _max_start(regimes0))
                    sim["maxit"] = 1
                    flag, out2 = ctx.solver.solve(shocks, init, sim, init_regime=[regime_hist[i0 + im]])
                    if flag:
                        return _Update(error_flag=flag, lik=np.inf, regimes=tuple(regs))
                    out = out2
                    regs = out.regimes
                    T2, R2, C2 = ctx.rule(out.ss[0])
                    up = step.update(T2, R2, C2)
                    if up["error_flag"]:
                        return _Update(error_flag=up["error_flag"], lik=up["lik"], regimes=tuple(regs))
                else:
                    return _Update(error_flag=330, lik=np.inf, regimes=tuple(regs))
    error_flag = 0
    if niter > maxiter and regs[0] != regimes0[0]:
        error_flag = 331
    lik = np.inf if error_flag else up["lik"]
    a = (out.piecewise[0:2, r] - ctx.ob.ys[r]).T
    T1, R1, C1 = ctx.rule(out.ss[0])
    T2n, R2n, C2n = ctx.rule(out.ss[1])
    P_t = up["P_upd"]
    P_next = T2n @ P_t @ T2n.T + R2n @ ctx.Q @ R2n.T
    return _Update(error_flag=error_flag, lik=float(lik), regimes=tuple(regs[:3]), a=a, a1=up["a1"],
                   P=(P_t, P_next), P1_2=up["P1_2"], v2=up["v2"], Fi2=up.get("Fi2"), Ki2=up.get("Ki2"),
                   T=(T1, T2n), R=(R1, R2n), C=(C1, C2n), etahat=up["etahat2"], alphahat=up["alphahat"])


def _as_tuple(guess):
    return tuple((tuple(int(x) for x in r), tuple(int(x) for x in s)) for r, s in guess)


def _engine(ctx: _PKFContext, t, step: _Step, regimes_pair) -> _Update:
    """kalman_update_engine.m: ``regimes_pair`` are the regimes predicted for t and t+1 by the
    previous period."""
    o = ctx.o
    base = ctx.base
    brute = bool(o["likelihood.brute_force_regime_guess"])
    brute_extra = bool(o["likelihood.brute_force_extra_regime_guess"])
    loss_fn = bool(o["likelihood.loss_function_regime_guess"])
    best = _algo(ctx, step)
    likvec, regvec = [], []
    if best.error_flag == 0:
        likvec, regvec = [best.lik], [best.regimes[0]]
    else:
        best.lik = np.inf
    info0 = best.error_flag
    is_base_pair = tuple(regimes_pair[:2]) == (base, base)
    if best.error_flag:
        if not is_base_pair:
            best = _algo(ctx, step, regimes0=list(regimes_pair[:2]))
        if best.error_flag == 0:
            likvec, regvec = [best.lik], [best.regimes[0]]
        else:
            best.lik = np.inf
        info1 = best.error_flag
    else:
        if not is_base_pair:
            alt = _algo(ctx, step, regimes0=list(regimes_pair[:2]))
            info1 = alt.error_flag
            if alt.error_flag == 0 and alt.regimes != best.regimes:
                likvec.append(alt.lik)
                regvec.append(alt.regimes[0])
            if alt.error_flag == 0 and alt.lik < best.lik:
                best = alt
        else:
            if t > int(o["likelihood.number_of_initial_periods_with_extra_regime_guess"]):
                info1 = 0
            else:
                info1 = 1
                brute, loss_fn = True, True
    if t < int(o["likelihood.first_period_binding_regime_allowed"]):
        return best
    diffstart, which = 0, 0
    if best.error_flag == 0 and len(regvec) == 1:
        diffs = [best.regimes[0][j][1][-1] - regimes_pair[0][j][1][-1] for j in range(ctx.k)]
        which = int(np.argmax(diffs))
        diffstart = diffs[which]
    if o["filter.use_relaxation"] and diffstart > o["filter.use_relaxation_tol_period"]:
        guess = [[list(rg), list(st)] for rg, st in best.regimes[0]]
        regR = best.regimes
        while _as_tuple(guess) == regR[0]:
            if regR[0][which][1][-1] <= 1:
                break
            st = guess[which][1]
            st[-1] = regR[0][which][1][-1] - 1
            if st[-1] == 1:
                guess[which][1] = [st[-1]]
                guess[which][0] = [0]
            elif len(st) > 1 and st[-2] == st[-1]:
                st[-2] -= 1
            resR = _algo(ctx, step, regimes0=[_as_tuple(guess), base], guess_regime=True)
            regR = resR.regimes
            if regR[0] not in regvec:
                likvec.append(resR.lik)
                regvec.append(regR[0])
            if resR.error_flag == 0 and resR.lik < best.lik:
                best = resR
        brute = loss_fn = False
    if (brute and (info0 and info1)) or (brute_extra and (info0 or info1)):
        use, info2 = None, 1
        if ctx.k == 1:
            for kk in range(1, 6):
                c1 = _algo(ctx, step, regimes0=[(((0, 1, 0), (1, 5, 5 + 4 * kk)),), base], guess_regime=True)
                info2 = c1.error_flag
                if info2 == 0:
                    use = c1
                    if not (best.error_flag == 0 and c1.regimes == best.regimes):
                        break
                c2 = _algo(ctx, step, regimes0=[(((1, 0), (1, 1 + 4 * kk)),), base], guess_regime=True)
                info2 = c2.error_flag
                if info2 == 0:
                    use = c2
                if use is not None:
                    info2 = 0
                    break
        else:
            use, info2 = _brute_force_two(ctx, step, best)
        if info2 == 0 and use is not None:
            info0 = info1 = 0
            if use.regimes[0] not in regvec:
                regvec.append(use.regimes[0])
                likvec.append(use.lik)
            if use.lik < best.lik:
                best = use
    if loss_fn and (info0 or info1):
        res_f = _findmin(ctx, step)
        if res_f is not None:
            c = _algo(ctx, step, regimes0=[res_f.regimes[0], base], guess_regime=True)
            if c.error_flag == 0 and c.regimes[0] not in regvec:
                regvec.append(c.regimes[0])
                likvec.append(c.lik)
            if c.error_flag == 0 and c.lik < best.lik:
                best = c
    if len(likvec) > 1:
        lv = np.asarray(likvec, float)
        mn = lv.min()
        with np.errstate(all="ignore"):
            best.lik = float(-2 * np.log(np.sum(np.exp(-(lv - mn) / 2))) + mn)
    if best.error_flag == 0:
        if np.isnan(best.lik):
            best.error_flag = 323
        elif np.isnan(best.a).any():
            best.error_flag = 324
    return best


def fminsearch_octave(fcn, x0, tol_x=1e-4, tol_f=1e-4, max_fun_evals=None, max_iter=None):
    """Octave 8's ``fminsearch`` (port in :mod:`pydyn.fminsearch`); returns the minimiser only."""
    from .fminsearch import fminsearch
    return fminsearch(fcn, x0, tol_x=tol_x, tol_f=tol_f, max_fun_evals=max_fun_evals, max_iter=max_iter)[0]

def _findmin(ctx, step):
    """``occbin.findmin`` + ``occbin.cost_function``: the regime guess from the shocks that minimise the
    distance between the model's first-period OccBin solution and the observables (scaled by the
    forecast variance of the observables), found with ``fminsearch``. Returns the OccBin solution
    at the minimum, or None if it fails."""
    di = step.di2
    Z = ctx.Z
    err_index = np.flatnonzero(np.diag(ctx.Q) != 0)
    obs = np.argmax(Z[di] != 0, axis=1)
    obs_decl = ctx.r[obs]
    current_obs = step.y2[di] + ctx.ob.ys[obs_decl]
    ZZ = Z[di]
    weights = np.diag(ZZ @ step.P1_t @ ZZ.T)
    init = np.zeros(ctx.n)
    init[ctx.r] = step.a_prev
    sim = dict(ctx.sim)

    def run(x):
        return ctx.solver.solve(np.asarray(x, float).reshape(1, -1), init, sim, exo_pos=err_index)

    def cost(x):
        flag, out = run(x)
        if flag:
            return 1.0e10
        return float(np.mean((out.piecewise[0, obs_decl] - current_obs) ** 2 / weights))

    y = fminsearch_octave(cost, np.zeros(len(err_index)))
    flag, out = run(y)
    return None if flag else out


def _brute_force_two(ctx, step, best):
    """The two-constraint brute-force regime guesses of kalman_update_engine.m."""
    base1 = ((0,), (1,))
    use, info2, c = None, 1, None

    def new_solution(c):
        return info2 == 0 and not (best.error_flag == 0 and c.regimes == best.regimes)

    for jk in (0, 1):
        for kk in range(1, 6):
            for jr in (0, 1):
                for kind in (1, 2):
                    cur = ((0, 1, 0), (1, 5, 5 + 4 * kk)) if kind == 1 else ((1, 0), (1, 1 + 4 * kk))
                    for kj in range(1, 2 + (1 if jk > 0 else 0)):
                        if jk == 0:
                            other = base1
                        elif kj == 1:
                            other = ((1, 0), (1, 1 + 4 * jk))
                        else:
                            other = ((0, 1, 0), (1, 5, 5 + 4 * jk))
                        g = (cur, other) if jr == 0 else (other, cur)
                        c = _algo(ctx, step, regimes0=[g, ctx.base], guess_regime=True)
                        info2 = c.error_flag
                        if info2 == 0:
                            use = c
                            if new_solution(c):
                                break
                    if new_solution(c):
                        break
                if new_solution(c):
                    break
            if use is not None:
                return use, 0
        if use is not None:
            return use, 0
    return use, info2


@dataclass
class PKFResult:
    LIK: float                       # minus the log likelihood (after the presample)
    lik: np.ndarray                  # per-period contributions (0.5 * Dynare's lik)
    a0: np.ndarray                   # (mm, T) updated (filtered) states a_t|t
    a1: np.ndarray                   # (mm, T + 1) one-step-ahead predictions a_t|t-1 (a1[:, T]: a_T+1|T)
    P0: np.ndarray                   # (T, mm, mm) P_t|t
    P1: np.ndarray                   # (T, mm, mm) P_t|t-1
    regimes: list                    # regime of each period t (regimes_(t) after the update)
    error_code: int = 0
    error_period: int = 0
    TT: np.ndarray = None            # (T + 2, mm, mm) transitions: TT[t] into period t + 1 (0-based)
    RR: np.ndarray = None
    CC: np.ndarray = None
    v: np.ndarray = None             # (pp, T) innovations (sequential ones for the univariate filter)
    Fi: np.ndarray = None
    Ki: np.ndarray = None


def _pkf_forward(ctx: _PKFContext, Y, a, P, data_index=None, presample=0, univariate=False, regimes_init=None,
                 first_period_occbin_update=None, check_inf=True) -> PKFResult:
    """Forward pass of the piecewise Kalman filter: missing_observations_kalman_filter.m with
    ``occbin_.status`` (multivariate, likelihood) or the forward loop of
    missing_DiffuseKalmanSmootherH3_Z.m with OccBin (univariate, smoother)."""
    o = ctx.o
    Z, H = ctx.Z, ctx.H
    pp, last = Y.shape
    mm = ctx.mm
    tol = ctx.kalman_tol
    if data_index is None:
        data_index = [np.flatnonzero(~np.isnan(Y[:, t])) for t in range(last)]
    fpou = int(o["likelihood.first_period_occbin_update"]) if first_period_occbin_update is None \
        else first_period_occbin_update
    use_updated = bool(o["likelihood.use_updated_regime"]) if not univariate else True
    TT = np.repeat(ctx.T0[None], last + 2, axis=0)
    RR = np.repeat(ctx.R0[None], last + 2, axis=0)
    CC = np.zeros((last + 2, mm))
    base = ctx.base
    regimes_ = [base] * (last + 3)
    if regimes_init is not None:
        regimes_[:len(regimes_init)] = list(regimes_init)
    a0 = np.zeros((mm, last))
    a1 = np.zeros((mm, last + 1))
    aK = np.zeros((mm, last + 1))            # Dynare's aK(1,:,:): piecewise one-step-ahead predictions
    P0 = np.zeros((last, mm, mm))
    P1 = np.zeros((last, mm, mm))
    vv = np.zeros((pp, last))
    Fi = np.zeros((pp, last))
    Ki = np.zeros((last, mm, pp))
    lik = np.zeros(last)
    Q = ctx.Q
    P = np.tril(P) + np.tril(P, -1).T
    a = np.asarray(a, float).copy()

    def fail(code, t):
        return PKFResult(np.inf, lik, a0, a1, P0, P1, regimes_[:last], code, t, TT, RR, CC, vv, Fi, Ki)

    ainit = Pinit = None
    for t in range(1, last + 1):              # 1-based as in Dynare
        i = t - 1
        a1[:, i] = a
        P1[i] = P
        C, R, T = CC[t], RR[t], TT[t]
        QQ = R @ Q @ R.T
        if t == 1:
            Pinit, ainit = P1[0].copy(), a1[:, 0].copy()
        di = data_index[i]
        if univariate and t < fpou:
            # missing_DiffuseKalmanSmootherH3_Z.m before first_period_occbin_update: the linear
            # univariate filter with the base-regime matrices
            if t == 1:
                a = T @ a + C
                P = T @ P @ T.T + QQ
                a1[:, 0], P1[0] = a, P
            at, Pt = a.copy(), P.copy()
            Hd = np.diag(H)
            for ii in di:
                Zi = Z[ii]
                vv[ii, i] = Y[ii, i] - Zi @ at
                Fi[ii, i] = Zi @ Pt @ Zi + Hd[ii]
                Ki[i][:, ii] = Pt @ Zi
                if Fi[ii, i] > tol:
                    at = at + Ki[i][:, ii] * vv[ii, i] / Fi[ii, i]
                    Pt = Pt - np.outer(Ki[i][:, ii], Ki[i][:, ii]) / Fi[ii, i]
                    Pt = (Pt + Pt.T) / 2
            a0[:, i], P0[i] = at, Pt
            Tn, Cn = TT[t + 1], CC[t + 1]
            a = Tn @ at + Cn
            P = Tn @ Pt @ Tn.T + RR[t + 1] @ Q @ RR[t + 1].T
            P = (P + P.T) / 2
            aK[:, i + 1] = a
            continue
        if t < fpou or not use_updated:
            # the linear branch (Dynare uses it before first_period_occbin_update)
            if len(di) == 0:
                a = T @ a + C
                P = T @ P @ T.T + QQ
            else:
                z = Z[di]
                v = Y[di, i] - z @ a
                F = z @ P @ z.T + H[np.ix_(di, di)]
                if t < fpou and _rcond(F) < tol:
                    sig = np.sqrt(np.diag(F))
                    if np.any(np.diag(F) < tol) or _rcond(F / np.outer(sig, sig)) < tol:
                        return fail(320, t)
                with np.errstate(all="ignore"):
                    iF = np.linalg.inv(F)
                    lik[i] = np.log(np.linalg.det(F)) + v @ iF @ v + len(di) * LOG2PI
                if t < fpou:
                    K = P @ z.T @ iF
                    P0[i] = P - K @ z @ P
                    a0[:, i] = a + K @ v
                    vv[di, i] = v
                    P = T @ (P - K @ z @ P) @ T.T + QQ
                    a = T @ (a + K @ v) + C
                    aK[:, i + 1] = a
        if t >= fpou:
            if t == 1:
                step = _Step(ctx, ainit, ainit, Pinit, Pinit, np.zeros(0, int), data_index[0], Y[:, 0],
                             TT[0], RR[0], CC[0], np.zeros(pp), np.zeros(pp), np.zeros((mm, pp)), univariate)
            else:
                step = _Step(ctx, a0[:, i - 1], a1[:, i - 1], P0[i - 1], P1[i - 1], data_index[i - 1], di,
                             Y[:, i], TT[i], RR[i], CC[i], vv[:, i - 1], Fi[:, i - 1], Ki[i - 1], univariate)
            step.P1_t = P
            res = _engine(ctx, t, step, (regimes_[i], regimes_[i + 1]))
            likx = res.lik
            if not univariate and t < int(o["likelihood.first_period_binding_regime_allowed"]) \
                    and res.regimes[0] != base:
                likx = np.inf
            if res.error_flag:
                return fail(res.error_flag, t)
            if check_inf and np.isinf(likx):
                return fail(320, t)
            if use_updated:
                lik[i] = likx
            regimes_[i:i + 3] = list(res.regimes)
            a0[:, i] = res.a[:, 0]
            a1[:, i] = res.a1[:, 1]
            a = res.a[:, 1].copy()
            aK[:, i + 1] = a
            vv[di, i] = res.v2[di]
            if univariate:
                Fi[di, i] = res.Fi2[di]
                Ki[i][:, di] = res.Ki2[:, di]
            TT[i], TT[i + 1] = res.T
            RR[i], RR[i + 1] = res.R
            CC[i], CC[i + 1] = res.C
            P0[i] = res.P[0]
            P1[i] = res.P1_2
            P = res.P[1]
    a1[:, last] = a
    lik = 0.5 * lik
    LIK = float(np.sum(lik[presample:]))
    out = PKFResult(LIK, lik, a0, a1, P0, P1, regimes_[:last + 1], 0, 0, TT, RR, CC, vv, Fi, Ki)
    out.ainit, out.Pinit, out.aK = ainit, Pinit, aK
    return out


def piecewise_kalman_filter(ctx: _PKFContext, Y, a, P, data_index=None, presample=0, kalman_tol=None) -> PKFResult:
    """missing_observations_kalman_filter.m with ``occbin_.status`` (the likelihood of
    dsge_likelihood.m): ``Y`` (pp x T) observations minus their steady state (and trends), ``a``/``P``
    the initial prediction and covariance (period 1). ``LIK`` is minus the log likelihood."""
    if kalman_tol is not None:
        ctx.kalman_tol = float(kalman_tol)
    res = _pkf_forward(ctx, Y, a, P, data_index, presample)
    if not res.error_code:
        res.regimes = res.regimes[:Y.shape[1]]
    return res


# ====================================================================== OccBin smoother
def _univariate_backward(ctx, a1, P1, v, Fi, Ki, TT, RR, data_index, ainit, Pinit):
    """Backward pass of missing_DiffuseKalmanSmootherH3_Z.m (non-diffuse part) with time-varying
    ``TT``/``RR`` (0-based: ``TT[t]`` is the transition into period t + 1)."""
    Z, Q, tol = ctx.Z, ctx.Q, ctx.kalman_tol
    mm, T = a1.shape[0], len(data_index)
    alphahat = np.zeros((mm, T))
    etahat = np.zeros((ctx.nexo, T))
    ri = np.zeros(mm)
    for t in range(T - 1, -1, -1):
        for i in np.asarray(data_index[t], int)[::-1]:
            if Fi[i, t] > tol:
                ri = Z[i] * v[i, t] / Fi[i, t] + ri - Z[i] * (Ki[t][:, i] @ ri) / Fi[i, t]
        alphahat[:, t] = a1[:, t] + P1[t] @ ri
        etahat[:, t] = Q @ RR[t].T @ ri
        ri = TT[t].T @ ri
    alphahat0 = ainit + Pinit @ ri
    return alphahat, etahat, alphahat0


def _tv_forward(ctx, Y, a_init, P_init, data_index, TT, RR, CC):
    """Forward univariate filter of missing_DiffuseKalmanSmootherH3_Z.m with given time-varying
    matrices (DSGE_smoother iterations, ``first_period_occbin_update = inf``)."""
    Z, Q, tol = ctx.Z, ctx.Q, ctx.kalman_tol
    Hd = np.diag(ctx.H)
    pp, T = Y.shape
    mm = ctx.mm
    a = np.zeros((mm, T))
    a1 = np.zeros((mm, T + 1))
    P1 = np.zeros((T, mm, mm))
    v, Fi, Ki = np.zeros((pp, T)), np.zeros((pp, T)), np.zeros((T, mm, pp))
    Pn = np.array(P_init, float)
    an = np.asarray(a_init, float).copy()
    for t in range(T):
        if t == 0:
            an = TT[0] @ an + CC[0]
            Pn = TT[0] @ Pn @ TT[0].T + RR[0] @ Q @ RR[0].T
        a1[:, t] = an
        P1[t] = Pn
        at, Pt = an.copy(), Pn.copy()
        for i in np.asarray(data_index[t], int):
            Zi = Z[i]
            v[i, t] = Y[i, t] - Zi @ at
            Fi[i, t] = Zi @ Pt @ Zi + Hd[i]
            Ki[t][:, i] = Pt @ Zi
            if Fi[i, t] > tol:
                at = at + Ki[t][:, i] * v[i, t] / Fi[i, t]
                Pt = Pt - np.outer(Ki[t][:, i], Ki[t][:, i]) / Fi[i, t]
                Pt = (Pt + Pt.T) / 2
        a[:, t] = at
        Tn, Cn = TT[t + 1], CC[t + 1]
        an = Tn @ at + Cn
        Pn = Tn @ Pt @ Tn.T + RR[t + 1] @ Q @ RR[t + 1].T
        Pn = (Pn + Pn.T) / 2
    a1[:, T] = an
    return a, a1, P1, v, Fi, Ki


@dataclass
class OccbinSmootherResult:
    """Output of :func:`occbin_smoother`. States are in deviations from the steady state, in
    declaration order, shape (T, n_endo); shocks (T, n_exo)."""
    smoothed: np.ndarray             # oo_.SmoothedVariables - steady state
    shocks: np.ndarray               # oo_.SmoothedShocks
    updated: np.ndarray              # oo_.UpdatedVariables - steady state (real-time PKF, a_t|t)
    filtered: np.ndarray             # oo_.FilteredVariables - steady state (real-time PKF, a_t+1|t)
    smoothed0: np.ndarray            # smoothed state in period 0
    regime_history: list             # oo_.occbin.smoother.regime_history (T + 1 entries)
    realtime_regime_history: list    # oo_.occbin.smoother.realtime_regime_history
    linear_smoothed: np.ndarray = None   # oo_.occbin.linear_smoother (deviations, declaration order)
    linear_shocks: np.ndarray = None
    error_flag: int = 0              # oo_.occbin.smoother.error_flag
    warning_flag: int = 0            # oo_.occbin.smoother.warning_flag
    iterations: int = 1
    regime_history_iter: list = field(default_factory=list)
    piecewise: np.ndarray = None     # simulation with the smoothed shocks (levels), oo_.occbin.smoother.simul.piecewise
    linear: np.ndarray = None        # the same ignoring the constraints (levels)
    ys: np.ndarray = None
    TT: np.ndarray = None            # (T + 1, n, n) regime-dependent transitions in declaration order (TT[t] into t + 1)
    RR: np.ndarray = None
    CC: np.ndarray = None
    converged: bool = True


def occbin_smoother(ob: Occbin, Y, Q, H, obs_idx, options=None, data_index=None, lik_init=1, qz_criterium=1 - 1e-6,
                    Harvey_scale_factor=10.0, kalman_tol=1e-10) -> OccbinSmootherResult:
    """OccBin smoother: Dynare's ``occbin.DSGE_smoother`` (with the default ``smoother_redux = false``:
    all endogenous variables are filter states). ``Y`` (pp x T) observations minus their steady
    state, ``Q`` covariance of the shocks, ``H`` of the measurement errors, ``obs_idx`` declaration
    indices of the observed variables.

    1. linear smoother (``oo_.occbin.linear_smoother``);
    2. real-time smoother: univariate piecewise Kalman filter (kalman_update_algo_3), then the
       backward pass with the regime-dependent matrices;
    3. simulation with the smoothed shocks from the smoothed initial state, and, while the regimes of
       the simulation differ from those used by the smoother, a smoother with the matrices of the
       new regimes (at most ``smoother.max_number_of_iterations`` passes), including Dynare's
       "conditional smoother" attempt.
    """
    from .perturbation import dr_ordering
    o = dict(DEFAULTS) if options is None else dict(options)
    o["kalman_tol"] = kalman_tol
    m = ob.model
    order_var, state_var, counts = dr_ordering(m.lead_lag_incidence)
    inv = np.empty(m.n_endo, int)
    inv[order_var] = np.arange(m.n_endo)
    obs_idx = np.asarray(obs_idx, int)
    pp, T = Y.shape
    n = ob.n
    Z = np.zeros((pp, n))
    Z[np.arange(pp), inv[obs_idx]] = 1.0
    ctx = _PKFContext(ob, order_var, Z, np.asarray(H, float), np.asarray(Q, float), o, prefix="smoother")
    if data_index is None:
        data_index = [np.flatnonzero(~np.isnan(Y[:, t])) for t in range(T)]
    state_dr = np.arange(counts["nstatic"], counts["nstatic"] + counts["npred"] + counts["nboth"])
    fpou_s = int(o["smoother.first_period_occbin_update"])
    if fpou_s < 1:
        raise ValueError("smoother_first_period_occbin_update must be a positive integer")
    T0, R0 = ctx.T0, ctx.R0
    if lik_init == 1:
        from .kalman import lyapunov_symm
        P_init = lyapunov_symm(T0, R0 @ ctx.Q @ R0.T, qz_criterium)
    elif lik_init == 2:
        P_init = Harvey_scale_factor * np.eye(n)
    else:
        raise NotImplementedError("the OccBin smoother is available for lik_init = 1, 2")
    a_init = np.zeros(n)
    ys = ob.ys
    base = ctx.base

    def to_decl(x):              # (n, ...) in DR order -> declaration order
        return x[inv]

    # --- 1. linear smoother
    TTl = np.repeat(T0[None], T + 2, axis=0)
    RRl = np.repeat(R0[None], T + 2, axis=0)
    CCl = np.zeros((T + 2, n))
    la, la1, lP1, lv, lFi, lKi = _tv_forward(ctx, Y, a_init, P_init, data_index, TTl, RRl, CCl)
    lin_alpha, lin_eta, lin_alpha0 = _univariate_backward(ctx, la1, lP1, lv, lFi, lKi, TTl, RRl, data_index,
                                                          a_init, P_init)
    # --- 2. real-time smoother
    rt = _pkf_forward(ctx, Y, a_init, P_init, data_index, univariate=True, check_inf=False,
                      first_period_occbin_update=fpou_s)
    rt_ok = rt.error_code == 0
    if rt_ok:
        alphahat, etahat, alphahat0 = _univariate_backward(ctx, rt.a1, rt.P1, rt.v, rt.Fi, rt.Ki, rt.TT, rt.RR,
                                                           data_index, rt.ainit, rt.Pinit)
        regime_history = list(rt.regimes[:T + 1])
        ahat, aK = rt.a0, rt.aK
    else:
        alphahat, etahat, alphahat0 = lin_alpha, lin_eta, lin_alpha0
        regime_history = [base] * (T + 1)
        ahat, aK = la, la1
    realtime_regimes = list(regime_history)
    regime_history0 = [list(regime_history)]
    sim = _sim_kwargs(o, "smoother")
    init_regime = list(regime_history)

    def simulate(eta, alpha0):
        shocks = np.vstack([eta.T, np.zeros((1, ctx.nexo))])
        kw = dict(sim)
        kw["periods"] = shocks.shape[0]
        return ctx.solver.solve(shocks, to_decl(alpha0), kw, init_regime=init_regime)

    def matrices(res):
        TT = np.array([t_[0][np.ix_(order_var, order_var)] for t_ in res.ss])
        RR = np.array([t_[1][order_var, :] for t_ in res.ss])
        CC = np.array([t_[2][order_var] for t_ in res.ss])
        return TT, RR, CC

    error_flag, warning_flag = 0, 0
    flag, out = simulate(etahat, alphahat0)
    last_sim_ok = True
    fbr = int(o["likelihood.first_period_binding_regime_allowed"])
    if flag:
        last_sim_ok, error_flag = False, 321
        new_regimes = regime_history
        TT = RR = CC = None
    else:
        new_regimes = list(out.regimes)
        if new_regimes[:fbr - 1] != regime_history[:fbr - 1]:
            error_flag, new_regimes = 322, regime_history
        TT, RR, CC = matrices(out)
    regime_history = new_regimes
    is_changed = regime_history0[0] != regime_history
    rt_converged = not (is_changed or error_flag)
    sto = dict(alphahat=alphahat, etahat=etahat, alphahat0=alphahat0, regime_history=list(regime_history))
    sto_TT = (TT, RR, CC)
    cond_converged = False
    if not rt_converged and not (lik_init == 2 and Harvey_scale_factor == 0) and TT is not None:
        # conditional smoother: initial state known (the smoothed one), Harvey_scale_factor = 0
        a_c = np.zeros(n)
        a_c[state_dr] = alphahat0[state_dr]
        rc = _pkf_forward(ctx, Y, a_c, np.zeros((n, n)), data_index, univariate=True, check_inf=False,
                          first_period_occbin_update=1, regimes_init=[regime_history[0]] + list(regime_history))
        if rc.error_code:
            error_flag = 322
        else:
            c_alpha, c_eta, c_alpha0 = _univariate_backward(ctx, rc.a1, rc.P1, rc.v, rc.Fi, rc.Ki, rc.TT, rc.RR,
                                                            data_index, rc.ainit, rc.Pinit)
            c_regimes = list(rc.regimes[:T + 1])
            regime_history = c_regimes
            flag, outc = simulate(c_eta, alphahat0)
            error_flag = 0
            if flag:
                error_flag = 321
                c_new = regime_history
            else:
                out = outc
                c_new = list(outc.regimes)
                if c_new[:fbr - 1] != regime_history[:fbr - 1]:
                    error_flag, c_new = 322, regime_history
            is_changed = c_regimes != c_new
            if not (is_changed or error_flag):
                cond_converged = True
                TT, RR, CC = matrices(outc)
                alphahat, etahat, alphahat0 = c_alpha, c_eta, c_alpha0
                regime_history = c_regimes
    maxiter = int(o["smoother.max_number_of_iterations"])
    it, is_periodic = 1, False
    fast = bool(o["smoother.fast"])
    is_changed_start = False
    cur_TT = sto_TT
    while is_changed and maxiter > it and not is_periodic and last_sim_ok:
        if it == 1:
            if rt_ok:
                regime_history = list(sto["regime_history"])
            error_flag = 0
            TT, RR, CC = cur_TT
        it += 1
        a_, a1_, P1_, v_, Fi_, Ki_ = _tv_forward(ctx, Y, a_init, P_init, data_index, TT, RR, CC)
        alphahat, etahat, alphahat0 = _univariate_backward(ctx, a1_, P1_, v_, Fi_, Ki_, TT, RR, data_index,
                                                           a_init, P_init)
        regime_history0.append(list(regime_history))
        sto_TT = (TT, RR, CC)
        flag, out2 = simulate(etahat, alphahat0)
        if flag:
            error_flag = 321
            break
        new = list(out2.regimes)
        if new[:fbr - 1] != regime_history[:fbr - 1]:
            error_flag = 322
            break
        out = out2
        regime_history = new
        TT, RR, CC = matrices(out2)
        prev = regime_history0[-1]
        is_changed = prev != regime_history
        d_reg = any(prev[k][j][0] != regime_history[k][j][0] for k in range(len(prev)) for j in range(ctx.k))
        is_changed_start = any(prev[k][j][1] != regime_history[k][j][1] for k in range(len(prev)) for j in range(ctx.k))
        if fast:
            is_changed = d_reg
        is_tmp = any(regime_history0[kk] == regime_history for kk in range(it - 1))
        is_periodic = is_changed and is_tmp
    regime_history0.append(list(regime_history))
    final_regimes = regime_history0[-1]
    converged = False
    if (((maxiter == it) or not last_sim_ok) and is_changed) or is_periodic or error_flag:
        if is_periodic:
            converged, error_flag, warning_flag = True, 0, 321
        elif not rt_ok:
            error_flag = 322
    else:
        converged, error_flag = True, 0
        if not cond_converged and not rt_converged:
            warning_flag = 320
        elif fast and is_changed_start:
            warning_flag = 323
    if rt_ok and not cond_converged and not rt_converged and not converged:
        warning_flag, error_flag = 322, 0
        alphahat, etahat, alphahat0 = sto["alphahat"], sto["etahat"], sto["alphahat0"]
        final_regimes = realtime_regimes
        is_changed = False
    if is_changed and is_periodic:
        TT, RR, CC = sto_TT
        final_regimes = regime_history0[-2]
    res = OccbinSmootherResult(
        smoothed=to_decl(alphahat).T, shocks=etahat.T, updated=to_decl(ahat).T, filtered=to_decl(aK[:, 1:T + 1]).T,
        smoothed0=to_decl(alphahat0), regime_history=final_regimes, realtime_regime_history=realtime_regimes,
        linear_smoothed=to_decl(lin_alpha).T, linear_shocks=lin_eta.T, error_flag=error_flag,
        warning_flag=warning_flag, iterations=it, regime_history_iter=regime_history0, ys=ys, converged=converged)
    if out is not None:
        res.piecewise = out.piecewise[:-1]
        res.linear = out.linear[:-1]
    if TT is not None:
        res.TT = np.array([np.zeros((n, n)) for _ in range(len(TT))])
        res.RR = np.zeros((len(TT), n, ctx.nexo))
        res.CC = np.zeros((len(TT), n))
        for k in range(len(TT)):
            res.TT[k][np.ix_(order_var, order_var)] = TT[k]
            res.RR[k][order_var] = RR[k]
            res.CC[k][order_var] = CC[k]
    return res


def inversion_filter_smoother(ivf: InversionFilterResult, ys, err_index, n_exo) -> OccbinSmootherResult:
    """Smoother output of the inversion filter (``smoother_inversion_filter``; Dynare's
    IVF_posterior ``atT``/``innov``): the filtered states and shocks are the smoothed ones, the
    system is assumed to start at the steady state. ``updated``/``filtered`` are NaN (Dynare
    does not provide them with the inversion filter)."""
    T, n = ivf.stateval.shape
    shocks = np.zeros((T, n_exo))
    shocks[:, np.asarray(err_index, int)] = ivf.filtered_errs
    nan = np.full((T, n), np.nan)
    return OccbinSmootherResult(smoothed=ivf.stateval.copy(), shocks=shocks, updated=nan, filtered=nan.copy(),
                                smoothed0=np.zeros(n), regime_history=list(ivf.regime_history),
                                realtime_regime_history=list(ivf.regime_history), error_flag=ivf.error_code,
                                ys=np.asarray(ys, float))


# ====================================================================== shock decomposition
def occbin_shock_decomposition(sm: OccbinSmootherResult, T0, R0, endo_names, exo_names, tinit=1, regime_shock=None):
    """occbin/shock_decomposition.m. ``sm`` is an OccBin smoother result (with ``TT``/``RR``/``CC``),
    ``T0``/``R0`` the linear rules (declaration order). Returns a dict of
    :class:`pydyn.smoother.ShockDecomposition` (layout ``(n_endo, n_exo + 2, T)``: shocks, initial
    conditions, smoothed variable; deviations from the steady state):

    * ``"decomp"`` (``oo_.occbin.smoother.decomp``): piecewise decomposition; the effect of the
      regime constants is in the initial-condition column, unless the model has a shock named
      ``EPS_REGIME`` (``regime_shock``) that receives it;
    * ``"wdecomp"`` (``oo_.occbin.smoother.wdecomp``): the regime effect spread over the columns
      in proportion to their absolute contributions;
    * ``"init_decomp"`` (``oo_.occbin.smoother.init_decomp``): initial conditions by variable, then
      the shocks and the smoothed variable (Dynare's field keeps only the n variable columns: its
      reordering by ``inv_order_var`` drops the last two);
    * ``"linear"``/``"linear_init"`` (``oo_.occbin.linear_smoother.decomp``/``init_decomp``).

    ``tinit`` (1-based) is the first period decomposed (Dynare's ``TINIT``)."""
    from .smoother import ShockDecomposition
    if sm.TT is None or sm.linear_smoothed is None:
        raise ValueError("the smoother result has no regime matrices / linear smoother (inversion filter?)")
    alpha = sm.smoothed.T                          # (n, T) declaration order
    eta = sm.shocks.T
    la, le = sm.linear_smoothed.T, sm.linear_shocks.T
    n, gend = alpha.shape
    nexo = eta.shape[0]
    t0 = int(tinit) - 1
    L = gend - t0
    # linear
    as_lin, att, deco = np.zeros((n, L)), np.zeros((n, L)), np.zeros((n, nexo + 2, L))
    deco_init = np.zeros((n, n + 2, L))
    deco_init[:, -1, :] = la[:, t0:]
    deco_init[np.arange(n), np.arange(n), 0] = la[:, t0]
    att[:, 0] = as_lin[:, 0] = la[:, t0]
    for j in range(1, L):
        e = le[:, j + t0]
        as_lin[:, j] = T0 @ as_lin[:, j - 1] + R0 @ e
        att[:, j] = T0 @ att[:, j - 1]
        deco[:, :nexo, j] = R0 * e[None, :] + T0 @ deco[:, :nexo, j - 1]
        deco_init[:, :n, j] = T0 @ deco_init[:, :n, j - 1]
        deco_init[:, n, j] = deco_init[:, n + 1, j] - deco_init[:, :n, j].sum(axis=1)
    deco[:, nexo] = att
    deco[:, nexo + 1] = as_lin
    # piecewise
    as_p, att_p, reg_p = np.zeros((n, L)), np.zeros((n, L)), np.zeros((n, L))
    deco_p = np.zeros((n, nexo + 2, L))
    deco_p_init = np.zeros((n, n + 2, L))
    deco_p_init[:, -1, :] = alpha[:, t0:]
    deco_p_init[np.arange(n), np.arange(n), 0] = alpha[:, t0]
    att_p[:, 0] = as_p[:, 0] = alpha[:, t0]
    for j in range(1, L):
        TM, RM, CONST = sm.TT[j + t0], sm.RR[j + t0], sm.CC[j + t0]
        e = eta[:, j + t0]
        as_p[:, j] = TM @ as_p[:, j - 1] + RM @ e + CONST
        att_p[:, j] = TM @ att_p[:, j - 1]
        reg_p[:, j] = TM @ reg_p[:, j - 1] + CONST
        deco_p[:, :nexo, j] = RM * e[None, :] + TM @ deco_p[:, :nexo, j - 1]
        deco_p_init[:, :n, j] = TM @ deco_p_init[:, :n, j - 1]
        deco_p_init[:, n, j] = deco_p_init[:, n + 1, j] - deco_p_init[:, :n, j].sum(axis=1)
    i_reg = list(exo_names).index(regime_shock) if regime_shock in list(exo_names) else (
        list(exo_names).index("EPS_REGIME") if "EPS_REGIME" in list(exo_names) else None)
    if i_reg is not None:
        deco_p[:, i_reg, :] = reg_p
    else:
        att_p = att_p + reg_p
    deco_p[:, nexo] = att_p
    deco_p[:, nexo + 1] = as_p
    decomp = deco_p.copy()
    if i_reg is None:
        deco_p[:, nexo] = att_p - reg_p
    rr = np.abs(deco_p[:, :-1, :])
    if i_reg is not None:
        rr[:, i_reg, :] = 0
    tot = rr.sum(axis=1, keepdims=True)
    tot[tot < 1e-10] = 1
    ww = rr / tot
    wdeco = deco_p.copy()
    if i_reg is not None:
        wdeco[:, i_reg, :] = 0
    for k in range(L):
        if np.any(reg_p[:, k]):
            wdeco[:, :-1, k] = wdeco[:, :-1, k] + reg_p[:, k][:, None] * ww[:, :, k]
    cols = list(exo_names) + ["initial_conditions", "smoothed"]
    icols = list(endo_names) + ["shocks", "smoothed"]
    mk = lambda z, c, kind: ShockDecomposition(z=z, endo_names=list(endo_names), columns=c, kind=kind,  # noqa: E731
                                               steady_state=sm.ys, first_period=int(tinit))
    return {"decomp": mk(decomp, cols, "shocks"), "wdecomp": mk(wdeco, cols, "shocks"),
            "init_decomp": mk(deco_p_init, icols, "initial_conditions"),
            "linear": mk(deco, cols, "shocks"), "linear_init": mk(deco_init, icols, "initial_conditions")}


def squeeze_occbin_shock_decomposition(decomps: dict, var_list) -> dict:
    """occbin/squeeze_shock_decomposition.m: keep the listed variables in every decomposition."""
    from .smoother import squeeze_shock_decomposition
    return {k: squeeze_shock_decomposition(v, var_list) for k, v in decomps.items()}


# ====================================================================== regime output
def regime_table(regime_history, periods=None, dynare_indexing=False):
    """The table written by occbin_write_regimes (write_regimes_to_xls.m) as a list of dicts:
    ``time``, ``regime_sequence`` and ``starting_period_of_regime`` (with suffixes ``_1``, ``_2`` for
    two constraints), the sequences written as MATLAB's ``int2str`` does ("0  1  0").

    ``periods`` (1-based, default all) selects the rows. Dynare 7.2 labels the rows with ``periods``
    but fills them with the first ``len(periods)`` regimes (it indexes ``regime_history(tp)`` with
    the row number): ``dynare_indexing=True`` reproduces that; by default row ``t`` holds the regime
    of period ``t``."""
    rh = list(regime_history)
    T = list(range(1, len(rh) + 1)) if periods is None else [int(p) for p in np.atleast_1d(periods)]
    k = len(rh[0]) if rh else 1
    rows = []
    for tp, t in enumerate(T):
        reg = rh[tp] if dynare_indexing else rh[t - 1]
        row = {"time": t}
        for j, (r, s) in enumerate(reg):
            suf = "" if k == 1 else f"_{j + 1}"
            row[f"regime_sequence{suf}"] = "  ".join(str(int(x)) for x in r)
            row[f"starting_period_of_regime{suf}"] = "  ".join(str(int(x)) for x in s)
        rows.append(row)
    return rows


def write_regime_table(path, regime_history, periods=None, dynare_indexing=False):
    """Write :func:`regime_table` as a CSV file (instead of Dynare's .xls)."""
    import csv
    rows = regime_table(regime_history, periods, dynare_indexing)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["time"])
        w.writeheader()
        w.writerows(rows)
    return rows


def plot_data(sm: OccbinSmootherResult, endo_names, var_list=None, levels=True):
    """Plotting-friendly paths: {variable: {"smoothed", "linear_smoothed", "updated", "filtered",
    "binding"}} with arrays of length T (levels if ``levels``, else deviations), and ``binding`` the
    (T, n_constraints) 0/1 indicator of the binding constraints in each period."""
    names = list(endo_names)
    idx = [names.index(v) for v in (names if var_list is None else var_list)]
    add = sm.ys if levels and sm.ys is not None else np.zeros(sm.smoothed.shape[1])
    T = sm.smoothed.shape[0]
    bind = regime_indicator(sm.regime_history, T)
    out = {}
    for i in idx:
        d = {"smoothed": sm.smoothed[:, i] + add[i], "updated": sm.updated[:, i] + add[i],
             "filtered": sm.filtered[:, i] + add[i], "binding": bind}
        if sm.linear_smoothed is not None:
            d["linear_smoothed"] = sm.linear_smoothed[:, i] + add[i]
        out[names[i]] = d
    return out
