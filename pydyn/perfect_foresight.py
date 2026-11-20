# This file is part of pydyn.
#
# It is a Python port of parts of Dynare 7.2 (Copyright (C) 1996-2026 Dynare Team,
# https://www.dynare.org), modified in 2026:
#   matlab/perfect-foresight-models/perfect_foresight_setup.m, make_ex_.m, make_y_.m,
#   perfect_foresight_solver.m (homotopy loop, scenario construction, linearization fallbacks),
#   perfect_foresight_solver_core.m, sim1.m, sim1_linear.m, linear_perfect_foresight_problem.m,
#   sim1_purely_backward.m, sim1_purely_forward.m, sim1_purely_static.m, solve_stacked_problem.m,
#   perfect_foresight_mcp_problem.m, compute_maxerror.m,
#   perfect_foresight_with_expectation_errors_setup.m, perfect_foresight_with_expectation_errors_solver.m,
#   and the way the preprocessor executes initval, endval, histval, shocks and mshocks
#   (preprocessor/src/NumericalInitialization.cc, Shocks.cc, ModelTree.cc for the ordering of
#   complementarity conditions), all GPL-3.0-or-later;
#   matlab/lmmcp/lmmcp.m (Copyright (C) 2005 Christian Kanzow and Stefania Petra, 2013 Christophe
#   Gouel, 2014-2022 Dynare Team: "Unlimited permission is granted to everyone to use, copy, modify
#   or distribute this software").
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Deterministic (perfect foresight) simulation.

``perfect_foresight(model)`` does what ``perfect_foresight_setup`` followed by
``perfect_foresight_solver`` do in Dynare 7.2 (or the ``perfect_foresight_with_expectation_errors``
pair when the .mod file uses it), reading initial and terminal conditions, shocks and options
from the .mod file:

* ``initval``, ``endval``, ``steady``, ``histval``, ``shocks`` (periods/values), ``mshocks``,
  ``shocks(learnt_in=...)``, ``endval(learnt_in=...)`` are replayed in the order of the file, so
  that the content of Dynare's ``oo_.steady_state``, ``oo_.exo_steady_state``,
  ``oo_.initial_steady_state`` at ``perfect_foresight_setup`` is reproduced. In particular,
  without ``steady`` after ``initval`` the path starts from (and, without ``endval``, ends at)
  the values written in ``initval``, which need not be a steady state.
* Solvers: Newton with a sparse LU factorisation on the stacked system (``sim1``), one linear solve
  for ``model(linear)`` / ``linear_approximation``, period-by-period solvers for purely backward,
  purely forward and static models, Dynare 7's homotopy on the size of the shocks with its two
  linearization fallbacks, and the LMMCP algorithm of Kanzow and Petra for models with
  complementarity conditions (``perfect_foresight_solver(lmmcp)``).

Layout of the result: ``endo`` has one row per period and ``maximum_lag + periods + maximum_lead``
rows, i.e. it is the transpose of Dynare's ``oo_.endo_simul``: with lags and leads row 0 is the
initial condition (period 0), rows 1..periods the simulation and the last row the terminal
condition. A model without lags has no initial row, a model without leads no terminal row. Leads
and lags beyond one period are carried by the auxiliary variables the preprocessor adds (extra
columns, not extra rows), as in Dynare. ``exo`` has the same rows.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, sparse
from scipy.sparse.linalg import splu

from . import options as _opt
from .expr import NAMESPACE
from .steady import DYNARE_TOLF, fill_auxiliary, steady_state

#: "vector": residuals and Jacobian of all periods in one vectorised numpy evaluation, assembled
#: on a fixed sparsity pattern; "loop": one evaluation of the generated functions per period
#: (the fallback used automatically when a model function cannot be evaluated on arrays).
#: Environment variable PYDYN_PF_STACKED.
STACKED = os.environ.get("PYDYN_PF_STACKED", "vector")

#: column ordering SuperLU uses for the stacked Jacobian. The matrix is block tridiagonal, so the
#: natural order keeps the fill inside the band and saves the ordering step: measured 3 times faster
#: than COLAMD on a 40-variable model with 1000 periods, the same on a 198-variable model with 200
#: periods. Environment variable PYDYN_PF_LU (NATURAL, COLAMD, MMD_AT_PLUS_A).
LU_ORDERING = os.environ.get("PYDYN_PF_LU", "NATURAL")

#: pydyn's default tolerance on the residuals when neither the .mod file nor the caller gives
#: ``tolf`` (Dynare's default is 1e-5)
DEFAULT_TOLF = 1e-9


class PerfectForesightError(Exception):
    """Raised when no solution is found; ``result`` holds the last iterate (status False)."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


@dataclass
class PerfectForesightResult:
    """Outcome of a deterministic simulation (see the module docstring for the layout)."""
    endo: np.ndarray                 # (maximum_lag + periods + maximum_lead, n_endo): oo_.endo_simul'
    exo: np.ndarray                  # same rows, n_exo columns: oo_.exo_simul
    iterations: int = 0              # oo_.deterministic_simulation.iterations (see perfect_foresight)
    max_residual: float = np.nan     # oo_.deterministic_simulation.error
    status: bool = True              # oo_.deterministic_simulation.status
    endo_names: list = field(default_factory=list)
    exo_names: list = field(default_factory=list)
    periods: int = 0
    maximum_lag: int = 1
    maximum_lead: int = 1
    steady_state: np.ndarray | None = None            # oo_.steady_state (terminal)
    initial_steady_state: np.ndarray | None = None    # oo_.initial_steady_state (None without endval)
    exo_steady_state: np.ndarray | None = None
    initial_exo_steady_state: np.ndarray | None = None
    params: np.ndarray | None = None
    homotopy_completion_share: float = 1.0
    homotopy_iterations: int = 0
    homotopy_linearization: bool = False
    homotopy_marginal_linearization: bool = False
    solver: str = ""
    info: dict = field(default_factory=dict)

    def __getitem__(self, name):
        if name in self.endo_names:
            return self.endo[:, self.endo_names.index(name)]
        return self.exo[:, self.exo_names.index(name)]

    def paths(self, deviation=None, exo=False, simulation_only=False) -> dict:
        """Dict name -> path. ``deviation``: None (levels), "initial" or "terminal" (difference
        from that steady state, i.e. Dynare's ``oo_.initial_steady_state`` or ``oo_.steady_state``;
        without ``endval`` both are the same). ``exo=True`` adds the exogenous variables,
        ``simulation_only=True`` drops the initial and terminal rows."""
        if deviation not in (None, "initial", "terminal"):
            raise ValueError("deviation must be None, 'initial' or 'terminal'")
        rows = slice(self.maximum_lag, self.maximum_lag + self.periods) if simulation_only else slice(None)
        base = bx = 0.0
        if deviation is not None:
            initial = deviation == "initial" and self.initial_steady_state is not None
            base = self.initial_steady_state if initial else self.steady_state
            bx = self.initial_exo_steady_state if initial else self.exo_steady_state
        y = (self.endo - base)[rows]
        out = {v: y[:, i] for i, v in enumerate(self.endo_names)}
        if exo:
            x = (self.exo - bx)[rows]
            out.update({v: x[:, i] for i, v in enumerate(self.exo_names)})
        return out

    def deviations(self, from_="initial", **kw) -> dict:
        return self.paths(deviation=from_, **kw)


# ----------------------------------------------------------------------------------- options
_SOLVER_KEYS = {
    "tolf": "dynatol.f", "tolx": "dynatol.x", "maxit": "simul.maxit", "stack_solve_algo": "stack_solve_algo",
    "no_homotopy": "no_homotopy", "linear_approximation": "linear_approximation",
    "endogenous_terminal_period": "endogenous_terminal_period", "lmmcp": "lmmcp.status",
    "homotopy_initial_step_size": "simul.homotopy_initial_step_size",
    "homotopy_min_step_size": "simul.homotopy_min_step_size",
    "homotopy_step_size_increase_success_count": "simul.homotopy_step_size_increase_success_count",
    "homotopy_max_completion_share": "simul.homotopy_max_completion_share",
    "homotopy_linearization_fallback": "simul.homotopy_linearization_fallback",
    "homotopy_marginal_linearization_fallback": "simul.homotopy_marginal_linearization_fallback",
    "homotopy_exclude_varexo": "simul.homotopy_exclude_varexo",
    "steady_tolf": "simul.steady_tolf", "constant_simulation_length": "pfwee.constant_simulation_length",
    "noprint": "noprint",
}
_SETUP_KEYS = {"periods": "periods", "endval_steady": "simul.endval_steady",
               "endval_steady_nocheck": "simul.endval_steady_nocheck", "datafile": "datafile"}
_DEFAULTS = {
    "dynatol.f": None, "dynatol.x": 1e-5, "simul.maxit": 50, "stack_solve_algo": 0, "no_homotopy": False,
    "linear_approximation": False, "endogenous_terminal_period": False, "lmmcp.status": False,
    "simul.homotopy_initial_step_size": 1.0, "simul.homotopy_min_step_size": 1e-3,
    "simul.homotopy_step_size_increase_success_count": 3, "simul.homotopy_max_completion_share": 1.0,
    "simul.homotopy_linearization_fallback": False, "simul.homotopy_marginal_linearization_fallback": 0.0,
    "simul.homotopy_exclude_varexo": [], "simul.steady_tolf": None, "pfwee.constant_simulation_length": False,
    "noprint": False, "periods": None, "simul.endval_steady": False, "simul.endval_steady_nocheck": False,
    "datafile": None,
}
#: options that Dynare itself only uses with bytecode/block or other linear solvers, with the reason
_NOTES = {
    "markowitz": "markowitz only matters for Dynare's bytecode solvers",
    "minimal_solving_periods": "minimal_solving_periods only matters for Dynare's bytecode solvers",
    "simul.robust_lin_solve": "robust_lin_solve (iterative fallbacks for singular stacked Jacobians) is not "
                              "implemented: a singular Jacobian makes the simulation fail",
    "simul.steady_solve_algo": "steady_solve_algo: pydyn has one steady-state solver",
    "simul.steady_maxit": "steady_maxit: pydyn has one steady-state solver",
    "simul.steady_tolx": "steady_tolx: pydyn has one steady-state solver",
    "simul.steady_markowitz": "steady_markowitz only matters for Dynare's bytecode solvers",
    "solve_algo": "solve_algo: period-by-period problems are solved by Newton's method (then scipy's hybr)",
}


def _resolve_options(model, pfwee, kw):
    names = (("perfect_foresight_with_expectation_errors_setup", "perfect_foresight_with_expectation_errors_solver")
             if pfwee else ("perfect_foresight_setup", "perfect_foresight_solver"))
    setup, solver = (_statement_options(model, n) for n in names)
    o = dict(_DEFAULTS)
    o.update({k: v for k, v in setup.items() if k in _DEFAULTS})
    o.update({k: v for k, v in solver.items() if k in _DEFAULTS})
    given = set(setup) | set(solver)
    for k, v in kw.items():
        key = _SOLVER_KEYS.get(k) or _SETUP_KEYS.get(k)
        if key is None:
            raise TypeError(f"perfect_foresight() got an unexpected keyword argument '{k}'")
        if v is not None:
            o[key] = v
            given.add(key)
    excl = o["simul.homotopy_exclude_varexo"]
    if isinstance(excl, dict):
        excl = excl.get("symbol_list", [])
    o["simul.homotopy_exclude_varexo"] = [excl] if isinstance(excl, str) else list(excl or [])
    if o["dynatol.f"] is None:
        o["dynatol.f"] = DEFAULT_TOLF
    o["dynatol.f"], o["dynatol.x"], o["simul.maxit"] = float(o["dynatol.f"]), float(o["dynatol.x"]), int(o["simul.maxit"])
    for k in ("simul.homotopy_initial_step_size", "simul.homotopy_min_step_size", "simul.homotopy_max_completion_share"):
        o[k] = float(o[k])
    m = o["simul.homotopy_marginal_linearization_fallback"]
    o["simul.homotopy_marginal_linearization_fallback"] = 0.01 if m is True else float(m or 0.0)
    # reports (once per model) for options written in the .mod file that are not acted upon
    algo = int(o["stack_solve_algo"])
    if "stack_solve_algo" in given:
        if not 0 <= algo <= 10:
            raise ValueError("stack_solve_algo must be between 0 and 10")
        _opt.warn_once(model, f"stack_solve_algo={algo}: pydyn has one solver for the stacked system, Newton's method with a "
                              "sparse LU factorisation (Dynare's stack_solve_algo=0, with SuperLU instead of UMFPACK); "
                              "every value of stack_solve_algo is solved that way", stacklevel=4)
    for st, src in zip(names, (setup, solver)):
        notes = [f"{k} ({_NOTES[k]})" for k in src if k in _NOTES]
        other = sorted(k for k in src if k not in _DEFAULTS and k not in _NOTES and k not in _opt.COSMETIC
                       and k != "stack_solve_algo")
        if notes or other:
            _opt.warn_once(model, f"perfect_foresight does not act on these {st} options from the .mod file: "
                                  + "; ".join(notes + other), stacklevel=4)
    return o


def _statement_options(model, name):
    """Options of the last statement ``name`` of the .mod file."""
    out = {}
    for st in model.statements:
        if st.get("statementName") == name:
            out = st.get("options", {}) or {}
    return dict(out)


def _check_homotopy_options(o):
    """Argument checks of perfect_foresight_solver.m."""
    init, mn, mx = (o["simul.homotopy_initial_step_size"], o["simul.homotopy_min_step_size"],
                    o["simul.homotopy_max_completion_share"])
    lin, marg = o["simul.homotopy_linearization_fallback"], o["simul.homotopy_marginal_linearization_fallback"]
    if o["no_homotopy"] and (init != 1 or mx != 1 or lin or marg != 0):
        raise ValueError("the no_homotopy option is incompatible with homotopy_initial_step_size, "
                         "homotopy_max_completion_share, homotopy_linearization_fallback and "
                         "homotopy_marginal_linearization_fallback")
    for nm, v in (("homotopy_initial_step_size", init), ("homotopy_min_step_size", mn),
                  ("homotopy_max_completion_share", mx), ("homotopy_marginal_linearization_fallback", marg)):
        if not 0 <= v <= 1:
            raise ValueError(f"{nm} must be in the [0,1] interval")
    if init < mn:
        raise ValueError("homotopy_initial_step_size must be greater than or equal to homotopy_min_step_size")
    if lin and marg > 0:
        raise ValueError("homotopy_linearization_fallback and homotopy_marginal_linearization_fallback "
                         "cannot be used together")
    if mx < 1 and not lin and marg == 0:
        raise ValueError("homotopy_max_completion_share is less than 1, so homotopy_linearization_fallback or "
                         "homotopy_marginal_linearization_fallback must also be given")


# ----------------------------------------------------------------------------------- setup
class _Workspace:
    """What Dynare holds when perfect_foresight_setup runs (oo_ and M_ fields used by it)."""

    def __init__(self, model, params):
        n, nx = model.n_endo, model.n_exo
        self.params = np.array(model.params if params is None else params, float)
        self.ss, self.xss = np.zeros(n), np.zeros(nx)          # oo_.steady_state, oo_.exo_steady_state
        self.ss0 = self.xss0 = None                            # oo_.initial_(exo_)steady_state
        self.det_shocks = []            # (exo index, first period, last period, value, type)
        self.learnt_shocks = []         # dicts: learnt_in, exo, p1, p2, type, value
        self.learnt_endval = []         # dicts: learnt_in, exo, type, value
        self.endo_histval = self.exo_histval = None
        self.dirty = True               # values set by initval/endval since the last steady
        self.pfwee = False
        self.has_setup = False


def _is_setup(st):
    return st.get("statementName") in ("perfect_foresight_setup", "perfect_foresight_with_expectation_errors_setup")


def _number(model, expr):
    try:
        return float(model.eval_expr(expr))
    except Exception:  # noqa: BLE001 - MATLAB matrices, workspace functions: the interpreter knows them
        return float(model._value(expr))


def _period(v, what):
    if isinstance(v, (int, np.integer)) or (isinstance(v, str) and re.fullmatch(r"\s*-?\d+\s*", v)):
        return int(v)
    raise NotImplementedError(f"{what} given as a date or an expression ({v!r}): dates are not supported")


def _replay(model, params=None, steady=None, extra_learnt_shocks=(), extra_learnt_endval=()):
    """Replay the statements of the .mod file up to the last perfect_foresight_setup."""
    sts = model.statements
    idx = [i for i, st in enumerate(sts) if _is_setup(st)]
    last = idx[-1] if idx else len(sts) - 1
    ws = _Workspace(model, params)
    ws.has_setup = bool(idx)
    force = (not idx) if steady is None else bool(steady)
    never = steady is False
    ie, ix = model._ie, model._ix
    counts = {}
    for st in sts[:last + 1]:
        if st.get("statementName") in ("initval", "endval"):      # endval(learnt_in=...) too: the loader mixes them
            for d in st.get("vals", []):
                counts[(st["statementName"], d["name"])] = counts.get((st["statementName"], d["name"]), 0) + 1

    def run_steady(nocheck=None):
        with np.errstate(all="ignore"):
            guess = fill_auxiliary(model, ws.ss, ws.xss, ws.params)
        ws.ss, ws.params = steady_state(model, ws.params, guess=guess, exo=ws.xss,
                                        check=None if nocheck is None else not nocheck)
        ws.dirty = False

    def set_values(st, stored):
        for d in st.get("vals", []):
            name = d["name"]
            # the values Dynare holds after the block, as computed by the model loader; a variable
            # set in several blocks is evaluated here (possible when the expression has no variable)
            if name in stored and counts.get((st["statementName"], name), 0) <= 1:
                val = float(stored[name])
            else:
                val = _number(model, d["value"])
            if name in ie:
                ws.ss[ie[name]] = val
            elif name in ix:
                ws.xss[ix[name]] = val
        # the driver the preprocessor writes sets the auxiliary variables right after the block
        with np.errstate(all="ignore"):
            ws.ss = fill_auxiliary(model, ws.ss, ws.xss, ws.params)
        ws.dirty = True

    for st in sts[:last + 1]:
        kind = st.get("statementName")
        if kind == "initval":
            set_values(st, model.initval)
        elif kind == "endval" and "learnt_in" in st:
            for d in st.get("vals", []):
                ws.learnt_endval.append(dict(learnt_in=_period(st["learnt_in"], "learnt_in"), exo=ix[d["name"]],
                                             type=d.get("type", "level"), value=_number(model, d["value"])))
        elif kind == "endval":
            if force and ws.dirty and ws.ss0 is None:
                run_steady()
            if ws.ss0 is None:
                ws.ss0, ws.xss0 = ws.ss.copy(), ws.xss.copy()
            set_values(st, model.endval)
        elif kind == "load_params_and_steady_state":
            for d in st.get("values", []):
                if d["name"] in ie and d["name"] in model.initval:
                    ws.ss[ie[d["name"]]] = model.initval[d["name"]]
                elif d["name"] in ix and d["name"] in model.initval:
                    ws.xss[ix[d["name"]]] = model.initval[d["name"]]
            ws.dirty = True
        elif kind == "steady" and not never:
            run_steady()
        elif kind == "histval":
            _histval(model, ws, st)
        elif kind in ("shocks", "mshocks") and "learnt_in" in st:
            lin = _period(st["learnt_in"], "learnt_in")
            if st.get("overwrite"):
                ws.learnt_shocks = [d for d in ws.learnt_shocks if d["learnt_in"] != lin]
            for d in st.get("learnt_shocks", []):
                for v in d["values"]:
                    ws.learnt_shocks.append(dict(learnt_in=lin, exo=ix[d["var"]], p1=_period(v["period1"], "period"),
                                                 p2=_period(v["period2"], "period"), type=v.get("type", "level"),
                                                 value=_number(model, v["value"])))
        elif kind in ("shocks", "mshocks") and not st.get("surprise"):
            if st.get("overwrite"):
                ws.det_shocks = []
            typ = "level" if kind == "shocks" else ("multiply_initial_steady_state" if st.get("relative_to_initval")
                                                    else "multiply_steady_state")
            for d in st.get("deterministic_shocks", []):
                if d["var"] not in ix:
                    raise NotImplementedError(f"shock on {d['var']}: deterministic exogenous variables are not supported")
                for v in d["values"]:
                    ws.det_shocks.append((ix[d["var"]], _period(v["period1"], "period"), _period(v["period2"], "period"),
                                          _number(model, v["value"]), typ))
        elif kind in ("initval_file", "histval_file", "perfect_foresight_controlled_paths", "shock_paths",
                      "smoother2histval"):
            raise NotImplementedError(f"{kind} is not implemented in pydyn.perfect_foresight")
        elif _is_setup(st):
            ws.pfwee = kind != "perfect_foresight_setup"
    if force and ws.dirty:
        run_steady()
    for d in extra_learnt_shocks:
        d = dict(d)
        p = d.pop("periods", None)
        if p is not None:
            d["p1"], d["p2"] = (p, p) if np.isscalar(p) else (p[0], p[-1])
        d["exo"] = ix[d.pop("var")] if "var" in d else d["exo"]
        d.setdefault("type", "level")
        ws.learnt_shocks.append(d)
    for d in extra_learnt_endval:
        d = dict(d)
        d["exo"] = ix[d.pop("var")] if "var" in d else d["exo"]
        d.setdefault("type", "level")
        ws.learnt_endval.append(d)
    return ws


_SIMPLE_AUX = re.compile(r"\s*([A-Za-z_]\w*)\s*(?:\(\s*(-?\d+)\s*\))?\s*")


def _histval(model, ws, st):
    """M_.endo_histval and M_.exo_histval as the histval block and dynamic_set_auxiliary_series
    build them. Variables the block does not mention are 0 in period 0 (not their steady state)."""
    info = model.info
    L = int(info.get("orig_maximum_lag_with_diffs_expanded", info.get("orig_maximum_lag", 1)))
    if info["maximum_lag"] == 0:
        return                                    # M_.endo_histval has no column: ignored by Dynare
    if L < 1:
        return
    series = {v: np.zeros(L) for v in model.orig_endo_names + model.exo_names}      # dates -L+1 .. 0
    for d in st.get("vals", []):
        lag = int(d["lag"])
        if not -L < lag <= 0:
            raise ValueError(f"histval: {d['name']}({lag}) is outside the lags of the model")
        if d["name"] in series:
            series[d["name"]][L - 1 + lag] = _number(model, d["value"])
    defs = {eq.get("lhs"): eq.get("rhs") for eq in model.raw.get("model", []) if isinstance(eq, dict)}
    for a in model.raw.get("aux_vars", []):
        name = model.endo_names[a["endo_index"] - 1]
        mo = _SIMPLE_AUX.fullmatch(str(defs.get(name, "")))
        if a.get("type") not in (0, 1, 2, 3) or mo is None or mo.group(1) not in series:
            raise NotImplementedError(f"histval with the auxiliary variable {name} = {defs.get(name)} is not supported")
        src, shift = series[mo.group(1)], int(mo.group(2) or 0)
        out = np.full(L, np.nan)
        for t in range(L):
            if 0 <= t + shift < L:
                out[t] = src[t + shift]
        series[name] = out
    h = np.array([series[v][-1] for v in model.endo_names])
    ws.endo_histval = np.where(np.isnan(h), 0.0, h)
    ws.exo_histval = np.array([series[v][-1] for v in model.exo_names])


def _make_paths(model, ws, periods):
    """make_ex_.m and make_y_.m: oo_.exo_simul and oo_.endo_simul (one row per period)."""
    lag, lead = model.info["maximum_lag"], model.info["maximum_lead"]
    rows = lag + periods + lead
    if ws.endo_histval is not None and ws.ss0 is not None:
        raise PerfectForesightError("histval and endval cannot be used simultaneously")
    x = np.tile(ws.xss, (rows, 1))
    y = np.tile(ws.ss, (rows, 1))
    if ws.ss0 is not None:
        x[:lag] = ws.xss0
        y[:lag] = ws.ss0
    if ws.endo_histval is not None and lag:
        x[:lag] = ws.exo_histval
        y[:lag] = ws.endo_histval
    msg = [model.exo_names[j] for j, p1, p2, _, _ in ws.det_shocks if p2 > periods or p1 > periods]
    if msg:
        raise PerfectForesightError("a shock is dated after the end of the simulation: at least one expected value for "
                                    f"{', '.join(sorted(set(msg)))} has been declared after the terminal period")
    for j, p1, p2, val, typ in ws.det_shocks:
        k1, k2 = p1 + lag - 1, p2 + lag - 1                      # 0-based rows
        if k1 < 0:
            raise PerfectForesightError("the model does not have lags, so you cannot set values for period 0")
        if typ == "multiply_steady_state":
            val = ws.xss[j] * val
        elif typ == "multiply_initial_steady_state":
            if ws.xss0 is None:
                raise PerfectForesightError("option relative_to_initval of mshocks cannot be used without an endval block")
            val = ws.xss0[j] * val
        x[k1:k2 + 1, j] = val
    return y, x


def exo_path(model, periods):
    """Exogenous path from initval/endval/histval and the deterministic ``shocks``/``mshocks``
    blocks (levels), as ``perfect_foresight_setup`` builds ``oo_.exo_simul``."""
    ws = _replay(model, steady=False)
    return _make_paths(model, ws, int(periods))[1]


# ----------------------------------------------------------------------------------- model functions
def _get_power_deriv_vec(x, p, k):
    """get_power_deriv for arrays (one element per period)."""
    dxp = x ** (p - k)
    for i in range(int(k)):
        dxp = dxp * (p - i)
    special = (np.abs(x) < 1e-12) & (p >= 0) & (k > p) & (np.abs(p - np.round(p)) < 1e-12)
    return np.where(special, 0.0, dxp) if np.any(special) else dxp


def _vector_functions(model):
    """The generated model functions bound to a numpy namespace that accepts one array per
    variable (inputs of shape (n, T)); None if they cannot be built."""
    fns = model.__dict__.get("_pf_vector_functions")
    if fns is None:
        try:
            ns = dict(NAMESPACE)
            ns["get_power_deriv"] = _get_power_deriv_vec
            ns.update(model.functions)
            ns["np"] = np
            exec(compile(model.source, f"<pydyn-vector:{model.json_dir}>", "exec"), ns)  # noqa: S102 - generated from the model
            fns = (ns["dynamic_resid"], ns["dynamic_jac"])
        except Exception:  # noqa: BLE001
            fns = False
        model.__dict__["_pf_vector_functions"] = fns
    return fns or None


class _Stacked:
    """Residuals and Jacobian of the stacked system F(y[t-1], y[t], y[t+1], x[t]) = 0, t = 1..T.

    ``eq_index`` reorders the equations within each period (complementarity problems)."""

    def __init__(self, model, params, ss, eq_index=None, stacked=None):
        self.model, self.params, self.ss = model, np.asarray(params, float), ss
        self.n = n = model.n_endo
        self.eq_index = None if eq_index is None else np.asarray(eq_index, np.intp)
        endo = model.jac_cols < 3 * n
        self.endo = endo
        self.er, self.ec = model.jac_rows[endo], model.jac_cols[endo]
        if self.eq_index is not None:
            pos = np.empty(n, np.intp)
            pos[self.eq_index] = np.arange(n)
            self.er = pos[self.er]
        self.mode = STACKED if stacked is None else stacked
        self.vector = _vector_functions(model) if self.mode == "vector" else None
        self._patterns = {}
        self.evaluations = 0

    def _pattern(self, T):
        p = self._patterns.get(T)
        if p is None:
            n = self.n
            tt = np.arange(T)[:, None]
            gr, gc = tt * n + self.er[None, :], (tt - 1) * n + self.ec[None, :]
            keep = (gc >= 0) & (gc < T * n)
            pat = sparse.csc_matrix((np.arange(1, keep.sum() + 1, dtype=float), (gr[keep], gc[keep])), shape=(T * n, T * n))
            ok = pat.nnz == keep.sum()            # no duplicate entries: the data vector maps one to one
            perm = pat.data.astype(np.intp) - 1 if ok else None
            p = self._patterns[T] = (keep, pat.indices.copy(), pat.indptr.copy(), perm, gr[keep], gc[keep])
        return p

    def _values(self, full, X, jac):
        """(res (T, n), vals (T, nnz) or None): vectorised if possible, else one call per period."""
        m, T = self.model, full.shape[0] - 2
        ss = self.ss
        if self.vector is not None:
            try:
                yy = np.concatenate([full[:-2].T, full[1:-1].T, full[2:].T])
                out = []
                for fn in self.vector[:2 if jac else 1]:
                    vals = fn(yy, X.T, self.params, ss)
                    a = np.empty((T, len(vals)))
                    for i, v in enumerate(vals):
                        a[:, i] = v
                    out.append(a)
                return out[0], (out[1] if jac else None)
            except (ValueError, TypeError, IndexError):
                self.vector = None                # e.g. a function that does not accept arrays
        res = np.empty((T, self.n))
        vals = np.empty((T, len(m.jac_rows))) if jac else None
        for t in range(T):
            yy = full[t:t + 3].ravel()
            res[t] = m.dynamic_resid(yy, X[t], self.params, ss)
            if jac:
                vals[t] = m.dynamic_jac_values(yy, X[t], self.params, ss)
        return res, vals

    def __call__(self, Y, y0, yT, X, jac=True):
        """``Y`` (T, n), ``X`` (T, n_exo). Returns (res of length T n, sparse Jacobian or None)."""
        T, n = Y.shape
        self.evaluations += 1
        full = np.vstack([y0[None, :], Y, yT[None, :]])
        with np.errstate(all="ignore"):
            res, vals = self._values(full, X, jac)
        if self.eq_index is not None:
            res = res[:, self.eq_index]
        if not jac:
            return res.ravel(), None
        return res.ravel(), self.assemble(vals, T)

    def assemble(self, vals, T):
        """Stacked Jacobian from the structural non-zeros ``vals`` (T, nnz) of each period."""
        n = self.n
        keep, indices, indptr, perm, gr, gc = self._pattern(T)
        v = vals[:, self.endo]
        if perm is not None:
            return sparse.csc_matrix((v[keep][perm], indices, indptr), shape=(T * n, T * n))
        return sparse.csc_matrix((v[keep], (gr, gc)), shape=(T * n, T * n))


def _lu_solve(A, b):
    """A \\ b with a sparse LU factorisation; non-finite values (singular matrix) are returned as such."""
    try:
        with np.errstate(all="ignore"):
            return splu(A.tocsc(), permc_spec=LU_ORDERING).solve(b)
    except (RuntimeError, ValueError):
        return np.full(b.shape, np.nan)


def _nan(n):
    return np.full(n, np.nan)


def _boundaries(model, endo, periods):
    lag, lead = model.info["maximum_lag"], model.info["maximum_lead"]
    y0 = endo[lag - 1].copy() if lag > 0 else _nan(model.n_endo)
    yT = endo[lag + periods].copy() if lead > 0 else _nan(model.n_endo)
    return y0, yT


def _maxerror(model, problem, endo, exo, periods):
    """compute_maxerror.m (the infinity norm keeps NaN, unlike max(abs()))."""
    lag = model.info["maximum_lag"]
    y0, yT = _boundaries(model, endo, periods)
    res, _ = problem(endo[lag:lag + periods], y0, yT, exo[lag:lag + periods], jac=False)
    return _norm_inf(res)


def _norm_inf(v):
    v = np.abs(v)
    return float(np.nan) if np.isnan(v).any() else (float(v.max()) if v.size else 0.0)


# ----------------------------------------------------------------------------------- solvers
def _sim1(model, problem, endo, exo, periods, o, log):
    """sim1.m: Newton's method on the stacked system."""
    lag, n = model.info["maximum_lag"], model.n_endo
    endo = endo.copy()
    y0, yT = _boundaries(model, endo, periods)
    Y = endo[lag:lag + periods].copy()
    X = exo[lag:lag + periods]
    T, tolf, maxit = periods, o["dynatol.f"], o["simul.maxit"]
    it, converged, err = 1, False, np.nan
    while not converged and it <= maxit:
        res, A = problem(Y, y0, yT, X[:T])
        if o["endogenous_terminal_period"] and it > 1:
            for k in range(T):
                if _norm_inf(res[k * n:(k + 1) * n]) < tolf / 1e7:
                    if k + 1 < T:
                        keep = (k + 1) * n
                        res, A = res[:keep], A[:keep, :keep]
                        yT = Y[k + 1].copy()
                        endo[lag + k + 1:lag + T] = Y[k + 1:]
                        Y, T = Y[:k + 1], k + 1
                    break
        err = _norm_inf(res)
        log(f"Iter: {it},\t err. = {err:g}")
        if err < tolf:
            converged = True
        else:
            Y = Y - _lu_solve(A, res).reshape(T, n)
        it += 1
    endo[lag:lag + T] = Y
    if o["endogenous_terminal_period"]:
        err = _maxerror(model, problem, endo, exo, periods)
    success = bool(converged and np.all(np.isfinite(Y)))
    return endo, success, err, it          # Dynare's count: residual evaluations + 1


def _sim1_linear(model, problem, endo, exo, periods, ss, xss, params, o, nocheck):
    """sim1_linear.m: one linear solve around the (terminal) steady state."""
    lag, n = model.info["maximum_lag"], model.n_endo
    y3 = np.tile(ss, 3)
    d1 = model.dynamic_resid(y3, xss, params, ss)
    if not nocheck and not np.max(np.abs(d1)) <= DYNARE_TOLF:
        raise PerfectForesightError("linear_approximation: the Jacobian is not evaluated at the steady state "
                                    f"(max |residual| at oo_.steady_state = {np.max(np.abs(d1)):.3e})")
    vals = model.dynamic_jac_values(y3, xss, params, ss)
    J = sparse.csr_matrix((vals, (model.jac_rows, model.jac_cols)), shape=(n, 3 * n + model.n_exo))
    dy, dx = endo - ss, exo - xss
    y0, yT = _boundaries(model, dy, periods)
    y0, yT = np.nan_to_num(y0), np.nan_to_num(yT)        # absent lag/lead columns are structural zeros

    def residuals(Y):
        full = np.vstack([y0[None, :], Y, yT[None, :]])
        z = np.hstack([full[:-2], full[1:-1], full[2:], dx[lag:lag + periods]])
        return (J @ z.T).T.ravel()

    Y = dy[lag:lag + periods].copy()
    res = residuals(Y)
    A = problem.assemble(np.tile(vals, (periods, 1)), periods)
    Y = Y - _lu_solve(A, res).reshape(periods, n)
    err = _norm_inf(residuals(Y))
    success = bool(np.all(np.isfinite(Y)) and np.isfinite(err))
    out = endo.copy()
    out[lag:lag + periods] = Y + ss
    return out, success, err, 1


def _solve_period(model, guess, fixed_lag, fixed_lead, x, params, ss, o):
    """One period of a purely backward, purely forward or static model: dynare_solve's
    convention that a guess whose residuals are already below tolf is returned unchanged, then
    Newton's method, then scipy's hybrid Powell method."""
    n = model.n_endo
    tolf, maxit = o["dynatol.f"], o["simul.maxit"]

    def stack(z):
        return np.concatenate([fixed_lag, z, fixed_lead])

    def f(z):
        with np.errstate(all="ignore"):
            return model.dynamic_resid(stack(z), x, params, ss)

    def jac(z):
        with np.errstate(all="ignore"):
            return model.dynamic_jac(stack(z), x, params, ss)[:, n:2 * n]

    z = np.array(guess, float)
    r = f(z)
    if np.all(np.isfinite(r)) and np.max(np.abs(r)) < tolf:
        return z, True
    for _ in range(maxit):
        if not np.all(np.isfinite(r)):
            break
        try:
            z = z - np.linalg.solve(jac(z), r)
        except np.linalg.LinAlgError:
            break
        r = f(z)
        if np.all(np.isfinite(r)) and np.max(np.abs(r)) < tolf:
            return z, True
    sol = optimize.root(f, np.array(guess, float), jac=jac, method="hybr", tol=min(tolf, 1e-10))
    r = f(sol.x)
    ok = bool(np.all(np.isfinite(r)) and np.max(np.abs(r)) < tolf)
    return (sol.x if ok else np.array(guess, float)), ok


def _sim1_recursive(model, endo, exo, periods, params, ss, o):
    """sim1_purely_backward.m, sim1_purely_forward.m, sim1_purely_static.m."""
    n, lag = model.n_endo, model.info["maximum_lag"]
    endo = endo.copy()
    nan = _nan(n)
    backward = model.info["maximum_endo_lag"] > 0
    forward = model.info["maximum_endo_lead"] > 0
    success = True
    if backward:
        for it in range(lag, lag + periods):
            y = endo[it - 1]
            endo[it], ok = _solve_period(model, y, y, nan, exo[it], params, ss, o)
            if not ok:
                success = False
                break
    elif forward:
        for it in range(lag + periods - 1, lag - 1, -1):
            yf = endo[it + 1]
            endo[it], ok = _solve_period(model, yf, nan, yf, exo[it], params, ss, o)
            if not ok:
                success = False
                break
    else:
        y = endo[0]
        for it in range(periods):
            y, ok = _solve_period(model, y, nan, nan, exo[it], params, ss, o)
            success = success and ok
            endo[it] = y
    return endo, success


# ----------------------------------------------------------------------------------- LMMCP
def complementarity_conditions(model, params=None):
    """(lower bounds, upper bounds, equation ordering) from the complementarity conditions of the
    model (``[mcp = 'r > 0']`` tags or the ``⟂`` syntax), as Dynare's
    ``dynamic_complementarity_conditions`` and ``M_.dynamic_mcp_equations_reordering``.
    Equation ``eq_index[i]`` is paired with variable ``i``."""
    n = model.n_endo
    lb, ub = np.full(n, -np.inf), np.full(n, np.inf)
    order = list(range(n))
    saved = model.params
    try:
        if params is not None:
            model.params = np.asarray(params, float)
        eqs = [eq for eq in model.raw.get("model", []) if isinstance(eq, dict)]
        for i, eq in enumerate(eqs):
            if "mcp" in (eq.get("tags") or {}) and not eq.get("complementarity_condition"):
                raise NotImplementedError("an [mcp=...] tag the preprocessor did not turn into a complementarity "
                                          "condition")
            cc = eq.get("complementarity_condition")
            if not cc:
                continue
            if len(eqs) != n or cc["variable"] not in model._ie:
                raise NotImplementedError("complementarity condition on something else than an endogenous variable")
            v = model._ie[cc["variable"]]
            if cc.get("lower_bound") not in (None, ""):
                lb[v] = model.eval_expr(cc["lower_bound"])
            if cc.get("upper_bound") not in (None, ""):
                ub[v] = model.eval_expr(cc["upper_bound"])
            j = order.index(i)
            order[v], order[j] = order[j], order[v]
    finally:
        model.params = saved
    return lb, ub, np.array(order, np.intp)


def has_complementarity(model):
    return any(isinstance(eq, dict) and (eq.get("complementarity_condition") or "mcp" in (eq.get("tags") or {}))
               for eq in model.raw.get("model", []))


def _lmmcp_phi(x, F, lb, ub, l1, l2, idx):
    n = len(x)
    y = np.zeros(2 * n)
    with np.errstate(all="ignore"):
        phi_u = np.sqrt((ub - x) ** 2 + F ** 2) - ub + x + F
        I = idx == 0
        y[:n][I] = -l1 * F[I]
        y[n:][I] = -l2 * F[I]
        I = idx == 1
        y[:n][I] = l1 * (-x[I] + lb[I] - F[I] + np.sqrt((x[I] - lb[I]) ** 2 + F[I] ** 2))
        y[n:][I] = l2 * np.maximum(0, x[I] - lb[I]) * np.maximum(0, F[I])
        I = idx == 2
        y[:n][I] = -l1 * phi_u[I]
        y[n:][I] = l2 * np.maximum(0, ub[I] - x[I]) * np.maximum(0, -F[I])
        I = idx == 3
        y[:n][I] = l1 * (np.sqrt((x[I] - lb[I]) ** 2 + phi_u[I] ** 2) - x[I] + lb[I] - phi_u[I])
        y[n:][I] = l2 * (np.maximum(0, x[I] - lb[I]) * np.maximum(0, F[I])
                         + np.maximum(0, ub[I] - x[I]) * np.maximum(0, -F[I]))
    return y


def _lmmcp_dphi(x, F, DF, lb, ub, l1, l2, idx):
    """An element of the C-subdifferential of Phi (subfunction DPhi of lmmcp.m)."""
    n, null = len(x), 1e-8
    with np.errstate(all="ignore"):
        beta_l = (np.abs(x - lb) <= null) & (np.abs(F) <= null)
        beta_u = (np.abs(ub - x) <= null) & (np.abs(F) <= null)
        z = (beta_l | beta_u).astype(float)
        alpha_l = (x - lb >= -null) & (F >= -null)
        alpha_u = (ub - x >= -null) & (F <= null)
        DFz = DF @ z
        Da, Db = np.zeros(n), np.zeros(n)
        s, e = np.zeros(n), np.zeros(n)                      # H2 = diag(s) DF + diag(e)
        I0, I1, I2, I3 = idx == 0, idx == 1, idx == 2, idx == 3
        Db[I0] = -1
        s[I0] = -1
        denz = np.maximum(null, np.sqrt(z ** 2 + DFz ** 2))
        if I1.any():
            den1 = np.maximum(null, np.sqrt((x - lb) ** 2 + F ** 2))
            a = I1 & ~beta_l
            Da[a], Db[a] = (x[a] - lb[a]) / den1[a] - 1, F[a] / den1[a] - 1
            b = I1 & beta_l
            Da[b], Db[b] = z[b] / denz[b] - 1, DFz[b] / denz[b] - 1
            a = I1 & alpha_l
            s[a], e[a] = x[a] - lb[a], F[a]
        if I2.any():
            den1 = np.maximum(null, np.sqrt((ub - x) ** 2 + F ** 2))
            a = I2 & ~beta_u
            Da[a], Db[a] = (ub[a] - x[a]) / den1[a] - 1, -F[a] / den1[a] - 1
            b = I2 & beta_u
            Da[b], Db[b] = -z[b] / denz[b] - 1, -DFz[b] / denz[b] - 1
            a = I2 & alpha_u
            s[a], e[a] = x[a] - ub[a], F[a]
        if I3.any():
            phi = -ub + x + F + np.sqrt((ub - x) ** 2 + F ** 2)
            den1 = np.maximum(null, np.sqrt((x - lb) ** 2 + phi ** 2))
            den3 = np.maximum(null, np.sqrt((ub - x) ** 2 + F ** 2))
            den4 = np.maximum(null, np.sqrt(z ** 2))
            ai, bi, ci, di = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
            a = I3 & ~beta_u
            ci[a], di[a] = (x[a] - ub[a]) / den3[a] + 1, F[a] / den3[a] + 1
            b = I3 & beta_u
            ci[b], di[b] = 1 + z[b] / denz[b], 1 + DFz[b] / denz[b]
            a = I3 & ~beta_l
            ai[a], bi[a] = (x[a] - lb[a]) / den1[a] - 1, phi[a] / den1[a] - 1
            b = I3 & beta_l
            ai[b], bi[b] = z[b] / den4[b] - 1, (ci[b] * z[b] + di[b] * DFz[b]) / den4[b] - 1
            Da[I3], Db[I3] = ai[I3] + bi[I3] * ci[I3], bi[I3] * di[I3]
            a = I3 & alpha_l & alpha_u
            s[a], e[a] = 2 * x[a] - lb[a] - ub[a], 2 * F[a]
            a = I3 & alpha_l & ~alpha_u
            s[a], e[a] = x[a] - lb[a], F[a]
            a = I3 & ~alpha_l & alpha_u
            s[a], e[a] = x[a] - ub[a], F[a]
    H1 = sparse.diags(Db) @ DF + sparse.diags(Da)
    H2 = sparse.diags(s) @ DF + sparse.diags(e)
    return sparse.vstack([l1 * H1, l2 * H2]).tocsc()


def _least_squares(H, rhs, mu=None):
    """min |H d - rhs| (H is 2n x n). Small problems (n < 100, as in lmmcp.m) use a dense
    regularised least-squares solve; large ones the normal equations with one refinement step."""
    n = H.shape[1]
    with np.errstate(all="ignore"):
        if mu is not None:
            M = np.vstack([H.toarray(), np.sqrt(mu) * np.eye(n)])
            if not np.all(np.isfinite(M)):
                return np.full(n, np.nan)
            return np.linalg.lstsq(M, np.concatenate([rhs, np.zeros(n)]), rcond=None)[0]
        N = (H.T @ H).tocsc()
        try:
            lu = splu(N)
            d = lu.solve(H.T @ rhs)
            d = d + lu.solve(H.T @ (rhs - H @ d))
        except (RuntimeError, ValueError):
            d = np.full(n, np.nan)
        if not np.all(np.isfinite(d)):
            d = sparse.linalg.lsqr(H, rhs, atol=1e-14, btol=1e-14, iter_lim=20 * n)[0]
        return d


def lmmcp(fun, x, lb, ub, tolfun=None, maxiter=500, log=None):
    """Port of Dynare's ``lmmcp.m`` (Kanzow and Petra's semismooth least-squares method for mixed
    complementarity problems, with the projected Levenberg-Marquardt preprocessor).

    ``fun(x, jac)`` returns ``(F, DF)`` (``DF`` sparse, or None when ``jac`` is False). Solves
    lb = x => F >= 0, lb < x < ub => F = 0, x = ub => F <= 0. Returns (x, F, exitflag, iterations)
    with exitflag 1 when the merit function 0.5 |Phi|^2 is below 0.5 tolfun^2."""
    beta, Big, delta, deltamin, eta, kwatch, lambda1, m = 0.55, 1e10, 5.0, 1.0, 0.95, 20, 0.1, 10
    null, presteps, sigma, sigma1, sigma2, tmin = 1e-10, 20, 1e-4, 0.5, 2.0, 1e-12
    tolfun = float(np.sqrt(np.finfo(float).eps)) if tolfun is None else float(tolfun)
    eps2, lambda2 = 0.5 * tolfun ** 2, 1 - lambda1
    log = log or (lambda s: None)
    x = np.maximum(lb, np.minimum(np.asarray(x, float), ub))
    n = len(x)
    idx = np.zeros(n, int)
    idx[(lb > -Big) & (ub > Big)] = 1
    idx[(lb < -Big) & (ub < Big)] = 2
    idx[(lb > -Big) & (ub < Big)] = 3
    small = n < 100

    def phi(x, F):
        return _lmmcp_phi(x, F, lb, ub, lambda1, lambda2, idx)

    def dphi(x, F, DF):
        return _lmmcp_dphi(x, F, DF, lb, ub, lambda1, lambda2, idx)

    def psi(P):
        v = 0.5 * float(P @ P)
        return v if np.isfinite(v) else np.inf

    def direction(DPhi, Phi, k, scale):
        mu = None
        if small:
            mu = 1e-16
            with np.errstate(all="ignore"):
                N = (DPhi.T @ DPhi).toarray()
                if not np.all(np.isfinite(N)) or np.linalg.cond(N, 1) > 1e25:
                    mu = scale / (k + 1)
        return _least_squares(DPhi, -Phi, mu)

    Fx, DFx = fun(x, True)
    Phix = phi(x, Fx)
    normPhix, Psix = float(np.linalg.norm(Phix)), psi(Phix)
    DPhix = dphi(x, Fx, DFx)
    DPsix = DPhix.T @ Phix
    start = (x.copy(), Fx, DFx, Phix, Psix, DPhix, DPsix)
    k = k_main = 0
    aux = np.zeros(m)
    aux[0] = Psix
    MaxPsi = Psix
    best = (k, x.copy(), Fx, Phix, Psix, DPhix, DPsix)
    # ---- preprocessor: projected Levenberg-Marquardt steps
    normp = 1.0
    while k < presteps and Psix > eps2 and normp > null:
        k += 1
        p = direction(DPhix, Phix, k, 1e-6)
        normp = float(np.linalg.norm(p))
        if not np.isfinite(normp):
            break
        lbnew = np.maximum(np.minimum(lb - x, 0), -delta)
        ubnew = np.minimum(np.maximum(ub - x, 0), delta)
        d = np.maximum(lbnew, np.minimum(p, ubnew))
        x = x + d
        Fx, DFx = fun(x, True)
        Phinew = phi(x, Fx)
        normnew = float(np.linalg.norm(Phinew))
        if normnew <= eta * normPhix:
            delta = max(deltamin, sigma2 * delta)
        elif normnew > 5 * eta * normPhix:
            delta = max(deltamin, sigma1 * delta)
        Phix, Psix, normPhix = Phinew, psi(Phinew), normnew
        DPhix = dphi(x, Fx, DFx)
        DPsix = DPhix.T @ Phix
        log(f"{k:4d} {Psix:24.5e} (preprocessor)")
    if Psix < eps2:
        return x, Fx, 1, k
    x, Fx, DFx, Phix, Psix, DPhix, DPsix = start
    x = x.copy()
    # ---- main algorithm: Levenberg-Marquardt direction with a nonmonotone Armijo rule
    while k < maxiter and Psix > eps2:
        d = direction(DPhix, Phix, k, 1e-1)
        if not np.all(np.isfinite(d)):
            break
        t = 1.0
        xnew = x + d
        Fnew, _ = fun(xnew, False)
        Phinew = phi(xnew, Fnew)
        Psinew = psi(Phinew)
        const = sigma * float(DPsix @ d)
        while not Psinew <= MaxPsi + const * t and t > tmin:
            t *= beta
            xnew = x + t * d
            Fnew, _ = fun(xnew, False)
            Phinew = phi(xnew, Fnew)
            Psinew = psi(Phinew)
        x, Phix, Psix = xnew, Phinew, Psinew
        Fx, DFx = fun(x, True)
        DPhix = dphi(x, Fx, DFx)
        DPsix = DPhix.T @ Phix
        k += 1
        k_main += 1
        aux[k_main % m] = Psix
        MaxPsi = Psix if k_main <= 5 else float(aux.max())
        if Psix < best[4]:
            best = (k, x.copy(), Fx, Phix, Psix, DPhix, DPsix)
        elif k - best[0] > kwatch:
            _, x, Fx, Phix, Psix, DPhix, DPsix = best
            x = x.copy()
            MaxPsi = Psix
        log(f"{k:4d} {Psix:24.5e} {t:11.7g}")
    if Psix <= eps2:
        flag = 1
    elif k >= maxiter:
        flag = 0
    else:
        flag = -1
    return x, Fx, flag, k


def _solve_mcp(model, problem, endo, exo, periods, lb, ub, o, log):
    """solve_stacked_problem.m with solve_algo=10 (LMMCP), then dynare_solve's and its own
    treatment of the residuals of equations whose variable sits on a bound."""
    lag, n = model.info["maximum_lag"], model.n_endo
    y0, yT = _boundaries(model, endo, periods)
    X = exo[lag:lag + periods]
    LB, UB = np.tile(lb, periods), np.tile(ub, periods)

    def fun(z, jac):
        return problem(z.reshape(periods, n), y0, yT, X, jac=jac)

    z, F, flag, it = lmmcp(fun, endo[lag:lag + periods].ravel(), LB, UB, tolfun=o.get("lmmcp.tolfun"),
                           maxiter=o.get("lmmcp.maxiter", 500), log=log)
    eps = np.finfo(float).eps
    bounded = np.isfinite(LB) | np.isfinite(UB)
    F = np.where(bounded & ((z <= LB + eps) | (z >= UB - eps)), 0.0, F)
    out = endo.copy()
    out[lag:lag + periods] = z.reshape(periods, n)
    return out, flag == 1 and bool(np.all(np.isfinite(z))), _norm_inf(F), it


# ----------------------------------------------------------------------------------- driver
class _Sim:
    """State of one perfect_foresight_solver call (the fields of oo_ and M_ it reads)."""

    def __init__(self, model, params, endo, exo, ss, xss, ss0, xss0, o, verbose=False, stacked=None):
        self.model, self.params, self.o = model, params, o
        self.endo, self.exo = endo, exo
        self.ss, self.xss, self.ss0, self.xss0 = ss, xss, ss0, xss0
        self.verbose = verbose
        self.stacked = stacked
        info = model.info
        self.lag, self.lead = info["maximum_lag"], info["maximum_lead"]
        self.mcp = None
        if o["lmmcp.status"]:
            if o["linear_approximation"] or o.get("linear"):
                raise NotImplementedError("lmmcp together with model(linear) or linear_approximation is not implemented")
            self.mcp = complementarity_conditions(model, params)
        self._problems = {}

    def log(self, s):
        if self.verbose:
            print(s)

    def problem(self, ss):
        key = ss.tobytes()
        p = self._problems.get(key)
        if p is None:
            if len(self._problems) > 4:
                self._problems.clear()
            p = self._problems[key] = _Stacked(self.model, self.params, ss,
                                               None if self.mcp is None else self.mcp[2], self.stacked)
        return p

    # perfect_foresight_solver_core.m
    def core(self, endo, exo, ss, xss, periods):
        """Returns (endo, success, maxerror, iterations)."""
        m, o, info = self.model, self.o, self.model.info
        has_lag, has_lead = info["maximum_endo_lag"] > 0, info["maximum_endo_lead"] > 0
        linear = o["linear_approximation"]
        if (linear or o["endogenous_terminal_period"]) and not (has_lag and has_lead):
            name = "linear_approximation" if linear else "endogenous_terminal_period"
            raise ValueError(f"option {name} is not available with purely backward, purely forward or static models")
        if o.get("linear") and has_lag and has_lead:
            linear = True
        problem = self.problem(ss)
        if self.mcp is not None:
            self.solver = "lmmcp"
            return _solve_mcp(m, problem, endo, exo, periods, self.mcp[0], self.mcp[1], o, self.log)
        if not (has_lag and has_lead):
            self.solver = "period by period"
            y, success = _sim1_recursive(m, endo, exo, periods, self.params, ss, o)
            return y, success, _maxerror(m, problem, y, exo, periods), 0
        if linear:
            self.solver = "linear"
            nocheck = bool(m.options.get("steady", {}).get("steadystate.nocheck", False))
            return _sim1_linear(m, problem, endo, exo, periods, ss, xss, self.params, o, nocheck)
        self.solver = "stacked Newton"
        return _sim1(m, problem, endo, exo, periods, o, self.log)

    def steady(self, guess, x):
        o = self.o
        try:
            tol = DYNARE_TOLF if o["simul.steady_tolf"] is None else float(o["simul.steady_tolf"])
            with np.errstate(all="ignore"):
                ys, _ = steady_state(self.model, self.params, guess=guess, exo=x, tol=tol,
                                     check=not o["simul.endval_steady_nocheck"])
        except Exception:  # noqa: BLE001 - any failure of the steady-state computation
            return None
        return ys if np.all(np.isfinite(ys)) else None

    # create_scenario of perfect_foresight_solver.m
    def scenario(self, share, shareorig, endoorig, exoorig, endobase, exobase, recompute, endo, ss, xss, periods):
        lag, lead = self.lag, self.lead
        w = share / shareorig
        exo = exoorig * w + exobase * (1 - w)
        for name in self.o["simul.homotopy_exclude_varexo"]:
            if name not in self.model._ix:
                raise ValueError("homotopy_exclude_varexo must only contain exogenous variable names")
            j = self.model._ix[name]
            exo[:, j] = exoorig[:, j]
        endo = endo.copy()
        endo[:lag] = w * endoorig[:lag] + (1 - w) * endobase[:lag]
        last = range(lag + periods, lag + periods + lead)
        ok = True
        if recompute:
            saved = endo[lag + periods:].copy()
            for j in last:
                ys = self.steady(endo[j], exo[j])
                if ys is None:
                    ys = self.steady(self.ss if self.ss0 is None else self.ss0, exo[j])
                if ys is None:
                    endo[lag + periods:] = saved
                    ok = False
                    break
                endo[j] = ys
            ss, xss = endo[-1].copy(), exo[-1].copy()
        else:
            endo[lag + periods:] = w * endoorig[lag + periods:] + (1 - w) * endobase[lag + periods:]
        return ok, endo, exo, ss, xss

    # homotopy_loop of perfect_foresight_solver.m
    def homotopy(self, max_share, shareorig, endoorig, exoorig, endobase, exobase, recompute, periods):
        o, lag, lead = self.o, self.lag, self.lead
        completed, step, counter, iteration = 0.0, min(o["simul.homotopy_initial_step_size"], max_share), 0, 0
        endo, exo, ss, xss = endoorig.copy(), exoorig.copy(), self.ss, self.xss
        maxerror, solver_iter = np.nan, 0
        sim = slice(lag, lag + periods)
        while step > o["simul.homotopy_min_step_size"]:
            iteration += 1
            saved_endo, saved_exo = endo, exo
            new_share = completed + step
            if new_share > max_share:
                new_share = max_share
                step = new_share - completed
            ok, endo, exo, ss, xss = self.scenario(new_share, shareorig, endoorig, exoorig, endobase, exobase,
                                                   recompute, endo, ss, xss, periods)
            success = False
            if ok:
                if completed == 0 and not (iteration == 1 and new_share == shareorig):
                    # guess: the terminal steady state (the base scenario for models without leads)
                    endo[sim] = endo[lag + periods] if lead > 0 else endobase[sim]
                endo, success, maxerror, solver_iter = self.core(endo, exo, ss, xss, periods)
            else:
                maxerror = np.nan
            if o["no_homotopy"] or (iteration == 1 and success and new_share == 1):
                if success:
                    completed = new_share
                break
            if iteration == 1:
                self.log("Entering the homotopy method iterations...")
            self.log(f"{iteration}\t | {new_share:1.5f}\t | {'succeeded' if success else 'failed'}\t | {maxerror:e}")
            if success:
                completed = new_share
                if completed >= max_share:
                    break
                counter += 1
                n_inc = o["simul.homotopy_step_size_increase_success_count"]
                if n_inc > 0 and counter >= n_inc:
                    counter = 0
                    step *= 2
            else:
                endo, exo = saved_endo, saved_exo
                counter = 0
                step /= 2
        return completed, endo, exo, ss, xss, iteration, maxerror, solver_iter

    # perfect_foresight_solver.m
    def solve(self, periods):
        m, o, lag, lead = self.model, self.o, self.lag, self.lead
        _check_homotopy_options(o)
        rows = lag + periods + lead
        if self.endo.shape != (rows, m.n_endo) or self.exo.shape != (rows, m.n_exo):
            raise ValueError(f"endo and exo must have {rows} rows (maximum_lag + periods + maximum_lead)")
        base_ss, base_x = (self.ss, self.xss) if self.ss0 is None else (self.ss0, self.xss0)
        endobase, exobase = np.tile(base_ss, (rows, 1)), np.tile(base_x, (rows, 1))
        if o["simul.endval_steady"]:
            if lead == 0:
                raise ValueError("option endval_steady cannot be used on a purely backward or static model")
            recompute = True
        elif self.ss0 is not None:
            recompute = True
            tol = DYNARE_TOLF if o["simul.steady_tolf"] is None else float(o["simul.steady_tolf"])
            for j in range(lag + periods, rows):
                with np.errstate(all="ignore"):
                    r = _norm_inf(m.static_resid(self.endo[j], self.exo[j], self.params))
                if not r <= tol:
                    recompute = False
                    break
        else:
            recompute = False
        endoorig, exoorig = self.endo, self.exo
        max_share = o["simul.homotopy_max_completion_share"]
        completed, endo, exo, ss, xss, iteration, maxerror, solver_iter = self.homotopy(
            max_share, 1.0, endoorig, exoorig, endobase, exobase, recompute, periods)
        # oo_.exo_simul is left as perfect_foresight_setup built it unless the simulation fails
        out = dict(endo=endo, exo=exoorig, ss=self.ss, xss=self.xss, status=False, linearization=False,
                   marginal=False, share=completed, homotopy_iterations=iteration, iterations=solver_iter,
                   recompute=recompute)
        marg = o["simul.homotopy_marginal_linearization_fallback"]
        if completed == 1:
            out.update(status=True)
            if recompute:
                out["ss"] = ss
        elif o["simul.homotopy_linearization_fallback"] and completed > 0:
            out["endo"] = endobase + (endo - endobase) / completed
            if recompute:
                out["ss"] = out["endo"][-1].copy()
            maxerror = _maxerror(m, self.problem(out["ss"]), out["endo"], out["exo"], periods)
            out.update(status=True, linearization=True)
        elif marg > 0 and completed > marg:
            extra_share = completed - marg
            ok, e_endo, e_exo, e_ss, e_xss = self.scenario(extra_share, 1.0, endoorig, exoorig, endobase, exobase,
                                                           recompute, endo, ss, xss, periods)
            if ok:
                e_endo, ok, _, _ = self.core(e_endo, e_exo, e_ss, e_xss, periods)
            if not ok:
                share2, e_endo, e_exo, e_ss, e_xss, _, _, _ = self.homotopy(
                    extra_share, 1.0, endoorig, exoorig, endobase, exobase, recompute, periods)
                ok = share2 == extra_share
            if ok:
                out["endo"] = endo + (endo - e_endo) * (1 - completed) / marg
                if recompute:
                    out["ss"] = out["endo"][-1].copy()
                maxerror = _maxerror(m, self.problem(out["ss"]), out["endo"], out["exo"], periods)
                out.update(marginal=True)
            else:
                out.update(ss=ss, xss=xss, exo=exo)
            out["status"] = bool(ok)
        else:
            out.update(ss=ss, xss=xss, exo=exo)
        out["maxerror"] = maxerror
        return out


def _model_is_linear(model):
    """``model(linear)`` is not in the preprocessor's JSON: read it from the .mod file."""
    cached = model.__dict__.get("_pf_linear")
    if cached is None:
        cached = False
        try:
            mod = model.json_dir.parents[2] / (model.json_dir.parents[1].name + ".mod")
            text = re.sub(r"/\*.*?\*/", " ", mod.read_text(errors="replace"), flags=re.S)
            text = re.sub(r"(//|%)[^\n]*", " ", text)
            cached = re.search(r"(?<![\w.])model\s*\(([^)]*)\)", text) is not None and any(
                re.search(r"(?<![\w=])linear(?![\w=])", mo.group(1))
                for mo in re.finditer(r"(?<![\w.])model(?:_options)?\s*\(([^)]*)\)", text))
        except (OSError, IndexError):
            cached = False
        model.__dict__["_pf_linear"] = cached
    return cached


def _pfwee_information(model, ws, periods, datafile):
    """perfect_foresight_with_expectation_errors_setup.m: ``shocks_info[exo, real time, information
    time]`` and ``terminal_info[exo, information time]``."""
    nx = model.n_exo
    terminal = np.full((nx, periods), np.nan)
    shocks = np.full((nx, periods, periods), np.nan)
    if datafile:
        path = model.json_dir.parents[2] / datafile
        with open(path) as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        names = [s.strip() for s in lines[0].split(",")][1:]
        raw = np.array([[float(s) if s.strip() else np.nan for s in ln.split(",")[1:]] for ln in lines[1:]])
        if raw.shape[0] - 2 != periods:
            raise ValueError(f"the number of rows in {datafile} does not match the periods setting")
        for i, name in enumerate(names):
            j, p = model._ix[name], int(raw[0, i])
            shocks[j, p - 1:, p - 1] = raw[p:-1, i]
            terminal[j, p - 1] = raw[-1, i]
        return shocks, terminal
    terminal[:, 0] = ws.xss
    shocks[:, :, 0] = ws.xss[:, None]
    for j, p1, p2, val, typ in ws.det_shocks:
        if typ == "multiply_steady_state":
            val = ws.xss[j] * val
        elif typ == "multiply_initial_steady_state":
            if ws.xss0 is None:
                raise PerfectForesightError("option relative_to_initval of mshocks cannot be used without an endval block")
            val = ws.xss0[j] * val
        if p1 < 1 or p2 > periods:
            raise PerfectForesightError("a shock is dated outside the simulation")
        shocks[j, p1 - 1:p2, 0] = val
    for p in range(periods):
        if p > 0:
            terminal[:, p] = terminal[:, p - 1]
            shocks[:, :, p] = shocks[:, :, p - 1]
        for d in ws.learnt_endval:
            if d["learnt_in"] != p + 1:
                continue
            j, v = d["exo"], d["value"]
            if d["type"] == "level":
                terminal[j, p] = v
            elif d["type"] == "add":
                terminal[j, p] = terminal[j, p - 1] + v
            elif d["type"] == "multiply":
                terminal[j, p] = terminal[j, p - 1] * v
            else:
                raise ValueError(f"unknown type {d['type']} in endval(learnt_in=...)")
            shocks[j, p:, p] = terminal[j, p]
        for d in ws.learnt_shocks:
            if d["learnt_in"] != p + 1:
                continue
            j, v, sl = d["exo"], d["value"], slice(d["p1"] - 1, d["p2"])
            if d["p1"] < 1 or d["p2"] > periods:
                raise PerfectForesightError("a learnt shock is dated outside the simulation")
            if d["type"] == "level":
                shocks[j, sl, p] = v
            elif d["type"] == "add":
                shocks[j, sl, p] = shocks[j, sl, p - 1] + v
            elif d["type"] == "multiply":
                shocks[j, sl, p] = shocks[j, sl, p - 1] * v
            elif d["type"] == "multiply_steady_state":
                shocks[j, sl, p] = terminal[j, p] * v
            else:
                raise ValueError(f"unknown type {d['type']} in shocks(learnt_in=...)")
    return shocks, terminal


def _solve_pfwee(model, ws, periods, o, verbose, stacked, datafile):
    """perfect_foresight_with_expectation_errors_solver.m."""
    lag, lead = model.info["maximum_lag"], model.info["maximum_lead"]
    shocks, terminal = _pfwee_information(model, ws, periods, datafile)
    ws_plain = ws
    saved = ws.det_shocks
    ws_plain.det_shocks = []                       # the setup command builds the paths without the shocks
    endo_simul, exo_simul = _make_paths(model, ws_plain, periods)
    ws.det_shocks = saved
    o = dict(o)
    if lead > 0:
        o["simul.endval_steady"] = True
    ss, xss = ws.ss.copy(), ws.xss.copy()
    ss0, xss0 = (ss.copy(), xss.copy()) if ws.ss0 is None else (ws.ss0, ws.xss0)
    const = bool(o["pfwee.constant_simulation_length"])
    info_period, increment, share0 = 1, 0, None
    out, solves = None, []
    while info_period <= periods:
        p = info_period - 1
        xss = terminal[:, p].copy()
        if const and increment > 0:
            endo_simul = np.vstack([endo_simul, np.tile(ss, (increment, 1))])
            exo_simul = np.vstack([exo_simul, np.full((increment, model.n_exo), np.nan)])
        endo = endo_simul[p:].copy()
        exo = exo_simul[p:].copy()
        sim_length = periods if const else periods - p
        known = periods - p
        exo[lag:lag + known] = shocks[:, p:, p].T
        exo[lag + known:] = xss
        sim = _Sim(model, ws.params, endo, exo, ss, xss, ss0, xss0, o, verbose, stacked)
        out = sim.solve(sim_length)
        out["solver"] = getattr(sim, "solver", "")
        solves.append((info_period, out["iterations"], out["maxerror"]))
        if not out["status"]:
            out["endo_full"], out["exo_full"] = endo_simul, exo_simul
            raise PerfectForesightError("perfect_foresight_with_expectation_errors_solver: failed to compute the solution "
                                        f"for information available at period {info_period}", out)
        if out["marginal"]:
            raise NotImplementedError("marginal linearization together with expectation errors is not implemented")
        if share0 is None:
            share0 = out["share"]
            o["simul.homotopy_max_completion_share"] = share0
        elif out["share"] != share0:
            raise PerfectForesightError("could not find a solution for information available at period "
                                        f"{info_period} with the same homotopy completion share as period 1")
        ss = out["ss"]
        endo_simul[p:] = out["endo"]
        exo_simul[p:] = out["exo"]
        increment = 1
        while (info_period + increment <= periods
               and np.array_equal(terminal[:, p], terminal[:, p + increment])
               and np.array_equal(shocks[:, p + increment:, p], shocks[:, p + increment:, p + increment])):
            increment += 1
        info_period += increment
    out = dict(out)
    out.update(endo=endo_simul, exo=exo_simul, ss=ss, xss=xss, ss0=ss0, xss0=xss0, solves=solves,
               shocks_info=shocks, terminal_info=terminal)
    return out


def perfect_foresight(model, periods=None, params=None, exo=None, y_init=None, y_term=None,
                      guess=None, tol=None, maxit=None, *, steady=None, expectation_errors=None,
                      learnt_shocks=None, learnt_endval=None, raise_on_failure=True, verbose=False,
                      stacked=None, lmmcp_tolfun=None, lmmcp_maxiter=None, **options) -> PerfectForesightResult:
    """Deterministic simulation: Dynare's ``perfect_foresight_setup`` + ``perfect_foresight_solver``
    (or the ``perfect_foresight_with_expectation_errors`` pair).

    Everything is read from the .mod file and can be overridden:

    ``periods``, ``params``; ``exo`` (full path, ``maximum_lag + periods + maximum_lead`` rows),
    ``y_init`` / ``y_term`` (initial and terminal conditions), ``guess`` (``periods`` rows);
    ``tol`` or ``tolf`` (default: the file's ``tolf``, else 1e-9; Dynare's own default is 1e-5),
    ``maxit``, ``tolx``, ``no_homotopy``, ``homotopy_initial_step_size``, ``homotopy_min_step_size``,
    ``homotopy_step_size_increase_success_count``, ``homotopy_max_completion_share``,
    ``homotopy_linearization_fallback``, ``homotopy_marginal_linearization_fallback``,
    ``homotopy_exclude_varexo``, ``linear_approximation``, ``endogenous_terminal_period``,
    ``lmmcp``, ``endval_steady``, ``endval_steady_nocheck``, ``steady_tolf``,
    ``constant_simulation_length``, ``stack_solve_algo``.

    ``steady``: None follows the .mod file when it has a ``perfect_foresight_setup`` command (a
    ``steady`` command after ``initval``/``endval`` makes the path start from/end at a steady state,
    otherwise the values of the block are used as they are), and computes the steady states when
    the file has no such command; True always computes them; False never does.

    ``expectation_errors``: None follows the file; with ``learnt_shocks`` (dicts with ``learnt_in``,
    ``var``, ``periods``, ``value`` and optionally ``type``: level, add, multiply,
    multiply_steady_state) and ``learnt_endval`` (``learnt_in``, ``var``, ``value``, ``type``) news
    can be added from Python: the counterpart of ``shocks(learnt_in=...)`` / ``endval(learnt_in=...)``.

    With complementarity conditions in the model and the ``lmmcp`` option the stacked problem is
    solved with the LMMCP algorithm (``lmmcp_tolfun``: tolerance on the 2-norm of its merit vector,
    default sqrt(eps) as in Dynare, where ``tolf`` and ``maxit`` are not used for this solver).

    Raises ``PerfectForesightError`` when no solution is found (``raise_on_failure=False`` returns
    the result with ``status`` False instead, like Dynare).
    """
    if tol is not None:
        options.setdefault("tolf", tol)
    if maxit is not None:
        options["maxit"] = maxit
    if periods is not None:
        options["periods"] = periods
    ws = _replay(model, params, steady, learnt_shocks or (), learnt_endval or ())
    pfwee = ws.pfwee if expectation_errors is None else bool(expectation_errors)
    if expectation_errors is None and (learnt_shocks or learnt_endval):
        pfwee = True
    if not pfwee and (ws.learnt_shocks or ws.learnt_endval) and ws.has_setup:
        _opt.warn_once(model, "shocks(learnt_in=...) / endval(learnt_in=...) blocks are only used by the "
                              "perfect_foresight_with_expectation_errors commands; they are ignored here, as in Dynare")
    o = _resolve_options(model, pfwee, options)
    if o["periods"] is None:
        o["periods"] = _statement_options(model, "perfect_foresight_setup").get("periods") or \
            _statement_options(model, "perfect_foresight_with_expectation_errors_setup").get("periods")
    if o["periods"] is None:
        raise ValueError("periods not given and not set in the .mod file")
    T = int(o["periods"])
    o["linear"] = _model_is_linear(model)
    if lmmcp_tolfun is not None:
        o["lmmcp.tolfun"] = float(lmmcp_tolfun)
    else:
        try:                                         # options_.lmmcp.TolFun = ... written in the .mod file
            o["lmmcp.tolfun"] = float(model.matlab_options["lmmcp"]["TolFun"])
        except Exception:  # noqa: BLE001
            pass
    if lmmcp_maxiter is not None:
        o["lmmcp.maxiter"] = int(lmmcp_maxiter)
    if has_complementarity(model) and not o["lmmcp.status"]:
        _opt.warn_once(model, "the model has complementarity conditions but perfect_foresight_solver is used without "
                              "the lmmcp option: as in Dynare the conditions are ignored and the unconstrained path "
                              "is computed (pass lmmcp=True to impose them)")
    info = model.info
    lag, lead = info["maximum_lag"], info["maximum_lead"]
    if o["datafile"] and not pfwee:
        raise NotImplementedError("perfect_foresight_setup(datafile=...) (an initval_file) is not implemented")
    if pfwee:
        if exo is not None or y_init is not None or y_term is not None or guess is not None:
            raise NotImplementedError("exo, y_init, y_term and guess cannot be combined with expectation errors")
        out = _solve_pfwee(model, ws, T, o, verbose, stacked, o["datafile"])
        ss0, xss0 = out["ss0"], out["xss0"]
    else:
        endo, x = _make_paths(model, ws, T)
        rows = lag + T + lead
        if exo is not None:
            x = np.array(exo, float)
            if x.shape != (rows, model.n_exo):
                raise ValueError(f"exo must have shape ({rows}, {model.n_exo})")
        if y_init is not None:
            if not lag:
                raise ValueError("the model has no lag: there is no initial condition")
            endo[:lag] = np.asarray(y_init, float)
        if y_term is not None:
            if not lead:
                raise ValueError("the model has no lead: there is no terminal condition")
            endo[lag + T:] = np.asarray(y_term, float)
        if guess is not None:
            g = np.asarray(guess, float)
            if g.shape != (T, model.n_endo):
                raise ValueError(f"guess must have shape ({T}, {model.n_endo})")
            endo[lag:lag + T] = g
        sim = _Sim(model, ws.params, endo, x, ws.ss, ws.xss, ws.ss0, ws.xss0, o, verbose, stacked)
        out = sim.solve(T)
        out["solver"] = getattr(sim, "solver", "")
        ss0, xss0 = ws.ss0, ws.xss0
    extra = {k: out[k] for k in ("solves", "shocks_info", "terminal_info", "recompute") if k in out}
    extra["tolf"] = o["dynatol.f"]
    res = PerfectForesightResult(
        endo=out["endo"], exo=out["exo"], iterations=int(out["iterations"] or 0), max_residual=float(out["maxerror"]),
        status=bool(out["status"]), endo_names=list(model.endo_names), exo_names=list(model.exo_names), periods=T,
        maximum_lag=lag, maximum_lead=lead, steady_state=out["ss"], initial_steady_state=ss0,
        exo_steady_state=out["xss"], initial_exo_steady_state=xss0, params=ws.params,
        homotopy_completion_share=float(out["share"]), homotopy_iterations=int(out["homotopy_iterations"]),
        homotopy_linearization=bool(out["linearization"]), homotopy_marginal_linearization=bool(out["marginal"]),
        solver=out.get("solver", ""), info=extra)
    if not res.status and raise_on_failure:
        raise PerfectForesightError(
            f"failed to solve the perfect foresight model (max residual {res.max_residual:.2e} after "
            f"{res.iterations} iterations of the last attempt, homotopy completed {res.homotopy_completion_share:.3g} "
            f"of the shocks in {res.homotopy_iterations} iterations)", res)
    return res


def unanticipated_shocks(model, surprises, periods=None, **kw) -> PerfectForesightResult:
    """A sequence of unanticipated shocks: ``surprises`` is a list of ``(period, {exo: value})``;
    each shock hits in ``period`` and is learnt in that same period (agents expect the exogenous
    variables to stay at their steady state afterwards). Shorthand for
    ``perfect_foresight(model, learnt_shocks=[...])``, i.e. Dynare's ``shocks(learnt_in=p)`` blocks
    with ``perfect_foresight_with_expectation_errors_solver``."""
    learnt = [dict(learnt_in=int(p), var=name, periods=int(p), value=float(v), type="level")
              for p, shocks in surprises for name, v in shocks.items()]
    return perfect_foresight(model, periods, learnt_shocks=learnt, expectation_errors=True, **kw)


# reachable as attributes of the function too: ``pydyn.perfect_foresight`` is the function, which
# hides this module in the package namespace
perfect_foresight.Result = PerfectForesightResult
perfect_foresight.unanticipated_shocks = unanticipated_shocks
perfect_foresight.lmmcp = lmmcp
perfect_foresight.complementarity_conditions = complementarity_conditions
perfect_foresight.exo_path = exo_path
