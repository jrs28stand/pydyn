# This file is part of pydyn. It is a Python port, modified in 2026, of these files of Dynare 7.2
# (Copyright (C) 2003-2026 Dynare Team, https://www.dynare.org, GPL-3.0-or-later):
#   matlab/optimal_policy/{dyn_ramsey_static, evaluate_planner_objective,
#     get_optimal_policy_discount_factor}.m,
#   the Ramsey branch of matlab/evaluate_steady_state.m,
#   matlab/discretionary_policy/{discretionary_policy, discretionary_policy_1,
#     discretionary_policy_engine, discretionary_policy_initialization}.m,
#   matlab/+osr/{run, objective, get_variance_of_endogenous_variables}.m,
#   matlab/matrix_solver/lyapunov_symm.m (fixed-point method 3).
# Released under the GNU General Public License, version 3 or later; see LICENSE. No warranty.
"""Optimal policy: Ramsey (``ramsey_model``), welfare (``evaluate_planner_objective``), discretion
(``discretionary_policy``) and optimal simple rules (``osr``), after Dynare 7.2.

Ramsey models need nothing special beyond the steady state: the preprocessor writes the first-order
conditions of the planner (with the Lagrange multipliers ``MULT_*``) into the model, and
:func:`pydyn.steady_state` routes such models to :func:`ramsey_steady_state`, a port of Dynare's
``dyn_ramsey_static.m``. ``first_order``, ``k_order``, moments, simulations and IRFs then work on the
Ramsey model as on any other.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import options as _opt

_EPS = float(np.finfo(float).eps)
#: options_.ramsey.maxit (Dynare's default)
RAMSEY_MAXIT = 500
#: options_.gstep(1): floor of the step in Dynare's forward-difference Jacobians
GSTEP = 1e-2


class OptimalPolicyError(ValueError):
    """Dynare stops with an error (its ``info`` code is in ``.info``)."""

    def __init__(self, message, info=None):
        super().__init__(message)
        self.info = info


# =========================================================================== Ramsey: model structure
def _statement(model, *kinds):
    for st in model.statements:
        if st.get("statementName") in kinds:
            return st
    return None


def _statements(model, *kinds):
    return [st for st in model.statements if st.get("statementName") in kinds]


def is_ramsey(model) -> bool:
    """The model is the Ramsey problem written by the preprocessor (``ramsey_model`` or ``ramsey_policy``)."""
    return any(a.get("type") == 6 for a in model.raw.get("aux_vars", [])) and \
        _statement(model, "ramsey_model", "ramsey_policy") is not None


def is_discretionary(model) -> bool:
    return _statement(model, "discretionary_policy") is not None


@dataclass
class RamseyInfo:
    """What Dynare keeps in ``M_.ramsey_orig_endo_nbr``, ``M_.ramsey_orig_eq_nbr``, ``options_.instruments``."""
    orig_endo_nbr: int          # M_.ramsey_orig_endo_nbr: variables (and FOC equations) before the multipliers
    orig_eq_nbr: int            # M_.ramsey_orig_eq_nbr: equations of the original model = number of multipliers
    mult: np.ndarray            # endogenous indices of the multipliers, in equation order
    instruments: list
    inst_idx: np.ndarray        # endogenous indices of the instruments
    recursive: np.ndarray       # auxiliary variables created after the multipliers (chain rule in A)
    discount_index: int         # parameter index of optimal_policy_discount_factor

    @property
    def inst_nbr(self):
        return self.orig_endo_nbr - self.orig_eq_nbr


def _instrument_names(model):
    out = []
    for st in _statements(model, "ramsey_model", "ramsey_policy", "discretionary_policy"):
        ins = st.get("options", {}).get("instruments")
        if ins:
            out = list(ins.get("symbol_list", ins) if isinstance(ins, dict) else ins)
    return out


def ramsey_info(model) -> RamseyInfo:
    cached = model.__dict__.get("_ramsey_info")
    if cached is not None:
        return cached
    if not is_ramsey(model):
        raise OptimalPolicyError("the model has no ramsey_model / ramsey_policy statement")
    aux = model.raw.get("aux_vars", [])
    mults = sorted((a for a in aux if a.get("type") == 6), key=lambda a: a["endo_index"])
    mult = np.array([a["endo_index"] - 1 for a in mults], int)
    n_r = int(mult.min())
    if not np.array_equal(mult, np.arange(n_r, n_r + len(mult))):
        raise NotImplementedError("Lagrange multipliers that are not contiguous in the list of variables")
    # multipliers are numbered after the original equations (eq_nbr), and declared in that order
    eqs = [a.get("eq_nbr") for a in mults]
    if eqs != sorted(eqs):
        raise NotImplementedError("Lagrange multipliers not in equation order")
    rec = np.array(sorted(a["endo_index"] - 1 for a in aux if a.get("type") != 6 and a["endo_index"] - 1 > mult.max()), int)
    names = _instrument_names(model)
    for nme in names:
        if nme not in model._ie:
            raise OptimalPolicyError(f"instrument {nme} is not an endogenous variable")
    if "optimal_policy_discount_factor" not in model._ip:
        raise OptimalPolicyError("optimal_policy_discount_factor is not a parameter of the model")
    info = RamseyInfo(orig_endo_nbr=n_r, orig_eq_nbr=len(mult), mult=mult, instruments=names,
                      inst_idx=np.array([model._ie[v] for v in names], int), recursive=rec,
                      discount_index=model._ip["optimal_policy_discount_factor"])
    model.__dict__["_ramsey_info"] = info
    return info


def planner_discount(model, params=None) -> float:
    """``get_optimal_policy_discount_factor``: the parameter ``optimal_policy_discount_factor``."""
    params = model.params if params is None else params
    if "optimal_policy_discount_factor" not in model._ip:
        raise OptimalPolicyError("the model has no planner discount factor (optimal_policy_discount_factor)")
    return float(params[model._ip["optimal_policy_discount_factor"]])


# =========================================================================== Ramsey steady state
def _has_steady_state_file(model) -> bool:
    """Dynare's options_.steadystate_flag: a steady_state_model block or a _steadystate.m file."""
    return model.steady_state_fn is not None or model.steady_state_model is not None or \
        model.steady_state_file is not None


class _initval_as:
    """Run a steady-state file with Dynare's ``ys_init``: pydyn's steady-state functions start from
    ``model.initval``, so it holds ys_init for the duration of the call."""

    def __init__(self, model, ys):
        self.model, self.ys = model, ys

    def __enter__(self):
        self.saved = self.model.initval
        self.model.initval = {**self.saved, **dict(zip(self.model.endo_names, map(float, self.ys)))}

    def __exit__(self, *exc):
        self.model.initval = self.saved


def _evaluate_steady_state_file(model, ys_init, x, params):
    """evaluate_steady_state_file.m for a Ramsey model (no auxiliary variables set, no check).
    Returns (ys, params); raises SteadyStateError when the file signals a failure."""
    from . import matlab
    from .steady import (SteadyStateError, _from_steady_state_model, steady_state_file_fn)
    with _initval_as(model, ys_init):
        if model.steady_state_fn is not None or model.steady_state_model is None:
            fn = model.steady_state_fn or steady_state_file_fn(model)
            try:
                out = fn(model, np.array(params, float), x)
            except matlab.Unsupported as exc:
                raise SteadyStateError(f"the steady-state file cannot be interpreted ({exc})") from None
            if isinstance(out, tuple):
                ys, p = np.asarray(out[0], float).ravel(), np.asarray(out[1], float).ravel()
            else:
                ys, p = np.asarray(out, float).ravel(), np.array(params, float)
            if ys.size < model.n_endo:          # MATLAB grows ys_init(1:k) = ...: the rest is ys_init
                ys = np.concatenate([ys, np.asarray(ys_init, float)[ys.size:]])
        else:
            with matlab.steady_context(model, params):
                ys, p = _from_steady_state_model(model, params, x)
    return ys, p


def _aux_rows(model, idx):
    """Rows of the static model holding the defining equations of auxiliary variables ``idx``. In
    a Ramsey model, auxiliary variables created before the multipliers (expectation operator, ...)
    have their equation among the original ones (after the FOCs); those created after are last."""
    if not is_ramsey(model):
        return np.asarray(idx, int)
    ri = ramsey_info(model)
    idx = np.asarray(idx, int)
    return np.where(idx < ri.orig_endo_nbr, idx + ri.orig_endo_nbr - ri.inst_nbr, idx)


def set_auxiliary_variables(model, ys, x, params):
    """``<fname>.set_auxiliary_variables``: auxiliary variables (not the multipliers) from their equations."""
    from .steady import fill_auxiliary
    return fill_auxiliary(model, ys, x, params)


def _multiplier_jacobian(model, ri, xx, x, params):
    """``ramsey_multipliers_static_g1``: derivatives of the FOC equations with respect to the
    multipliers, through the auxiliary variables created after them (chain rule, as the preprocessor's
    computeRamseyMultipliersDerivatives)."""
    J = model.static_jac(xx, x, params)
    F = np.arange(ri.orig_endo_nbr)
    A = J[np.ix_(F, ri.mult)]
    if ri.recursive.size:
        rows = _aux_rows(model, ri.recursive)
        dR = -np.linalg.solve(J[np.ix_(rows, ri.recursive)], J[np.ix_(rows, ri.mult)])
        A = A + J[np.ix_(F, ri.recursive)] @ dR
    return A


def _mldivide(A, y):
    """MATLAB's ``A \\ y``: LU for square matrices, least squares (QR) otherwise."""
    from scipy import linalg
    if A.shape[0] == A.shape[1]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", linalg.LinAlgWarning)
                return linalg.solve(A, y)
        except (linalg.LinAlgError, linalg.LinAlgWarning):
            pass
    return linalg.lstsq(A, y, lapack_driver="gelsy")[0]


class _RamseyStatic:
    """``dyn_ramsey_static_1``: residuals of the Ramsey steady-state problem given the original
    variables (no steady-state file) or the instruments (steady-state file), the multipliers being
    the least-squares solution of the (linear in them) first-order conditions."""

    def __init__(self, model, x, ys_init, params, ssflag):
        self.model, self.x, self.ys_init, self.params, self.ssflag = model, x, np.array(ys_init, float), params, ssflag
        self.ri = ramsey_info(model)
        self.has_aux = any(a.get("type") != 6 for a in model.raw.get("aux_vars", []))
        self.n_orig = model.raw.get("orig_endo_nbr", model.n_endo)

    def __call__(self, z):
        return self.evaluate(z)[0]

    def evaluate(self, z):
        from .steady import SteadyStateError
        m, ri, x, params = self.model, self.ri, self.x, self.params
        z = np.asarray(z, float)
        if self.ssflag:
            ys = self.ys_init.copy()
            ys[ri.inst_idx] = z
            try:
                with np.errstate(all="ignore"):
                    full, params = _evaluate_steady_state_file(m, ys, x, params)
            except SteadyStateError:
                return np.ones(ri.inst_nbr) + np.sum(np.abs(ys[:self.n_orig])), np.full(m.n_endo, np.nan), params
            if not np.all(np.isfinite(full[:self.n_orig])):
                # MATLAB would see complex values here (penalty 1 + sum|imag|); pydyn gets NaN
                return np.ones(ri.inst_nbr), np.full(m.n_endo, np.nan), params
            z = full
        xx = np.zeros(m.n_endo)
        xx[:self.n_orig] = z[:self.n_orig]
        with np.errstate(all="ignore"):
            if self.has_aux:
                xx = set_auxiliary_variables(m, xx, x, params)
            res = m.static_resid(xx, x, params)
            A = _multiplier_jacobian(m, ri, xx, x, params)
            y = res[:ri.orig_endo_nbr]
            if not (np.all(np.isfinite(A)) and np.all(np.isfinite(y))):
                mult = np.full(len(ri.mult), np.nan)
            else:
                mult = -_mldivide(A, y)
            resids1 = y + A @ mult
            if ri.inst_nbr == 1:
                r1 = np.array([np.sqrt(resids1 @ resids1)])
            else:
                if np.all(np.isfinite(A)) and np.all(np.isfinite(y)):
                    r = np.linalg.qr(np.column_stack([A, y]).T, mode="r")
                    d = np.abs(np.diag(r)[:A.shape[1]])
                    if np.any(d <= 1e-10 * max(1.0, np.max(np.abs(A)))):
                        # the first FOCs' multiplier coefficients are linearly dependent: the last row of R,
                        # Dynare's residual, is then not unique (Dynare's sparse QR and LAPACK's differ)
                        _opt.warn_once(m, "Ramsey steady state: the residual dyn_ramsey_static takes from a QR factorisation "
                                          "is not unique for this model (dependent first-order conditions come first); "
                                          "the steady-state search can differ from Dynare's. Declaring the instruments "
                                          "last can avoid this")
                    r = np.vstack([r, np.zeros((A.shape[1] + 1 - r.shape[0], r.shape[1]))]) if r.shape[0] < A.shape[1] + 1 else r
                    k = A.shape[0] + np.arange(1 - ri.inst_nbr, 1) - 1
                    r1 = r[-1, k]
                else:
                    r1 = np.full(ri.inst_nbr, np.nan)
        if self.ssflag:
            resids = r1
        else:
            resids = np.concatenate([res[ri.orig_endo_nbr + np.arange(self.n_orig - ri.inst_nbr)], r1])
        ss = np.zeros(m.n_endo)
        ss[:ri.orig_endo_nbr] = xx[:ri.orig_endo_nbr]
        ss[ri.mult] = mult
        if self.has_aux:
            with np.errstate(all="ignore"):
                ss = set_auxiliary_variables(m, ss, x, params)
        return resids, ss, params


def _numjac(fun, z, f0):
    """The forward-difference Jacobian of dynare_solve.m / trust_region.m without analytic Jacobian."""
    n = z.size
    dh = np.maximum(np.abs(z), GSTEP) * _EPS ** (1 / 3)
    J = np.empty((f0.size, n))
    for j in range(n):
        zz = z.copy()
        zz[j] += dh[j]
        J[:, j] = (fun(zz) - f0) / dh[j]
    return J


def _csolve_numgrad(fun):
    """csolve.m's numerical gradient (gradfun = '')."""
    def grad(z):
        f0 = fun(z)
        n = z.size
        g = np.empty((f0.size, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1e-6
            g[:, i] = (fun(z + e) - f0) / 1e-6
        return f0, g
    return grad


def ramsey_steady_state(model, params=None, guess=None, x=None, tol=None, tolx=None, solve_algo=None,
                        maxit=RAMSEY_MAXIT, _single_pass=False):
    """Steady state of a Ramsey model as Dynare's ``evaluate_steady_state`` computes it
    (``options_.ramsey_policy`` branch, ``dyn_ramsey_static.m``). Returns ``(ys, params)``.

    ``guess`` is Dynare's ``ys_init`` (default: the ``initval`` values, 0 elsewhere, as in
    ``oo_.steady_state`` before any ``steady``). Without ``guess``, when the .mod file has a ``steady``
    command, the computation is run twice, the second time from the first result, as Dynare does
    (``steady``, then again in ``stoch_simul``): with a steady-state file this refines the instruments.

    * With a ``steady_state_model`` block, a ``_steadystate.m`` file or a ``steady_state_fn``, the
      block gives the steady state conditional on the instruments (whose starting values come from
      ``initval``); the instruments are found by ``csolve`` (one instrument) or ``dynare_solve``
      (several) so that the first-order conditions hold with least-squares multipliers.
    * Otherwise the original variables are solved for by ``dynare_solve`` (``solve_algo``, default 4,
      with Dynare's forward-difference Jacobian), the multipliers again by least squares; an initial
      guess that already solves the whole static model (residual norm below ``tol``) is kept.
    """
    if guess is None and "steady" in model.options and not _single_pass:
        # the .mod file runs steady before the solution commands: Dynare's stoch_simul (resol) then
        # computes the steady state again, starting from the one steady found (oo_.steady_state)
        ys, p1 = ramsey_steady_state(model, params, None, x, tol, tolx, solve_algo, maxit, _single_pass=True)
        return ramsey_steady_state(model, p1, ys, x, tol, tolx, solve_algo, maxit, _single_pass=True)
    from .steady import SteadyStateError, dynare_solve, DYNARE_TOLF, PORTED_SOLVE_ALGOS
    from .occbin_filter import csolve
    ri = ramsey_info(model)
    so = model.options.get("steady", {})
    if tol is None:
        tol = so.get("solve_tolf", model.matlab_options.get("solve_tolf") if hasattr(model.matlab_options, "get") else None)
        tol = DYNARE_TOLF if tol is None else float(np.ravel(tol)[0])
    tolx = float(so.get("solve_tolx", _EPS ** (2 / 3))) if tolx is None else float(tolx)
    if solve_algo is None:
        solve_algo = int(so.get("solve_algo", 4))
    params = np.array(model.params if params is None else params, float)
    x = model.exo_steady() if x is None else np.asarray(x, float)
    if not np.isfinite(params[ri.discount_index]):
        warnings.warn("evaluate_steady_state: the planner_discount is NaN/Inf. That will cause problems.")
    ssflag = _has_steady_state_file(model)
    if guess is None:
        ys_init = np.array([model.initval.get(v, 0.0) for v in model.endo_names], float)
    else:
        ys_init = np.array(guess, float)
    has_aux = any(a.get("type") != 6 for a in model.raw.get("aux_vars", []))
    with np.errstate(all="ignore"):
        if has_aux and not ssflag:
            ys_init = set_auxiliary_variables(model, ys_init, x, params)
        lo, hi = ri.orig_endo_nbr, ri.orig_endo_nbr + ri.orig_eq_nbr
        if ssflag:
            if not ri.instruments:
                raise OptimalPolicyError("a Ramsey model with a steady-state file needs instruments "
                                         "(ramsey_model(instruments=...))")
            try:
                ys, p1 = _evaluate_steady_state_file(model, ys_init, x, params)
            except SteadyStateError as exc:
                raise OptimalPolicyError(f"steady-state file failed: {exc}", 19) from None
            resids = model.static_resid(ys, x, p1)
            if np.any(np.isnan(resids[lo:hi])):
                raise OptimalPolicyError("The steady state file computation for the Ramsey problem resulted in NaNs "
                                         "(Dynare's error 84): if the initial values of the instruments are not "
                                         "admissible, change them in an initval block", 84)
            if np.max(np.abs(resids[lo:hi])) > tol:
                raise OptimalPolicyError("The steady state file does not solve the steady state for the Ramsey problem "
                                         "(Dynare's error 85)", 85)
        # dyn_ramsey_static
        nl = _RamseyStatic(model, x, ys_init, params, ssflag)
        check = 0
        if not ssflag and np.linalg.norm(model.static_resid(ys_init, x, params)) < tol:
            ys = ys_init
        elif ssflag:
            z0 = ys_init[ri.inst_idx]
            if len(ri.instruments) != ri.inst_nbr:
                # dyn_ramsey_static solves for the declared instruments, dyn_ramsey_static_1 returns
                # inst_nbr = ramsey_orig_endo_nbr - ramsey_orig_eq_nbr residuals: a non-square system that
                # Dynare's solvers only handle when the starting point already solves it
                f0 = nl(z0)
                if not (np.all(np.isfinite(f0)) and np.max(np.abs(f0)) < tol):
                    raise NotImplementedError(
                        f"{len(ri.instruments)} instruments declared for {ri.inst_nbr} missing equation(s), and the "
                        "initial values of the instruments do not solve the Ramsey steady state: Dynare would "
                        "run its solver on a non-square system")
                z, check = z0, 0
            elif len(ri.instruments) == 1:
                z, rc = csolve(nl, z0, _csolve_numgrad(nl), tol, maxit)
                if rc in (1, 3):
                    check = 81
                if rc == 4:
                    check = 87
            else:
                z, err = _dynare_solve_nojac(nl, z0, tol, tolx, maxit, solve_algo, dynare_solve, PORTED_SOLVE_ALGOS)
                check = 81 if err else 0
            ys_tmp = ys_init.copy()
            ys_tmp[ri.inst_idx] = z
            _, params = _evaluate_steady_state_file(model, ys_tmp, x, params)
            nl.params = params
            ys = nl.evaluate(z)[1]
        else:
            n_orig = model.raw.get("orig_endo_nbr", model.n_endo)
            z, err = _dynare_solve_nojac(nl, ys_init[:n_orig], tol, tolx, maxit, solve_algo, dynare_solve, PORTED_SOLVE_ALGOS)
            check = 81 if err else 0
            ys = nl.evaluate(z)[1]
        if check:
            raise OptimalPolicyError(f"The steady state for the Ramsey problem could not be computed (Dynare's error {check})", check)
        resids = model.static_resid(ys, x, params)
    if np.any(np.isnan(resids[lo:])):
        raise OptimalPolicyError("The steady state computation for the Ramsey problem resulted in NaNs (Dynare's error 82)", 82)
    if np.any(np.isnan(resids[:lo])):
        raise OptimalPolicyError("The steady state computation for the Ramsey problem resulted in NaNs in the "
                                 "auxiliary equations (Dynare's error 83)", 83)
    if np.max(np.abs(resids)) > tol:
        raise OptimalPolicyError("The steady state for the Ramsey problem could not be computed (Dynare's error 81): "
                                 f"max |residual| = {np.max(np.abs(resids)):.3e}", 81)
    return ys, params


def _dynare_solve_nojac(fun, z0, tol, tolx, maxit, solve_algo, dynare_solve, ported):
    """dynare_solve.m with options_.jacobian_flag = false (as dyn_ramsey_static sets it)."""
    if solve_algo not in (4, 9):
        if solve_algo in (13, 14):
            raise OptimalPolicyError("DYNARE_SOLVE: option solve_algo=13 needs computed Jacobian")
        raise NotImplementedError(f"Ramsey steady state with solve_algo={solve_algo}: pydyn has ported "
                                  "solve_algo 4 and 9 for this case")

    def fj(z):
        f0 = fun(z)
        return f0, _numjac(fun, np.asarray(z, float), f0)
    z, err, _ = dynare_solve(fj, z0, tol, tolx, maxit, solve_algo)
    return z, err


# =========================================================================== planner objective
class PlannerObjective:
    """The ``planner_objective`` expression as the preprocessor writes it (``<fname>.objective.static_*``):
    value, gradient and Hessian with respect to the endogenous variables (declaration order)."""

    def __init__(self, model):
        from .expr import NAMESPACE
        st = _statement(model, "planner_objective")
        if st is None:
            raise OptimalPolicyError("there is no planner_objective statement in the .mod file")
        block = st.get("static_model_simple")
        if block is None:
            raise NotImplementedError("planner_objective without the preprocessor's static_model_simple block")
        self.model = model
        src, meta = model._build(model._tr_static(), block, "objective")
        ns = dict(NAMESPACE)
        ns.update(model.functions)
        ns["np"] = np
        exec(compile(src, "<pydyn planner_objective>", "exec"), ns)  # noqa: S102 - generated from the model
        self._ns, self._meta = ns, meta
        self.n = model.n_endo

    def _call(self, fn, y, params):
        x = np.zeros(self.model.n_exo)
        params = self.model.params if params is None else params
        return np.array(self._ns[fn](np.asarray(y, float), x, np.asarray(params, float), None), dtype=float)

    def value(self, y, params=None) -> float:
        return float(self._call("objective_resid", y, params)[0])

    def gradient(self, y, params=None) -> np.ndarray:
        dims, rows, cols = self._meta["objective_jac"]
        g = np.zeros(self.n)
        np.add.at(g, cols, self._call("objective_jac", y, params))
        return g

    def hessian(self, y, params=None) -> np.ndarray:
        """Full symmetric Hessian (build_two_dim_hessian)."""
        rows, cols = self._meta["objective_hess"]
        H = np.zeros((self.n, self.n))
        vals = self._call("objective_hess", y, params)
        for (i, j), v in zip(cols.tolist(), vals.tolist()):
            H[i, j] = v
            H[j, i] = v
        return H


def planner_objective(model) -> PlannerObjective:
    po = model.__dict__.get("_planner_objective")
    if po is None:
        po = model.__dict__["_planner_objective"] = PlannerObjective(model)
    return po


def _hessian_eq_zero(model) -> bool:
    """M_.hessian_eq_zero: the dynamic model has no second derivatives (a linear model)."""
    meta = model._meta.get("dynamic_hess")
    return meta is None or len(meta[0]) == 0


def _akc(A, B, C):
    """A_times_B_kronecker_C."""
    return A @ np.kron(B, C)


def lyapunov_symm_fixed_point(model, a, b, tol=1e-10, qz_criterium=1 + 1e-6, complex_threshold=1e-15):
    """lyapunov_symm.m with method=3 (fixed point iterations, as evaluate_planner_objective calls it).
    MATLAB keeps the last solution in a persistent variable and starts the next call (of the same
    size) from it with at most 300 iterations (2000 for the first call); pydyn keeps it per model.
    When the iterations do not converge, Dynare switches to the Schur method for the rest of the session."""
    from .moments import lyapunov_symm
    state = model.__dict__.setdefault("_lyap_fp", {"X": None, "method": 3})
    if state["method"] == 3:
        X = state["X"]
        if X is None or len(X) != len(b):
            X, max_it = b.copy(), 2000
        else:
            max_it = 300
        it, evol = 0, 100.0
        at = a.T
        while evol > tol and it < max_it:
            X_old = X
            X = a @ X @ at + b
            evol = float(np.max(np.sum(np.abs(X - X_old), axis=0)))
            it += 1
        state["X"] = X
        if it >= max_it:
            print(f"lyapunov_symm:: convergence not achieved in solution of Lyapunov equation after {it} "
                  "iterations, switching method from 3 to 0")
            state["method"] = 0
        else:
            return X
    return lyapunov_symm(a, b, qz_criterium, complex_threshold)[0]


@dataclass
class PlannerObjectiveValue:
    """``oo_.planner_objective_value``. ``conditional`` is a float, except under Ramsey policy where Dynare
    reports two values: with the initial Lagrange multipliers at zero and at their steady state."""
    unconditional: float
    conditional: object
    zero_initial_multiplier: float | None = None
    steady_initial_multiplier: float | None = None


def _histval_vector(model, mult_idx):
    """M_.endo_histval (lag 0 column), or None without histval: as the histval block and
    dynamic_set_auxiliary_series build it (variables not set are 0, not their steady state)."""
    from types import SimpleNamespace
    from .perfect_foresight import _histval
    sts = _statements(model, "histval")
    if not sts:
        return None
    ws = SimpleNamespace(endo_histval=None, exo_histval=None)
    _histval(model, ws, sts[-1], model._stmt_values.get(id(sts[-1]), {}))
    if ws.endo_histval is None:              # a model without lags: M_.endo_histval is empty
        return None
    return np.asarray(ws.endo_histval, float)


def _initial_state(model, ys, dr, nstatic, nspred, histval=None, u=None):
    """get_initial_state of evaluate_planner_objective.m: (yhat_L_SS, yhat_L_0, u)."""
    mult = [a["endo_index"] - 1 for a in model.raw.get("aux_vars", []) if a.get("type") == 6]
    non_mult = np.ones(model.n_endo, bool)
    non_mult[mult] = False
    y_ss = np.array(ys, float)
    y_0 = np.zeros(model.n_endo)
    hv = _histval_vector(model, mult) if histval is None else np.asarray(histval, float)
    if hv is not None:
        y_ss[non_mult] = hv[non_mult]
        y_0[non_mult] = hv[non_mult]
    else:
        y_0[non_mult] = ys[non_mult]
    sv = dr["order_var"][nstatic:nstatic + nspred]
    yhat_0 = y_0[sv] - ys[sv]
    yhat_ss = y_ss[sv] - ys[sv]
    if u is None:
        u = model.exo_steady().copy()
        if model.det_shocks:
            # Dynare's code: positions in the concatenated list of periods index the list of shocks
            periods, entries = [], []
            for e in model.det_shocks:
                periods.extend(range(e[1], e[2] + 1))
                entries.append(e)
            if any(p == 0 for p in periods):
                print("evaluate_planner_objective: M_.det_shocks contains values for the predetermined t=0 period. "
                      "Dynare will ignore them. Use histval to set the value of lagged innovations.")
            if any(p > 1 for p in periods):
                print("evaluate_planner_objective: Shock values for periods not contained in the initial information "
                      "set (t=1) have been provided. Note that they will be ignored.")
            for k in [k for k, p in enumerate(periods) if p == 1]:
                if k >= len(entries):
                    raise OptimalPolicyError("evaluate_planner_objective: index exceeds the number of deterministic "
                                             "shocks (Dynare fails here too)")
                val = entries[k][3]
                u[entries[k][0]] = float(np.ravel(val)[0])
    return yhat_ss, yhat_0, np.asarray(u, float)


def evaluate_planner_objective(model, dr=None, order=None, u=None, histval=None, ramsey=None, discretionary=None,
                               params=None, Sigma_e=None, print_results=False) -> PlannerObjectiveValue:
    """Welfare (``evaluate_planner_objective``) from a decision rule.

    ``dr``: rules in Dynare's layout (``first_order(m).as_dr()``, ``k_order(m)``, or
    :func:`discretionary_policy`'s result); computed at ``order`` (default: the .mod file's stoch_simul
    order, else 2) when omitted. Orders 1 and 2 are available (Dynare needs its ``k_order_welfare``
    MEX beyond). ``u`` is the shock vector of the first period of the conditional welfare: Dynare takes
    ``oo_.exo_simul(1,:)``, which is the first *simulated* shock when ``stoch_simul`` ran with
    ``periods > 0``, else the exogenous steady state (the default here), or the deterministic shocks of
    period 1. ``histval`` (default: the .mod file's last ``histval`` block) sets the initial state.

    Unconditional welfare is ``E[U]/(1-beta)``; conditional welfare starts from the deterministic
    steady state (or ``histval``), under Ramsey policy with the multipliers at their steady state and
    at zero."""
    from .perturbation import dr_ordering
    ramsey = is_ramsey(model) if ramsey is None else ramsey
    discretionary = is_discretionary(model) if discretionary is None else discretionary
    so = _opt.stoch_options(model)
    if hasattr(dr, "as_dr"):
        dr = dr.as_dr()
    if u is None and not model.det_shocks and int(so.get("periods", 0) or 0) > 0:
        _opt.warn_once(model, "evaluate_planner_objective: the .mod file simulates (periods > 0), so Dynare's conditional "
                              "welfare starts from the first simulated shock (oo_.exo_simul(1,:)); pydyn uses the "
                              "exogenous steady state unless u= is given")
    if order is None:
        order = int(dr["order"]) if dr is not None and "order" in dr else int(so.get("order", 2))
    if dr is None:
        if discretionary:
            dr = discretionary_policy(model).dr
        elif order == 1:
            from .perturbation import first_order
            dr = first_order(model, loglinear=False).as_dr()
        else:
            from .korder import k_order
            dr = k_order(model, order=order, loglinear=False)
    if params is None:
        params = dr["params"] if dr.get("params") is not None else model.params
    params = np.asarray(params, float)
    Sigma = np.asarray(model.Sigma_e if Sigma_e is None else Sigma_e, float)
    beta = planner_discount(model, params)
    if beta >= 1:
        print("evaluate_planner_objective: the planner discount factor is not strictly smaller than 1. "
              "Unconditional welfare will not be finite.")
    po = planner_objective(model)
    ys = np.asarray(dr["ys"], float)
    if dr.get("loglinear"):
        raise NotImplementedError("evaluate_planner_objective with loglinear rules")
    _, _, cnt = dr_ordering(model.lead_lag_incidence)
    nstatic, nspred = cnt["nstatic"], cnt["npred"] + cnt["nboth"]
    ov = np.asarray(dr["order_var"], int)
    n = model.n_endo
    ghx, ghu = np.asarray(dr["ghx"], float), np.asarray(dr["ghu"], float)
    nx = ghu.shape[1]
    gy, gu = np.zeros((n, nspred)), np.zeros((n, nx))
    gy[ov], gu[ov] = ghx, ghu
    Gy, Gu = ghx[nstatic:nstatic + nspred], ghu[nstatic:nstatic + nspred]
    U = po.value(ys, params)
    Uy = po.gradient(ys, params)[None, :]
    I = np.eye(nspred)
    hez = _hessian_eq_zero(model)
    if order == 1 and not discretionary:
        EW = U / (1 - beta)
        Wbar = U / (1 - beta)
        Wy = np.linalg.solve((I - beta * Gy).T, (Uy @ gy).T).T
        Wu = Uy @ gu + beta * Wy @ Gu
        yss, y0, u = _initial_state(model, ys, dr, nstatic, nspred, histval, u)
        W_SS = float((Wbar + Wy @ yss + Wu @ u)[0])
        W_0 = float((Wbar + Wy @ y0 + Wu @ u)[0])
    elif order == 2 or discretionary:
        lq = discretionary or hez
        Uyy = po.hessian(ys, params).reshape(1, -1, order="F")
        Uyygygy, Uyygugu, Uyygugy = _akc(Uyy, gy, gy), _akc(Uyy, gu, gu), _akc(Uyy, gu, gy)
        from .moments import theoretical_moments
        var_list = [model.endo_names[i] for i in ov[nstatic:nstatic + nspred]]
        mom_order = 1 if lq else 2
        if not lq:
            for k in ("ghxx", "ghxu", "ghuu", "ghs2"):
                if k not in dr:
                    raise OptimalPolicyError("evaluate_planner_objective at order 2 needs second-order rules")
            if so.get("pruning"):
                print("evaluate_planner_objective: pruning option is not supported and will be ignored")
        if nspred:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mo = theoretical_moments(model, dr, var_list, Sigma_e=Sigma, order=mom_order, pruning=False,
                                         nodecomposition=True, conditional_variance_decomposition=[], ar=0)
            Ey = np.array(mo.mean, float)
            Vy = np.array(mo.var, float)
        else:
            Ey, Vy = np.zeros(0), np.zeros((0, 0))
        if not lq and (np.any(np.isnan(Ey)) or np.any(np.isnan(Vy))):
            print("evaluate_planner_objective: encountered NaN moments in the endogenous variables; "
                  "they are replaced by a large number (options_.huge_number)")
        Ey[np.isnan(Ey)] = 1e7
        Vy[np.isnan(Vy)] = 1e7
        Euu = Sigma.reshape(-1, order="F")
        Wbar = U / (1 - beta)
        Wy = np.linalg.solve((I - beta * Gy).T, (Uy @ gy).T).T
        Wu = Uy @ gu + beta * Wy @ Gu
        qz = _opt.qz_criterium(model)
        if not lq:
            gyy = np.zeros((n, nspred * nspred)); gyy[ov] = dr["ghxx"]
            gyu = np.zeros((n, nspred * nx)); gyu[ov] = dr["ghxu"]
            guu = np.zeros((n, nx * nx)); guu[ov] = dr["ghuu"]
            gss = np.zeros(n); gss[ov] = np.ravel(dr["ghs2"])
            Gyy = np.asarray(dr["ghxx"], float)[nstatic:nstatic + nspred]
            Gyu = np.asarray(dr["ghxu"], float)[nstatic:nstatic + nspred]
            Guu = np.asarray(dr["ghuu"], float)[nstatic:nstatic + nspred]
            Gss = np.ravel(dr["ghs2"])[nstatic:nstatic + nspred]
            Eyhat = Ey - ys[ov[nstatic:nstatic + nspred]]
            EU = U + Uy @ gy @ Eyhat + 0.5 * ((Uyygygy + Uy @ gyy) @ Vy.reshape(-1, order="F")
                                              + (Uyygugu + Uy @ guu) @ Euu + Uy @ gss)
            rhs = (Uyygygy + Uy @ gyy + beta * Wy @ Gyy).reshape(nspred, nspred, order="F")
            Wyy = lyapunov_symm_fixed_point(model, np.sqrt(beta) * Gy.T, rhs, qz_criterium=qz).reshape(1, -1, order="F")
            Wuu = Uyygugu + Uy @ guu + beta * (_akc(Wyy, Gu, Gu) + Wy @ Guu)
            Wss = (Uy @ gss + beta * (Wy @ Gss + Wuu @ Euu)) / (1 - beta)
            Wyu = Uyygugy + Uy @ gyu + beta * (_akc(Wyy, Gu, Gy) + Wy @ Gyu)
        else:
            EU = U + 0.5 * Uyygugu @ Euu
            if model.info.get("maximum_endo_lag", 0) and nspred:
                Eyhat = Ey - ys[ov[nstatic:nstatic + nspred]]
                EU = EU + Uy @ gy @ Eyhat + 0.5 * Uyygygy @ Vy.reshape(-1, order="F")
            Wyy = lyapunov_symm_fixed_point(model, np.sqrt(beta) * Gy.T, Uyygygy.reshape(nspred, nspred, order="F"),
                                            qz_criterium=qz).reshape(1, -1, order="F") if nspred else np.zeros((1, 0))
            Wuu = Uyygugu + beta * _akc(Wyy, Gu, Gu)
            Wss = beta * Wuu @ Euu / (1 - beta)
            Wyu = Uyygugy + beta * _akc(Wyy, Gu, Gy)
        EW = float(np.ravel(EU)[0]) / (1 - beta)
        yss, y0, u = _initial_state(model, ys, dr, nstatic, nspred, histval, u)

        def W(yh):
            yh, uu = yh[:, None], u[:, None]
            return float(np.ravel(Wbar + Wy @ yh + Wu @ uu + _akc(Wyu, yh, uu)
                                  + 0.5 * (Wss + _akc(Wyy, yh, yh) + _akc(Wuu, uu, uu)))[0])
        W_SS, W_0 = W(yss), W(y0)
    else:
        raise NotImplementedError("evaluate_planner_objective beyond order 2 needs Dynare's k_order_welfare "
                                  "(a MEX file pydyn does not have)")
    if ramsey:
        out = PlannerObjectiveValue(EW, {"zero_initial_multiplier": W_0, "steady_initial_multiplier": W_SS},
                                    zero_initial_multiplier=W_0, steady_initial_multiplier=W_SS)
    else:
        out = PlannerObjectiveValue(EW, W_SS)
    if print_results:
        print(format_planner_objective(out, ramsey, discretionary))
    return out


def evaluate_planner_objective_perfect_foresight(model, pf, params=None) -> PlannerObjectiveValue:
    """``evaluate_planner_objective`` after a perfect foresight simulation of a Ramsey model: the
    planner objective summed backwards along the simulated path (``pf``, a
    :class:`pydyn.PerfectForesightResult`), from ``U(T)/(1-beta)`` in the last simulated period.
    ``conditional`` is the welfare of the path, ``unconditional`` the terminal value ``U(T)/(1-beta)``."""
    params = np.asarray(pf.params if params is None and getattr(pf, "params", None) is not None else
                        (model.params if params is None else params), float)
    beta = planner_discount(model, params)
    if beta >= 1:
        print("evaluate_planner_objective: the planner discount factor is not strictly smaller than 1. "
              "Unconditional welfare will not be finite.")
    po = planner_objective(model)
    endo = np.asarray(pf.endo, float)
    lead, lag = int(model.info.get("maximum_lead", 1)), int(model.info.get("maximum_lag", 1))
    T = endo.shape[0]
    EW = po.value(endo[T - lead - 1], params) / (1 - beta)
    W = EW
    for t in range(T - lead, lag, -1):              # MATLAB t = T-lead : -1 : 1+lag
        W = po.value(endo[t - 1], params) + beta * W
    return PlannerObjectiveValue(EW, W)


def format_planner_objective(v: PlannerObjectiveValue, ramsey: bool, discretionary: bool = False) -> str:
    if ramsey:
        return (f"\nApproximated value of unconditional welfare:  {v.unconditional:10.8f}\n"
                f"\nApproximated value of conditional welfare:\n"
                f"    - with initial Lagrange multipliers set to 0: {v.zero_initial_multiplier:10.8f}\n"
                f"    - with initial Lagrange multipliers set to steady state: {v.steady_initial_multiplier:10.8f}\n")
    kind = " with discretionary policy" if discretionary else ""
    return (f"\nApproximated value of unconditional welfare{kind}:  {v.unconditional:10.8f}\n"
            f"\nApproximated value of conditional welfare{kind}: {v.conditional:10.8f}\n")


# =========================================================================== discretionary policy
@dataclass
class DiscretionaryPolicyResult:
    """Outcome of :func:`discretionary_policy`: ``dr`` (Dynare's ``oo_.dr``: ``ys``, ``ghx``, ``ghu``,
    ``order_var``, ``state_var``, order 1), the transition ``H`` and impact ``G`` in declaration order
    (y_t = H y_{t-1} + G e_t), the welfare (``oo_.planner_objective_value``) and the iterations."""
    dr: dict
    H: np.ndarray
    G: np.ndarray
    instruments: list
    welfare: PlannerObjectiveValue | None = None
    iterations: int = 0

    def as_dr(self):
        return self.dr

    @property
    def ghx(self):
        return self.dr["ghx"]

    @property
    def ghu(self):
        return self.dr["ghu"]


class DiscretionaryPolicyError(OptimalPolicyError):
    pass


def _sylvester_doubling(d, g, h, tol, maxit):
    v = d.copy()
    for _ in range(maxit):
        vadd = g @ v @ h
        v = v + vadd
        if np.linalg.norm(vadd, 1) <= tol * np.linalg.norm(v, 1):
            break
        g = g @ g
        h = h @ h
    return v


def _sylvester_hessenberg_schur(d, g, h):
    """Fallback of discretionary_policy_engine: v = d + g v h (solved directly)."""
    n, m = d.shape
    from scipy import linalg
    # vec(v) = vec(d) + (h' kron g) vec(v)
    K = np.eye(n * m) - np.kron(h.T, g)
    return linalg.solve(K, d.reshape(-1, order="F")).reshape(n, m, order="F")


def discretionary_policy_engine(AAlag, AA0, AAlead, BB, bigw, instr_id, beta, maxit=3000, tol=np.sqrt(_EPS),
                                qz_criterium=1.000001, H00=None):
    """discretionary_policy_engine.m (Dennis 2007): iterate on the policy rule and value function of
    the linear-quadratic problem until the rule converges. ``instr_id`` are 0-based. Returns
    (H, G, retcode, iterations)."""
    eq_nbr, endo_nbr = AAlag.shape
    exo_nbr = BB.shape[1]
    instr_id = np.asarray(instr_id, int)
    y = np.setdiff1d(np.arange(endo_nbr), instr_id)
    instr_nbr = instr_id.size
    A0, A1, A2 = AA0[:, y], -AAlag[:, y], -AAlead[:, y]
    A3, A4, A5 = -AA0[:, instr_id], -AAlead[:, instr_id], -BB
    W, Q = bigw[np.ix_(y, y)], bigw[np.ix_(instr_id, instr_id)]
    A6 = -AAlag[:, instr_id]
    aux = np.any(A6 != 0, axis=0)
    naux = int(aux.sum())
    ny, mm = eq_nbr, eq_nbr + naux
    A00 = np.zeros((mm, mm)); A00[:ny, :ny] = A0; A00[ny:, ny:] = np.eye(naux)
    A11 = np.zeros((mm, mm)); A11[:ny, :ny] = A1; A11[:ny, ny:] = A6[:, aux]
    A22 = np.zeros((mm, mm)); A22[:ny, :ny] = A2
    A33 = np.zeros((mm, instr_nbr)); A33[:ny] = A3; A33[ny:, aux] = np.eye(naux)
    A44 = np.zeros((mm, instr_nbr)); A44[:ny] = A4
    A55 = np.zeros((mm, exo_nbr)); A55[:ny] = A5
    WW = np.zeros((mm, mm)); WW[:ny, :ny] = W
    endo_augm_id = np.setdiff1d(np.arange(endo_nbr + naux), instr_id)
    H0 = np.zeros((endo_nbr + naux, endo_nbr + naux))
    if H00 is not None:
        H0[:endo_nbr, :endo_nbr] = H00
    H10 = H0[np.ix_(endo_augm_id, endo_augm_id)]
    F10 = H0[np.ix_(instr_id, endo_augm_id)]
    it = 0
    H1, F1 = H10, F10
    with np.errstate(all="ignore"):
        while True:
            it += 1
            P = _sylvester_doubling(WW + beta * F1.T @ Q @ F1, beta * H1.T, H1, tol, maxit)
            if not np.all(np.isfinite(P)):
                P = _sylvester_hessenberg_schur(WW + beta * F1.T @ Q @ F1, beta * H1.T, H1)
                if not np.all(np.isfinite(P)):
                    return None, None, 2, it
            D = A00 - A22 @ H1 - A44 @ F1
            Dinv = np.linalg.inv(D)
            A3DPD = A33.T @ Dinv.T @ P @ Dinv
            F1 = -np.linalg.solve(Q + A3DPD @ A33, A3DPD @ A11)
            H1 = Dinv @ (A11 + A33 @ F1)
            NQ = np.max(np.abs(np.vstack([H1, F1]) - np.vstack([H10, F10])))
            if np.isnan(NQ):
                rcode = 3
            elif it > maxit:
                rcode = 2
            elif NQ < tol:
                rcode = 1
            else:
                rcode = 0
            if rcode:
                break
            H10, F10 = H1, F1
    if rcode == 3:
        return None, None, 63, it
    if rcode == 2:
        return None, None, 61, it
    if np.max(np.abs(np.linalg.eigvals(H1))) > qz_criterium:
        return None, None, 62, it
    F2 = -np.linalg.solve(Q + A3DPD @ A33, A3DPD @ A55)
    H2 = Dinv @ (A55 + A33 @ F2)
    H = np.zeros((endo_nbr + naux, endo_nbr + naux))
    G = np.zeros((endo_nbr + naux, exo_nbr))
    H[np.ix_(endo_augm_id, endo_augm_id)] = H1
    H[np.ix_(instr_id, endo_augm_id)] = F1
    G[endo_augm_id] = H2
    G[instr_id] = F2
    if naux:                                        # lagged instruments: their auxiliary columns
        H[:, instr_id[aux]] = H[:, H.shape[1] - naux:]
    return H[:endo_nbr, :endo_nbr], G[:endo_nbr], 0, it


_DP_MESSAGES = {61: "Discretionary policy: maximum number of iterations has been reached. Procedure failed.",
                62: "Discretionary policy: some eigenvalues greater than options_.qz_criterium. Model potentially unstable.",
                63: "Discretionary policy: NaN elements are present in the solution. Procedure failed.",
                64: "discretionary_policy: the derivatives of the objective function contain NaN.",
                65: "discretionary_policy: the model must be written in deviation form and not have constant terms or "
                    "an analytical steady state meeds to be provided.",
                66: "discretionary_policy: the objective function must have zero first order derivatives.",
                2: "discretionary_policy: the Sylvester equation of the value function could not be solved."}


def discretionary_policy(model, params=None, tol=None, maxit=None, qz_criterium=None, H0=None, welfare=True,
                         u=None, warm_start=True) -> DiscretionaryPolicyResult:
    """``discretionary_policy`` (Dynare 7.2's discretionary_policy.m / discretionary_policy_1.m): the
    optimal policy under discretion of a linear-quadratic problem (Dennis 2007), with the
    ``instruments`` and ``planner_objective`` of the .mod file. ``tol`` (``discretionary_tol``,
    default 1e-7) and ``maxit`` (``dp.maxit``, 3000) default to the .mod file's options.

    As in Dynare the objective's first derivatives and Hessian are taken at zero, the model's at the
    steady state given by a steady-state file or block, else at zero (the model must then be written
    in deviations). Like Dynare's persistent variable, the previous solution of the same model starts
    the iterations (``warm_start=False``, or ``H0``, changes that). ``welfare`` adds
    :func:`evaluate_planner_objective` (``u``: see there)."""
    from .perturbation import dr_ordering
    from .steady import SteadyStateError, DYNARE_TOLF
    opts = {**model.options.get("discretionary_policy", {})}
    so = _opt.stoch_options(model)
    if opts.get("loglinear") or so.get("loglinear"):
        raise DiscretionaryPolicyError("discretionary_policy is not compatible with `loglinear` option set to 1")
    if int(opts.get("order", 1)) != 1:
        raise DiscretionaryPolicyError("discretionary_policy: only linear-quadratic problems can be solved")
    tol = float(opts.get("discretionary_tol", 1e-7)) if tol is None else float(tol)
    maxit = int(opts.get("dp.maxit", 3000)) if maxit is None else int(maxit)
    qz = _opt.qz_criterium(model, qz_criterium)
    params = np.array(model.params if params is None else params, float)
    n, nx = model.n_endo, model.n_exo
    eq_nbr = model.jac_dims[0]
    instr_nbr = n - eq_nbr
    if instr_nbr == 0:
        raise DiscretionaryPolicyError("discretionary_policy:: There are no available instruments, because the model "
                                       "has as many equations as variables.")
    names = _instrument_names(model)
    if len(names) < instr_nbr:
        raise DiscretionaryPolicyError("discretionary_policy:: There are fewer declared instruments than omitted equations.")
    if len(names) > instr_nbr:
        raise DiscretionaryPolicyError("discretionary_policy:: There are more declared instruments than omitted equations.")
    for v in names:
        if v not in model._ie:
            raise DiscretionaryPolicyError(f"discretionary_policy:: instrument {v} not found")
    instr_id = np.array([model._ie[v] for v in names], int)
    beta = planner_discount(model, params)
    x = model.exo_steady()
    if _has_steady_state_file(model):
        ys0 = np.array([model.initval.get(v, 0.0) for v in model.endo_names], float)
        try:
            with np.errstate(all="ignore"):
                ys, params = _evaluate_steady_state_file(model, ys0, x, params)
        except SteadyStateError as exc:
            raise DiscretionaryPolicyError(f"steady-state file failed: {exc}", 19) from None
        if model.raw.get("aux_vars"):
            ys = set_auxiliary_variables(model, ys, x, params)
        if not model.options.get("steady", {}).get("steadystate.nocheck"):
            res = model.static_resid(ys, x, params)
            if not np.max(np.abs(res)) <= DYNARE_TOLF or np.any(np.isnan(res)):
                raise DiscretionaryPolicyError("the steady state given by the steady-state file does not solve the "
                                               "static model (Dynare's error 19)", 19)
    else:
        ys = np.zeros(n)
    po = planner_objective(model)
    y0 = np.zeros(n)
    Uy = po.gradient(y0, params)
    if np.any(np.isnan(Uy)):
        raise DiscretionaryPolicyError(_DP_MESSAGES[64], 64)
    if np.any(Uy != 0):
        raise DiscretionaryPolicyError(_DP_MESSAGES[66], 66)
    Wm = po.hessian(y0, params)
    yy = np.tile(ys, 3)
    tolf = float(model.options.get("steady", {}).get("solve_tolf", DYNARE_TOLF))
    resid = model.dynamic_resid(yy, np.zeros(nx), params, ys)
    if not np.max(np.abs(resid)) <= tolf:
        raise DiscretionaryPolicyError(_DP_MESSAGES[65], 65)
    J = model.dynamic_jac(yy, np.zeros(nx), params, ys)
    if H0 is None and warm_start:
        H0 = model.__dict__.get("_dp_Hold")
    H, G, info, it = discretionary_policy_engine(J[:, :n], J[:, n:2 * n], J[:, 2 * n:3 * n], J[:, 3 * n:], Wm,
                                                 instr_id, beta, maxit, tol, qz, H0)
    if info:
        raise DiscretionaryPolicyError(_DP_MESSAGES.get(info, f"discretionary_policy failed (info {info})"), info)
    model.__dict__["_dp_Hold"] = H
    order_var, state_var, _ = dr_ordering(model.lead_lag_incidence)
    ghx = H[np.ix_(order_var, state_var)] if model.info.get("maximum_endo_lag", 0) else np.zeros((n, 0))
    dr = {"ys": ys, "order_var": order_var, "state_var": state_var, "ghx": ghx, "ghu": G[order_var],
          "order": 1, "params": params}
    out = DiscretionaryPolicyResult(dr=dr, H=H, G=G, instruments=names, iterations=it)
    if welfare:
        out.welfare = evaluate_planner_objective(model, dr, order=1, u=u, ramsey=False, discretionary=True, params=params)
    return out


# =========================================================================== optimal simple rules
@dataclass
class OSRResult:
    """``oo_.osr``: optimal parameters, objective value at the optimum, optimiser information; the
    model's parameters are set to the optimum, and ``dr`` holds the rules there."""
    optim_params: dict
    objective_function: float
    x: np.ndarray
    initial_objective: float
    exitflag: int = 0
    message: str = ""
    iterations: int = 0
    func_count: int = 0
    dr: dict | None = None
    error_indicator: int = 1


class OSRError(OptimalPolicyError):
    pass


#: resol's error codes that osr.objective penalises with info(2) (when finite)
_OSR_INFO2 = {3, 4, 5, 6, 19, 20, 21, 23, 26, 81, 84, 85}


def _osr_weights(model):
    """M_.osr.variable_weights and M_.osr.variable_indices (0-based), as the preprocessor writes them:
    a covariance weight is set at (name1, name2) only."""
    W = np.zeros((model.n_endo, model.n_endo))
    idx = []
    for a, b, v in model.options.get("optim_weights", {}).get("weights", []):
        i = model._ie[a]
        j = model._ie[b] if b else i
        W[i, j] = v
        idx += [i] if b is None else [i, j]
    return W, idx


def variance_of_endogenous_variables(model, dr, i_var, Sigma_e=None, qz_criterium=None, schur_vec_tol=1e-11):
    """osr.get_variance_of_endogenous_variables: the variance of the variables in rows ``i_var`` of the
    decision rules (DR order), Inf where they are not stationary."""
    from .moments import lyapunov_symm
    from .perturbation import dr_ordering
    Sigma = np.asarray(model.Sigma_e if Sigma_e is None else Sigma_e, float)
    ghx = np.asarray(dr["ghx"], float)[i_var]
    ghu = np.asarray(dr["ghu"], float)[i_var]
    _, _, cnt = dr_ordering(model.lead_lag_incidence)
    ns, nspred = cnt["nstatic"], cnt["npred"] + cnt["nboth"]
    A = np.asarray(dr["ghx"], float)[ns:ns + nspred]
    B = np.asarray(dr["ghu"], float)[ns:ns + nspred]
    vx, u = lyapunov_symm(A, B @ Sigma @ B.T, _opt.qz_criterium(model, qz_criterium))
    n = len(i_var)
    if u.shape[1] > 0:
        i_stat = np.flatnonzero(np.any(np.abs(ghx @ u) < schur_vec_tol, axis=1))
    else:
        i_stat = np.arange(n)
    out = np.full((n, n), np.inf)
    ix = np.ix_(i_stat, i_stat)
    out[ix] = ghx[i_stat] @ vx @ ghx[i_stat].T + ghu[i_stat] @ Sigma @ ghu[i_stat].T
    return out


def _resol(model, params, order):
    """resol.m: steady state and decision rules at ``params``. Returns (dr, info)."""
    from .perturbation import BlanchardKahnError, first_order
    from .steady import SteadyStateError, steady_state
    from .korder import k_order, KOrderError
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                ys, p = steady_state(model, params)
                if order == 1:
                    dr = first_order(model, ys=ys, params=p, loglinear=False).as_dr()
                    dr["params"] = p
                else:
                    dr = k_order(model, order=order, ys=ys, params=p, loglinear=False)
    except BlanchardKahnError as e:
        return None, list(getattr(e, "info", (3, np.nan)))
    except OptimalPolicyError as e:
        return None, [e.info or 20, np.nan]
    except (SteadyStateError, KOrderError, np.linalg.LinAlgError, ValueError, ArithmeticError):
        return None, [20, np.nan]
    return dr, [0]


class _OSRProblem:
    """The set-up of osr.run and the objective osr.objective."""

    def __init__(self, model, param_names=None, order=None, opt_algo=None):
        st_opts = dict(model.options.get("osr", {}))
        self.model, self.st_opts = model, st_opts
        self.opt_algo = st_opts.get("osr.opt_algo", 4) if opt_algo is None else opt_algo
        if order is None:
            order = int(st_opts.get("order", _opt.stoch_options(model).get("order", 2)))
        self.order = order
        if param_names is None:
            param_names = list(model.options.get("osr_params", {}).get("symbol_list", []))
        if not param_names:
            raise OSRError("OSR: no parameters to optimise (osr_params)")
        self.param_names = list(param_names)
        self.i_params = np.array([model._ip[p] for p in param_names], int)
        kinds = [s["statementName"] for s in model.statements]
        # the preprocessor's planner_discount_declared: a planner_objective statement before osr
        self.use_planner = "planner_objective" in kinds and \
            kinds.index("planner_objective") < (kinds.index("osr") if "osr" in kinds else len(kinds))
        if not self.use_planner:
            if order != 1:
                raise OSRError("OSR: OSR without planner_objective only supports order=1.")
            W, i_var = _osr_weights(model)
            if np.any(np.isinf(W)):
                raise OSRError("OSR: At least one of the optim_weights is infinite.")
            if not i_var:
                raise OSRError("OSR: there is neither a planner_objective nor optim_weights")
            self.i_var = np.unique(i_var)
            self.W = W[np.ix_(self.i_var, self.i_var)]
        elif st_opts.get("analytic_derivation"):
            raise OSRError("OSR: analytic_derivation is only available at order=1 with optim_weights.")
        if model.info.get("maximum_lead", 1) == 0:
            raise OSRError("OSR: Backward or static model: no point in using OSR")
        self.params0 = np.array(model.params, float)
        if np.any(np.isnan(self.params0[self.i_params])):
            raise OSRError("OSR: At least one of the initial parameter values for osr_params is NaN")
        self.qz = _opt.qz_criterium(model)

    def __call__(self, x):
        """``(loss, info)`` as osr.objective returns them (loss 1e8 and info[3] the penalty on failure)."""
        model = self.model
        p = self.params0.copy()
        p[self.i_params] = x
        dr, info = _resol(model, p, self.order)
        if info[0]:
            i2 = info[1] if len(info) > 1 else np.nan
            pen = i2 if (info[0] in _OSR_INFO2 and np.isfinite(i2)) else 0.1
            return 1e8, [info[0], i2, 0, pen]
        if not self.use_planner:
            vx = variance_of_endogenous_variables(model, dr, dr_inv(dr)[self.i_var], qz_criterium=self.qz)
            nz = self.W != 0                      # sparse weights: 0 * Inf does not occur
            loss = float(np.sum(self.W[nz] * vx[nz]))
        else:
            with np.errstate(all="ignore"):
                loss = -evaluate_planner_objective(model, dr, order=self.order, ramsey=False, discretionary=False,
                                                   params=p).unconditional
        if np.isinf(loss):
            return 1e8, [174, 0, 0, 0.1]
        if np.isnan(loss):
            return 1e8, [175, 0, 0, 0.1]
        return loss, [0]


def _parse_optim_opt(text):
    """``optim=('Name',value,...)`` as the preprocessor writes it: ''Name'',value pairs."""
    import re
    out = {}
    for name, val in re.findall(r"'+([A-Za-z_]\w*)'+\s*,\s*('+[^']*'+|[^,]+)", str(text)):
        val = val.strip()
        if val.startswith("'"):
            out[name] = val.strip("'")
        else:
            try:
                out[name] = float(val)
            except ValueError:
                out[name] = val
    return out


def osr_objective(model, values, param_names=None, order=None) -> float:
    """The OSR loss at the parameter ``values`` (osr.objective; 1e8 when the model cannot be solved there)."""
    return _OSRProblem(model, param_names, order)(np.asarray(values, float))[0]


def osr(model, param_names=None, opt_algo=None, order=None, bounds=None, maxit=None, tolf=None,
        optim_opt=None, print_results=False) -> OSRResult:
    """``osr`` (Dynare 7.2's osr.run / osr.objective): the values of ``osr_params`` that minimise the
    weighted (co)variances of ``optim_weights`` (order 1), or, when the .mod file has a
    ``planner_objective`` before ``osr``, maximise the unconditional welfare
    (:func:`evaluate_planner_objective`; orders 1 and 2).

    The minimiser is Dynare's default, csminwel (``opt_algo=4``; options ``MaxIter``, ``TolFun``,
    ``InitialInverseHessian``, ``NumgradAlgorithm``, ``NumgradEpsilon`` through ``optim_opt`` as a dict,
    or the deprecated ``maxit``/``tolf``). Other values of ``opt_algo`` (MATLAB's fmincon/fminunc/
    fminsearch, CMA-ES, simulated annealing, ...) are refused: pydyn does not have them. As in Dynare,
    ``osr_params_bounds`` need a constrained optimiser, so they are refused too; :func:`osr_objective`
    evaluates the loss at any parameter values.

    The model's parameters are set to the optimum (as Dynare does); the result's ``dr`` holds the rules
    there."""
    prob = _OSRProblem(model, param_names, order, opt_algo)
    opt_algo, st_opts = prob.opt_algo, prob.st_opts
    if st_opts.get("analytic_derivation"):
        raise NotImplementedError("osr(analytic_derivation): the analytic derivatives of the moments "
                                  "(identification.get_perturbation_params_derivs) are not ported")
    if bounds is None:
        bl = model.options.get("osr_params_bounds", {}).get("bounds", [])
        if bl:
            bounds = np.column_stack([np.full(len(prob.param_names), -np.inf), np.full(len(prob.param_names), np.inf)])
            for name, lo, hi in bl:
                bounds[prob.param_names.index(name)] = (lo, hi)
    oo = dict(optim_opt or {})
    if st_opts.get("optim_opt") and optim_opt is None:
        oo = _parse_optim_opt(st_opts["optim_opt"])
        known = {"MaxIter", "TolFun", "InitialInverseHessian", "NumgradAlgorithm", "NumgradEpsilon", "verbosity", "SaveFiles"}
        if opt_algo == 4 and set(oo) - known:
            warnings.warn(f"csminwel: Unknown option(s) {sorted(set(oo) - known)}")      # as Dynare
    if maxit is None and st_opts.get("osr.maxit") is not None:
        maxit = int(st_opts["osr.maxit"])
    if tolf is None and st_opts.get("osr.tolf") is not None:
        tolf = float(st_opts["osr.tolf"])
    if maxit is not None or tolf is not None:
        warnings.warn("OSR: The use of maxit and tolf is now deprecated. Please use the optim-option instead.")
        if opt_algo != 4:
            raise OSRError("OSR: The maxit and tolf options are not supported when not using the default opt_algo=4.")
        if maxit is not None:
            oo["MaxIter"] = maxit
        if tolf is not None:
            oo["TolFun"] = tolf
    x0 = prob.params0[prob.i_params]
    loss0, info0 = prob(x0)
    if info0[0]:
        print(f"OSR: the model cannot be solved at the initial values (Dynare's error {info0[0]})")
    if opt_algo in (5, 6, 10, 11):
        raise OSRError(f"OSR: OSR does not support opt_algo={opt_algo}.")
    if bounds is not None and not (opt_algo in (1, 2, 5, 9) or isinstance(opt_algo, str)):
        raise OSRError("OSR: OSR with bounds on parameters requires a constrained optimizer, i.e. opt_algo= 1,2 or 9.")
    if opt_algo != 4:
        raise NotImplementedError(f"osr with opt_algo={opt_algo}: pydyn has Dynare's default minimiser "
                                  "(csminwel, opt_algo=4) only")
    from .estimation import csminwel as _cw
    from types import SimpleNamespace

    def csminwel(fcn, x0, **kw):
        d = _cw(lambda x: (lambda f, info: (f, info[0], info[3] if info[0] else 0.0))(*fcn(x)), x0, **kw)
        return SimpleNamespace(fval=d["f"], x=np.asarray(d["x"], float), iterations=d["iterations"],
                               func_count=d["fcount"], retcode=d["retcode"], message=d["message"])
    H0 = oo.get("InitialInverseHessian")
    res = csminwel(prob, x0, H0=None if H0 is None else np.asarray(H0, float),
                   crit=float(oo.get("TolFun", 1e-7)), nit=int(oo.get("MaxIter", 1000)),
                   method=int(oo.get("NumgradAlgorithm", 2)), epsilon=float(oo.get("NumgradEpsilon", 1e-6)))
    model.params[prob.i_params] = res.x
    dr, info = _resol(model, model.params, prob.order)
    out = OSRResult(optim_params=dict(zip(prob.param_names, res.x.tolist())), objective_function=res.fval, x=res.x,
                    initial_objective=loss0, exitflag=res.retcode, message=res.message, iterations=res.iterations,
                    func_count=res.func_count, dr=dr, error_indicator=int(bool(info[0])))
    if print_results:
        print("\nOPTIMAL VALUE OF THE PARAMETERS\n" + "\n".join(f"{k:>12s} {v:10.6f}" for k, v in out.optim_params.items())
              + f"\n\nObjective function : {res.fval:16.6g}\n")
    return out


def dr_inv(dr):
    """inv_order_var: DR position of each variable (declaration order)."""
    ov = np.asarray(dr["order_var"], int)
    inv = np.empty_like(ov)
    inv[ov] = np.arange(ov.size)
    return inv
