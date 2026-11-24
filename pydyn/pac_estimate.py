# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# Behaviour follows Dynare 7.2: matlab/+pac/+estimate/{init, nls, iterative_ols}.m, matlab/optimization/{gauss_newton,
# jacobian}.m, matlab/utilities/estimation/{opt4nls, rewrite_equation_with_tables, write_residuals_routine,
# write_ssr_routine, get_estimated_parameters_indices}.m, matlab/{islagof, get_lhs_and_rhs,
# get_variables_and_parameters_in_equation}.m (Copyright (C) 2018-2026 Dynare Team, GPL-3.0-or-later).
"""Estimation of PAC equations: nonlinear least squares (``pac.estimate.nls``) and iterated OLS
(``pac.estimate.iterative_ols``).

Plain numerics on arrays and dseries (:mod:`pydyn.dseries`); the interface with the live ``M_`` / ``oo_`` is in
:mod:`pydyn.results_nx`.  Dynare rewrites the equation as MATLAB text and writes residual functions to disk; here the
equation (from the JSON of the preprocessor, with the PAC expectation replaced by its defining equation) is translated to
a Python function of the parameter vector with :class:`pydyn.expr.Translator`.  Where the minimiser is deterministic
and ported (csminwel, simplex, Gauss-Newton) the iterates are Dynare's; ``fmincon``, ``fminunc``, ``fminsearch`` and
``lsqnonlin`` are Octave optim / MATLAB toolbox routines that pydyn does not port: they are run with SciPy to a tight
tolerance and agree with the minimiser of the sum of squares, not with the iterates of those routines; ``annealing`` and
``particleswarm`` (random search) are refused."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import expr as _expr
from .dates import Date
from .dseries import DSeries


class PacEstimateError(Exception):
    """A MATLAB ``error()`` of Dynare's routines (the message is Dynare's)."""


class PacEstimateUnsupported(Exception):
    """What pydyn does not do."""


_ID = re.compile(r"[A-Za-z_]\w*")


def identifiers(text):
    """get_variables_and_parameters_in_expression: the names of an expression (functions included; they are
    intersected with the model's names)."""
    return set(_ID.findall(text))


# ------------------------------------------------------------------------------------------------ series evaluation
def _namespace():
    return dict(_expr.NAMESPACE)


def series_function(text, resolve, shiftable):
    """Python function of an environment dict for the Dynare expression ``text``; ``resolve(name, shift)`` gives the
    Python source of a symbol (or None)."""
    tr = _expr.Translator(resolve, shiftable=shiftable)
    src = tr(text)
    code = compile(src, "<pac>", "eval")
    ns = _namespace()

    def f(env):
        with np.errstate(all="ignore"):
            return eval(code, ns, env)
    return f


def set_auxiliary_series(ds, aux_equations, param_names, params):
    """``dynamic_set_auxiliary_series``: the auxiliary variables of ``aux_equations`` ([(name, rhs)] in order) from the
    series of ``ds``; a value that needs an observation before the sample is NaN."""
    cols = {nm: ds.data[:, j].copy() for j, nm in enumerate(ds.names)}
    n = ds.nobs

    def get(name, shift):
        if name not in cols:
            raise PacEstimateError(f"Variable {name} is missing in the database.")
        x = cols[name]
        if shift == 0:
            return x
        out = np.full(n, np.nan)
        k = -shift
        if k > 0:
            if k < n:
                out[k:] = x[:n - k]
        elif -k < n:
            out[:n + k] = x[-k:]
        return out
    names = list(ds.names)
    pidx = {nm: i for i, nm in enumerate(param_names)}
    shiftable = set(cols) | {a for a, _ in aux_equations}
    # the definitions are evaluated in dependency order (the preprocessor writes them recursively)
    defined = set(ds.names) - {a for a, _ in aux_equations}
    pending = list(aux_equations)
    while pending:
        progress = False
        for item in list(pending):
            name, rhs = item
            deps = {x for x in identifiers(rhs) if x in shiftable and x != name}
            if not deps <= defined:
                continue
            f = series_function(rhs, lambda nm, sh: (f"_g({nm!r}, {sh})" if nm in shiftable else
                                                     (f"_p[{pidx[nm]}]" if nm in pidx else None)), shiftable)
            v = f({"_g": get, "_p": params})
            v = np.broadcast_to(np.asarray(v, float), (n,)).copy()
            if name not in cols:
                names.append(name)
            cols[name] = v
            defined.add(name)
            pending.remove(item)
            progress = True
        if not progress:
            raise PacEstimateError("Variable " + sorted({x for _, r in pending for x in identifiers(r)
                                                         if x in shiftable and x not in defined})[0] + " is missing in the database.")
    return DSeries(np.column_stack([cols[nm] for nm in names]), names, ds.init)


# ------------------------------------------------------------------------------------------------ initialisation
@dataclass
class Prepared:
    pacmodl: str
    lhs: str
    rhs: str
    rname: str
    pnames: list
    enames: list
    xnames: list
    ipnames: list                       # 0-based positions in the parameter vector, in the order of ``params``
    params: list                        # [(name, value)] in that order
    data: DSeries
    range: list                         # Dates
    islagged: bool
    extra: dict = field(default_factory=dict)


def _rows(ds, dates):
    """Row positions of consecutive dates in a series (an error as dseries subsref when outside the sample)."""
    first = ds.init
    k0 = dates[0] - first
    k1 = dates[-1] - first
    if k0 < 0:
        raise PacEstimateError(f"dseries::subsref: Indices are out of bounds! Subsample cannot start before {first}.")
    if k1 >= ds.nobs:
        raise PacEstimateError(f"dseries::subsref: Indices are out of bounds! Subsample cannot end after {first + (ds.nobs - 1)}.")
    return k0, k1


def check_range(rng):
    if not rng or any(rng[i + 1] - rng[i] != 1 for i in range(len(rng) - 1)):
        raise PacEstimateUnsupported("the range of dates of the estimation must be a sequence of consecutive periods")


def init_estimation(*, kind, eqname, params, data, rng, M, json_model, json_original, param_names, endo_names, exo_names,
                    orig_endo_nbr, aux_equations, pac_model, aux_vars, params_vector):
    """pac.estimate.init.  ``kind`` is 'nls' or 'iterative_ols'; ``params`` is a list [(name, value)] in the user's order;
    ``json_model`` / ``json_original`` the lists of equations of modfile.json / modfile-original.json; ``pac_model`` the
    M_.pac.<name> fields as plain Python (see results_nx); ``aux_vars`` is a pydyn.var_model.AuxVars."""
    def eqn(eqs, name):
        for e in eqs:
            if (e.get("tags") or {}).get("name") == name:
                return e
        raise PacEstimateError(f"Equation {name} is not defined.")
    RHS = eqn(json_original, eqname)["rhs"]
    if "pac_expectation" not in RHS.lower():
        raise PacEstimateError("This is not a PAC equation.")
    m = re.search(r"\(model_name\s*=\s*(\w+)\)", RHS)
    pacmodl = m.group(1)
    e = eqn(json_model, eqname)
    lhs, rhs = e["lhs"], e["rhs"]
    pm = pac_model
    if "aux_ids" in pm:
        auxrhs = [(json_model[i - 1]["lhs"], json_model[i - 1]["rhs"]) for i in pm["aux_ids"]]
        paren = True
    elif "aux_id" in pm:
        i = pm["aux_id"]
        auxrhs = [(json_model[i - 1]["lhs"], json_model[i - 1]["rhs"])]
        paren = False
    else:
        raise PacEstimateError("Cannot find auxiliary variables for PAC expectation.")
    for a, r in auxrhs:
        rhs = re.sub(r"\b" + re.escape(a) + r"\b", lambda _m, r=r, paren=paren: f"({r})" if paren else r, rhs)
    ids = identifiers(lhs) | identifiers(rhs)
    pnames = sorted(ids & set(param_names))
    enames = sorted(ids & set(endo_names))
    xnames = sorted(ids & set(exo_names))
    # get_estimated_parameters_indices
    missing = [p for p, _ in params if p not in pnames]
    if missing:
        raise PacEstimateError("The estimated parameters must be used in equation %s." % eqname)
    ipn_user = [param_names.index(p) + 1 for p, _ in params]            # 1-based, the user's order
    ipn = list(ipn_user)
    if kind == "iterative_ols":
        ipn = [int(x) for x in np.asarray(pm["ec"]["params"]).ravel()] + [int(x) for x in np.asarray(pm["ar"]["params"]).ravel()]
        if "optim_additive" in pm:
            v = np.asarray(pm["optim_additive"]["params"], float).ravel()
            ipn += [int(x) for x in v[~np.isnan(v)]]
        if "additive" in pm:
            v = np.asarray(pm["additive"]["params"], float).ravel()
            ipn += [int(x) for x in v[~np.isnan(v)]]
        if pm.get("share_of_optimizing_agents_index") is not None:
            ipn.append(int(pm["share_of_optimizing_agents_index"]))
        ipn = [x if x in ipn_user else None for x in ipn]
    ipn = [x for x in ipn if x is not None]
    # [~, permutation] = ismember(ipnames__, ipnames_); params = orderfields(params, permutation)
    perm = [(ipn.index(x) + 1) if x in ipn else 0 for x in ipn_user]
    if kind == "iterative_ols":
        if sorted(perm) != list(range(1, len(perm) + 1)):
            raise PacEstimateUnsupported("iterative_ols: the estimated parameters are not all parameters of the PAC "
                                         "equation's error correction, autoregressive or share terms")
        params = [params[k - 1] for k in perm]
    # the auxiliary variables of the data
    if len(endo_names) > orig_endo_nbr:
        data = set_auxiliary_series(data, aux_equations, param_names, params_vector)
    for n in enames:
        if n not in data.names:
            raise PacEstimateError(f"Variable {n} is missing in the database.")
    check_range(rng)
    k0, k1 = _rows(data, rng)
    cols = [data.names.index(n) for n in enames]
    if np.any(np.isnan(data.data[k0:k1 + 1][:, cols])):
        raise PacEstimateError("Some variable values are missing in the database.")
    for n in xnames:
        if n not in data.names:
            raise PacEstimateError(f"Variable {n} is missing in the database.")
    if len(xnames) == 1:
        rname = xnames[0]
        if not np.all(np.isnan(data.data[:, data.names.index(rname)])):
            raise PacEstimateError(f"The residual ({rname}) must have NaN values in the provided database.")
    else:
        tmp = data.data[k0:k1 + 1][:, [data.names.index(n) for n in xnames]]
        idx = [j for j in range(len(xnames)) if not np.any(np.isnan(tmp[:, j]))]
        if len(idx) == len(xnames):
            raise PacEstimateError("There is no residual in this equation, all the exogenous variables are observed.")
        if len(idx) < len(xnames) - 1:
            raise PacEstimateError("It is not allowed to have more than one residual in a PAC equation")
        rname = [xnames[j] for j in range(len(xnames)) if j not in idx][0]
    rhs = _remove_residual(rhs, rname)
    return Prepared(pacmodl, lhs, rhs, rname, pnames, enames, xnames, [x - 1 for x in ipn],
                    [(p, v) for p, v in params], data, rng, "(-1)" in rhs)


def _remove_residual(rhs, rname):
    """regexprep(rhs, rname, ''): the residual disappears with one of its neighbouring plus signs (Dynare leaves the sign
    and relies on MATLAB accepting a leading or doubled one); a residual that is not an additive term is refused."""
    pat = r"\b" + re.escape(rname) + r"\b"
    ms = list(re.finditer(pat, rhs))
    if len(ms) != 1:
        raise PacEstimateUnsupported(f"the residual {rname} must appear exactly once in the equation")
    m = ms[0]
    before, after = rhs[:m.start()], rhs[m.end():]
    if after.startswith("+"):
        return before + after[1:]
    if before.endswith("+") or before == "" and after == "":
        return before[:-1] + after
    raise PacEstimateUnsupported(f"the residual {rname} is not an additive term of the equation")


# ------------------------------------------------------------------------------------------------ residual functions
def residual_function(prep, param_names, endo_names, exo_names, update, base_params):
    """``r(p)`` and ``ssr(p)`` of Dynare's written routines: the estimated parameters ``p`` are put in the parameter vector,
    the PAC parameters recomputed (``update``), the residual ``lhs - rhs`` evaluated on the data of the range (and the period
    before)."""
    data, rng = prep.data, prep.range
    k0, k1 = _rows(data, [rng[0] - 1] + rng)
    D = data.data[k0:k1 + 1]
    pidx = {n: i for i, n in enumerate(param_names)}
    colidx = {n: data.names.index(n) for n in data.names}
    shiftable = set(endo_names) | set(exo_names)
    lagged = prep.islagged

    def resolve(name, shift):
        if name in pidx and name not in shiftable:
            return f"P[{pidx[name]}]"
        if name in shiftable:
            if name not in colidx:
                raise PacEstimateError(f"Variable {name} is missing in the database.")
            k = colidx[name]
            if lagged:
                if shift == 0:
                    return f"D[1:, {k}]"
                if shift == -1:
                    return f"D[:-1, {k}]"
                raise PacEstimateUnsupported(f"{name}({shift}): only lags of one period appear in a PAC equation")
            if shift != 0:
                raise PacEstimateUnsupported(f"{name}({shift}) in an equation without lags")
            return f"D[:, {k}]"
        return None
    # the identifiers of the equation that are neither parameters nor variables must be functions of the namespace
    fl = series_function(prep.lhs, resolve, shiftable)
    fr = series_function(prep.rhs, resolve, shiftable)
    ip = np.array(prep.ipnames, int)

    def full(p):
        P = np.array(base_params, float)
        P[ip] = p
        return update(P)

    def r(p):
        P = full(p)
        env = {"P": P, "D": D}
        return np.asarray(fl(env), float) - np.asarray(fr(env), float)

    def ssr(p):
        x = r(p)
        return float(x @ x)

    def lhs_values(p):
        P = full(p)
        return np.asarray(fl({"P": P, "D": D}), float)
    return r, ssr, full, lhs_values


# ------------------------------------------------------------------------------------------------ optimisation
def opt4nls(varargin):
    """opt4nls: ``(noprint, options)`` from key/value pairs; 'noprint' is consumed here."""
    noprint = False
    opt = {}
    if len(varargin) % 2:
        raise PacEstimateError("Options must come by key/value pairs.")
    i = 0
    while i < len(varargin) - 1 or (i == 0 and len(varargin) == 2):
        k, v = varargin[i], varargin[i + 1]
        if k == "noprint":
            noprint = bool(np.asarray(v).ravel()[0]) if not isinstance(v, str) else bool(v)
        else:
            opt[k] = v
        i += 2
        if i >= len(varargin):
            break
    return noprint, opt


def jacobian(fun, x, h):
    """Dynare's jacobian.m: ``J(:, i) = (r - fun(x + h e_i)) / h`` (the opposite sign of the derivative)."""
    r = fun(x)
    J = np.zeros((r.size, x.size))
    for i in range(x.size):
        z = np.array(x, float)
        z[i] += h
        J[:, i] = (r - fun(z)) / h
    return r, J


def gauss_newton(fun, x0):
    """gauss_newton.m"""
    max_iter, h, xtol = 100, np.sqrt(np.finfo(float).eps), 1e-6
    phi = 0.5 * (np.sqrt(5.0) + 1.0)
    x0 = np.array(x0, float)
    counter = 0

    def ssr(x):
        r = fun(x)
        return float(r @ r)
    while True:
        r0, J = jacobian(fun, x0, h)
        d = np.linalg.pinv(J) @ r0
        x1 = x0 + d
        r1 = fun(x1)
        s0, s1 = float(r0 @ r0), float(r1 @ r1)
        if s1 > s0:
            l1, l2 = 0.0, 1.0
            L1 = l2 - (l2 - l1) / phi
            L2 = l1 + (l2 - l1) / phi
            while abs(L1 - L2) > 1e-6:
                if ssr(x0 + L1 * d) < ssr(x0 + L2 * d):
                    l2 = L2
                else:
                    l1 = L1
                L1 = l2 - (l2 - l1) / phi
                L2 = l1 + (l2 - l1) / phi
            x1 = x0 + 0.5 * (l1 + l2) * d
        noconv = np.max(np.abs(x1 - x0)) > xtol
        counter += 1
        x0 = x1
        if not noconv or counter > max_iter:
            break
    return x0, s1


def minimise(ssrfun, resfun, x0, optimizer, bounds, opt, csminwel_defaults):
    """The minimisation step of nls.m for each optimiser; returns ``(x, ssr)``."""
    from scipy import optimize as so
    x0 = np.array(x0, float)
    algo = optimizer or "csminwel"
    if algo == "GaussNewton":
        return gauss_newton(resfun, x0)
    if algo == "csminwel":
        from .estimation import csminwel
        crit, nit, method, eps = csminwel_defaults
        H0 = 1e-4 * np.eye(x0.size)
        for k, v in opt.items():
            if k == "MaxIter":
                nit = int(v)
            elif k == "InitialInverseHessian":
                H0 = np.array(v, float) if not isinstance(v, str) else _eval_matrix(v)
            elif k == "TolFun":
                crit = float(v)
            elif k == "NumgradAlgorithm":
                method = int(v)
            elif k == "NumgradEpsilon":
                eps = float(v)
            elif k in ("verbosity", "SaveFiles"):
                pass
            else:
                import warnings
                warnings.warn(f"csminwel: Unknown option ({k})!")
        res = csminwel(lambda x: (ssrfun(np.asarray(x, float)), 0, 0.0), x0, H0=H0, crit=crit, nit=nit, method=method,
                       epsilon=eps)
        return np.asarray(res["x"], float), float(res["f"])
    if algo == "simplex":
        from .simplex import simplex_optimization_routine, simplex_options
        o = simplex_options({k: v for k, v in opt.items()})
        res = simplex_optimization_routine(lambda z: (ssrfun(np.asarray(z, float)), True), x0, o)
        return np.asarray(res.x, float), float(res.fval)
    if algo in ("annealing", "particleswarm"):
        raise PacEstimateUnsupported(f"the optimiser '{algo}' is a random search: its result cannot be matched")
    tol = 1e-12
    lo, hi = (None, None) if bounds is None else (bounds[:, 0], bounds[:, 1])
    if algo == "fmincon":
        bnds = [(l, u) for l, u in zip(lo, hi)]
        bnds = [(None if np.isinf(l) else l, None if np.isinf(u) else u) for l, u in bnds]
        res = so.minimize(ssrfun, x0, method="L-BFGS-B", bounds=bnds, options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000})
        res = so.minimize(ssrfun, res.x, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 2000}) \
            if not res.success else res
        x = res.x if bounds is None else np.clip(res.x, lo, hi)
        return np.asarray(x, float), float(ssrfun(x))
    if algo == "fminunc":
        res = so.minimize(ssrfun, x0, method="BFGS", options={"gtol": 1e-9, "maxiter": 1000})
        return np.asarray(res.x, float), float(res.fun)
    if algo == "fminsearch":
        res = so.minimize(ssrfun, x0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 5000,
                                                                       "maxfev": 20000})
        return np.asarray(res.x, float), float(res.fun)
    if algo == "lsqnonlin":
        lm = "levenberg-marquardt" in [str(v) for v in opt.values()]
        if lm:
            res = so.least_squares(resfun, x0, method="lm", xtol=tol, ftol=tol, gtol=tol)
        else:
            res = so.least_squares(resfun, x0, bounds=(lo, hi), xtol=tol, ftol=tol, gtol=tol)
        r = resfun(res.x)
        return np.asarray(res.x, float), float(r @ r)
    raise PacEstimateError(f"{algo} is not an implemented optimization routine.")


def _eval_matrix(text):
    raise PacEstimateUnsupported("the option InitialInverseHessian given as a string")


def nls_statistics(resfun, x, SSR):
    """The covariance matrix of nls.m (White and Domowitz, theorem 3.2) and the t-statistics."""
    r, J = jacobian(resfun, x, 1e-6)
    T = r.size
    A = 2.0 * (J.T @ J) / T
    J = J * r[:, None]
    B = J.T @ J
    l = int(np.floor(T ** 0.25 + 0.5))
    for tau in range(1, l + 1):
        B = B + (1 - tau / (l + 1)) * (J[tau:].T @ J[:-tau] + J[:-tau].T @ J[tau:])
    B = (4.0 / T) * B
    Ai = np.linalg.inv(A)
    C = Ai @ B @ Ai / T
    return r, C, x / np.sqrt(np.diag(C))


# ------------------------------------------------------------------------------------------------ iterative OLS
def _setdiff_idx(a, b):
    """``[~, i] = setdiff(a, b)``: the positions (0-based) in ``a`` of the values that are not in ``b``, by increasing value."""
    bs = {x for x in b if not (isinstance(x, float) and np.isnan(x))}
    vals = sorted({x for x in a if x not in bs})
    return [list(a).index(v) for v in vals]


def islagof(aux, endo_names, orig_endo_nbr, v1, v2):
    """matlab/islagof.m on 1-based indices of endogenous variables."""
    if v1 == v2:
        return False
    a1, a2 = v1 > orig_endo_nbr, v2 > orig_endo_nbr
    if not a1 and not a2:
        return False
    if a1 and a2:
        i1, i2 = aux.info(v1), aux.info(v2)
        ll1, ll2 = i1["type"] in (1, 0), i2["type"] in (1, 0)
        if ll1 and ll2:
            return bool(i1.get("orig_index") and i2.get("orig_index") and i1["orig_lead_lag"] < i2["orig_lead_lag"])
        d1 = i1["type"] in (8, 9)
        if d1:                                    # Dynare tests auxinfo1 twice
            b = False
            if i1["type"] == 9 and i2["type"] == 8:
                while i1["type"] == 9:
                    if i1["orig_index"] == v2:
                        b = True
                    i1 = aux.info(i1["orig_index"])
            return b
    if a1 and not a2:
        i1 = aux.info(v1)
        return i1["type"] == 1 and i1["orig_index"] == v2
    if not a1 and a2:
        i2 = aux.info(v2)
        return i2["type"] == 0 and i2["orig_index"] == v1
    return False


def _lag(ds, k):
    from .dseries import _lag as lag
    return lag(ds, float(k))


def _cols(ds, names, rng):
    k0, k1 = _rows(ds, rng)
    return ds.data[k0:k1 + 1][:, [ds.names.index(n) for n in names]]


def iterative_ols(prep, *, pm, M_params, param_names, endo_names, exo_names, orig_endo_nbr, aux, update, share_bounds):
    """iterative_ols.m.  ``pm`` holds M_.pac.<name> as plain Python; ``update(P)`` is pac.update.parameters on a parameter
    vector.  Returns the estimates in the order of ``prep.params``, the residual, the ssr and the new parameter vector."""
    data, rng, rhs = prep.data, prep.range, prep.rhs
    params0 = np.array([v for _, v in prep.params], float)
    ipnames = [i + 1 for i in prep.ipnames]                       # 1-based
    P = np.array(M_params, float)
    nob = "non_optimizing_behaviour" in pm
    additive_pm = pm.get("additive")
    is_exo = additive_pm is not None and np.asarray(additive_pm["vars"]).size > 1
    oa = pm.get("optim_additive")
    is_optim_exo = oa is not None and np.asarray(oa["vars"]).size > 0
    pnames_est = {p for p, _ in prep.params}
    share_index = None
    estimate_share = False
    share = 1.0
    nob_params = None
    nobp = None
    if nob:
        nobp = pm["non_optimizing_behaviour"]
        npar = np.asarray(nobp["params"], float).ravel()
        noparams = np.isnan(npar)
        nob_params = np.full(npar.size, np.nan)
        if not np.all(noparams):
            est = [param_names[int(i) - 1] in pnames_est for i in npar[~noparams]]
            if any(est):
                raise PacEstimateError("The estimation of non optimizing behaviour parameters is not yet allowed.")
            nob_params[noparams] = 1.0
            nob_params[~noparams] = P[npar[~noparams].astype(int) - 1]
        else:
            nob_params[noparams] = 1.0
        nob_params = nob_params * np.asarray(nobp["scaling_factor"], float).ravel()
        share_index = int(pm["share_of_optimizing_agents_index"])
        estimate_share = param_names[share_index - 1] in pnames_est
        if not estimate_share:
            share = float(P[share_index - 1])
            if share > 1 or share < 0:
                raise PacEstimateError("The share of optimizing agents shoud be in (0,1).")
    # additive terms
    additive = dict(params=np.zeros(0), vars=np.zeros(0, int), isendo=np.zeros(0, bool), lags=np.zeros(0),
                    scaling_factor=np.zeros(0), estimation=np.zeros(0, bool))
    if is_exo:
        rid = exo_names.index(prep.rname) + 1
        a = {k: np.atleast_1d(np.asarray(additive_pm[k])).ravel() for k in ("params", "vars", "isendo", "lags", "scaling_factor")}
        drop = [j for j in range(a["vars"].size) if int(a["vars"][j]) == rid and not bool(a["isendo"][j])]
        keep = [j for j in range(a["vars"].size) if j not in drop]
        additive = {k: v[keep] for k, v in a.items()}
        additive["estimation"] = np.array([x in ipnames for x in additive["params"]], bool)
    optim_additive = dict(params=np.zeros(0), vars=np.zeros(0, int), isendo=np.zeros(0, bool), lags=np.zeros(0),
                          scaling_factor=np.zeros(0), estimation=np.zeros(0, bool))
    if is_optim_exo:
        a = {k: np.atleast_1d(np.asarray(oa[k])).ravel() for k in ("params", "vars", "isendo", "lags", "scaling_factor")}
        optim_additive = dict(a)
        optim_additive["estimation"] = np.array([x in ipnames for x in a["params"]], bool)
    # data for the PAC expectation
    hpi = [int(i) for i in np.asarray(pm["h_param_indices"]).ravel()]
    listofvariables, blocks = [], []
    isconstant = False
    for i, hp in enumerate(hpi):
        name = re.escape(param_names[hp - 1])
        match = re.search(r"(?P<var>((\w*)|\w*\(-1\)))\*" + name, rhs)
        if not match:
            match = re.search(name + r"\*(?P<var>((\w*\(-1\))|(\w*)))", rhs)
        if match:
            v = match.group("var")
            if "(-1)" not in v:
                listofvariables.append(v)
                if i >= len(listofvariables):
                    raise PacEstimateUnsupported("iterative_ols: Dynare itself fails here (the constant of the PAC "
                                                 "expectation is not the last h parameter)")
                blocks.append((listofvariables[i], data.data[:, data.names.index(listofvariables[i])]))
            else:
                listofvariables.append(v[:-4])
                if i >= len(listofvariables):
                    raise PacEstimateUnsupported("iterative_ols: Dynare itself fails here (the constant of the PAC "
                                                 "expectation is not the last h parameter)")
                blocks.append((listofvariables[i], _lag(_one(data, v[:-4]), 1).data[:, 0]))
        elif param_names[hp - 1] == f"h_{prep.pacmodl}_constant":
            isconstant = True
    names_pac = [b[0] for b in blocks]
    if len(set(names_pac)) != len(names_pac):
        raise PacEstimateError("dseries::horzcat: I cannot concatenate dseries objects with common variable names "
                               f"({', '.join(sorted({n for n in names_pac if names_pac.count(n) > 1}))})!")
    k0, k1 = _rows(data, rng)
    dataPAC = np.column_stack([b[1][k0:k1 + 1] for b in blocks]) if blocks else np.zeros((k1 - k0 + 1, 0))
    if isconstant:
        dataPAC = np.column_stack([np.ones(dataPAC.shape[0]), dataPAC])

    def series(name, lag):
        ds = _one(data, name)
        if lag:
            ds = _lag(ds, lag)
        return ds.data[k0:k1 + 1, 0]

    dataNOB = np.zeros((k1 - k0 + 1, 0))
    if nob:
        cols = []
        for i in range(np.asarray(nobp["vars"]).size):
            isendo = bool(np.atleast_1d(nobp["isendo"])[i])
            var = (endo_names if isendo else exo_names)[int(np.atleast_1d(nobp["vars"])[i]) - 1]
            lag = int(np.atleast_1d(nobp["lags"])[i])
            cols.append(series(var, lag))
        dataNOB = np.column_stack(cols) if cols else dataNOB

    def additive_data(spec):
        est_cols, est_names, cal = [], [], 0.0
        any_cal = any_est = False
        for i in range(np.asarray(spec["vars"]).size):
            isendo = bool(spec["isendo"][i])
            var = (endo_names if isendo else exo_names)[int(spec["vars"][i]) - 1]
            lag = -int(spec["lags"][i])
            if spec["estimation"][i]:
                any_est = True
                est_names.append(var)
                est_cols.append(spec["scaling_factor"][i] * _lag(_one(data, var), lag).data[k0:k1 + 1, 0])
            else:
                any_cal = True
                tmp = _lag(_one(data, var), lag).data[k0:k1 + 1, 0]
                if not np.isnan(spec["params"][i]):
                    tmp = P[int(spec["params"][i]) - 1] * tmp
                cal = cal + spec["scaling_factor"][i] * tmp
        return (np.column_stack(est_cols) if est_cols else None), (cal if any_cal else None), any_est

    xdata_add, cal_add, any_est_x = (None, None, False)
    if is_exo:
        xdata_add, cal_add, any_est_x = additive_data(additive)
    xdata_opt, cal_opt, any_est_optim = (None, None, False)
    if is_optim_exo:
        xdata_opt, cal_opt, any_est_optim = additive_data(optim_additive)
    ec = pm["ec"]
    ecv = [int(x) for x in np.asarray(ec["vars"]).ravel()]
    if bool(np.atleast_1d(ec["isendo"])[1]):
        ecvars = ecv
    else:
        ecvars = ecv[::-1]
    ecterm = _lag(_one(data, endo_names[ecvars[0] - 1]), 1).data[:, 0] - _lag(_one(data, endo_names[ecvars[1] - 1]), 1).data[:, 0]
    cols_x = [ecterm]
    arp = np.asarray(pm["ar"]["params"]).ravel()
    arv = np.asarray(pm["ar"]["vars"]).ravel()
    lhs_var = int(np.asarray(pm["lhs_var"]).ravel()[0])
    for i in range(arp.size):
        if islagof(aux, endo_names, orig_endo_nbr, int(arv[i]), lhs_var):
            cols_x.append(data.data[:, data.names.index(endo_names[int(arv[i]) - 1])])
    XDATA = np.column_stack([c[k0:k1 + 1] for c in cols_x])
    if is_optim_exo and any_est_optim:
        XDATA = np.column_stack([XDATA, xdata_opt])
    if is_exo and any_est_x:
        XDATA = np.column_stack([XDATA, xdata_add])
    ip = list(ipnames)
    params_id_0 = [ip.index(share_index)] if (nob and share_index in ip) else []
    excl1 = ([share_index] if nob else []) + [float(x) for x in optim_additive["params"]] + [float(x) for x in additive["params"]]
    excl5 = ([share_index] if nob else []) + [float(x) for x in additive["params"]]
    params_id_1 = _setdiff_idx(ip, excl1)
    params_id_5 = _setdiff_idx(ip, excl5)
    params_id_2 = sorted(set(range(len(ip))) - set(params_id_1))
    params_id_3 = sorted(set(params_id_2) - set(params_id_0))
    params_id_4 = params_id_1 + params_id_3
    params0_ = params0[params_id_4]
    if estimate_share:
        share = float(params0[params_id_0[0]])
    if share > 1 or share < 0:
        raise PacEstimateError("Initial value for the share of optimizing agents shoud be in (0,1).")
    P[np.array(ip) - 1] = params0
    n5 = len(params_id_5)

    def pac_expectations(P):
        P = update(P)
        pe = dataPAC @ P[np.array(hpi) - 1]
        corr = 0.0
        gl = pm.get("growth_linear_comb")
        if gl is not None:
            for g in gl:
                gv = g["constant"]
                if g["param_id"] > 0:
                    gv = gv * P[int(g["param_id"]) - 1]
                if g["exo_id"] > 0:
                    gv = gv * _lag(_one(data, exo_names[int(g["exo_id"]) - 1]), abs(int(g["lag"]))).data[:, 0]
                    gv = gv[k0:k1 + 1]
                elif g["endo_id"] > 0:
                    gv = gv * _lag(_one(data, endo_names[int(g["endo_id"]) - 1]), abs(int(g["lag"]))).data[:, 0]
                    gv = gv[k0:k1 + 1]
                corr = corr + gv
            corr = corr * P[int(pm["growth_neutrality_param_index"]) - 1]
        return pe + corr, P

    PacE, P = pac_expectations(P)
    lhs_name = endo_names[lhs_var - 1]
    Y0 = data.data[k0:k1 + 1, data.names.index(lhs_name)]
    noconv, counter = True, 0
    log = []
    while noconv:
        counter += 1
        Y = Y0.copy()
        if nob:
            Y = Y - share * PacE
            Y = Y - (1 - share) * (dataNOB @ nob_params)
        else:
            Y = Y - PacE
        if is_exo and cal_add is not None:
            Y = Y - cal_add
        if is_optim_exo and cal_opt is not None:
            Y = Y - share * cal_opt
        p1 = np.linalg.lstsq(XDATA, Y, rcond=None)[0]
        p1[:n5] = p1[:n5] / share
        r = Y - XDATA[:, :n5] @ p1[:n5] * share
        if is_optim_exo and any_est_optim:
            r = r - XDATA[:, n5:] @ p1[n5:]
        ssr = float(r @ r)
        crit = float(np.max(np.abs(params0_ - p1)))
        noconv = crit > 1e-6
        log.append((counter, crit, ssr))
        params0_ = p1
        if estimate_share:
            P[np.array(ip)[params_id_4] - 1] = params0_
            PacE, P = pac_expectations(P)
            Y = Y0 - dataNOB @ nob_params
            if is_exo:
                if cal_add is not None:
                    Y = Y - cal_add
                if any_est_x:
                    Y = Y - XDATA[:, len(params_id_1) - 1:] @ params0_[len(params_id_1) - 1:]
            Z = XDATA[:, :n5] @ params0_[:n5] + PacE - dataNOB @ nob_params
            if is_optim_exo and cal_opt is not None:
                Z = Z + cal_opt
            share = float(np.linalg.lstsq(Z.reshape(-1, 1), Y, rcond=None)[0][0])
            share = max(min(share, share_bounds[1]), share_bounds[0])
            if share < np.finfo(float).eps:
                raise PacEstimateError(f"On iteration {counter} the share of optimizing agents is found to be zero. "
                                       "Please increase the default lower bound for this parameter.")
            P[ip[params_id_0[0]] - 1] = share
        else:
            if len(params0_) != len(ip):
                raise PacEstimateError("=: nonconformant arguments")
            P[np.array(ip) - 1] = params0_
            PacE, P = pac_expectations(P)
    return dict(ssr=ssr, residual=r, estimator=params0_, iterations=log, P=P)


def _one(ds, name):
    if name not in ds.names:
        raise PacEstimateError(f"dseries::extract: Variable {name} is unknown in dseries object!")
    j = ds.names.index(name)
    return DSeries(ds.data[:, [j]], [name], ds.init)
