# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# Behaviour follows Dynare 7.2: matlab/+pac/+update/parameters.m, matlab/pac-tools/{hVectors,a2alpha,
# buildGmatrixWithAlphaAndBeta,iota}.m (Copyright (C) 2018-2025 Dynare Team, GPL-3.0-or-later).
"""PAC expectations: the h vectors of the auxiliary model (VAR or trend component model) and the growth neutrality
correction, written into the parameters of the model."""
from __future__ import annotations

import numpy as np

from .matlab import Cell, Struct, farr
from .submodels import StructArray


class PacError(Exception):
    """A MATLAB ``error()`` of Dynare's PAC functions (the message is Dynare's)."""


EPS = np.finfo(float).eps


def a2alpha(a):
    a = np.asarray(a, float).ravel()
    m = a.size
    alpha = np.zeros(m)
    alpha[m - 1] = a[m - 1]
    if m > 2:
        alpha[1:m - 1] = a[1:m - 1] - a[2:m]
    if m < 2:
        raise PacError("Index exceeds the number of array elements.")
    alpha[0] = a[0] - a[1] - 1
    return alpha


def build_G(params):
    """buildGmatrixWithAlphaAndBeta.m: ``params`` = [lag polynomial coefficients (EC parameter first); beta]."""
    params = np.asarray(params, float).ravel()
    m = params.size - 1
    if m < 1:
        raise PacError("Input argument has to be a vector with at least two elements. The last element is the discount factor.")
    alpha = a2alpha(params[:m])[::-1]
    beta = params[-1]
    if beta < EPS or beta > 1 - EPS:
        raise PacError("beta has to be a discount factor!")
    G = np.zeros((m, m))
    G[:m - 1, 1:m] = np.eye(m - 1)
    G[m - 1, :] = -alpha
    G[m - 1, :] = G[m - 1, :] * np.cumprod(beta * np.ones(m))[::-1]
    return G, alpha, beta


def iota(n, idx):
    i = np.zeros((n, 1))
    i[np.atleast_1d(idx).astype(int) - 1] = 1
    return i


def h_vectors(params, H, auxmodel, kind, idx, want_lrcp=False):
    """hVectors.m: the weights of the PAC expectation on the state of the companion VAR (column), and the
    parameter of the growth neutrality correction."""
    G, alpha, beta = build_G(params)
    A = np.concatenate([alpha, [1.0]])
    A_1 = np.polyval(A, 1.0)
    A_b = np.polyval(A, beta)
    m = alpha.size
    n = H.shape[0]
    tmp = np.eye(n * m) - np.kron(G, H.T)
    im = iota(m, m)
    rhs = np.kron(im, iota(n, idx))
    if kind == "ll":
        h = A_1 * A_b * (np.kron(im, H).T @ np.linalg.solve(tmp, rhs))
    elif kind == "dd":
        h = A_1 * A_b * (np.kron(im.T @ np.linalg.inv(np.eye(m) - G), H.T) @ np.linalg.solve(tmp, rhs))
    elif kind == "dl":
        h = A_1 * A_b * (np.kron(im.T @ np.linalg.inv(np.eye(m) - G), H.T - np.eye(H.shape[0])) @ np.linalg.solve(tmp, rhs))
    else:
        raise PacError("Unknown kind value in PAC model.")
    lrcp = None
    if want_lrcp:
        if kind == "ll":
            lrcp = np.nan
        else:
            IG = np.eye(m) - G
            d = A_1 * A_b * float((im.T @ np.linalg.inv(IG @ IG) @ im).item())
            lrcp = float(1 - np.sum(np.asarray(params, float).ravel()[1:-1]) - d)
    return h.ravel(), lrcp


class Context:
    """The live ``M_`` and ``oo_`` (read-only) and what is needed of the model."""

    def __init__(self, M, oo, endo_names, param_names, aux):
        self.M, self.oo, self.endo_names, self.param_names, self.aux = M, oo, endo_names, param_names, aux

    def aux_types(self, name, types):
        """isauxiliary(name, types)"""
        if name not in self.endo_names:
            return False
        i = self.endo_names.index(name) + 1
        if i <= self.aux.orig:
            return False
        a = self.aux.info(i)
        return a is not None and a["type"] in types


def _f(v):
    return np.asarray(farr(v), float).ravel(order="F")


def _i(v):
    return _f(v).astype(int)


def _islogical(x):
    return isinstance(x, (bool, np.bool_))


def _isnumeric(x):
    return isinstance(x, (int, float, np.floating, np.integer)) and not _islogical(x)


def _bgp(sec, i):
    try:
        return sec["bgp"].items[i]
    except (KeyError, IndexError):
        raise PacError("reference to non-existent field 'bgp'") from None


def update_parameters(ctx, pacname, params):
    """pac.update.parameters.m: ``params`` (M_.params) with the h parameters (and the growth neutrality parameter) set."""
    M, oo = ctx.M, ctx.oo
    params = np.array(params, float)
    pacs = M.get("pac")
    if not isinstance(pacs, Struct) or pacname not in pacs:
        raise PacError(f"PAC model {pacname} is not defined in the model block!")
    pm = pacs[pacname]
    if "model_consistent_expectations" in pm and bool(_f(pm["model_consistent_expectations"])[0]):
        raise PacError("This function cannot be used with Model Consistent Expectation. Try pac.mce.parameters instead.")
    typ, auxname = pm["auxiliary_model_type"], pm["auxiliary_model_name"]
    if not isinstance(M.get(typ), Struct) or auxname not in M[typ]:
        raise PacError(f"Unknown auxiliary model ({auxname}) in PAC model ({pacname})!")
    varmodel = M[typ][auxname]
    if not isinstance(oo.get(typ), Struct) or auxname not in oo[typ]:
        raise PacError(f"Auxiliary model {auxname} has to be estimated first!")
    varcalib = oo[typ][auxname]
    if "CompanionMatrix" not in varcalib or np.any(np.isnan(_f(varcalib["CompanionMatrix"]))):
        raise PacError(f"Auxiliary model {auxname} has to be estimated first.")
    H = np.asarray(farr(varcalib["CompanionMatrix"]), float)
    comps = pm["components"].elems if isinstance(pm.get("components"), StructArray) else []
    nc = len(comps)
    if nc > 0 and typ == "trend_component":
        raise PacError("Composite PAC target not allowed with trend component model.")
    max_lag = int(_f(pm["max_lag"])[0])
    ecp = _i(pm["ec"]["params"])
    arp = _i(pm["ar"]["params"])
    pacvalues = params[np.concatenate([ecp, arp[:max_lag]]) - 1]
    lst = [str(x) for x in varmodel["list_of_variables_in_companion_var"].items]
    names = ctx.endo_names

    def find_in_var(endo_idx, message):
        """The 1-based position in the companion VAR of the variable ``endo_idx`` (or of its auxiliary variable)."""
        nm = names[endo_idx - 1]
        if nm in lst:
            return lst.index(nm) + 1
        ad = [j for j, x in enumerate(lst) if ctx.aux_types(x, (8, 10))]
        if not ad:
            raise PacError(message)
        for j in ad:
            a = ctx.aux.info(names.index(lst[j]) + 1)
            if a["endo_index"] == endo_idx:
                return j + 1
            if a["type"] == 8 and a["orig_index"] == endo_idx:
                return j + 1
        raise PacError(message)

    if nc:
        ids = [find_in_var(int(_f(c["endo_var"])[0]), "Cannot find the trend variable in the Companion VAR model.") for c in comps]
    else:
        ecv = _i(pm["ec"]["vars"])
        ist = np.asarray(farr(pm["ec"]["istarget"])).ravel().astype(bool)
        sel = ecv[ist]
        if sel.size != 1:
            raise PacError("Expected exactly one target in the error correction term of the PAC equation.")
        ids = [find_in_var(int(sel[0]), "Cannot find the trend variable in the auxiliary VAR / Trend component model.")]
    isconstant = False
    if typ == "var":
        isconstant = bool(_f(varmodel["isconstant"])[0]) if "isconstant" in varmodel else False
    if not nc:
        if typ == "var":
            nonst = np.asarray(farr(varmodel["nonstationary"])).ravel().astype(bool)
            kind = ["dd"] if nonst[ids[0] - 1] else ["ll"]
            if isconstant:
                ids[0] += 1
        else:
            kind = ["dd"]
    else:
        if isconstant:
            ids = [i + 1 for i in ids]
    if not nc:
        if pm.get("kind"):
            kind = [pm["kind"]]
    else:
        kind = []
        for c in comps:
            if not c.get("kind"):
                raise PacError("kind declaration is mandatory for each component in pac_target_info.")
            kind.append(c["kind"])
    beta = params[int(_f(pm["discount_index"])[0]) - 1]
    growth_flag = "growth_str" in pm or any("growth_str" in c for c in comps)
    gamma = params[int(_f(pm["share_of_optimizing_agents_index"])[0]) - 1] if "non_optimizing_behaviour" in pm else 1.0
    h, gn = [], []
    for i in range(len(ids)):
        hv, lr = h_vectors(np.concatenate([pacvalues, [beta]]), H, typ, kind[i], ids[i], want_lrcp=growth_flag)
        h.append(hv)
        gn.append(lr)
    # update the parameters with h
    if typ == "var":
        if isconstant:
            if "h_param_indices" in pm:
                params[_i(pm["h_param_indices"]) - 1] = h[0]
            else:
                for i, c in enumerate(comps):
                    params[_i(c["h_param_indices"]) - 1] = h[i]
        else:
            if "h_param_indices" in pm:
                hp = _i(pm["h_param_indices"])
                params[hp[0] - 1] = 0.0
                params[hp[1:] - 1] = h[0]
            else:
                for i, c in enumerate(comps):
                    hp = _i(c["h_param_indices"])
                    params[hp[0] - 1] = 0.0
                    params[hp[1:] - 1] = h[i]
    else:
        params[_i(pm["h_param_indices"]) - 1] = h[0]
    if growth_flag:
        for j in range(len(ids)):
            if np.isnan(gn[j]):
                continue
            gg = -(gn[j] - 1)
            cc = 1.0 - gg * gamma
            ll = 0.0
            if "optim_additive" in pm:
                sec = pm["optim_additive"]
                tmp0 = 0.0
                ps, sf = _f(sec["params"]), _f(sec["scaling_factor"])
                for i in range(ps.size):
                    b = _bgp(sec, i)
                    if np.isnan(ps[i]) and _islogical(b) and b:
                        tmp0 += sf[i]
                    elif not np.isnan(ps[i]) and _islogical(b) and b:
                        tmp0 += params[int(ps[i]) - 1] * sf[i]
                    elif not _islogical(b):
                        raise PacError("It is not possible to provide a value for the mean of an exogenous variable "
                                       "appearing in the optimal part of the PAC equation.")
                cc = cc - tmp0 * gamma
            if gamma < 1:
                if "non_optimizing_behaviour" in pm and "params" in pm["non_optimizing_behaviour"]:
                    sec = pm["non_optimizing_behaviour"]
                    tmp0 = tmp1 = 0.0
                    ps, sf = _f(sec["params"]), _f(sec["scaling_factor"])
                    for i in range(ps.size):
                        b = _bgp(sec, i)
                        if np.isnan(ps[i]) and _islogical(b) and b:
                            tmp0 += sf[i]
                        elif not np.isnan(ps[i]) and _islogical(b) and b:
                            tmp0 += params[int(ps[i]) - 1] * sf[i]
                        elif not _islogical(b) and _isnumeric(b) and np.isnan(ps[i]):
                            tmp1 += sf[i] * b
                        elif not _islogical(b) and _isnumeric(b) and not np.isnan(ps[i]):
                            tmp1 += sf[i] * ps[i] * b
                    cc = cc - (1.0 - gamma) * tmp0
                    ll = -(1.0 - gamma) * tmp1 / gamma
            if "additive" in pm:
                sec = pm["additive"]
                tmp0 = tmp1 = 0.0
                ps, sf = _f(sec["params"]), _f(sec["scaling_factor"])
                for i in range(ps.size):
                    b = _bgp(sec, i)
                    if np.isnan(ps[i]) and _islogical(b) and b:
                        tmp0 += sf[i]
                    elif not np.isnan(ps[i]) and _islogical(b) and b:
                        tmp0 += params[int(ps[i]) - 1] * sf[i]
                    elif not _islogical(b) and _isnumeric(b) and np.isnan(ps[i]):
                        tmp1 += sf[i] * b
                    elif not _islogical(b) and _isnumeric(b) and not np.isnan(ps[i]):
                        tmp1 += sf[i] * ps[i] * b
                cc = cc - tmp0
                ll = ll - tmp1 / gamma
            if "growth_neutrality_param_index" in pm:
                params[int(_f(pm["growth_neutrality_param_index"])[0]) - 1] = cc / gamma
            else:
                params[int(_f(comps[j]["growth_neutrality_param_index"])[0]) - 1] = cc / gamma
    return params


def _comp_has_growth_param(pm):
    comps = pm["components"].elems if isinstance(pm.get("components"), StructArray) else []
    return [c for c in comps if "growth_neutrality_param_index" in c and np.size(farr(c["growth_neutrality_param_index"])) > 0]


def mce_parameters(M, pacname, params):
    """pac.mce.parameters.m: the new M_.params (alpha parameters and the growth neutrality correction) of a PAC equation
    under model consistent expectations."""
    params = np.array(params, float)
    if not isinstance(M.get("pac"), Struct) or pacname not in M["pac"]:
        raise PacError(f"PAC model {pacname} is not defined in the model block!")
    pm = M["pac"][pacname]
    if not bool(_f(pm["model_consistent_expectations"])[0]):
        raise PacError("This function is to be used only for PAC Model Consistent Expectations.")
    print(f"PAC model {pacname} is used in equation {pm['eq_name']}.")
    print()
    max_lag = int(_f(pm["max_lag"])[0])
    p = np.full(2 + max_lag, np.nan)
    p[0] = params[_i(pm["ec"]["params"])[0] - 1]
    arp = _i(pm["ar"]["params"])
    if arp.size != max_lag:
        raise PacError("=: nonconformant arguments")
    p[1:1 + max_lag] = params[arp - 1]
    p[-1] = params[int(_f(pm["discount_index"])[0]) - 1]
    G, alpha, beta = build_G(p)
    params[_i(pm["mce"]["alpha"]) - 1] = alpha[::-1]
    comps = pm["components"].elems if isinstance(pm.get("components"), StructArray) else []
    if "growth_neutrality_param_index" in pm or _comp_has_growth_param(pm):
        gamma = params[int(_f(pm["share_of_optimizing_agents_index"])[0]) - 1] if "non_optimizing_behaviour" in pm else 1.0
        A = np.concatenate([alpha, [1.0]])
        A_1 = np.polyval(A, 1.0)
        A_b = np.polyval(A, beta)
        m = alpha.size
        IG = np.eye(m) - G
        d = A_1 * A_b * float((iota(m, m).T @ np.linalg.inv(IG @ IG) @ iota(m, m)).item())
        cc = 1 / gamma - (np.sum(p[1:-1]) + d)
        if "optim_additive" in pm:
            sec = pm["optim_additive"]
            tmp0 = 0.0
            ps, sf = _f(sec["params"]), _f(sec["scaling_factor"])
            for i in range(ps.size):
                b = _bgp(sec, i)
                if np.isnan(ps[i]) and _islogical(b) and b:
                    tmp0 += sf[i]
                elif not np.isnan(ps[i]) and _islogical(b) and b:
                    raise PacError("Undefined variable 'equations' (Dynare 7.2's pac.mce.parameters has an undefined "
                                   "variable in this branch)")
                elif not _islogical(b):
                    raise PacError("It is not possible to provide a value for the mean of an exogenous variable "
                                   "appearing in the optimal part of the PAC equation.")
            cc = cc - tmp0
        if gamma < 1 and "non_optimizing_behaviour" in pm and "params" in pm["non_optimizing_behaviour"]:
            sec = pm["non_optimizing_behaviour"]
            tmp0 = tmp1 = 0.0
            ps, sf = _f(sec["params"]), _f(sec["scaling_factor"])
            for i in range(ps.size):
                b = _bgp(sec, i)
                if np.isnan(ps[i]) and _islogical(b) and b:
                    tmp0 += sf[i]
                elif not np.isnan(ps[i]) and _islogical(b) and b:
                    tmp0 += params[int(ps[i]) - 1] * sf[i]
                elif not _islogical(b) and _isnumeric(b) and np.isnan(ps[i]):
                    tmp1 += sf[i] * b
                elif not _islogical(b) and _isnumeric(b) and not np.isnan(ps[i]):
                    tmp1 += sf[i] * ps[i] * b
            cc = cc - (1 - gamma) * tmp0 / gamma
        if "additive" in pm:
            sec = pm["additive"]
            tmp0 = tmp1 = 0.0
            ps, sf = _f(sec["params"]), _f(sec["scaling_factor"])
            for i in range(ps.size):
                b = _bgp(sec, i)
                if np.isnan(ps[i]) and _islogical(b) and b:
                    tmp0 += sf[i]
                elif not np.isnan(ps[i]) and _islogical(b) and b:
                    tmp0 += params[int(ps[i]) - 1] * sf[i]
                elif not _islogical(b) and _isnumeric(b) and np.isnan(ps[i]):
                    tmp1 += sf[i] * b
                elif not _islogical(b) and _isnumeric(b) and not np.isnan(ps[i]):
                    tmp1 += sf[i] * ps[i] * b
            cc = cc - tmp0 / gamma
        if "growth_neutrality_param_index" in pm:
            params[int(_f(pm["growth_neutrality_param_index"])[0]) - 1] = cc
        else:
            for c in comps:
                if "growth_neutrality_param_index" in c and np.size(farr(c["growth_neutrality_param_index"])) > 0:
                    params[int(_f(c["growth_neutrality_param_index"])[0]) - 1] = cc
    return params


# ----------------------------------------------------------------------------- write_expectations.m / print_expectations.m
def _aux_chain(ctx, variable):
    """The transformations (outermost last) of an auxiliary variable and the original variable name
    (write_expectations.m): only ``diff`` (type 8) auxiliary variables are known from the JSON output."""
    transformations = []
    while True:
        if variable not in ctx.endo_names:
            break
        idx = ctx.endo_names.index(variable) + 1
        a = ctx.aux.info(idx)
        if a is None:
            break
        if a["type"] == 8:
            transformations.append("diff")
            variable = ctx.endo_names[int(a["orig_index"]) - 1]
        elif a["type"] == 10:
            if "unary_op" not in a:
                raise PacError("This case is not implemented.")
            transformations.append(a["unary_op"])
            variable = ctx.endo_names[int(a["orig_index"]) - 1]
        else:
            raise PacError("This case is not implemented.")
    return variable, transformations


def _g(x):
    return x.items if isinstance(x, Cell) else x


def write_expectations(ctx, name, kind, iscrlf=True, two_outputs=False):
    """Dynare's write_expectations(name, kind, iscrlf) with aggregate = true: the unrolled expectation term (text)
    and the growth neutrality correction."""
    M = ctx.M
    field = "var_expectation" if kind == "var" else "pac"
    em = M[field][name]
    if "model_consistent_expectations" in em and bool(_f(em["model_consistent_expectations"])[0]):
        return "", 0
    if two_outputs and kind == "var":
        raise PacError("Cannot return more than one argument if the expectation model is a VAR.")
    pn = ctx.param_names
    typ, auxname = em["auxiliary_model_type"], em["auxiliary_model_name"]
    aux = M[typ][auxname]
    maxlag = int(np.max(_f(aux["max_lag"])))
    if typ == "trend_component":
        maxlag += 1
    ident = 0
    ts = abs(float(_f(em["time_shift"])[0])) if kind == "var" else 0
    timeindices = np.arange(maxlag) + ts
    comps = em["components"].elems if isinstance(em.get("components"), StructArray) else []
    expression = None
    if kind == "var" and typ == "var":
        ident += 1
        expression = pn[_i(em["param_indices"])[ident - 1] - 1]
    if kind == "pac" and typ == "var":
        ident += 1
        if "h_param_indices" in em:
            expression = pn[_i(em["h_param_indices"])[ident - 1] - 1]
        else:
            def term(c):
                p = pn[_i(c["h_param_indices"])[ident - 1] - 1]
                return p if c["coeff_str"] == "1" else f"{c['coeff_str']}*{p}"
            expression = term(comps[0])
            for c in comps[1:]:
                expression = f"{expression}+{term(c)}"
    lst = [str(x) for x in aux["list_of_variables_in_companion_var"].items]
    for i in range(1, maxlag + 1):
        for j in range(len(lst)):
            ident += 1
            variable, transformations = _aux_chain(ctx, lst[j])
            if kind == "var":
                parameter = pn[_i(em["param_indices"])[ident - 1] - 1]
            elif "h_param_indices" in em:
                parameter = pn[_i(em["h_param_indices"])[ident - 1] - 1]
            else:
                def term2(c):
                    p = pn[_i(c["h_param_indices"])[ident - 1] - 1]
                    return p if c["coeff_str"] == "1" else f"{c['coeff_str']}*{p}"
                parameter = term2(comps[0])
                for c in comps[1:]:
                    parameter = f"{parameter}+{term2(c)}"
                parameter = f"({parameter})"
            if kind == "var":
                if timeindices[i - 1]:
                    variable = f"{variable}(-{int(timeindices[i - 1])})"
            else:
                variable = f"{variable}(-{i})"
            for t in reversed(transformations):
                variable = f"{t}({variable})"
            nl = "\n" if iscrlf else ""
            if ident == 1:
                expression = f"{parameter}*{variable}{nl}"
            else:
                expression = f"{expression} + {parameter}*{variable}{nl}"
    growth = ""
    if "growth_neutrality_param_index" in em:
        gp = pn[int(_f(em["growth_neutrality_param_index"])[0]) - 1]
        n_lc = len(em["growth_linear_comb"].elems) if isinstance(em.get("growth_linear_comb"), StructArray) else 0
        growth = f"{gp}*{em['growth_str']}" if n_lc == 1 else f"{gp}*({em['growth_str']})"
    elif comps:
        for c in comps:
            if c.get("kind") == "ll" or "growth_neutrality_param_index" not in c:
                continue
            if not c.get("growth_str"):
                continue
            gp = pn[int(_f(c["growth_neutrality_param_index"])[0]) - 1]
            n_lc = len(c["growth_linear_comb"].elems) if isinstance(c.get("growth_linear_comb"), StructArray) else 0
            body = f"{gp}*{c['growth_str']}" if n_lc == 1 else f"{gp}*({c['growth_str']})"
            if c["coeff_str"] != "1":
                body = f"{c['coeff_str']}*{body}"
            growth = body if not growth else f"{growth}+{body}"
    if not two_outputs and growth:
        expression = f"{expression} + {growth}"
    return expression, growth
