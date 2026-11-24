# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# Behaviour follows Dynare 7.2: matlab/+pac/+estimate/{nls, iterative_ols}.m (Copyright (C) 2018-2026 Dynare Team,
# GPL-3.0-or-later).
"""``pac.estimate.nls`` and ``pac.estimate.iterative_ols`` on the live workspace of :mod:`pydyn.results`."""
from __future__ import annotations

import json
import warnings

import numpy as np

from . import dseries as _ds
from . import pac as _pac
from . import pac_estimate as _pe
from .matlab import Cell, MatlabError, Struct, Unsupported, farr, _F
from .submodels import StructArray


def _R():
    from . import results
    return results


def _plain(v):
    """A Struct of M_.pac as nested dicts of numpy arrays / str / lists."""
    if isinstance(v, Struct):
        return {k: _plain(v[k]) for k in v.keys()}
    if isinstance(v, StructArray):
        return [_plain(e) for e in v.elems]
    if isinstance(v, Cell):
        return [_plain(x) for x in v.items]
    if isinstance(v, str):
        return v
    try:
        a = v if isinstance(v, np.ndarray) else np.asarray(farr(v))
    except Exception:  # noqa: BLE001
        return v
    return a.ravel() if a.ndim == 2 and 1 in a.shape else a


class PacEstimateMixin:
    def _functions_pacest(self):
        return {"pac.estimate.nls": self.fn_pac_estimate_nls,
                "pac.estimate.iterative_ols": self.fn_pac_estimate_iterative_ols}

    # ------------------------------------------------------------------------------------------ common
    def _pe_inputs(self, kind, args):
        self._need_sub(f"pac.estimate.{kind}")
        self.pull()
        if len(args) < 4:
            raise MatlabError("Not enough input arguments.")
        eqname, params, data, rng = args[:4]
        if not isinstance(eqname, str):
            raise MatlabError("pac.estimate: the first argument must be the name of the equation")
        if not isinstance(params, Struct):
            raise MatlabError("pac.estimate: the second argument must be a structure")
        if not isinstance(data, _ds.DSeries):
            raise MatlabError("pac.estimate: the third argument must be a dseries object")
        dates = _ds.dates_of(rng)
        if dates is None:
            raise MatlabError("pac.estimate: the fourth argument must be a range of dates")
        plist = []
        for k in params.keys():
            v = np.asarray(farr(params[k]), float).ravel()
            if v.size != 1:
                raise MatlabError("pac.estimate: the initial value of a parameter must be a scalar")
            plist.append((k, float(v[0])))
        m = self.model
        jdir = m.json_dir
        try:
            jm = json.loads((jdir / "modfile.json").read_text())["model"]
            jo = json.loads((jdir / "modfile-original.json").read_text())["model"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise Unsupported(f"pac.estimate: the JSON output of the preprocessor is not available ({exc})", "file") from None
        if isinstance(jm, dict):
            jm = [jm]
        orig = int(m.raw.get("orig_endo_nbr", m.n_endo))
        endo = list(m.endo_names)
        aux_eqs = []
        auxn = {endo[a["endo_index"] - 1] for a in m.raw.get("aux_vars", []) if a.get("type") != 6}
        for e in jm:
            if not e.get("tags") and e["lhs"] in auxn:
                aux_eqs.append((e["lhs"], e["rhs"]))
        return eqname, plist, data, dates, jm, jo, orig, endo, aux_eqs

    def _pe_pm(self, jo, eqname):
        import re
        RHS = next((e["rhs"] for e in jo if (e.get("tags") or {}).get("name") == eqname), None)
        if RHS is None:
            raise MatlabError(f"Equation {eqname} is not defined.")
        mm = re.search(r"\(model_name\s*=\s*(\w+)\)", RHS)
        if "pac_expectation" not in RHS.lower() or not mm:
            raise MatlabError("This is not a PAC equation.")
        name = mm.group(1)
        pm = self.M_["pac"][name] if self._has("pac", name) else None
        if pm is None:
            raise MatlabError(f"PAC model {name} is not defined in the model block!")
        d = _plain(pm)
        if "components" in d:
            d["aux_ids"] = [int(np.asarray(c["aux_id"]).ravel()[0]) for c in d["components"]]
        elif "aux_id" in d:
            d["aux_id"] = int(np.asarray(d["aux_id"]).ravel()[0])
        for k in ("share_of_optimizing_agents_index", "growth_neutrality_param_index"):
            if k in d:
                d[k] = int(np.asarray(d[k]).ravel()[0])
        if "growth_linear_comb" in d:
            d["growth_linear_comb"] = [{kk: (float(np.asarray(vv).ravel()[0])) for kk, vv in g.items()} for g in d["growth_linear_comb"]]
        return name, d

    def _pe_update(self, name):
        ctx = self._pac_context()

        def update(P):
            try:
                return _pac.update_parameters(ctx, name, P)
            except _pac.PacError as exc:
                raise MatlabError(str(exc)) from None
        return update

    # ------------------------------------------------------------------------------------------ nls
    def fn_pac_estimate_nls(self, args, nargout):
        R = _R()
        eqname, plist, data, dates, jm, jo, orig, endo, aux_eqs = self._pe_inputs("nls", args)
        optimizer = args[4] if len(args) > 4 else None
        rest = list(args[5:])
        if optimizer is not None and not isinstance(optimizer, str) and not (isinstance(optimizer, np.ndarray) and optimizer.size == 0):
            raise MatlabError("pac.estimate.nls: the optimizer must be a character array")
        if isinstance(optimizer, np.ndarray):
            optimizer = None
        m = self.model
        name, pm = self._pe_pm(jo, eqname)
        try:
            prep = _pe.init_estimation(kind="nls", eqname=eqname, params=plist, data=data, rng=dates, M=self.M_,
                                       json_model=jm, json_original=jo, param_names=list(m.param_names),
                                       endo_names=endo, exo_names=list(m.exo_names), orig_endo_nbr=orig,
                                       aux_equations=aux_eqs, pac_model=pm, aux_vars=self._aux, params_vector=self._params())
            if bool(np.atleast_1d(pm["ec"]["istarget"])[1]):
                raise MatlabError(f"\nThe error correction term in PAC equation ({name}) is not correct.\nThe error "
                                  "correction term should be the difference between a trend\nand the level of the "
                                  f"endogenous variable.")
            update = self._pe_update(name)
            base = self._params()
            r_fun, ssr_fun, full, lhs_values = _pe.residual_function(prep, list(m.param_names), endo, list(m.exo_names),
                                                                    update, base)
            ecp = int(np.asarray(pm["ec"]["params"]).ravel()[0]) - 1
            ecname = m.param_names[ecp]
            names = [p for p, _ in prep.params]
            bounds = None
            if optimizer in ("lsqnonlin", "fmincon", "annealing", "particleswarm"):
                bounds = np.tile([-np.inf, np.inf], (len(names), 1))
                for i, nm in enumerate(names):
                    if nm == ecname:
                        bounds[i] = [0.0, 1.0]
            if optimizer in ("fmincon", "fminunc", "fminsearch", "lsqnonlin"):
                warnings.warn(f"pac.estimate.nls: '{optimizer}' is run with SciPy to a tight tolerance: the minimiser of the "
                              "sum of squares is that of Dynare's routine, its iterates are not", stacklevel=2)
            noprint, opt = _pe.opt4nls(rest)
            x0 = np.array([v for _, v in prep.params], float)
            ssr0 = ssr_fun(x0)
            if not np.isfinite(ssr0):
                raise MatlabError("Cannot evaluate the Sum of Squared Residuals on the initial guess.")
            o = self.options_
            cs = (1e-7, 1000, 2, 1e-6)
            try:
                cs = (float(farr(o["csminwel"]["tolerance"]["f"]).ravel()[0]), int(farr(o["csminwel"]["maxiter"]).ravel()[0]),
                      int(farr(o["gradient_method"]).ravel()[0]), float(farr(o["gradient_epsilon"]).ravel()[0]))
            except (KeyError, TypeError, IndexError, AttributeError):
                pass
            x1, SSR = _pe.minimise(ssr_fun, r_fun, x0, optimizer, bounds, opt, cs, rng=self.rng)
            r, C, student = _pe.nls_statistics(r_fun, x1, SSR)
        except _pe.PacEstimateError as exc:
            raise MatlabError(str(exc)) from None
        except _pe.PacEstimateUnsupported as exc:
            raise Unsupported(f"pac.estimate.nls: {exc}", "function") from None
        T = r.size
        lhs = lhs_values(x1)
        first = prep.range[0]
        base_new = full(x1)
        self._set_params(base_new)
        s = Struct()
        s["lhs"] = _ds.DSeries(lhs, ["lhs"], first)
        s["fit"] = _ds.DSeries(lhs - r, ["fit"], first)
        s["residual"] = _ds.DSeries(r, ["residual"], first)
        s["ssr"] = _F(SSR)
        s["s2"] = _F(SSR / T)
        s["R2"] = _F(1 - np.var(r, ddof=1) / np.var(lhs, ddof=1))
        s["parnames"] = Cell([nm for nm, _ in prep.params], (len(prep.params), 1))
        s["estimator"] = R.col(x1)
        s["covariance"] = R.mat(C)
        s["student"] = R.col(student)
        self.set_oo(f"pac.{name}", self._pe_merge(name, s))
        return []

    def _pe_merge(self, name, s):
        cur = self.oo_.get("pac")
        old = cur.get(name) if isinstance(cur, Struct) else None
        if isinstance(old, Struct):
            out = old.clone()
            for k in s.keys():
                out[k] = s[k]
            return out
        return s

    # ------------------------------------------------------------------------------------------ iterative OLS
    def fn_pac_estimate_iterative_ols(self, args, nargout):
        R = _R()
        if len(args) > 4:
            raise MatlabError("Too many input arguments.")
        eqname, plist, data, dates, jm, jo, orig, endo, aux_eqs = self._pe_inputs("iterative_ols", args)
        m = self.model
        name, pm = self._pe_pm(jo, eqname)
        try:
            prep = _pe.init_estimation(kind="iterative_ols", eqname=eqname, params=plist, data=data, rng=dates, M=self.M_,
                                       json_model=jm, json_original=jo, param_names=list(m.param_names),
                                       endo_names=endo, exo_names=list(m.exo_names), orig_endo_nbr=orig,
                                       aux_equations=aux_eqs, pac_model=pm, aux_vars=self._aux, params_vector=self._params())
            try:
                lb = float(farr(self.options_["pac"]["estimation"]["ols"]["share_of_optimizing_agents"]["lb"]).ravel()[0])
                ub = float(farr(self.options_["pac"]["estimation"]["ols"]["share_of_optimizing_agents"]["ub"]).ravel()[0])
            except (KeyError, TypeError, IndexError, AttributeError):
                lb, ub = 0.0, 1.0
            res = _pe.iterative_ols(prep, pm=pm, M_params=self._params(), param_names=list(m.param_names), endo_names=endo,
                                    exo_names=list(m.exo_names), orig_endo_nbr=orig, aux=self._aux,
                                    update=self._pe_update(name), share_bounds=(lb, ub))
        except _pe.PacEstimateError as exc:
            raise MatlabError(str(exc)) from None
        except _pe.PacEstimateUnsupported as exc:
            raise Unsupported(f"pac.estimate.iterative_ols: {exc}", "function") from None
        for k, crit, ssr in res["iterations"]:
            self.warnings.append(f"Iter. {k},\t crit: {crit:10.5f}\t ssr: {ssr:10.8f}")
        self._set_params(res["P"])
        n = np.asarray(res["estimator"]).size
        s = Struct()
        s["ssr"] = _F(res["ssr"])
        s["residual"] = R.col(res["residual"])
        s["estimator"] = R.col(res["estimator"])
        s["parnames"] = Cell([nm for nm, _ in prep.params], (len(prep.params), 1))
        s["covariance"] = R.mat(np.full((n, n), np.nan))
        s["student"] = R.col(np.full(n, np.nan))
        self.set_oo(f"pac.{name}", self._pe_merge(name, s))
        return []
