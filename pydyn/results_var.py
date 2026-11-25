# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# Behaviour follows Dynare 7.2: matlab/get_companion_matrix.m, matlab/+var_expectation/*.m, matlab/+pac/*.m,
# matlab/pac-tools/*.m (Copyright (C) 2018-2025 Dynare Team, GPL-3.0-or-later).
"""VAR, trend component, VAR expectation and PAC models on the live workspace of :mod:`pydyn.results`."""
from __future__ import annotations

import numpy as np

from . import matlab as _ml
from . import pac as _pac
from . import submodels as _sub
from . import var_model as _vm
from .matlab import Cell, MatlabError, Struct, Unsupported, farr, farrn, _F


def _R():
    from . import results
    return results


def _ints(v):
    return np.asarray(farr(v), float).ravel(order="F").astype(int)


def _flt(v):
    return np.asarray(farr(v), float).ravel(order="F")


def _isstr(v):
    return isinstance(v, str)


class VarModelMixin:
    """Methods of :class:`pydyn.results.Results` for ``var_model``, ``trend_component_model``,
    ``var_expectation_model`` and ``pac_model``."""

    _sub_loaded = False

    # ------------------------------------------------------------------------------------------ statements
    def _ensure_submodels(self, kind=""):
        """Read what the preprocessor writes for these models (see :mod:`pydyn.submodels`) into M_ and keep the
        generated functions."""
        if self._sub_loaded:
            return
        R = _R()
        try:
            gen = _sub.run_generator(self.mod_file)
        except _sub.SubModelError as exc:
            raise R.ResultsError(f"<{kind}>: {exc}") from None
        m = self.model
        if gen.endo_names != list(m.endo_names) or gen.param_names != list(m.param_names):
            raise R.ResultsError(f"<{kind}>: the second run of the preprocessor (to write the VAR / PAC description) does "
                                 "not give the same variables and parameters as the JSON output being used; run the .mod file "
                                 "itself, not an old JSON folder")
        try:
            newM = _sub.apply_lines(gen.lines, self.M_)
        except _sub.SubModelError as exc:
            raise R.ResultsError(f"<{kind}>: {exc}") from None
        self.set_global("M_", newM)
        self._sub_funcs = {}
        for name, text in gen.functions.items():
            try:
                body, funcs = _ml.parse(text)
            except _ml.ParseError as exc:
                raise R.ResultsError(f"<{kind}>: cannot parse the generated {name}.m: {exc}") from None
            self._sub_funcs[name] = funcs
        self._sub_loaded = True
        self._aux = _vm.AuxVars(m.raw.get("aux_vars", []), m.raw.get("orig_endo_nbr", m.n_endo))

    def st_var_model(self, st):
        self._ensure_submodels("var_model")

    def st_trend_component_model(self, st):
        self._ensure_submodels("trend_component_model")

    def st_var_expectation_model(self, st):
        self._ensure_submodels("var_expectation_model")

    def st_pac_model(self, st):
        self._ensure_submodels("pac_model")

    def st_pac_target_info(self, st):
        self._ensure_submodels("pac_target_info")

    # ------------------------------------------------------------------------------------------ the functions
    def _functions_var(self):
        F = {"get_companion_matrix": self.fn_get_companion_matrix,
             "var_expectation.initialize": self.fn_varexp_initialize,
             "var_expectation.update": self.fn_varexp_update,
             "pac.initialize": self.fn_pac_initialize,
             "pac.update.expectation": self.fn_pac_update_expectation,
             "pac.update.parameters": self.fn_pac_update_parameters,
             "pac.update.equation": self.fn_pac_update_expectation,
             "pac.print": self.fn_pac_print,
             "var_expectation.print": self.fn_varexp_print,
             "pac.mce.parameters": self.fn_pac_mce_parameters,
             "pac.bgp.set": self.fn_pac_bgp_set,
             "pac.bgp.get": self.fn_pac_bgp_get,
             f"{self.name}.varmatrices": self._gen_function("varmatrices"),
             f"{self.name}.trend_component_ar_a0": self._gen_function("trend_component_ar_a0"),
             f"{self.name}.pac_target_coefficients": self._gen_function("pac_target_coefficients"),
             **{f"{self.name}.{fn}": self._gen_function(fn, lazy=True) for fn in _sub.MODEL_FUNCTIONS},
             "get_ar_ec_matrices": self.fn_get_ar_ec_matrices,
             "simult_": self.fn_simult_,
             "get_companion_matrix_legacy": self.fn_get_companion_matrix_legacy}
        return F

    #: package functions of Dynare that pydyn refuses, with the reason
    _REFUSED = (("pac.expectations.", "pac.expectations.*: pydyn does not write the evaluation functions of the PAC expectation"),
                ("pac.", "this function of the pac package is not supported"),
                ("var_expectation.", "this function of the var_expectation package is not supported"),
                ("equation.", "the equation package (equation.evaluate, equation.cherrypick, ...) is not supported"))

    def var_refusal(self, path):
        """A function that refuses with a clear message for a package function that pydyn does not have, else None."""
        if path.startswith(f"{self.name}."):                   # +<model>/ functions the preprocessor writes and pydyn does not run
            def refuse(args, nargout, _p=path):
                raise Unsupported(f"{_p}: pydyn does not run this function of the model's package", "function")
            return refuse
        for prefix, why in self._REFUSED:
            if path.startswith(prefix):
                def refuse(args, nargout, _p=path, _w=why):
                    raise Unsupported(f"{_p}: {_w}", "function")
                return refuse
        return None

    # ------------------------------------------------------------------------------------------ helpers
    def _need_sub(self, what):
        if not self._sub_loaded:
            raise MatlabError(f"{what}: no VAR / trend component / PAC model has been declared")

    def _sub_struct(self, kind, name, what="Unknown type of auxiliary model."):
        M = self.M_
        d = M.get(kind) if isinstance(M, Struct) else None
        if not isinstance(d, Struct) or name not in d:
            raise MatlabError(what)
        return d[name]

    def _has(self, kind, name):
        d = self.M_.get(kind)
        return isinstance(d, Struct) and name in d

    def _gen_function(self, fname, lazy=False):
        """The function the preprocessor writes (``<model>.varmatrices`` ...), callable from the .mod file.  With ``lazy``
        (the model's own functions ``<model>.dynamic_resid``...) the preprocessor's text is read at the first call."""
        def run(args, nargout):
            if lazy:
                self._ensure_submodels(f"{self.name}.{fname}")
            self._need_sub(f"{self.name}.{fname}")
            return self._call_generated(fname, args, nargout)
        return run

    def _call_generated(self, fname, args, nargout):
        funcs = self._sub_funcs.get(fname)
        if not funcs:
            raise MatlabError(f"{fname}.m was not generated by the preprocessor")
        main = next(iter(funcs.values()))
        return self.I.call_user(main, funcs, args, nargout)

    def _params(self):
        return _flt(self.M_["params"]).copy()

    def _set_params(self, p):
        self.set_M("params", _R().col(p))

    def _endo_names(self):
        return list(self.model.endo_names)

    # ------------------------------------------------------------------------------------------ get_companion_matrix
    def fn_get_companion_matrix(self, args, nargout):
        """``[A0, A0star, AR, B] = get_companion_matrix(name[, type])``"""
        R = _R()
        if not 1 <= len(args) <= 2 or not all(_isstr(a) for a in args):
            raise MatlabError("get_companion_matrix: the name (and type) of the auxiliary model are character arrays")
        self._need_sub("get_companion_matrix")
        name = args[0]
        if len(args) < 2:
            if self._has("var", name):
                typ = "var"
            elif self._has("trend_component", name):
                typ = "trend_component"
            else:
                raise MatlabError("Unknown type of auxiliary model.")
        else:
            typ = args[1]
        if typ not in ("var", "trend_component"):
            raise MatlabError("Unknown type of auxiliary model.")
        mod = self._sub_struct(typ, name)
        params = self._params()
        names = self._endo_names()
        lhs = _ints(mod["lhs"])
        n = lhs.size
        if typ == "var":
            structural = bool(farr(mod["structural"]).ravel()[0]) if "structural" in mod else False
            out = self._call_generated("varmatrices", [name, R.col(params), np.bool_(structural)], 3)
            AR = np.asarray(farrn(out[0]), float)
            if AR.ndim == 2:
                AR = AR[:, :, None]
            constant = _flt(out[2])
            C, isconstant = _vm.companion_var(AR, constant)
            self.set_M(f"var.{name}.isconstant", np.bool_(isconstant))
            self.set_oo(f"var.{name}.CompanionMatrix", R.mat(C))
            self.set_M(f"var.{name}.list_of_variables_in_companion_var", R.cellstr([names[i - 1] for i in lhs]))
            res = [np.zeros((0, 0)), np.zeros((0, 0)), _arr_nd(AR), np.zeros((0, 0))]
            return res[:max(nargout, 1)] if nargout else []
        out = self._call_generated("trend_component_ar_a0", [name, R.col(params)], 3)
        AR = np.asarray(farrn(out[0]), float)
        if AR.ndim == 2:
            AR = AR[:, :, None]
        A0 = np.asarray(farr(out[1]), float)
        A0star = np.asarray(farr(out[2]), float)
        targets = np.asarray(farr(mod["targets"]), float).ravel() != 0
        q = int(targets.sum())
        m = n - q
        target_eqn = _ints(mod["target_eqn"])
        ecm = np.flatnonzero(~targets)
        for i in range(m):
            if not self._aux.difference_order(int(lhs[ecm[i]])):
                raise MatlabError(f"{name} is not a trend component model. The LHS variables should be in differences")
        C, B = _vm.companion_trend_component(AR, A0.reshape(m, m) if A0.size == m * m else A0, A0star, n, target_eqn, targets)
        self.set_oo(f"trend_component.{name}.CompanionMatrix", R.mat(C))
        lst = [names[i - 1] for i in lhs]
        for i in range(m):
            a = self._aux.info(int(lhs[ecm[i]]))
            if a is None or a["type"] != 8:
                raise MatlabError("This is a bug. Please contact the Dynare Team.")
            lst[ecm[i]] = names[int(a["orig_index"]) - 1]
        self.set_M(f"trend_component.{name}.list_of_variables_in_companion_var", R.cellstr(lst))
        res = [R.mat(A0), R.mat(A0star), _arr_nd(AR), _arr_nd(B)]
        return res[:max(nargout, 1)] if nargout else []

    # ------------------------------------------------------------------------------------------ legacy: AR and EC matrices from g1
    def _model_type_of(self, name, given=None):
        if given is None:
            given = "var"
            if not self._has("var", name):
                given = "trend_component"
                if not self._has("trend_component", name):
                    raise MatlabError(f"Could not find {name} in M_.var or M_.trend_component. First declare it via the var_model "
                                      "or trend_component_model statement.")
        elif not self._has(given, name):
            raise MatlabError(f"Could not find M_.{given}.{name}. First declare it via the var_model or trend_component_model statement.")
        return given

    def fn_get_ar_ec_matrices(self, args, nargout):
        """``get_ar_ec_matrices(model_name[, model_type])``: oo_.<type>.<name>.{ar, ec, ar_idx, ec_idx} from the Jacobian of the
        equations of the model (at ones)."""
        R = _R()
        self._need_sub("get_ar_ec_matrices")
        if not 1 <= len(args) <= 2 or not all(_isstr(a) and a for a in args):
            raise MatlabError("The first argument must be a non-empty string")
        name = args[0]
        typ = self._model_type_of(name, args[1] if len(args) > 1 else None)
        mod = self.M_[typ][name]
        m = self.model
        n = m.n_endo
        self.pull()
        g1 = m.dynamic_jac(np.ones(3 * n), np.ones(m.n_exo), self._params(), np.zeros(n))
        tags = [str(t) for t in mod["eqtags"].items]
        eqs = m.raw.get("model", [])
        rows = np.zeros(len(tags), int)
        for i, t in enumerate(tags):
            hit = [j + 1 for j, e in enumerate(eqs) if t in [str(v) for v in (e.get("tags") or {}).values()]]
            if len(hit) > 1:
                raise Unsupported(f"get_ar_ec_matrices: equation tag {t} is used by several equations", "function")
            rows[i] = hit[0] if hit else 0
        sel = -1 * g1[np.maximum(rows, 1) - 1, :][:, :3 * n]
        sel[rows == 0, :] = 0.0
        if int(m.info["maximum_endo_lead"]) > 0:
            # Dynare re-indexes the already reduced Jacobian here (g1(g1rows, :) twice): not reproduced
            raise Unsupported("get_ar_ec_matrices: models with leads", "function")
        lhs = _ints(mod["lhs"])
        if sel.shape[0] != lhs.size:
            raise MatlabError("Assertion failed.")
        vae = [(_ints(c["var"]), _ints(c["lag"])) for c in mod["rhs"]["vars_at_eq"].items]
        try:
            ar, ec, ar_idx, ec_idx = _vm.ar_ec_matrices(sel, lhs, vae, _flt(mod["max_lag"]), self._aux, n)
        except _vm.VarModelError as exc:
            raise MatlabError(str(exc)) from None
        self.set_oo(f"{typ}.{name}.ar", _arr_nd(ar))
        self.set_oo(f"{typ}.{name}.ec", _arr_nd(ec))
        self.set_oo(f"{typ}.{name}.ar_idx", R.col(ar_idx))
        self.set_oo(f"{typ}.{name}.ec_idx", R.col(ec_idx) if ec_idx.size else np.zeros((0, 0)))
        return []

    def fn_get_companion_matrix_legacy(self, args, nargout):
        """``[A0, AR, B] = get_companion_matrix_legacy(name[, type])``"""
        R = _R()
        self._need_sub("get_companion_matrix_legacy")
        if not 1 <= len(args) <= 2 or not all(_isstr(a) for a in args):
            raise MatlabError("get_companion_matrix_legacy: the name (and type) of the auxiliary model are character arrays")
        name = args[0]
        if len(args) < 2:
            if self._has("var", name):
                typ = "var"
            elif self._has("trend_component", name):
                typ = "trend_component"
            else:
                raise MatlabError("Unknown type of auxiliary model.")
        else:
            typ = args[1]
        self.fn_get_ar_ec_matrices([name, typ], 0)
        calib = self.oo_[typ][name]
        AR = np.asarray(farrn(calib["ar"]), float)
        if AR.ndim == 2:
            AR = AR[:, :, None]
        ec = np.asarray(farrn(calib["ec"]), float)
        if ec.ndim == 2:
            ec = ec[:, :, None]
        p = AR.shape[2]
        n = AR.shape[0]
        mod = self.M_[typ][name]
        names = self._endo_names()
        lhs = _ints(mod["lhs"])
        if typ == "var":
            C = np.zeros((n * p, n * p))
            C[0:n, 0:n] = AR[:, :, 0]
            for i in range(2, p + 1):
                C[0:n, (i - 1) * n:i * n] = AR[:, :, i - 1]
                C[(i - 1) * n:i * n, (i - 2) * n:(i - 1) * n] = np.eye(n)
            self.set_oo(f"var.{name}.CompanionMatrix", R.mat(C))
            self.set_M(f"var.{name}.list_of_variables_in_companion_var", R.cellstr([names[i - 1] for i in lhs]))
            res = [np.zeros((0, 0)), _arr_nd(AR), np.zeros((0, 0))]
            return res[:max(nargout, 1)] if nargout else []
        targets = np.asarray(farr(mod["targets"]), float).ravel() != 0
        q = int(targets.sum())
        m = n - q
        tg = np.flatnonzero(targets)
        ecm = np.flatnonzero(~targets)
        for i in range(m):
            if not self._aux.difference_order(int(lhs[ecm[i]])):
                raise MatlabError(f"Model {name} is not a Trend component  model! LHS variables should be in difference")
        tv = _ints(mod["target_vars"])
        tv = tv[tv > 0]
        reorder = [list(tv).index(v) + 1 if v in tv else 0 for v in lhs[tg]]
        if 0 in reorder:
            raise MatlabError("Index (_,0): out of bound; value 0 out of bound")
        tg = tg[np.array(reorder) - 1]
        A0 = ec[ecm, :, 0]
        ARm = AR[np.ix_(ecm, ecm, np.arange(p))]
        B = np.zeros((m + q, m + q, p + 1))
        B[np.ix_(ecm, ecm, [0])] = (np.eye(m) + A0 + ARm[:, :, 0])[:, :, None]
        B[np.ix_(ecm, tg, [0])] = (-A0)[:, :, None]
        B[np.ix_(tg, tg, [0])] = np.eye(q)[:, :, None]
        for i in range(2, p + 1):
            B[np.ix_(ecm, ecm, [i - 1])] = (ARm[:, :, i - 1] - ARm[:, :, i - 2])[:, :, None]
        B[np.ix_(ecm, ecm, [p])] = (-ARm[:, :, p - 1])[:, :, None]
        C = np.zeros((n * (p + 1), n * (p + 1)))
        for i in range(1, p + 1):
            C[0:n, (i - 1) * n:i * n] = B[:, :, i - 1]
            C[i * n:(i + 1) * n, (i - 1) * n:i * n] = np.eye(n)
        C[0:n, p * n:(p + 1) * n] = B[:, :, p]
        self.set_oo(f"trend_component.{name}.CompanionMatrix", R.mat(C))
        lst = [names[i - 1] for i in lhs]
        for i in range(m):
            a = self._aux.info(int(lhs[ecm[i]]))
            if a is None or a["type"] != 8:
                raise MatlabError("This is a bug. Please contact the Dynare Team.")
            lst[ecm[i]] = names[int(a["orig_index"]) - 1]
        self.set_M(f"trend_component.{name}.list_of_variables_in_companion_var", R.cellstr(lst))
        res = [R.mat(A0), _arr_nd(ARm), _arr_nd(B)]
        return res[:max(nargout, 1)] if nargout else []

    # ------------------------------------------------------------------------------------------ VAR expectation
    def _aux_type(self, auxname):
        if self._has("var", auxname):
            return "var"
        if self._has("trend_component", auxname):
            return "trend_component"
        raise MatlabError("Unknown type of auxiliary model.")

    def fn_varexp_initialize(self, args, nargout):
        self._need_sub("var_expectation.initialize")
        if len(args) != 1 or not _isstr(args[0]):
            raise MatlabError("var_expectation.initialize: the name of the model is a character array")
        ve = self._sub_struct("var_expectation", args[0], f"VAR_EXPECTATION_MODEL {args[0]} is not defined!")
        auxname = ve["auxiliary_model_name"]
        typ = self._aux_type(auxname)
        self.set_M(f"var_expectation.{args[0]}.auxiliary_model_type", typ)
        self.fn_get_companion_matrix([auxname, typ], 0)
        return []

    def fn_varexp_update(self, args, nargout):
        R = _R()
        self._need_sub("var_expectation.update")
        if len(args) != 1 or not _isstr(args[0]) or not args[0]:
            raise MatlabError("First input argument must be a row character array!")
        name = args[0]
        if not self._has("var_expectation", name):
            raise MatlabError(f"VAR_EXPECTATION_MODEL {name} is not defined!")
        ve = self.M_["var_expectation"][name]
        typ = ve.get("auxiliary_model_type")
        auxname = ve["auxiliary_model_name"]
        if typ is None:
            raise MatlabError("reference to non-existent field 'auxiliary_model_type'")
        if not self._has(typ, auxname):
            raise MatlabError(f"Unknown VAR ({auxname}) in VAR_EXPECTATION_MODEL ({name})!")
        aux = self.M_[typ][auxname]
        oo = self.oo_
        if typ not in oo or auxname not in oo[typ]:
            raise MatlabError(f"Auxiliary model {auxname} has to be estimated or calibrated first!")
        calib = oo[typ][auxname]
        if "CompanionMatrix" not in calib or np.any(np.isnan(farr(calib["CompanionMatrix"]))):
            raise MatlabError(f"Auxiliary model {auxname} has to be estimated first.\nPlease use *get_companion_matrix command first.")
        C = np.asarray(farr(calib["CompanionMatrix"]), float)
        if "discount_value" in ve:
            discount = float(farr(ve["discount_value"]).ravel()[0])
        elif "discount_index" in ve:
            discount = float(self._params()[int(farr(ve["discount_index"]).ravel()[0]) - 1])
        else:
            raise MatlabError("This is most likely a bug. Pleasse conntact the Dynare Team.")
        if discount <= 0:
            raise MatlabError("The discount factor must be positive.")
        if discount > 1:
            raise MatlabError("The discount cannot be greater than one.")
        timeshift = float(farr(ve["time_shift"]).ravel()[0]) if "time_shift" in ve else 0.0
        if timeshift > 0:
            raise MatlabError("Option time_shift of the var_expectation command cannot be positive.")
        names = self._endo_names()
        expr = ve["expr"]
        evars = _ints(expr["vars"])
        lst = [str(x) for x in aux["list_of_variables_in_companion_var"].items]
        ids = np.zeros(evars.size, int)
        for i, v in enumerate(evars):
            nm = names[v - 1]
            if nm not in lst:
                raise MatlabError(f"Cannot find variable {nm} in the companion VAR")
            ids[i] = lst.index(nm) + 1
        if "isconstant" in aux and bool(farr(aux["isconstant"]).ravel()[0]):
            ids = ids + 1
        horizon = _flt(ve["horizon"])
        n = C.shape[0]
        alpha = np.zeros(n)
        alpha[ids - 1] = _flt(expr["constants"])
        eparams = _flt(expr["params"])
        sel = ~np.isnan(eparams)
        alpha[ids[sel] - 1] = eparams[sel]
        try:
            par = _vm.var_expectation_parameters(C, alpha.reshape(1, -1), discount, horizon, timeshift).ravel()
        except _vm.VarModelError as exc:
            raise MatlabError(str(exc)) from None
        pidx = _ints(ve["param_indices"])
        p = self._params()
        if typ == "var":
            if bool(farr(aux["isconstant"]).ravel()[0]):
                p[pidx - 1] = par
            else:
                p[pidx[0] - 1] = 0.0
                p[pidx[1:] - 1] = par
        else:
            p[pidx - 1] = par
        self._set_params(p)
        return []

    # ------------------------------------------------------------------------------------------ PAC
    def fn_pac_initialize(self, args, nargout):
        self._need_sub("pac.initialize")
        if len(args) != 1 or not _isstr(args[0]):
            raise MatlabError("pac.initialize: the name of the PAC model is a character array")
        name = args[0]
        pm = self._sub_struct("pac", name, f"reference to non-existent field '{name}'")
        auxname = pm.get("auxiliary_model_name", "")
        mce = not auxname
        self.set_M(f"pac.{name}.model_consistent_expectations", np.bool_(mce))
        if not mce:
            self.fn_get_companion_matrix([auxname, pm["auxiliary_model_type"]], 0)
        pm = self.M_["pac"][name]
        for fld in ("additive", "optim_additive", "non_optimizing_behaviour"):
            if fld in pm:
                k = _flt(pm[fld]["params"]).size
                self.set_M(f"pac.{name}.{fld}.bgp", Cell([np.bool_(False)] * k, (1, k)))
        return []

    def fn_pac_update_expectation(self, args, nargout):
        if len(args) != 1:
            raise MatlabError("Invalid call to pac.update.expectation")
        self.fn_pac_update_parameters(args, 0)
        return []

    def fn_pac_update_parameters(self, args, nargout):
        self._need_sub("pac.update.parameters")
        if not args or not _isstr(args[0]):
            raise MatlabError("First input argument must be a row character array!")
        try:
            newp = _pac.update_parameters(self._pac_context(), args[0], self._params())
        except _pac.PacError as exc:
            raise MatlabError(str(exc)) from None
        self._set_params(newp)
        return []

    def fn_pac_mce_parameters(self, args, nargout):
        self._need_sub("pac.mce.parameters")
        if len(args) != 1 or not _isstr(args[0]):
            raise MatlabError("Input argument must be a row character array!")
        try:
            newp = _pac.mce_parameters(self.M_, args[0], self._params())
        except _pac.PacError as exc:
            raise MatlabError(str(exc)) from None
        self._set_params(newp)
        return []

    def fn_pac_bgp_get(self, args, nargout):
        """``pac.bgp.get(pacmodel, paceq, kind, id)``: M_.pac.<pacmodel>.<kind>.bgp{id}"""
        self._need_sub("pac.bgp.get")
        if len(args) != 4:
            raise MatlabError("Invalid call to pac.bgp.get")
        pm = self._sub_struct("pac", args[0], f"reference to non-existent field '{args[0]}'")
        if args[2] not in pm or "bgp" not in pm[args[2]]:
            raise MatlabError(f"reference to non-existent field '{args[2]}'")
        try:
            return [pm[args[2]]["bgp"].items[int(farr(args[3]).ravel()[0]) - 1]]
        except IndexError:
            raise MatlabError("index out of bound") from None

    def fn_pac_bgp_set(self, args, nargout):
        """``pac.bgp.set(pacmodel, paceq, variable, nonzeromean)``"""
        self._need_sub("pac.bgp.set")
        if len(args) != 4 or not all(_isstr(a) for a in args[:3]):
            raise MatlabError("Invalid call to pac.bgp.set")
        pacmodl, paceq, variable, nonzeromean = args
        pm = self._sub_struct("pac", pacmodl, f"reference to non-existent field '{pacmodl}'")
        names, exo = self._endo_names(), list(self.model.exo_names)
        xflag = False
        if variable in names:
            ide = names.index(variable) + 1
        elif variable in exo:
            ide, xflag = exo.index(variable) + 1, True
        else:
            raise MatlabError(f"Endogenous/Exogenous variable {variable} is unknown.")
        for typ in ("additive", "optim_additive", "non_optimizing_behaviour"):
            if typ not in pm:
                continue
            sec = pm[typ]
            if "additive" not in pm:
                raise MatlabError("reference to non-existent field 'additive'")
            if "bgp" not in pm["additive"]:
                k = _flt(sec["params"]).size
                sec = _R().struct_set(sec, "bgp", Cell([np.bool_(False)] * k, (1, k)))
            vars_ = _ints(sec["vars"])
            if ide in vars_:
                ie = int(np.flatnonzero(vars_ == ide)[0])
                isendo = bool(np.asarray(farr(sec["isendo"])).ravel()[ie])
                if isendo == xflag:
                    raise MatlabError("Variable type issue.")
                items = list(sec["bgp"].items)
                items[ie] = nonzeromean
                sec = _R().struct_set(sec, "bgp", Cell(items, sec["bgp"].shape))
                self.set_M(f"pac.{pacmodl}.{typ}", sec)
                return []
            if typ != "additive" or True:
                pm = pm
        import warnings
        warnings.warn(f"{variable} is not an exogenous variable in equation {paceq}.")
        return []

    # ------------------------------------------------------------------------------------------ print
    def fn_pac_print(self, args, nargout):
        """``pac.print(pacmodel, eqname[, withcalibration])``"""
        if len(args) not in (2, 3):
            raise MatlabError("Invalid call to pac.print")
        return self._print_expectations(args[1], args[0], "pac", True if len(args) < 3 else bool(farr(args[2]).ravel()[0]))

    def fn_varexp_print(self, args, nargout):
        """``var_expectation.print(name[, withcalibration])``"""
        if len(args) not in (1, 2):
            raise MatlabError("Invalid call to var_expectation.print")
        return self._print_expectations("fake", args[0], "var", True if len(args) < 2 else bool(farr(args[1]).ravel()[0]))

    def _print_expectations(self, eqname, name, kind, withcalibration):
        """print_expectations.m: <model>/model/<kind>-expectations/<name>-{parameters,expression}.inc (and, for PAC with a growth
        neutrality correction, -growth-neutrality-correction.inc).  The dseries evaluation function that Dynare also writes
        (+<model>/+<kind>_expectations/+<name>/evaluate.m) is NOT written."""
        import datetime
        self._need_sub("print_expectations")
        if not (_isstr(eqname) and _isstr(name)):
            raise MatlabError("First input argument must be a row character array.")
        field = "var_expectation" if kind == "var" else "pac"
        if not self._has(field, name):
            raise MatlabError(f"{'VAR_EXPECTATION_MODEL' if kind == 'var' else 'PAC_EXPECTATION_MODEL'} {name} is not defined.")
        em = self.M_[field][name]
        if "auxiliary_model_name" not in em or not em["auxiliary_model_name"]:
            if kind == "pac":
                raise MatlabError("print method does not work in PAC/MCE.")
        typ = em["auxiliary_model_type"]
        if not self._has(typ, em["auxiliary_model_name"]):
            raise MatlabError(f"Unknown VAR/TREND_COMPONENT model ({em['auxiliary_model_name']}) in "
                              f"{'VAR_EXPECTATION_MODEL' if kind == 'var' else 'PAC_EXPECTATION_MODEL'} ({name})!")
        pn = list(self.model.param_names)
        params = self._params()
        outdir = self.model.json_dir.parent / f"{kind}-expectations"
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        base = outdir / f"{name}-parameters.inc"
        comps = em["components"].elems if isinstance(em.get("components"), _sub.StructArray) else []

        def hidx():
            if kind == "var":
                return _ints(em["param_indices"])
            if "h_param_indices" in em:
                return _ints(em["h_param_indices"])
            return np.concatenate([_ints(c["h_param_indices"]) for c in comps])
        idx = hidx()
        decl = "parameters" + "".join(f" {pn[i - 1]}" for i in idx)
        out = [f"// This file has been generated by Dynare ({stamp}).\n\n", f"{decl};\n\n"]
        if withcalibration:
            for i in idx:
                out.append(f"{pn[i - 1]} = {params[i - 1]:.16f};\n")
        growth_correction = False
        if kind == "pac":
            if "growth_neutrality_param_index" in em:
                g = int(_flt(em["growth_neutrality_param_index"])[0])
                out.append("\n")
                out.append(f"parameters {pn[g - 1]};\n\n")
                if withcalibration:
                    out.append(f"{pn[g - 1]} = {params[g - 1]:.16f};\n")
                growth_correction = True
            else:
                for c in comps:
                    if "growth_neutrality_param_index" in c and np.size(farr(c["growth_neutrality_param_index"])) > 0:
                        g = int(_flt(c["growth_neutrality_param_index"])[0])
                        out.append("\n")
                        out.append(f"parameters {pn[g - 1]};\n\n")
                        if withcalibration:
                            out.append(f"{pn[g - 1]} = {params[g - 1]:.16f};\n")
                        growth_correction = True
        base.write_text("".join(out))
        print()
        print(f"Parameters declarations and calibrations are saved in {self._rel(base)}.")
        try:
            if kind == "var":
                expression, _ = _pac.write_expectations(self._pac_context(), name, kind, True)
            else:
                expression, growth = _pac.write_expectations(self._pac_context(), name, kind, True, two_outputs=True)
        except _pac.PacError as exc:
            raise MatlabError(str(exc)) from None
        fn = outdir / f"{name}-expression.inc"
        fn.write_text(f"// This file has been generated by Dynare ({stamp}).\n" + expression)
        print(f"Expectation unrolled expression is saved in {self._rel(fn)}.")
        if kind == "pac" and growth_correction:
            fg = outdir / f"{name}-growth-neutrality-correction.inc"
            fg.write_text(f"// This file has been generated by Dynare ({stamp}).\n" + growth)
            print(f"Growth neutrality correction is saved in {self._rel(fg)}.")
        import warnings
        warnings.warn("print_expectations: pydyn does not write the dseries evaluation function "
                      f"+{self.name}/+{kind}_expectations/+{name}/evaluate.m")
        print()
        return []

    def _rel(self, path):
        return f"{self.name}/model/{path.parent.name}/{path.name}"

    def _pac_context(self):
        return _pac.Context(self.M_, self.oo_, self._endo_names(), list(self.model.param_names), self._aux)


def _arr_nd(a):
    from .matlab import ret
    return ret(np.asarray(a, float).copy())
