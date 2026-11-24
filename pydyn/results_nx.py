# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
#
# Behaviour follows Dynare 7.2 (matlab/send_{endogenous,exogenous}_variables_to_workspace.m, send_irfs_to_workspace.m,
# smoother2histval.m; Copyright (C) Dynare Team, GPL-3.0-or-later).
"""Small commands of the results layer: ``send_*_to_workspace`` and ``smoother2histval``."""
from __future__ import annotations

import numpy as np

from .matlab import Cell, MatlabError, Struct, Unsupported, farr, _F


def _R():
    from . import results
    return results


from .results_pacest import PacEstimateMixin
from .results_fd import ForecastDecompMixin


class _Omit(Exception):
    pass


class NextMixin(PacEstimateMixin, ForecastDecompMixin):
    _RESULTS_MAX_FIELD = 4_000_000             # bytes: larger top-level fields of oo_ are not written to <name>_results.mat
    def _functions_nx(self):
        return {**self._functions_pacest(), "forecasts.run": self.fn_forecasts_run,
                "send_endogenous_variables_to_workspace": self.fn_send_endo,
                "send_exogenous_variables_to_workspace": self.fn_send_exo,
                "send_irfs_to_workspace": self.fn_send_irfs,
                "collect_latex_files": self.fn_collect_latex_files, "system": self.fn_system}

    # ------------------------------------------------------------------ <fname>/Output/<fname>_results.mat
    def _save_results(self):
        """Dynare's driver ends with ``save([M_.dname '/Output/' M_.fname '_results.mat'], 'oo_', 'M_', 'options_')`` (unless
        ``nosave``): other files of the Dynare test suite load it.  Values pydyn cannot represent in a MAT-file (struct arrays
        such as M_.aux_vars, unknown values) are left out and listed in ``pydyn_omitted``: ``load`` gives them back as unknown
        values, which refuse when they are used."""
        import hashlib
        import json
        from scipy.io import savemat
        if getattr(self, "nosave", False):
            return
        R = _R()
        omitted = []

        def conv(v, path):
            try:
                return R._to_mat_file(v, path)
            except (Unsupported, MatlabError):
                if isinstance(v, Struct) and not hasattr(v, "get_field"):
                    out = {}
                    for k, x in v.items():
                        try:
                            out[k] = conv(x, f"{path}.{k}")
                        except _Omit:
                            pass
                    return out
                omitted.append(path)
                raise _Omit() from None

        def nbytes(x):
            if isinstance(x, np.ndarray):
                return x.nbytes
            if isinstance(x, dict):
                return sum(nbytes(y) for y in x.values())
            return 0

        data = {}
        for name, v in (("oo_", self.oo_), ("M_", self.M_), ("options_", self.options_)):
            try:
                data[name] = conv(v, name)
            except _Omit:
                return
            for k in [k for k, x in data[name].items() if nbytes(x) > self._RESULTS_MAX_FIELD]:     # posterior densities, ...
                del data[name][k]
                omitted.append(f"{name}.{k}")
        if omitted:
            data["pydyn_omitted"] = np.array(omitted, dtype=object).reshape(-1, 1)
        base = self.model.json_dir.parents[1]
        path = base / "Output" / f"{self.model.json_dir.parents[1].name}_results.mat"
        rec_path = base / self._WRITTEN
        try:
            rec = json.loads(rec_path.read_text()) if rec_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            rec = {}
        key = f"{base.name}/Output/{path.name}"
        if path.exists():
            if not (path.is_file() and not path.is_symlink() and rec.get(key) == hashlib.sha256(path.read_bytes()).hexdigest()):
                return                      # not written by pydyn.dynare(): left alone
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            savemat(str(path), data, do_compression=False, oned_as="column", long_field_names=True)
        except (OSError, ValueError, TypeError, NotImplementedError):          # a value the MAT writer cannot take: no file
            path.unlink(missing_ok=True)
            return
        rec[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        rec_path.write_text(json.dumps(rec, indent=0))

    def fn_collect_latex_files(self, args, nargout):
        """``collect_latex_files`` written as a function call in a .mod file: like the statement, pydyn writes no LaTeX."""
        from .options import warn_once
        if args or nargout:
            raise MatlabError("collect_latex_files: invalid call")
        warn_once(self.model, "collect_latex_files: LaTeX output is not implemented in pydyn; the call writes no file "
                              "(the model, parameters and priors are available as Python objects)", stacklevel=2)
        return []

    def fn_system(self, args, nargout):
        raise Unsupported("system: pydyn.dynare() does not run external programs (such as pdflatex on the TeX binder, "
                          "which pydyn does not write)", "function")

    # ------------------------------------------------------------------ forecasts.run
    def fn_forecasts_run(self, args, nargout):
        """``forecast = forecasts.run(var_list_, M_, options_, oo_, 'smoother')``: the forecast from the end-of-sample smoothed
        state of the last ``estimation`` (the one ``estimation(forecast=N)`` stores in oo_.forecast); the task 'simul' is not
        supported."""
        R = _R()
        if len(args) not in (5, 6) or not isinstance(args[4], str):
            raise MatlabError("forecasts.run: invalid call")
        if args[4] != "smoother":
            raise Unsupported(f"forecasts.run with the task '{args[4]}' (only 'smoother')", "function")
        if len(args) == 6:
            raise Unsupported("forecasts.run with a dataset_info argument", "function")
        res = getattr(self, "_smooth_res", None)
        if res is None or self.oo_.get("SmoothedVariables") is None:
            raise MatlabError("Reference to non-existent field 'SmoothedVariables'.")
        if not isinstance(args[1], Struct) or not isinstance(args[3], Struct):
            raise MatlabError("forecasts.run: M_ and oo_ must be structures")
        if args[1] is not self.M_ and args[1] != self.M_ or args[3] is not self.oo_ and args[3] != self.oo_:
            raise Unsupported("forecasts.run with an M_ or oo_ that is not the current one", "function")
        names = args[0].items if isinstance(args[0], Cell) else []
        names = [str(x) for x in names] or list(self.model.endo_names[:self.model.raw.get("orig_endo_nbr", self.model.n_endo)])
        h = int(np.asarray(farr(self.options_["forecast"]), float).ravel()[0])
        from .forecast import forecast_at
        import warnings
        cs = self.opt("forecasts.conf_sig", 0.9)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = forecast_at(self.model, smoothed=res, horizon=h, conf_sig=float(cs), var_list=names)
        except (NotImplementedError, ValueError, KeyError) as exc:
            raise MatlabError(f"forecasts.run: {exc}") from None
        return [self._forecast_struct(fc)]

    # ------------------------------------------------------------------ send_*_to_workspace
    def fn_send_endo(self, args, nargout):
        """``assignin('base', M_.endo_names{i}, oo_.endo_simul(i,:)')`` for every endogenous variable."""
        R = _R()
        self.pull()
        sim = self.oo_.get("endo_simul")
        if sim is None:
            raise MatlabError("Reference to non-existent field 'endo_simul'.")
        sim = np.atleast_2d(np.asarray(farr(sim), float))
        for i, nm in enumerate(self.model.endo_names):
            if i >= sim.shape[0]:
                raise MatlabError(f"index ({i + 1},_): out of bound {sim.shape[0]}")
            self.I.base_assign(nm, R.col(sim[i]))
        return []

    def fn_send_exo(self, args, nargout):
        """``assignin('base', M_.exo_names{i}, oo_.exo_simul(:,i)')`` for every exogenous variable (a row)."""
        R = _R()
        self.pull()
        sim = self.oo_.get("exo_simul")
        sim = np.atleast_2d(np.asarray(farr(sim), float)) if sim is not None else np.zeros((0, 0))
        for i, nm in enumerate(self.model.exo_names):
            if i >= sim.shape[1]:
                raise MatlabError(f"index (_,{i + 1}): out of bound {sim.shape[1]}")
            self.I.base_assign(nm, R.row(sim[:, i]))
        return []

    def fn_send_irfs(self, args, nargout):
        """``assignin('base', name, oo_.irfs.(name)')`` for every field of oo_.irfs."""
        irfs = self.oo_.get("irfs")
        if isinstance(irfs, Struct):
            for nm, v in irfs.items():
                a = np.asarray(farr(v), float)
                self.I.base_assign(nm, _R().mat(a.T) if a.ndim == 2 else v)
        return []

    # ------------------------------------------------------------------ smoother2histval
    def st_smoother_2_histval(self, st):
        """``smoother2histval(options)`` (matlab/smoother2histval.m): the smoothed values of ``period`` (and the
        ``maximum_lag - 1`` periods before it) become ``M_.endo_histval`` / ``M_.exo_histval``, or, with ``outfile``, a
        dseries saved in a MAT-file (``histval_file`` reads it)."""
        R = _R()
        from . import dseries as _ds
        from .dates import Date
        opts = st.get("options") or {}
        self.pull()
        m = self.model
        sv, ss = self.oo_.get("SmoothedVariables"), self.oo_.get("SmoothedShocks")
        if not isinstance(sv, Struct):
            raise R.ResultsError("MATLAB error in the .mod file: Could not find smoothed variables, did you compute them? "
                                 '(either by the "smoother" option of the "estimation" command, or the "calib_smoother" command')
        first = next(iter(sv.values()), None)
        post_metropolis = isinstance(first, Struct)
        if post_metropolis:
            sv, ss = sv["Mean"], ss["Mean"]
        names_v, names_x = list(sv), list(ss) if isinstance(ss, Struct) else []
        n = np.asarray(farr(sv[names_v[0]])).size
        ml = int(m.info["maximum_lag"])
        oml = int(m.info.get("orig_maximum_lag", ml))
        if n < int(m.info["maximum_endo_lag"]):
            raise R.ResultsError("MATLAB error in the .mod file: Not enough observations to create initial conditions")
        invars = opts.get("invars", names_v + names_x)
        invars = [invars] if isinstance(invars, str) else list(invars.get("symbol_list", invars)) if isinstance(invars, dict) else list(invars)
        period = int(opts["period"]) if "period" in opts else n
        if "period" in opts:
            if period > n:
                raise R.ResultsError("MATLAB error in the .mod file: The period that you indicated is beyond the data sample")
            if period < int(m.info["maximum_endo_lag"]):
                raise R.ResultsError("MATLAB error in the .mod file: The period that you indicated is too small to "
                                     "construct initial conditions")
        outvars = opts.get("outvars", invars)
        outvars = [outvars] if isinstance(outvars, str) else list(outvars.get("symbol_list", outvars)) if isinstance(outvars, dict) else list(outvars)
        if len(invars) != len(outvars):
            raise R.ResultsError("MATLAB error in the .mod file: The number of input and output variables is not the same")
        outfile = opts.get("outfile")
        loglin = bool(self.oo_["Smoother"]["loglinear"]) if isinstance(self.oo_.get("Smoother"), Struct) and \
            "loglinear" in self.oo_["Smoother"] else False
        ssv = np.asarray(farr(self.oo_["steady_state"]), float).ravel(order="F")
        xsv = np.asarray(farr(self.oo_["exo_steady_state"]), float).ravel(order="F") if m.n_exo else np.zeros(0)
        if outfile is None:
            eh = np.repeat(ssv.reshape(-1, 1), ml, axis=1)
            xh = np.repeat(xsv.reshape(-1, 1), ml, axis=1) if m.n_exo else np.zeros((0, ml))
        else:
            data = np.zeros((oml, len(invars)))
            for i, nm in enumerate(outvars):
                if nm in m.endo_names:
                    data[:, i] = ssv[m.endo_names.index(nm)]
        for i, nm in enumerate(invars):
            if nm in m.endo_names:
                s = np.asarray(farr(sv[nm]), float).ravel()
                s = np.exp(s) if loglin else s
            elif nm in m.exo_names:
                s = np.asarray(farr(ss[nm]), float).ravel()
            else:
                raise R.ResultsError("MATLAB error in the .mod file: smoother2histval: unknown input variable")
            if outfile is None:
                v = s[period - ml:period]
                j, jx = (m.endo_names.index(outvars[i]) if outvars[i] in m.endo_names else None,
                         m.exo_names.index(outvars[i]) if outvars[i] in m.exo_names else None)
                if j is not None:
                    eh[j, :] = v
                if jx is not None:
                    xh[jx, :] = v
                if j is None and jx is None:
                    raise R.ResultsError(f"MATLAB error in the .mod file: smoother2histval: output variable {outvars[i]} does not exist.")
            else:
                data[:, i] = s[period - oml:period]
        if outfile is None:
            self.set_M("endo_histval", R.mat(eh))
            self.set_M("exo_histval", R.mat(xh) if m.n_exo else np.zeros((0, 0)))
            return
        import os
        d, f = os.path.split(outfile)
        base, ext = os.path.splitext(f)
        if ext not in ("", ".mat"):
            raise R.ResultsError("MATLAB error in the .mod file: smoother2hisvtval: if outfile has an extension, it must be .mat")
        if d:
            raise Unsupported("smoother2histval: an outfile with a folder (Dynare writes [dir fname] without a separator)", "file")
        _ds.HOST = self
        self.dseries_method(_ds.DSeries(data, invars, Date(1, 1)), "save", [base])
