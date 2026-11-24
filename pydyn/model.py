# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Model object built from the Dynare preprocessor's JSON output."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import marshal
import os
import pickle
import re
import warnings
from pathlib import Path

import numpy as np

from . import dates as _dates
from . import external, matlab
from .expr import FAST_ERRORS, FAST_NAMESPACE, NAMESPACE, Translator, safe_name, temp_name
from .options import ACCUMULATED_OPTIONS

#: "fast": generated model functions run on Python floats with a numpy fallback; "numpy": numpy
#: scalars only (the original behaviour). Environment variable PYDYN_CODEGEN overrides it.
CODEGEN = os.environ.get("PYDYN_CODEGEN", "fast")
#: evaluate each exp(y[i]) / log(y[i]) once per generated function
HOIST = os.environ.get("PYDYN_HOIST", "1") != "0"
#: keep compiled model functions on disk (see Model); PYDYN_CACHE=0 turns it off
CACHE = os.environ.get("PYDYN_CACHE", "1") != "0"
#: part of the cache key: bump the number when the generated code changes
_CACHE_TAG = b"pydyn-codegen-1:" + importlib.util.MAGIC_NUMBER


_CACHE_MAX_FILES = 200


def cache_dir() -> Path:
    """Folder for compiled model functions: ``$PYDYN_CACHE_DIR``, else ``~/.cache/pydyn``."""
    return Path(os.environ.get("PYDYN_CACHE_DIR") or (Path.home() / ".cache" / "pydyn"))


def _pairs(items, key_a, key_b):
    return [(d[key_a], d[key_b]) for d in items]


class ModelError(Exception):
    pass


class Model:
    """A Dynare model: symbols, calibration, and compiled residual/derivative functions.

    Variables are held in declaration order. Dynamic functions take a stacked vector
    ``yy = [y(-1); y; y(+1)]`` of length ``3 * n_endo``.
    """

    def __init__(self, json_dir: str | Path, functions: dict | None = None, steady_state_fn=None,
                 cache: bool | None = None, run_statements: bool = True):
        """``functions`` maps names of external functions used in the model or in the
        ``steady_state_model`` block to Python callables. ``steady_state_fn`` is the Python
        counterpart of a Dynare ``_steadystate.m`` file: ``fn(model, params, exo) -> ys`` or
        ``(ys, params)``; it may raise ``SteadyStateError``. ``cache`` (default: on unless the
        environment variable PYDYN_CACHE is 0) keeps the compiled model functions in
        the user cache folder (``~/.cache/pydyn`` or ``$PYDYN_CACHE_DIR``), keyed by a hash of the JSON files,
        so that loading the same model again skips translation and compilation.

        ``run_statements=False`` (used by ``pydyn.results.dynare``) builds the model and its empty
        state but does not run the statements of the .mod file: the caller runs them, one at a time,
        through ``_exec_statement`` (see ``pydyn.results``)."""
        json_dir = Path(json_dir)
        self.functions = dict(functions or {})
        self.steady_state_fn = steady_state_fn
        self.json_dir = json_dir
        # MATLAB function files next to the .mod file that the steady_state_model block calls are
        # interpreted automatically; functions passed by the caller take precedence
        self.matlab_functions = {}
        ssm = json_dir / "steady_state_model.json"
        if ssm.exists() and len(json_dir.parents) > 2:
            used = set(re.findall(r"[A-Za-z_]\w*", ssm.read_text()))
            self.matlab_functions = matlab.helper_functions(json_dir.parents[2], used, set(self.functions) | set(NAMESPACE))
            self.functions.update(self.matlab_functions)
        text = (json_dir / "modfile.json").read_text()
        try:
            mf = json.loads(text)
        except json.JSONDecodeError:
            # Dynare 7.2 writes a misplaced brace in matched_irfs blocks: "period2": }5, ...
            # ... and unquoted negative bounds in estimated_params_bounds: "lower_bound": (-1)
            text = re.sub(r'("period2":\s*)\}(\d+)', r"\1\2", text)
            text = re.sub(r'"(lower_bound|upper_bound)":\s*\((-[^)"]*)\)', r'"\1": "\2"', text)
            # ... a backslash that is not a JSON escape (equation tag [eq='\#1']) ...
            text = re.sub(r'\\(.)', lambda mo: mo.group(0) if mo.group(1) in '"\\/bfnrtu' else "\\\\" + mo.group(1), text)
            # ... a doubled closing quote after period2 in shocks(learnt_in/dates) paths ...
            text = re.sub(r'("period2"\s*:\s*"[^"\n]*")"', r"\1", text)
            # ... a missing comma in model_comparison lists, a missing quote in matched_irfs_weights ...
            text = re.sub(r'("name":\s*"[^"\n]*")("prior":)', r"\1, \2", text)
            text = re.sub(r'("exo\d":\s*)([A-Za-z_]\w*")', r'\1"\2', text)
            # ... an unquoted MATLAB variable as the colormap option of plot_shock_decomposition ...
            text = re.sub(r'("[\w.]*colormap":\s*)([A-Za-z_]\w*)(?=\s*[,}])', r'\1"\2"', text)
            # ... an unquoted Inf as a numeric option value (estimation(dsge_var=Inf)) ...
            text = re.sub(r'(":\s*)(-?Inf)(?=\s*[,}\]])', r'\1"\2"', text)
            # ... evaluate_planner_objective(options) written with its options before the statement ...
            text = re.sub(r'\n\s*,\s*"options":\s*(\{[^{}]*\})\s*\{"statementName":\s*"evaluate_planner_objective"\}',
                          r'\n{"statementName": "evaluate_planner_objective", "options": \1}', text)
            # ... and numbers written as .8 or 006 in option lists
            text = re.sub(r'([:\[,]\s*)\.(\d)', r"\g<1>0.\2", text)
            text = re.sub(r'([:\[,]\s*)0+(\d+)(?=\s*[,\]}])', r"\1\2", text)
            mf = json.loads(text)
        self._cache = CACHE if cache is None else bool(cache)
        self.raw = mf

        self.endo_names = [d["name"] for d in mf["endogenous"]]
        self.exo_names = [d["name"] for d in mf["exogenous"]]
        self.param_names = [d["name"] for d in mf["parameters"]]
        self.n_endo, self.n_exo = len(self.endo_names), len(self.exo_names)
        self._ie = {v: i for i, v in enumerate(self.endo_names)}
        self._ix = {v: i for i, v in enumerate(self.exo_names)}
        self._ip = {v: i for i, v in enumerate(self.param_names)}
        # varexo_det: appended to the exogenous vector of the generated functions (x[n_exo + i]); the
        # stochastic solvers see them fixed at their steady-state values (see pydyn.varexo_det)
        self.exo_det_names = [d["name"] for d in mf.get("exogenous_deterministic", [])]
        self.n_exo_det = len(self.exo_det_names)
        self._ixd = {v: i for i, v in enumerate(self.exo_det_names)}

        info = mf["model_info"]
        lli = np.array(info["lead_lag_incidence"], dtype=int).reshape(self.n_endo, -1)
        # the preprocessor omits the lag (lead) column for models without lags (leads)
        cols = [lli[:, 0] if info["maximum_endo_lag"] > 0 else np.zeros(self.n_endo, int)]
        cols.append(lli[:, 1 if info["maximum_endo_lag"] > 0 else 0])
        cols.append(lli[:, -1] if info["maximum_endo_lead"] > 0 else np.zeros(self.n_endo, int))
        self.lead_lag_incidence = np.column_stack(cols)
        if info["maximum_endo_lag"] > 1 or info["maximum_endo_lead"] > 1:
            raise NotImplementedError("expected the preprocessor to reduce the model to one lead/lag")
        self.info = info

        self.varobs = list(mf.get("varobs", []))
        self.statements = mf.get("statements", [])
        self._deriv_keys = [k for k in mf if k.startswith("dynamic_g") and k.endswith("_sparse_indices")]
        self.unhandled_statements: list[str] = []

        self._compile()
        self._load_steady_state_model()
        mod_dir, name = json_dir.parents[2], json_dir.parents[1].name
        self.steady_state_file = mod_dir / f"{name}_steadystate.m"
        if not self.steady_state_file.exists():
            self.steady_state_file = None
        self._init_state()
        if run_statements:
            self._run_statements()

    # ------------------------------------------------------------------ compilation
    def _tr_dynamic(self):
        n = self.n_endo

        def resolve(name, shift):
            if name in self._ie:
                if shift not in (-1, 0, 1):
                    raise ValueError(f"{name}({shift}) outside one lead/lag")
                return f"y[{(shift + 1) * n + self._ie[name]}]"
            if name in self._ix:
                if shift != 0:
                    raise ValueError("lagged exogenous should have been removed by the preprocessor")
                return f"x[{self._ix[name]}]"
            if name in self._ixd:
                if shift != 0:
                    raise ValueError("lagged exogenous should have been removed by the preprocessor")
                return f"x[{self.n_exo + self._ixd[name]}]"
            if name in self._ip:
                return f"params[{self._ip[name]}]"
            return None

        return Translator(resolve, steady=self._steady_ref("ss[{}]"),
                          shiftable=set(self._ie) | set(self._ix) | set(self._ixd))

    def _steady_ref(self, fmt):
        def steady(v):
            if v in self._ie:
                return fmt.format(self._ie[v])
            if v in self._ip:                       # steady_state(parameter) is the parameter
                return f"params[{self._ip[v]}]"
            raise ModelError(f"steady_state({v}): unknown symbol")
        return steady

    def _tr_static(self):
        def resolve(name, shift):
            if name in self._ie:
                return f"y[{self._ie[name]}]"
            if name in self._ix:
                return f"x[{self._ix[name]}]"
            if name in self._ixd:
                return f"x[{self.n_exo + self._ixd[name]}]"
            if name in self._ip:
                return f"params[{self._ip[name]}]"
            return None

        # in the static model STEADY_STATE(v) is v itself
        return Translator(resolve, steady=self._steady_ref("y[{}]"),
                          shiftable=set(self._ie) | set(self._ix) | set(self._ixd))

    @staticmethod
    def _locals(block):
        out = []
        for d in block.get("model_local_variables", []):
            out.append((d.get("variable", d.get("name")), d.get("value")))
        return out

    _HOIST = re.compile(r"(?<![\w.])(exp|log)\(y\[(\d+)\]\)")
    _CHUNK = 500

    def _build(self, tr, block, name):
        """Source of the residual and derivative functions of a (static|dynamic) block, and the
        positions of the values they return.

        Each function takes ``(y, x, params, ss)`` and returns a flat list: the residuals, or the
        structural non-zeros of a derivative in the order of the preprocessor's JSON. The same
        source runs with Python floats (expr.FAST_NAMESPACE) and with numpy (expr.NAMESPACE)."""
        tr.temps = {d["temporary_term"] for k, v in block.items() if k.startswith("temporary_terms_") for d in v}
        tr.temps |= external.term_names(block)
        assign = [f"    {safe_name(k)} = {tr(v)}" for k, v in self._locals(block)]

        def temps(key):
            lines = [(d["temporary_term"], f"    {temp_name(d['temporary_term'])} = {tr(d['value'])}")
                     for d in block.get(key, [])]
            ext = block.get("external_functions_" + key, [])
            if not ext:
                return [ln for _, ln in lines]
            return external.order_lines(external.section_lines(ext, tr, self._ext_names), lines)

        def function(fn, body, values):
            hoisted = {}

            def hoist(mo):          # exp(y[i]) and log(y[i]) recur in most equations: evaluate once
                return hoisted.setdefault(mo.group(0), f"_hx_{mo.group(1)}{mo.group(2)}")
            if HOIST:
                body = [self._HOIST.sub(hoist, ln) for ln in body]
                values = [self._HOIST.sub(hoist, v) for v in values]
            out = [f"def {name}_{fn}(y, x, params, ss):"] + [f"    {v} = {k}" for k, v in hoisted.items()] + body
            if len(values) <= self._CHUNK:
                out.append("    return [" + ", ".join(values) + "]\n")
            else:                   # keep list displays (and the evaluation stack) of moderate size
                out.append("    v = [" + ", ".join(values[:self._CHUNK]) + "]")
                for i in range(self._CHUNK, len(values), self._CHUNK):
                    out.append("    v += [" + ", ".join(values[i:i + self._CHUNK]) + "]")
                out.append("    return v\n")
            return out

        meta = {}
        body = assign + temps("temporary_terms_")
        lines = function("resid", body, [f"({tr(d['residual']['lhs'])}) - ({tr(d['residual'].get('rhs', '0'))})"
                                         for d in block["residuals"]])
        jac = block["jacobian"]
        body = body + temps("temporary_terms_jacobian")
        lines += function("jac", body, [tr(e["val"]) for e in jac["entries"]])
        meta[f"{name}_jac"] = (tuple(jac["dims"]), np.array([e["indices"][0] - 1 for e in jac["entries"]], dtype=np.intp),
                               np.array([e["indices"][1] - 1 for e in jac["entries"]], dtype=np.intp))
        for fn, key, ttkey, order in (("hess", "hessian", "temporary_terms_hessian", 2),
                                      ("third", "third_derivative", "temporary_terms_third_derivative", 3)):
            body = body + temps(ttkey)
            entries = block.get(key, {}).get("entries", [])
            lines += function(fn, body, [tr(e["val"]) for e in entries])
            meta[f"{name}_{fn}"] = (np.array([e["indices"][0] - 1 for e in entries], dtype=np.int32),
                                    np.array([[i - 1 for i in e["indices"][1:]] for e in entries],
                                             dtype=np.int32).reshape(len(entries), order))
        return "\n".join(lines), meta

    def _compile(self):
        dyn_b = (self.json_dir / "dynamic.json").read_bytes()
        sta_b = (self.json_dir / "static.json").read_bytes()
        cached, cache_file = None, None
        if self._cache:
            h = hashlib.sha256()
            for part in (_CACHE_TAG, repr((self.endo_names, self.exo_names, self.param_names, HOIST,
                                           self.exo_det_names)).encode(),
                         dyn_b, sta_b):
                h.update(len(part).to_bytes(8, "little"))
                h.update(part)
            # The cache holds executable code, so it lives in the user's own cache folder and never
            # next to the model: a folder received from someone else must not be able to supply it.
            cache_file = cache_dir() / f"{h.hexdigest()[:40]}.pkl"
            try:
                with open(cache_file, "rb") as fh:
                    cached = pickle.load(fh)  # noqa: S301 - written by this package in the user's cache folder
                code = marshal.loads(cached["code"])
            except Exception:  # noqa: BLE001 - missing or unreadable cache: regenerate
                cached = None
        if cached is None:
            dyn_t, sta_t = dyn_b.decode(), sta_b.decode()
            self._ext_names = external.function_names([dyn_t, sta_t])
            if self._ext_names:
                dyn_t, sta_t = (external.substitute(t, self._ext_names) for t in (dyn_t, sta_t))
            try:
                dyn, sta = json.loads(dyn_t)["dynamic_model"], json.loads(sta_t)["static_model"]
            except json.JSONDecodeError:
                dyn = json.loads(external.repair_json(dyn_t))["dynamic_model"]
                sta = json.loads(external.repair_json(sta_t))["static_model"]
            src_d, meta = self._build(self._tr_dynamic(), dyn, "dynamic")
            src_s, meta_s = self._build(self._tr_static(), sta, "static")
            meta.update(meta_s)
            src = src_d + "\n" + src_s
            code = compile(src, f"<pydyn:{self.json_dir}>", "exec")
            meta["external_functions"] = self._ext_names
            cached = {"source": src, "meta": meta}
            if cache_file is not None:
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    old = sorted(cache_file.parent.glob("*.pkl"), key=lambda f: f.stat().st_mtime)
                    for stale in old[:max(0, len(old) - _CACHE_MAX_FILES + 1)]:      # keep the folder bounded
                        stale.unlink()
                    tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
                    with open(tmp, "wb") as fh:
                        pickle.dump({**cached, "code": marshal.dumps(code)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp, cache_file)
                except OSError:
                    pass                                             # read-only location: work without a cache
        self.source = cached["source"]
        self._meta = meta = cached["meta"]
        self._ext_names = meta.get("external_functions", [])
        self._exec(code)
        self.jac_dims, self.jac_rows, self.jac_cols = meta["dynamic_jac"]
        # with varexo_det: full derivatives (columns [y(-1), y, y(+1), x, x_det]) are kept in
        # _det_view; the attributes and methods below give the stochastic view without the x_det columns
        self._det_view = None
        if self.n_exo_det:
            nx = 3 * self.n_endo + self.n_exo
            full = {"dynamic_jac": (self.jac_dims, self.jac_rows, self.jac_cols)}
            keep = {"dynamic_jac": self.jac_cols < nx}
            for fn in ("dynamic_hess", "dynamic_third"):
                rows, cols = meta[fn]
                full[fn] = (rows, cols)
                keep[fn] = np.all(cols < nx, axis=1) if len(rows) else np.zeros(0, bool)
            self._det_view = {"full": full, "keep": keep}
            k = keep["dynamic_jac"]
            self.jac_dims = (self.jac_dims[0], nx)
            self.jac_rows, self.jac_cols = self.jac_rows[k], self.jac_cols[k]
        self._jac_entries = list(zip(self.jac_rows.tolist(), self.jac_cols.tolist()))
        # highest derivative order the preprocessor computed (a linear model has an empty Hessian)
        self.max_deriv_order = 1 + sum(f"dynamic_g{o}_sparse_indices" in self._deriv_keys for o in (2, 3))
        if self.jac_dims[1] != 3 * self.n_endo + self.n_exo:
            raise NotImplementedError(f"unexpected dynamic Jacobian layout {self.jac_dims}")

    def _exec(self, code):
        """Bind the generated functions twice: with Python floats (fast) and with numpy scalars."""
        self._fn, self._ff = dict(NAMESPACE), dict(FAST_NAMESPACE)
        for ns in (self._fn, self._ff):
            ns.update(self.functions)
            ns["np"] = np
            self._bind_external(ns)
            exec(code, ns)  # noqa: S102 - generated from the model

    def _bind_external(self, ns):
        """Names the generated code uses for external functions (see pydyn.external)."""
        if self._ext_names:
            table = self.__dict__.get("_ext_table")
            if table is None:
                mod_dir = self.json_dir.parents[2] if len(self.json_dir.parents) > 2 else None
                table = self._ext_table = external.ExternalFunctions(self.functions, mod_dir)
            external.bind(ns, self._ext_names, table)

    # ------------------------------------------------------------------ statements
    def _param_translator(self, extra=None):
        def resolve(name, shift):
            if name in self._ip:
                return f"params[{self._ip[name]}]"
            return None
        return Translator(resolve)

    _LEADING_ZERO = re.compile(r"(?<![\w.])0\d")

    def eval_expr(self, s: str, local=None) -> float:
        s = str(s)
        if self._LEADING_ZERO.search(s):
            s = re.sub(r"(?<![\w.])0+(\d+\.?\d*)", r"\1", s)
        ns = self.__dict__.get("_eval_ns")
        if ns is None:              # built once: copying NAMESPACE for every expression is the slow part
            ns = self._eval_ns = dict(NAMESPACE)
            ns.update(self.functions)
            self._eval_tr = self._param_translator()
        loc = {safe_name(k): v for k, v in getattr(self, "workspace", {}).items()}
        loc["params"] = self.params
        if local:
            loc.update(local)
        return float(eval(self._eval_tr(s), ns, loc))  # noqa: S307 - model expression

    def _load_steady_state_model(self):
        p = self.json_dir / "steady_state_model.json"
        self.steady_state_model = None
        if p.exists():
            self.steady_state_model = _pairs(json.loads(p.read_text())["steady_state_model"], "lhs", "rhs")

    _IDENT = re.compile(r"[A-Za-z_]\w*")

    def _eval_with_vars(self, s: str) -> float:
        """Value of an initval/endval right-hand side.  As in Dynare, a reference to an endogenous or
        exogenous variable reads the current content of oo_.steady_state / oo_.exo_steady_state:
        the value given earlier (in this block or a previous initval/endval block), the steady
        state computed by an earlier ``steady``, or 0 if the variable was never set."""
        s = str(s)
        local = {}
        for name in set(self._IDENT.findall(s)):
            if name in self._ie or name in self._ix:
                local[name] = self._matlab.oo_value(name)
        return self._value(s, local)

    def _set_oo_auxiliary(self):
        """The driver Dynare writes calls set_auxiliary_variables after an initval or endval block:
        the auxiliary variables in oo_.steady_state follow the values just set (e.g. AUX_EXO_LEAD = x)."""
        if not self.raw.get("aux_vars"):
            return
        from .steady import fill_auxiliary
        y = np.array([self._oo.get(v, 0.0) for v in self.endo_names], float)
        x = np.array([self._oo.get(v, 0.0) for v in self.exo_names], float)
        try:
            with np.errstate(all="ignore"):
                y = fill_auxiliary(self, y, x, self.params)
        except Exception:  # noqa: BLE001 - parameters not set yet, functions failing: leave them
            return
        for name in self.aux_names:
            self._oo[name] = float(y[self._ie[name]])

    def _oo_assign(self, values):
        """MATLAB code assigned oo_.steady_state / oo_.exo_steady_state: later steady-state
        computations start from (and use the exogenous values of) the new content, as in Dynare."""
        (self.endval if self._endval_seen else self.initval).update(values)

    def _init_state(self):
        """The state the statements of the .mod file fill (parameters NaN, no shocks, no options)."""
        self.params = np.full(len(self.param_names), np.nan)
        self.workspace: dict[str, float] = {}      # scalar MATLAB variables of the .mod file
        self.initval: dict[str, float] = {}
        self.endval: dict[str, float] = {}
        self.det_shocks: list = []                 # (exo index, first period, last period, value), 1-based
        self.det_shocks_exo_det: list = []         # the same for varexo_det variables (index in exo_det_names)
        self.Sigma_e = np.zeros((self.n_exo, self.n_exo))
        #: Dynare's M_.Correlation_matrix: set by 'corr' statements of shocks blocks only (not by
        #: 'var e, u = ...'); estimation rebuilds Sigma_e from it when standard deviations are estimated
        self.Correlation_matrix = np.eye(self.n_exo)
        self.measurement_errors: dict[str, float] = {}     # M_.H diagonal: variances by observed variable
        #: M_.H off-diagonal entries and M_.Correlation_matrix_ME (set by 'corr' statements only), keyed by the pair of
        #: observed variables in sorted-name order
        self.me_cov: dict[tuple, float] = {}
        self.me_corr: dict[tuple, float] = {}
        self.occbin_constraints = []
        self.options: dict[str, dict] = {}
        #: (statement, its own options) in file order, for the statements whose options accumulate in
        #: Dynare's options_ (see pydyn.options.accumulated)
        self.options_sequence: list[tuple[str, dict]] = []
        self._options_parsed: dict[str, dict] = {}
        self.matlab_options = matlab.Struct()      # options_.x = ... assignments (see pydyn.options.HONOURED_MATLAB_OPTIONS)
        self.matlab_options_log: list = []         # (index of the last Dynare statement run, options_ field, value) per assignment
        self.matlab_skipped: list = []             # MATLAB statements that were not executed: (text, kind, reason)
        self._n_surprise = 0
        self._oo: dict[str, float] = {}            # what Dynare holds in oo_.steady_state / oo_.exo_steady_state
        # MATLAB code that writes oo_.steady_state / oo_.exo_steady_state (or makes it unknown):
        # (index of the last Dynare statement run before it, "steady_state" | "exo_steady_state" | "unknown", values)
        self._oo_events: list = []
        self._stmt_values: dict = {}               # id(statement) -> values an initval/endval block left in oo_
        self._endval_seen = False
        #: set when MATLAB code read a numerically computed steady state: the calibration from that
        #: point on is only as accurate as the steady-state solver (Dynare's tolerance)
        self.calibration_tolerance = None

    def _run_statements(self):
        """Apply the statements of the .mod file in order.  Native MATLAB lines are interpreted by
        ``pydyn.matlab`` together with the Dynare statements, so that MATLAB control flow around
        parameter assignments (``if calib == 1 ... end``) behaves as in Dynare."""
        # every file goes through the session, with or without MATLAB lines: it also follows which
        # parameters and steady-state values an earlier steady/stoch_simul has changed in Dynare
        self._matlab = matlab.ModSession(self, ModelError)
        self._matlab.run(self.statements)
        if self._n_surprise > 1:
            warnings.warn("several shocks(surprise) blocks: they are merged; Dynare's 'overwrite' "
                          "option is not visible in the preprocessor output")
        if self.unhandled_statements:
            warnings.warn(f"statements not handled: {sorted(set(self.unhandled_statements))}")

    def _value(self, s, local=None) -> float:
        """Value of an expression in a Dynare statement (Dynare passes these to MATLAB).  The fast
        path is plain Python on numpy scalars; expressions that index a MATLAB matrix, call a MATLAB
        function defined next to the .mod file, or whose result is not finite (1/0, a NaN operand,
        a complex result) are evaluated by the MATLAB interpreter, which has MATLAB's semantics."""
        try:
            v = self.eval_expr(s, local)
            if v - v == 0:                          # finite
                return v
        except (NameError, TypeError, SyntaxError, ArithmeticError, ValueError):
            pass
        return self._matlab.eval_value(s, local)

    def _exec_statement(self, st):
        """One Dynare statement of the .mod file."""
        kind = st["statementName"]
        if kind == "param_init":
            self.params[self._ip[st["name"]]] = self._value(st["value"])
        elif kind == "initval" or (kind == "endval" and "learnt_in" not in st):
            target = self.initval if kind == "initval" else self.endval
            self._endval_seen = self._endval_seen or kind == "endval"
            stored = self._stmt_values[id(st)] = {}
            for d in st["vals"]:
                target[d["name"]] = self._oo[d["name"]] = stored[d["name"]] = self._eval_with_vars(d["value"])
                self._matlab.oo_written.add(d["name"])
            self._set_oo_auxiliary()
        elif kind == "endval":
            pass                                   # endval(learnt_in=...): read by perfect_foresight
        elif kind == "load_params_and_steady_state":
            stored = self._stmt_values[id(st)] = {}
            for d in st.get("values", []):
                if d["name"] in self._ip:
                    self.params[self._ip[d["name"]]] = self.eval_expr(d["value"])
                else:
                    self.initval[d["name"]] = self._oo[d["name"]] = stored[d["name"]] = self.eval_expr(d["value"])
                    self._matlab.oo_written.add(d["name"])
        elif kind == "shocks" and st.get("surprise"):
            self._n_surprise += 1
            self.options.setdefault("shock_paths", []).append(st)
        elif kind == "shocks" and "learnt_in" in st:
            pass                                   # shocks(learnt_in=...): news for perfect_foresight, not Sigma_e
        elif kind == "shocks":
            self._shocks(st)
        elif kind == "occbin_constraints":
            self.occbin_constraints = st["constraints"]
        elif kind == "homotopy" and st.get("values") and all(
                d["name"] in self._ip or d["name"] in self._ix for d in st["values"]):
            self._homotopy = st["values"]          # applied by the next steady(homotopy_mode=...)
        elif kind in ("steady", "check", "occbin_setup", "occbin_graph", "rplot"):
            if kind in ACCUMULATED_OPTIONS:
                self._accumulate_options(kind, st.get("options", {}))
            else:
                self.options.setdefault(kind, st.get("options", {}))
            if kind == "steady" and st.get("options", {}).get("homotopy_mode") and getattr(self, "_homotopy", None):
                # Dynare's steady moves these parameters and exogenous variables to their final values
                # (it stops with an error if the homotopy fails); later statements see the final values
                for d in self._homotopy:
                    val = self.eval_expr(d["final_value"])
                    if d["name"] in self._ip:
                        self.params[self._ip[d["name"]]] = val
                    else:
                        self.initval[d["name"]] = self._oo[d["name"]] = val
        elif kind in ("stoch_simul", "estimation", "occbin_solver", "estimated_params",
                      "estimated_params_init", "varobs", "perfect_foresight_setup", "perfect_foresight_solver", "calib_smoother",
                          "shock_decomposition", "initial_condition_decomposition", "shock_groups",
                          "observation_trends", "estimated_params_bounds",
                          "plot_shock_decomposition", "realtime_shock_decomposition",
                          "squeeze_shock_decomposition", "init2shocks", "filter_initial_state",
                          "perfect_foresight_with_expectation_errors_setup",
                          "perfect_foresight_with_expectation_errors_solver", "histval", "mshocks",
                          "initval_file", "histval_file", "perfect_foresight_controlled_paths"):
            if kind in ACCUMULATED_OPTIONS:
                self._accumulate_options(kind, st.get("options", {}))
            else:
                self.options[kind] = st.get("options", {}) if kind != "estimated_params" else st
            if kind == "mshocks" and "learnt_in" not in st:
                self._det_values(st)
            elif kind == "histval":
                self._histval_values(st)
            elif kind == "perfect_foresight_controlled_paths":
                stored = self._stmt_values[id(st)] = {}
                for i, path in enumerate(st.get("paths", [])):
                    for k, v in enumerate(path.get("values", [])):
                        stored[(i, k)] = self._value(v["value"])
        elif kind in ("ramsey_model", "ramsey_policy", "discretionary_policy", "osr", "evaluate_planner_objective",
                      "planner_objective", "osr_params", "optim_weights", "osr_params_bounds"):
            # optimal policy: read by pydyn.optimal_policy (ramsey_policy and discretionary_policy also act
            # as stoch_simul, whose options they take)
            self.options[kind] = st.get("options", {}) if kind not in ("osr_params", "optim_weights", "osr_params_bounds") else st
            if kind == "osr_params_bounds":
                self.options[kind] = {"bounds": [(d["parameter"], self._value(d["bounds"][0]), self._value(d["bounds"][1]))
                                                 for d in st.get("bounds", [])]}
            elif kind == "optim_weights":
                self.options[kind] = {"weights": [(d.get("name", d.get("name1")), d.get("name2"), self._value(d["value"]))
                                                  for d in st.get("weights", [])]}
            if kind in ("ramsey_model", "ramsey_policy", "discretionary_policy"):
                ins = st.get("options", {}).get("instruments") or {}
                names = list(ins.get("symbol_list", []) if isinstance(ins, dict) else ins)
                if names or "instruments" not in self.matlab_options:   # options_.instruments, read by steady-state files
                    self.matlab_options["instruments"] = matlab.Cell(names, (len(names), 1) if names else (0, 0))
            if kind in ("ramsey_policy", "discretionary_policy", "osr"):
                # these commands write their options into options_ (kept by later commands) and run stoch_simul
                self._accumulate_options("stoch_simul", {k: v for k, v in st.get("options", {}).items()
                                                         if k not in ("instruments", "osr.opt_algo", "planner_discount",
                                                                      "analytic_derivation", "optim_opt", "osr.maxit",
                                                                      "osr.tolf", "discretionary_tol", "dp.maxit",
                                                                      "planner_discount_latex_name")})
                self.options.setdefault(kind + "_all", []).append(st)
        elif kind == "conditional_forecast_paths":
            # values evaluated where the block stands, as constrained_paths_ is in Dynare (read by pydyn.forecast)
            stored = self._stmt_values[id(st)] = {}
            for i, path in enumerate(st.get("paths", [])):
                for k, v in enumerate(path.get("values", [])):
                    stored[(i, k)] = self._value(v["value"])
        elif kind in ("forecast", "conditional_forecast", "plot_conditional_forecast"):
            pass                                   # read by pydyn.forecast.run_forecasts
        elif kind in ("native", "verbatim"):
            pass                                   # run by pydyn.matlab.ModSession
        elif kind == "heteroskedastic_shocks":
            pass                                   # read by pydyn.heteroskedastic (heteroskedastic_filter)
        elif kind == "estimation_data":
            self.options[kind] = st.get("options", {})          # the data command: read by pydyn.dates / Estimation
        elif kind in ("write_latex_dynamic_model", "write_latex_static_model",
                      "write_latex_original_model", "write_latex_steady_state_model",
                      "collect_latex_files", "write_latex_definitions", "write_latex_parameter_table",
                      "write_latex_prior_table"):
            # no .tex file is written: say so once instead of ignoring the statement silently
            from .options import warn_once
            warn_once(self, f"{kind}: LaTeX output is not implemented in pydyn; the statement writes no file "
                            "(the model, parameters and priors are available as Python objects)", stacklevel=2)
        else:
            self.unhandled_statements.append(kind)

    def _accumulate_options(self, kind, opts):
        """Dynare writes the options of each statement into options_, where they stay until a later
        statement sets them again: model.options[kind] holds the options of all the statements of
        that kind so far (later ones override), options_sequence the statements in file order."""
        self.options[kind] = {**self.options.get(kind, {}), **opts}
        self.options_sequence.append((kind, dict(opts)))
        self._options_parsed[kind] = dict(self.options[kind])

    def _det_value(self, text, n):
        """Value of a deterministic shock over ``n`` periods: a number, or (``values (v);``) a MATLAB
        vector with one value per period, as Dynare's make_ex_ assigns it."""
        try:
            return self._value(text)
        except ModelError:
            v = self._matlab.eval_value(text, vector=True)
            if v.size == 1:
                return float(v[0])
            if v.size != n:
                raise ModelError(f"deterministic shock value {text!r} has {v.size} elements for {n} periods") from None
            return v

    def _det_values(self, st):
        """The values of the periods/values lines of a shocks or mshocks block, evaluated where the
        block stands in the file (stored for perfect_foresight)."""
        out = self._stmt_values[id(st)] = {}
        for d in st.get("deterministic_shocks", []):
            vals = out.setdefault(d["var"], [])
            for v in d["values"]:
                try:
                    n = int(v["period2"]) - int(v["period1"]) + 1
                except (TypeError, ValueError):            # dates (pydyn.dates)
                    try:
                        n = (_dates.Date.parse(v["period2"]) - _dates.Date.parse(v["period1"])) + 1
                    except (ValueError, TypeError):
                        n = 1
                vals.append(self._det_value(v["value"], n))

    def _histval_values(self, st):
        """histval values evaluated where the block stands (perfect_foresight uses them); a value
        pydyn cannot compute is left to perfect_foresight, which reports it."""
        out = self._stmt_values[id(st)] = {}
        for d in st.get("vals", []):
            try:
                out[(d["name"], int(d["lag"]))] = self._value(d["value"])
            except (ModelError, ValueError, TypeError, KeyError):
                pass

    def _shocks(self, st):
        """One shocks block, applied in Dynare's order: reset, variances, covariances, correlations."""
        ix, S = self._ix, self.Sigma_e
        if st.get("overwrite"):
            S[:] = 0
            self.Correlation_matrix = np.eye(self.n_exo)
            self.measurement_errors.clear()
            self.me_cov.clear()
            self.me_corr.clear()
        for key, power in (("variance", 1), ("stderr", 2)):
            for d in st.get(key, []):
                val = self._value(d[key]) ** power
                if d["name"] in ix:
                    S[ix[d["name"]], ix[d["name"]]] = val
                else:                               # measurement error on an observed variable
                    self.measurement_errors[d["name"]] = val
        for key in ("covariance", "correlation"):
            for d in st.get(key, []):
                if d["name"] not in ix and d["name2"] not in ix:         # measurement errors: M_.H(i,j), Correlation_matrix_ME
                    a, b = d["name"], d["name2"]
                    pair = (a, b) if a <= b else (b, a)
                    val = self._value(d[key])
                    if key == "correlation":
                        self.me_corr[pair] = val
                        val *= np.sqrt(self.measurement_errors.get(a, 0.0) * self.measurement_errors.get(b, 0.0))
                    self.me_cov[pair] = val
                    continue
                if d["name"] not in ix or d["name2"] not in ix:
                    self.unhandled_statements.append("shocks: covariance between a shock and a measurement error")
                    continue
                i, j = ix[d["name"]], ix[d["name2"]]
                val = self._value(d[key])
                if key == "correlation":
                    self.Correlation_matrix[i, j] = self.Correlation_matrix[j, i] = val
                    val *= np.sqrt(S[i, i] * S[j, j])
                S[i, j] = S[j, i] = val
        self._det_values(st)
        for d in st.get("deterministic_shocks", []):
            for v, val in zip(d["values"], self._stmt_values[id(st)][d["var"]]):
                p1, p2 = (_dates.Date.parse(x) if _dates.is_date_value(x) else int(x) for x in (v["period1"], v["period2"]))
                if d["var"] in self._ixd:        # shocks on varexo_det (M_.det_shocks with exo_det = true)
                    self.det_shocks_exo_det.append((self._ixd[d["var"]], p1, p2, val))
                    continue
                self.det_shocks.append((ix[d["var"]], p1, p2, val))
        for key in ("skewness", "coskewness"):
            if st.get(key):
                self.unhandled_statements.append(f"shocks: {key}")

    # ------------------------------------------------------------------ measurement errors
    def me_H(self, varobs=None) -> np.ndarray:
        """``M_.H`` (calibrated measurement-error covariance matrix) in ``varobs`` order."""
        varobs = list(self.varobs if varobs is None else varobs)
        extra = (set(self.measurement_errors) | {x for p in self.me_cov for x in p}) - set(varobs)
        if extra:
            raise ModelError(f"measurement error declared on non-observed variable(s) {sorted(extra)}")
        k = {v: i for i, v in enumerate(varobs)}
        H = np.zeros((len(varobs), len(varobs)))
        for v, x in self.measurement_errors.items():
            H[k[v], k[v]] = x
        for (a, b), x in self.me_cov.items():
            H[k[a], k[b]] = H[k[b], k[a]] = x
        return H

    def me_corr_matrix(self, varobs=None) -> np.ndarray:
        """``M_.Correlation_matrix_ME`` in ``varobs`` order (identity but for the 'corr' statements)."""
        varobs = list(self.varobs if varobs is None else varobs)
        k = {v: i for i, v in enumerate(varobs)}
        C = np.eye(len(varobs))
        for (a, b), x in self.me_corr.items():
            C[k[a], k[b]] = C[k[b], k[a]] = x
        return C

    def set_me_H(self, H, varobs=None, C=None) -> None:
        """Replace the calibrated measurement-error covariances by the matrix ``H`` (``varobs`` order)."""
        varobs = list(self.varobs if varobs is None else varobs)
        H = np.asarray(H, float)
        self.measurement_errors = {v: float(H[i, i]) for i, v in enumerate(varobs) if H[i, i] != 0}
        self.me_cov = {}
        for i in range(len(varobs)):
            for j in range(i):
                if H[i, j] != 0:
                    a, b = varobs[i], varobs[j]
                    self.me_cov[(a, b) if a <= b else (b, a)] = float(H[i, j])
        if C is not None:
            C = np.asarray(C, float)
            self.me_corr = {}
            for i in range(len(varobs)):
                for j in range(i):
                    if C[i, j] != 0:
                        a, b = varobs[i], varobs[j]
                        self.me_corr[(a, b) if a <= b else (b, a)] = float(C[i, j])

    # ------------------------------------------------------------------ parameters
    def get_param(self, name: str) -> float:
        return float(self.params[self._ip[name]])

    def set_param(self, name: str, value: float) -> None:
        """Counterpart of Dynare's ``set_param_value``. Takes effect at the next solve."""
        self.params[self._ip[name]] = float(value)

    def set_params(self, values: dict | None = None, **kw) -> None:
        for k, v in {**(values or {}), **kw}.items():
            self.set_param(k, v)

    @property
    def linear(self) -> bool:
        """``model(linear)`` (or ``model_options(linear)``) in the .mod file. The preprocessor's JSON
        output does not record it, so the .mod file next to the JSON folder is read (comments
        removed); False when it cannot be found."""
        cached = self.__dict__.get("_linear")
        if cached is None:
            cached = False
            try:
                mod = self.json_dir.parents[2] / (self.json_dir.parents[1].name + ".mod")
                text = re.sub(r"/\*.*?\*/", " ", mod.read_text(errors="replace"), flags=re.S)
                text = re.sub(r"(//|%)[^\n]*", " ", text)
                cached = any(re.search(r"(?<![\w=])linear(?![\w=])", mo.group(1))
                             for mo in re.finditer(r"(?<![\w.])model(?:_options)?\s*\(([^)]*)\)", text))
            except (OSError, IndexError):
                cached = False
            self.__dict__["_linear"] = cached
        return cached

    @property
    def orig_endo_names(self) -> list:
        """Variables declared in the .mod file, without the auxiliary ones the preprocessor adds."""
        return self.endo_names[: self.raw.get("orig_endo_nbr", self.n_endo)]

    @property
    def aux_names(self) -> list:
        return self.endo_names[self.raw.get("orig_endo_nbr", self.n_endo):]

    @property
    def param_dict(self) -> dict:
        return dict(zip(self.param_names, self.params.tolist()))

    def set_shock_stderr(self, name: str, value: float) -> None:
        """Counterpart of Dynare's ``set_shock_stderr_value``: sets the variance ``Sigma_e[i, i]``
        only, so the covariances with other shocks are kept (their correlation changes)."""
        i = self._ix[name]
        self.Sigma_e[i, i] = float(value) ** 2

    # ------------------------------------------------------------------ evaluation
    def _values(self, fn, y, x, params, ss):
        """Values returned by generated function ``fn`` as a float array.

        Python-float evaluation first; if it hits something Python raises on but numpy does not
        (log of a negative number, overflow, division by zero, a complex power), the numpy version
        is used, which gives nan/inf exactly as before."""
        y = np.asarray(y, float)
        x = np.zeros(self.n_exo) if x is None else np.asarray(x, float)
        if self.n_exo_det and x.shape[0] == self.n_exo:
            x = np.concatenate([x, self.exo_det_steady()])
        params = self.params if params is None else np.asarray(params, float)
        if CODEGEN == "fast":
            try:
                return np.array(self._ff[fn](y.tolist(), x.tolist(), params.tolist(),
                                             None if ss is None else np.asarray(ss, float).tolist()), dtype=float)
            except FAST_ERRORS:
                pass
        return np.array(self._fn[fn](y, x, params, ss), dtype=float)

    def _dense(self, fn, vals):
        dims, rows, cols = self._meta[fn]
        g = np.zeros(dims)
        g[rows, cols] = vals
        return g

    def static_resid(self, y, x=None, params=None):
        return self._values("static_resid", y, x, params, None)

    def static_jac(self, y, x=None, params=None):
        return self._dense("static_jac", self._values("static_jac", y, x, params, None))

    def dynamic_resid(self, yy, x=None, params=None, ss=None):
        return self._values("dynamic_resid", yy, x, params, ss)

    def dynamic_jac(self, yy, x=None, params=None, ss=None):
        """Dense Jacobian, columns ordered [y(-1), y, y(+1), exogenous]."""
        g = np.zeros(self.jac_dims)
        g[self.jac_rows, self.jac_cols] = self.dynamic_jac_values(yy, x, params, ss)
        return g

    def dynamic_jac_values(self, yy, x=None, params=None, ss=None):
        """Structural non-zeros of the dynamic Jacobian, at ``(jac_rows, jac_cols)``."""
        v = self._values("dynamic_jac", yy, x, params, ss)
        return v if self._det_view is None else v[self._det_view["keep"]["dynamic_jac"]]

    def dynamic_full_derivs(self, order, yy, x=None, params=None, ss=None):
        """``(rows, cols, vals)`` of the order-``order`` derivatives including the varexo_det columns
        (columns ordered [y(-1), y, y(+1), exogenous, deterministic exogenous], 0-based)."""
        fn = {1: "dynamic_jac", 2: "dynamic_hess", 3: "dynamic_third"}[order]
        vals = self._values(fn, yy, x, params, ss)
        if order == 1:
            _, rows, cols = self._meta["dynamic_jac"]
            return rows.astype(np.int32), cols.astype(np.int32)[:, None], vals
        rows, cols = self._meta[fn]
        return rows, cols, vals

    def dynamic_sparse_derivs(self, order, yy, x=None, params=None, ss=None):
        """Order-``order`` derivatives of the dynamic model as (rows, cols, vals).

        ``cols`` has shape (nnz, order); indices are 0-based with columns ordered
        [y(-1), y, y(+1), exogenous]. For order >= 2 each symmetric entry appears once.
        """
        if order == 1:
            return (self.jac_rows.astype(np.int32), self.jac_cols.astype(np.int32)[:, None],
                    self.dynamic_jac_values(yy, x, params, ss))
        fn = {2: "dynamic_hess", 3: "dynamic_third"}.get(order)
        if fn is None:
            raise NotImplementedError("derivatives above third order")
        rows, cols = self._meta[fn]
        vals = self._values(fn, yy, x, params, ss)
        if self._det_view is not None:
            k = self._det_view["keep"][fn]
            rows, cols, vals = rows[k], cols[k], vals[k]
        return rows, cols, vals

    def exo_det_steady(self):
        """Steady-state values of the varexo_det variables (initval, else 0), as oo_.exo_det_steady_state."""
        return np.array([self.initval.get(v, 0.0) for v in self.exo_det_names], float)

    def exo_steady(self, terminal=False):
        src = {**self.initval, **self.endval} if terminal else self.initval
        return np.array([src.get(v, 0.0) for v in self.exo_names])
