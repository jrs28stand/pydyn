"""Model object built from the Dynare preprocessor's JSON output."""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np

from . import matlab
from .expr import NAMESPACE, Translator, safe_name, temp_name


def _pairs(items, key_a, key_b):
    return [(d[key_a], d[key_b]) for d in items]


class ModelError(Exception):
    pass


class Model:
    """A Dynare model: symbols, calibration, and compiled residual/derivative functions.

    Variables are held in declaration order. Dynamic functions take a stacked vector
    ``yy = [y(-1); y; y(+1)]`` of length ``3 * n_endo``.
    """

    def __init__(self, json_dir: str | Path, functions: dict | None = None, steady_state_fn=None):
        """``functions`` maps names of external functions used in the model or in the
        ``steady_state_model`` block to Python callables. ``steady_state_fn`` is the Python
        counterpart of a Dynare ``_steadystate.m`` file: ``fn(model, params, exo) -> ys`` or
        ``(ys, params)``; it may raise ``SteadyStateError``."""
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
            mf = json.loads(text)
        dyn = json.loads((json_dir / "dynamic.json").read_text())["dynamic_model"]
        sta = json.loads((json_dir / "static.json").read_text())["static_model"]
        self.raw = mf

        self.endo_names = [d["name"] for d in mf["endogenous"]]
        self.exo_names = [d["name"] for d in mf["exogenous"]]
        self.param_names = [d["name"] for d in mf["parameters"]]
        self.n_endo, self.n_exo = len(self.endo_names), len(self.exo_names)
        self._ie = {v: i for i, v in enumerate(self.endo_names)}
        self._ix = {v: i for i, v in enumerate(self.exo_names)}
        self._ip = {v: i for i, v in enumerate(self.param_names)}
        if mf.get("exogenous_deterministic"):
            raise NotImplementedError("deterministic exogenous variables are not supported")

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

        self._compile(dyn, sta)
        self._load_steady_state_model()
        mod_dir, name = json_dir.parents[2], json_dir.parents[1].name
        self.steady_state_file = mod_dir / f"{name}_steadystate.m"
        if not self.steady_state_file.exists():
            self.steady_state_file = None
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
            if name in self._ip:
                return f"params[{self._ip[name]}]"
            return None

        return Translator(resolve, steady=self._steady_ref("ss[{}]"),
                          shiftable=set(self._ie) | set(self._ix))

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
            if name in self._ip:
                return f"params[{self._ip[name]}]"
            return None

        # in the static model STEADY_STATE(v) is v itself
        return Translator(resolve, steady=self._steady_ref("y[{}]"),
                          shiftable=set(self._ie) | set(self._ix))

    @staticmethod
    def _locals(block):
        out = []
        for d in block.get("model_local_variables", []):
            out.append((d.get("variable", d.get("name")), d.get("value")))
        return out

    def _build(self, tr, block, name):
        """Return source for residual and Jacobian functions of a (static|dynamic) block."""
        tr.temps = {d["temporary_term"] for k, v in block.items() if k.startswith("temporary_terms_") for d in v}
        assign = []
        for k, v in self._locals(block):
            assign.append(f"    {safe_name(k)} = {tr(v)}")
        tt_res = [f"    {temp_name(d['temporary_term'])} = {tr(d['value'])}" for d in block["temporary_terms_"]]
        tt_jac = [f"    {temp_name(d['temporary_term'])} = {tr(d['value'])}" for d in block["temporary_terms_jacobian"]]
        for key in ("external_functions_temporary_terms_", "external_functions_temporary_terms_jacobian"):
            if block.get(key):
                raise NotImplementedError("external functions are not supported")

        neq = len(block["residuals"])
        lines = [f"def {name}_resid(y, x, params, ss):"] + assign + tt_res
        lines.append(f"    r = np.empty({neq})")
        for i, d in enumerate(block["residuals"]):
            e = d["residual"]
            rhs = e.get("rhs", "0")
            lines.append(f"    r[{i}] = ({tr(e['lhs'])}) - ({tr(rhs)})")
        lines.append("    return r\n")

        jac = block["jacobian"]
        lines += [f"def {name}_jac(y, x, params, ss):"] + assign + tt_res + tt_jac
        lines.append(f"    g = np.zeros(({jac['dims'][0]}, {jac['dims'][1]}))")
        for e in jac["entries"]:
            r, c = e["indices"]
            lines.append(f"    g[{r - 1}, {c - 1}] = {tr(e['val'])}")
        lines.append("    return g\n")

        tt_prev = assign + tt_res + tt_jac
        for fn, key, ttkey, order in (("hess", "hessian", "temporary_terms_hessian", 2),
                                      ("third", "third_derivative", "temporary_terms_third_derivative", 3)):
            tt_prev = tt_prev + [f"    {temp_name(d['temporary_term'])} = {tr(d['value'])}" for d in block.get(ttkey, [])]
            lines += [f"def {name}_{fn}(y, x, params, ss):"] + tt_prev
            lines.append("    rows, cols, vals = [], [], []")
            for e in block.get(key, {}).get("entries", []):
                r, *c = e["indices"]
                lines.append(f"    rows.append({r - 1}); cols.append({tuple(i - 1 for i in c)}); vals.append({tr(e['val'])})")
            lines.append("    return rows, cols, vals\n")
        return "\n".join(lines)

    def _compile(self, dyn, sta):
        src = self._build(self._tr_dynamic(), dyn, "dynamic") + "\n" + self._build(self._tr_static(), sta, "static")
        ns = dict(NAMESPACE)
        ns.update(self.functions)
        ns["np"] = np
        exec(compile(src, f"<pydyn:{self.json_dir}>", "exec"), ns)  # noqa: S102 - generated from the model
        self.source = src
        self._f = ns
        self.jac_dims = tuple(dyn["jacobian"]["dims"])
        self._jac_entries = [(e["indices"][0] - 1, e["indices"][1] - 1) for e in dyn["jacobian"]["entries"]]
        # highest derivative order the preprocessor computed (a linear model has an empty Hessian)
        self.max_deriv_order = 1 + sum(f"dynamic_g{o}_sparse_indices" in self._deriv_keys for o in (2, 3))
        if self.jac_dims[1] != 3 * self.n_endo + self.n_exo:
            raise NotImplementedError(f"unexpected dynamic Jacobian layout {self.jac_dims}")

    # ------------------------------------------------------------------ statements
    def _param_translator(self, extra=None):
        def resolve(name, shift):
            if name in self._ip:
                return f"params[{self._ip[name]}]"
            return None
        return Translator(resolve)

    def eval_expr(self, s: str, local=None) -> float:
        s = re.sub(r"(?<![\w.])0+(\d+\.?\d*)", r"\1", str(s)) if re.search(r"(?<![\w.])0\d", str(s)) else str(s)
        ns = dict(NAMESPACE)
        ns.update(self.functions)
        ns.update({safe_name(k): v for k, v in getattr(self, "workspace", {}).items()})
        ns["params"] = self.params
        if local:
            ns.update(local)
        return float(eval(self._param_translator()(str(s)), ns))  # noqa: S307 - model expression

    def _load_steady_state_model(self):
        p = self.json_dir / "steady_state_model.json"
        self.steady_state_model = None
        if p.exists():
            self.steady_state_model = _pairs(json.loads(p.read_text())["steady_state_model"], "lhs", "rhs")

    def _eval_with_vars(self, s: str) -> float:
        """Evaluate an initval-style expression that may refer to variables set earlier."""
        local = {safe_name(k): v for k, v in self.initval.items() if k not in self._ip}
        return self.eval_expr(s, local)

    def _run_statements(self):
        """Apply the statements of the .mod file in order.  Native MATLAB lines are interpreted by
        ``pydyn.matlab`` together with the Dynare statements, so that MATLAB control flow around
        parameter assignments (``if calib == 1 ... end``) behaves as in Dynare."""
        self.params = np.full(len(self.param_names), np.nan)
        self.workspace: dict[str, float] = {}      # scalar MATLAB variables of the .mod file
        self.initval: dict[str, float] = {}
        self.endval: dict[str, float] = {}
        self.det_shocks: list = []                 # (exo index, first period, last period, value), 1-based
        self.Sigma_e = np.zeros((self.n_exo, self.n_exo))
        self.measurement_errors: dict[str, float] = {}
        self.occbin_constraints = []
        self.options: dict[str, dict] = {}
        self.matlab_options = matlab.Struct()      # options_.x = ... assignments (stored, not acted upon)
        self.matlab_skipped: list = []             # MATLAB statements that were not executed: (text, kind, reason)
        self._n_surprise = 0
        self._matlab = None
        if any(st["statementName"] in ("native", "verbatim") for st in self.statements):
            self._matlab = matlab.ModSession(self, ModelError)
            self._matlab.run(self.statements)
        else:
            for st in self.statements:
                self._exec_statement(st)
        if self._n_surprise > 1:
            warnings.warn("several shocks(surprise) blocks: they are merged; Dynare's 'overwrite' "
                          "option is not visible in the preprocessor output")
        if self.unhandled_statements:
            warnings.warn(f"statements not handled: {sorted(set(self.unhandled_statements))}")

    def _value(self, s) -> float:
        """Value of an expression in a Dynare statement.  Expressions that index a MATLAB matrix or
        call a MATLAB function defined next to the .mod file are passed to the MATLAB interpreter."""
        try:
            return self.eval_expr(s)
        except (NameError, TypeError, SyntaxError):
            if self._matlab is None:
                self._matlab = matlab.ModSession(self, ModelError)
            return self._matlab.eval_value(s)

    def _exec_statement(self, st):
        """One Dynare statement of the .mod file."""
        kind = st["statementName"]
        if kind == "param_init":
            self.params[self._ip[st["name"]]] = self._value(st["value"])
        elif kind == "initval":
            for d in st["vals"]:
                self.initval[d["name"]] = self._eval_with_vars(d["value"])
        elif kind == "endval":
            for d in st["vals"]:
                self.endval[d["name"]] = self._eval_with_vars(d["value"])
        elif kind == "load_params_and_steady_state":
            for d in st.get("values", []):
                if d["name"] in self._ip:
                    self.params[self._ip[d["name"]]] = self.eval_expr(d["value"])
                else:
                    self.initval[d["name"]] = self.eval_expr(d["value"])
        elif kind == "shocks" and st.get("surprise"):
            self._n_surprise += 1
            self.options.setdefault("shock_paths", []).append(st)
        elif kind == "shocks":
            self._shocks(st)
        elif kind == "occbin_constraints":
            self.occbin_constraints = st["constraints"]
        elif kind == "homotopy" and st.get("values") and all(
                d["name"] in self._ip or d["name"] in self._ix for d in st["values"]):
            self._homotopy = st["values"]          # applied by the next steady(homotopy_mode=...)
        elif kind in ("steady", "check", "occbin_setup", "occbin_graph", "rplot"):
            self.options.setdefault(kind, st.get("options", {}))
            if kind == "steady" and st.get("options", {}).get("homotopy_mode") and getattr(self, "_homotopy", None):
                # Dynare's steady moves these parameters and exogenous variables to their final values
                # (it stops with an error if the homotopy fails); later statements see the final values
                for d in self._homotopy:
                    val = self.eval_expr(d["final_value"])
                    if d["name"] in self._ip:
                        self.params[self._ip[d["name"]]] = val
                    else:
                        self.initval[d["name"]] = val
        elif kind in ("stoch_simul", "estimation", "occbin_solver", "estimated_params",
                      "estimated_params_init", "varobs", "perfect_foresight_setup", "perfect_foresight_solver", "calib_smoother",
                          "shock_decomposition", "initial_condition_decomposition", "shock_groups",
                          "observation_trends", "estimated_params_bounds"):
            self.options[kind] = st.get("options", {}) if kind != "estimated_params" else st
        elif kind in ("native", "verbatim"):
            pass                                   # run by pydyn.matlab.ModSession
        elif kind in ("write_latex_dynamic_model", "write_latex_static_model",
                      "write_latex_original_model", "write_latex_steady_state_model",
                      "collect_latex_files", "write_latex_definitions"):
            pass
        else:
            self.unhandled_statements.append(kind)

    def _shocks(self, st):
        """One shocks block, applied in Dynare's order: reset, variances, covariances, correlations."""
        ix, S = self._ix, self.Sigma_e
        if st.get("overwrite"):
            S[:] = 0
            self.measurement_errors.clear()
        for key, power in (("variance", 1), ("stderr", 2)):
            for d in st.get(key, []):
                val = self.eval_expr(d[key]) ** power
                if d["name"] in ix:
                    S[ix[d["name"]], ix[d["name"]]] = val
                else:                               # measurement error on an observed variable
                    self.measurement_errors[d["name"]] = val
        for key in ("covariance", "correlation"):
            for d in st.get(key, []):
                if d["name"] not in ix or d["name2"] not in ix:
                    self.unhandled_statements.append("shocks: measurement-error covariance")
                    continue
                i, j = ix[d["name"]], ix[d["name2"]]
                val = self.eval_expr(d[key])
                if key == "correlation":
                    val *= np.sqrt(S[i, i] * S[j, j])
                S[i, j] = S[j, i] = val
        for d in st.get("deterministic_shocks", []):
            for v in d["values"]:
                self.det_shocks.append((ix[d["var"]], int(v["period1"]), int(v["period2"]), self.eval_expr(v["value"])))
        for key in ("skewness", "coskewness", "multiplicative_deterministic_shocks"):
            if st.get(key):
                self.unhandled_statements.append(f"shocks: {key}")

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
        i = self._ix[name]
        self.Sigma_e[i, :] = 0.0
        self.Sigma_e[:, i] = 0.0
        self.Sigma_e[i, i] = float(value) ** 2

    # ------------------------------------------------------------------ evaluation
    def _x0(self, x):
        return np.zeros(self.n_exo) if x is None else np.asarray(x, float)

    def static_resid(self, y, x=None, params=None):
        return self._f["static_resid"](np.asarray(y, float), self._x0(x), self.params if params is None else params, None)

    def static_jac(self, y, x=None, params=None):
        return self._f["static_jac"](np.asarray(y, float), self._x0(x), self.params if params is None else params, None)

    def dynamic_resid(self, yy, x=None, params=None, ss=None):
        return self._f["dynamic_resid"](np.asarray(yy, float), self._x0(x), self.params if params is None else params, ss)

    def dynamic_jac(self, yy, x=None, params=None, ss=None):
        """Dense Jacobian, columns ordered [y(-1), y, y(+1), exogenous]."""
        return self._f["dynamic_jac"](np.asarray(yy, float), self._x0(x), self.params if params is None else params, ss)

    def dynamic_sparse_derivs(self, order, yy, x=None, params=None, ss=None):
        """Order-``order`` derivatives of the dynamic model as (rows, cols, vals).

        ``cols`` has shape (nnz, order); indices are 0-based with columns ordered
        [y(-1), y, y(+1), exogenous]. For order >= 2 each symmetric entry appears once.
        """
        yy, x = np.asarray(yy, float), self._x0(x)
        params = self.params if params is None else params
        if order == 1:
            g = self._f["dynamic_jac"](yy, x, params, ss)
            rows = np.array([r for r, _ in self._jac_entries], dtype=np.int32)
            cols = np.array([[c] for _, c in self._jac_entries], dtype=np.int32)
            return rows, cols, g[rows, cols[:, 0]]
        fn = {2: "dynamic_hess", 3: "dynamic_third"}.get(order)
        if fn is None:
            raise NotImplementedError("derivatives above third order")
        rows, cols, vals = self._f[fn](yy, x, params, ss)
        return (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32).reshape(len(rows), order),
                np.array(vals, float))

    def exo_steady(self, terminal=False):
        src = {**self.initval, **self.endval} if terminal else self.initval
        return np.array([src.get(v, 0.0) for v in self.exo_names])
