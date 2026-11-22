# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Acceptance sweep: run Dynare's test-suite .mod files through pydyn and compare with Dynare's results under Octave.

    python tools/accept_sweep.py run     [--jobs 1] [--only substr ...] [--force] [--timeout 600]
    python tools/accept_sweep.py one     <test/path.mod>            # one file, prints the result (debugging)
    python tools/accept_sweep.py report                              # docs/ACCEPTANCE.md from the stored results

Environment: PYDYN_ACCEPT_REF (default /home/claude/accept_ref: the .mat/.json files of tools/accept_octave.py),
DYNARE_SRC (default /home/claude/dl/src72), DYNARE_PREPROCESSOR.  pydyn's verdicts go to <ref>/_pydyn/<path>.json.

How a file is judged
--------------------
Every field of Dynare's ``oo_`` that the stored run holds and that the sweep knows (``FIELDS`` below: oo_ field ->
producer returning pydyn's value) is compared; the producers are the only place that knows pydyn's API, so they can be
pointed at the ``oo_`` layer later.  Comparison: ``max|a-b| <= atol + rtol * max|b|`` over the whole field, with
``rtol = 1e-8`` (``atol = 1e-10``) for deterministic quantities, 1e-5 for steady states solved numerically (Dynare stops
at tolf), 1e-6 for perfect-foresight paths (Newton stops at tolf).  Fields that depend on random numbers (simulated
moments, higher-order IRFs, MCMC output, forecasts with bands) are skipped, and so are fields Dynare leaves unset.

Parameters: decision rules, moments and IRFs are evaluated at Dynare's final ``M_.params`` / ``M_.Sigma_e`` and its
steady state, so that estimation and steady-state-solver differences are not mixed into them; the steady state is
compared separately from pydyn's own computation, and ``M_.params`` is compared with pydyn's calibration after the
.mod file's MATLAB code for files that do not change the parameters through estimation.
Verdicts: match, partial, refused, wrong, pydyn-error, dynare-failed, xfail, no-comparison (loaded, nothing to compare).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, os.environ.get("PYDYN_ACCEPT_PYDYN", str(ROOT)))   # a different copy of pydyn (e.g. the results-layer branch)
sys.path.insert(0, str(TOOLS))
from accept_manifest import load_manifest  # noqa: E402
from accept_octave import prepare_scratch, relid  # noqa: E402

REF = Path(os.environ.get("PYDYN_ACCEPT_REF", "/home/claude/accept_ref"))
SRC = Path(os.environ.get("DYNARE_SRC", "/home/claude/dl/src72"))
os.environ["PYDYN_CACHE_DIR"] = os.environ.get("PYDYN_ACCEPT_CACHE", "/var/tmp/pydyn_accept_cache")   # not the user cache: it may hold code compiled by other pydyn versions
os.environ.setdefault("DYNARE_PREPROCESSOR", str(SRC / "build-octave/preprocessor/src/dynare-preprocessor"))

RTOL, ATOL = 1e-8, 1e-10
RTOL_NUMERIC_SS, RTOL_PF = 1e-5, 1e-6


class Skip(Exception):
    """A field that is not compared on purpose (random numbers, option combination); the message says why."""


class NotProduced(Exception):
    """pydyn has no counterpart of this field."""


class SoftMatch(Exception):
    """Agreement to Dynare's solver tolerance only."""


class DifferentRoot(Exception):
    """pydyn and Dynare ended at different solutions of the static model (several steady states)."""


class Refusal(Exception):
    """pydyn refuses the request (Unsupported / NotImplementedError / ModelError ...)."""


# ------------------------------------------------------------------------------------------ reference access
def load_ref(path):
    d = loadmat(path, squeeze_me=True, struct_as_record=False)
    return d["oo_"], d["M_"], d["options_"]


def sget(obj, path, default=None):
    """oo_-style attribute path 'dr.ghx'; mat_struct or dict."""
    for p in path.split("."):
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(p)
        elif hasattr(obj, "_fieldnames"):
            obj = getattr(obj, p, None)
        else:
            return default
    return default if obj is None else obj


def fields_of(s):
    return list(getattr(s, "_fieldnames", []))


def is_struct(x):
    return hasattr(x, "_fieldnames")


def nonempty(x):
    if x is None:
        return False
    if is_struct(x):
        return len(fields_of(x)) > 0
    if isinstance(x, np.ndarray):
        return x.size > 0
    if isinstance(x, (str, bytes)):
        return len(x) > 0
    return True


def names_of(x):
    """Cell array / char matrix of names -> list of str."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    a = np.atleast_1d(x)
    if a.dtype.kind in "US" and a.ndim == 2:
        return [str(r).strip() for r in a]
    return [str(v).strip() for v in a.ravel()]


# ------------------------------------------------------------------------------------------ comparison
def to_float(x):
    a = np.asarray(x)
    if a.dtype == object:
        raise TypeError("object array")
    return a.astype(complex) if np.iscomplexobj(a) else a.astype(float)


def compare(a, b, rtol=RTOL, atol=ATOL, name=""):
    """(ok, max abs difference, message) of pydyn's ``a`` against Dynare's ``b`` (sizes must agree; shapes may differ in singleton axes)."""
    a, b = to_float(a), to_float(b)
    if a.size != b.size:
        return False, np.inf, f"size {a.shape} vs Dynare {b.shape}"
    a = a.reshape(b.shape) if a.shape != b.shape and (a.size == b.size and (1 in a.shape or 1 in b.shape or a.ndim != b.ndim)) else a
    if a.shape != b.shape:
        return False, np.inf, f"shape {a.shape} vs Dynare {b.shape}"
    if b.size == 0:
        return True, 0.0, ""
    both_bad = ~np.isfinite(a) & ~np.isfinite(b)      # 0/0 and x/0 of zero-variance variables: NaN or +-Inf in both, signs are noise
    one_bad = (~np.isfinite(a)) != (~np.isfinite(b))
    if np.any(one_bad):
        return False, np.inf, "finite in one program, NaN/Inf in the other"
    m = ~both_bad
    if not m.any():
        return True, 0.0, ""
    d = np.abs(a[m] - b[m])
    scale = np.max(np.abs(b[m]))
    mx = float(np.max(d)) if d.size else 0.0
    return bool(mx <= atol + rtol * scale), mx, ""


# ------------------------------------------------------------------------------------------ context
class Ctx:
    """Everything the producers need, computed lazily and cached."""

    def __init__(self, mod, entry, scratch, oo, M, opts):
        import pydyn
        self.pydyn = pydyn
        self.mod, self.entry, self.scratch = mod, entry, scratch
        self.oo, self.M, self.opts = oo, M, opts
        self.cache = {}
        self.notes = []
        self.m = None

    # -- loading
    def load(self):
        pydyn = self.pydyn
        path = self.scratch / "tests" / self.mod
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.m = pydyn.Model(pydyn.run_preprocessor(path, extra_args=tuple(self.entry.get("dynare_args", []))))
        self.warn_notes(w)
        self.kinds = [s["statementName"] for s in self.m.statements]
        self.kindset = set(self.kinds)
        self.ref_params = np.atleast_1d(np.asarray(sget(self.M, "params", []), float))
        self.ref_ys = np.atleast_1d(np.asarray(sget(self.oo, "steady_state", []), float))
        self.dr_ys = np.atleast_1d(np.asarray(sget(self.oo, "dr.ys", []), float))        # what the decision rules were computed around
        self.ref_var_list = names_of(sget(self.oo, "var_list"))
        self.ref_Sigma = sget(self.M, "Sigma_e")
        self.ref_endo = names_of(sget(self.M, "endo_names"))
        self.ref_exo = names_of(sget(self.M, "exo_names"))
        if self.ref_endo and self.ref_endo != list(self.m.endo_names):
            self.notes.append("endogenous names differ from Dynare's")
        if self.loglinear():
            # stoch_simul(loglinear) leaves oo_.dr.ys and oo_.steady_state in logs (stoch_simul.m, for the display of the rules)
            from pydyn.perturbation import log_mask
            mask = log_mask(self.m)
            for name in ("dr_ys", "ref_ys"):
                v = getattr(self, name)
                if v.size == mask.size and np.all(np.isfinite(v[mask])):
                    setattr(self, name, np.where(mask, np.exp(np.where(mask, v, 0.0)), v))
        self.calibration_params = np.array(self.m.params, float)
        self.calibration_Sigma = np.array(self.m.Sigma_e, float)
        return self.m

    def warn_notes(self, w):
        for x in w:
            msg = str(x.message)
            if "UnsupportedOption" in x.category.__name__ or "not handled" in msg:
                n = msg[:200]
                if n not in self.notes:
                    self.notes.append(n)

    # -- helpers
    def has(self, *kinds):
        return bool(self.kindset & set(kinds))

    def sync_to_dynare(self):
        """Put Dynare's final parameters and shock covariance into the model (decision rules, moments, IRFs are
        evaluated at the point Dynare ended with)."""
        m = self.m
        if self.ref_params.size == len(m.params):
            ok = np.isfinite(self.ref_params)
            if not ok.all() and not any("NaN" in n for n in self.notes):
                # M_.params holds NaN for parameters Dynare never set: the .mod file's body did not run (e.g. it sits inside
                # `if ~isoctave`, as in optimal_policy/OSR/osr_obj_corr_algo_*.mod: the Octave reference is empty)
                self.notes.append("Dynare's M_.params contains NaN (parameters never set): the Octave run did not execute the .mod file's body")
            m.params[ok] = self.ref_params[ok]
        if self.ref_Sigma is not None and np.size(self.ref_Sigma) == m.Sigma_e.size:
            m.Sigma_e[:, :] = np.asarray(self.ref_Sigma, float).reshape(m.Sigma_e.shape)

    def stoch_opts(self):
        return self.m.options.get("stoch_simul", {})

    def order(self):
        """Order Dynare actually solved to: read from the stored decision rules (options_.order is 2 by default even where the
        solver ran at first order, e.g. discretionary_policy)."""
        dr = sget(self.oo, "dr")
        stored = 0
        if dr is not None and nonempty(dr):
            fs = fields_of(dr)
            stored = 3 if "ghxxx" in fs else 2 if "ghxx" in fs else 1 if "ghx" in fs else 0
        if self.has("stoch_simul"):
            # a stoch_simul statement solves to its order (Dynare's default is 2); a later command (calib_smoother, forecast,
            # conditional_forecast, ...) may have replaced oo_.dr by a first-order one, but oo_.mean, oo_.var, ... stay those of
            # the stoch_simul run (conditional_forecasts/fs2000_calib_smoother_cf)
            return max(stored, int(self.stoch_opts().get("order", 2)))
        if stored:
            return stored
        o = sget(self.opts, "order")
        return int(o) if o is not None and np.size(o) else int(self.stoch_opts().get("order", 1))

    def loglinear(self):
        v = sget(self.opts, "loglinear")
        return bool(np.any(v)) if v is not None and np.size(v) else bool(self.stoch_opts().get("loglinear", False))

    def rule_ys(self):
        """Steady state the decision rules were computed around: oo_.dr.ys (oo_.steady_state holds zero multipliers for Ramsey)."""
        for cand in (self.dr_ys, self.ref_ys):
            if cand.size == self.m.n_endo and np.all(np.isfinite(cand)):
                return cand
        return None

    def check_rules_belong(self, ys):
        """Dynare's decision rules are those of the parameter values current at the last solve; MATLAB code may change the
        parameters afterwards (pydyn then ends with the later values).  If the stored steady state is not a steady state of the
        model at the final parameters, the rules are not comparable."""
        if ys is None or self.has("estimation", "osr", "identification", "method_of_moments"):
            return
        try:
            r = np.max(np.abs(self.m.static_resid(np.asarray(ys, float), self.m.exo_steady(), np.array(self.m.params, float))))
        except Exception:  # noqa: BLE001
            return
        if np.isfinite(r) and r > 1e-5:
            raise Skip(f"Dynare's rules belong to other parameter values than the final ones (static residual {r:.1e} at oo_.dr.ys)")

    def sol1(self):
        if "sol1" not in self.cache:
            self.sync_to_dynare()
            ys = self.rule_ys()
            if self.has("discretionary_policy"):
                from pydyn import optimal_policy as op
                self.cache["sol1"] = op.discretionary_policy(self.m)
                return self.cache["sol1"]
            self.check_rules_belong(ys)

            def solve(**kw):
                return (self.pydyn.first_order(self.m, ys=ys, params=np.array(self.m.params, float), loglinear=self.loglinear(), **kw)
                        if ys is not None else self.pydyn.first_order(self.m, loglinear=self.loglinear(), **kw))
            try:
                self.cache["sol1"] = solve()
            except self.pydyn.perturbation.BlanchardKahnError:
                # Dynare's stored oo_.dr may come from an earlier stoch_simul whose options_.qz_criterium differed from the last
                # one (decision_rules/ar_qz_test: the last call is expected to fail and is caught by a try/catch)
                alt = []
                for _k, fld, val in getattr(self.m, "matlab_options_log", []):
                    if fld == "qz_criterium" and val not in alt:
                        alt.append(val)
                for q in alt:
                    try:
                        self.cache["sol1"] = solve(qz_criterium=float(q))
                    except self.pydyn.perturbation.BlanchardKahnError:
                        continue
                    self.notes.append(f"decision rules solved with qz_criterium={float(q)!r} (an earlier options_.qz_criterium of the file; "
                                      "Dynare's stored oo_.dr is from that call, the later call fails in Dynare as well)")
                    break
                else:
                    raise
        return self.cache["sol1"]

    def drk(self):
        """Decision-rule dict at the order of the stoch_simul run (k_order for order >= 2)."""
        if "drk" not in self.cache:
            order = self.order()
            if order <= 1:
                self.cache["drk"] = self.sol1().as_dr()
            else:
                self.sync_to_dynare()
                self.cache["drk"] = self.pydyn.k_order(self.m, order=order, ys=self.rule_ys(), params=np.array(self.m.params, float),
                                                       loglinear=self.loglinear())
        return self.cache["drk"]

    def moments(self):
        if "mom" not in self.cache:
            from pydyn import moments as mo
            self.sol1()
            vl = [v for v in self.ref_var_list if v in self.m.endo_names] or None
            self.cache["mom"] = mo.theoretical_moments(self.m, self.drk(), var_list=vl, order=self.order())
        return self.cache["mom"]

    def pf_result(self):
        if "pf" not in self.cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kw = {}
                info = self.m.info
                if (info["maximum_endo_lag"] == 0 or info["maximum_endo_lead"] == 0) and self.m.options.get("perfect_foresight_solver", {}).get("dynatol.f") is None \
                        and self.m.options.get("perfect_foresight_solver", {}).get("tolf") is None:
                    # purely forward/backward/static models are solved period by period to dynatol.f, Dynare's 1e-5 (pydyn's default is
                    # 1e-9): Dynare's own path is only that accurate (solve_algo_12_14/purely_forward_reference: 1.2e-4 apart)
                    kw["tolf"] = 1e-5
                self.cache["pf"] = self.pydyn.perfect_foresight(self.m, raise_on_failure=False, **kw)
        return self.cache["pf"]

    def dynare_unsolved(self):
        """Dynare's own run failed (Blanchard-Kahn, steady state) without an error: oo_.dr exists but holds no decision rules
        (only possible with the noprint option, which also hides the message)."""
        dr = sget(self.oo, "dr")
        return self.has("stoch_simul") and dr is not None and "ghx" in fields_of(dr) and not nonempty(sget(dr, "ghx"))

    def ref_point_inconsistent(self):
        """Dynare's stored oo_.steady_state is not a root of the static model at M_.params (estimation files whose optimizer
        ended at parameters the stored results do not belong to)."""
        try:
            ref = np.asarray(sget(self.oo, "steady_state"), float).ravel()
            self.sync_to_dynare()
            r = np.max(np.abs(self.m.static_resid(ref, self.m.exo_steady(), np.array(self.m.params, float))))
            return not r < 1e-4
        except Exception:  # noqa: BLE001
            return False

    def stat_set(self):
        return self.has("stoch_simul") and self.kinds


# ------------------------------------------------------------------------------------------ producers
# Each producer returns an ndarray, a list of ndarrays (cell array) or a dict name -> ndarray (struct); it may raise
# Skip / NotProduced.  Registry: FIELDS[oo_ path] = dict(get=producer, needs=<statement names, any of>, rtol=..).
FIELDS: dict[str, dict] = {}


def field(path, needs=(), rtol=RTOL, atol=ATOL, when=None, rtol_soft=None):
    def deco(fn):
        FIELDS[path] = dict(get=fn, needs=set(needs), rtol=rtol, atol=atol, when=when, rtol_soft=rtol_soft)
        return fn
    return deco


def not_produced(path, why, needs=()):
    def fn(c):
        raise NotProduced(why)
    FIELDS[path] = dict(get=fn, needs=set(needs), rtol=RTOL, atol=ATOL, when=None)


STOCH = ("stoch_simul",)


@field("steady_state")
def f_steady(c):
    c.sync_to_dynare()
    c.check_rules_belong(steady_ref(c.ref_ys, c)) if c.has("stoch_simul", "ramsey_model", "ramsey_policy", "discretionary_policy") and c.dr_ys.size else None
    if c.has("discretionary_policy") and c.m.steady_state_model is None and c.m.steady_state_fn is None and c.m.steady_state_file is None:
        # Dynare's discretionary_policy_1.m takes ys = zeros (the model must be in deviation form) and has no steady-state solver:
        # the model has fewer equations than variables (the instruments are free), so there is nothing to solve for
        from pydyn import optimal_policy
        ys, p = optimal_policy.discretionary_policy(c.m, welfare=False).dr["ys"], np.array(c.m.params, float)
    else:
        try:
            ys, p = c.pydyn.steady_state(c.m, params=np.array(c.m.params, float))
        except c.pydyn.steady.SteadyStateError as exc:
            ref = steady_ref(c.ref_ys, c)
            try:
                with np.errstate(all="ignore"):
                    r_dy = np.max(np.abs(c.m.static_resid(np.asarray(ref, float), c.m.exo_steady(), np.array(c.m.params, float))))
            except Exception:  # noqa: BLE001
                r_dy = np.nan
            if not r_dy < 1e-4:
                raise Skip(f"Dynare never solved the static model in this file (its oo_.steady_state is the initial zero vector: static residual "
                           f"{r_dy:.1e}); pydyn's search does not find one either ({str(exc)[:80]})")
            raise
    numerical = c.m.steady_state_model is None and c.m.steady_state_fn is None and c.m.steady_state_file is None
    FIELDS["steady_state"]["rtol"] = RTOL_NUMERIC_SS if numerical else RTOL
    FIELDS["steady_state"]["rtol_soft"] = None
    if c.ref_endo != list(c.m.endo_names):
        raise NotProduced("endogenous variable names differ from Dynare's, steady state not compared")
    ys = np.asarray(ys, float)
    ref_ys = steady_ref(c.ref_ys, c)
    if ys.shape == ref_ys.shape and not np.allclose(ys, ref_ys, rtol=FIELDS["steady_state"]["rtol"], atol=1e-8):
        # a different root of the static model?  Compare the static residuals of both points.
        try:
            x = c.m.exo_steady()
            r_py = np.max(np.abs(c.m.static_resid(ys, x, np.array(p, float))))
            r_dy = np.max(np.abs(c.m.static_resid(ref_ys, x, np.array(p, float))))
        except Exception:  # noqa: BLE001
            r_py = r_dy = np.nan
        if r_dy < 1e-6 and r_py < 1e-6:
            raise DifferentRoot(f"both solve the static model (residual pydyn {r_py:.1e}, Dynare {r_dy:.1e}): another root, "
                                f"largest difference {np.max(np.abs(ys - ref_ys)):.2e}")
        if r_dy < 1e-4 and r_py < 1e-6:
            raise SoftMatch(f"Dynare's numerical steady state has static residual {r_dy:.1e}, pydyn's {r_py:.1e}; "
                            f"largest difference {np.max(np.abs(ys - ref_ys)):.2e}")
        if not r_dy < 1e-4:                       # also NaN: the stored point (initial zeros) cannot even be evaluated
            raise Skip(f"Dynare's oo_.steady_state is not a steady state (static residual {r_dy:.1e}): the file never solved for it "
                       "(initval values); pydyn's residual %.1e" % r_py)
        c.notes.append(f"steady state: static residual pydyn {r_py:.1e}, at Dynare's point {r_dy:.1e}")
    return ys


def steady_ref(ref, c):
    """oo_.dr.ys, the point the decision rules were computed around: Dynare leaves oo_.steady_state at the initval values (or zero
    multipliers for Ramsey) when stoch_simul alone computes the steady state."""
    if c.dr_ys.size == np.size(ref) and np.all(np.isfinite(c.dr_ys)) and c.has("stoch_simul", "ramsey_model", "ramsey_policy", "discretionary_policy"):
        return c.dr_ys           # oo_.steady_state is stale (initval values, zero multipliers) after a bare stoch_simul
    return ref


FIELDS["steady_state"]["refx"] = steady_ref


@field("exo_steady_state")
def f_exo_ss(c):
    if getattr(getattr(c.m, "_matlab", None), "path_edits", False) and c.has("perfect_foresight_with_expectation_errors_solver",
                                                                              "perfect_foresight_solver"):
        raise Skip("MATLAB code in the file writes oo_ (simulation paths, steady states) and solves again: Dynare's final value is the last "
                   "solve's (pydyn does not run such assignments and warns)")
    if c.has("perfect_foresight_with_expectation_errors_solver"):
        # the solver leaves the exogenous steady state of the last learnt endval (shocks(learnt_in)/endval(learnt_in)) in oo_
        return np.asarray(c.pf_result().exo_steady_state, float)
    return np.asarray(c.m.exo_steady(terminal=c.has("endval")), float)


@field("dr.ghx", STOCH + ("ramsey_model", "ramsey_policy", "discretionary_policy"))
def f_ghx(c):
    other_solver(c)
    return c.drk()["ghx"] if c.order() > 1 else c.sol1().ghx


@field("dr.ghu", STOCH + ("ramsey_model", "ramsey_policy", "discretionary_policy"))
def f_ghu(c):
    other_solver(c)
    return c.drk()["ghu"] if c.order() > 1 else c.sol1().ghu


for _k in ("ghxx", "ghxu", "ghuu", "ghs2", "ghxxx", "ghxxu", "ghxuu", "ghuuu", "ghxss", "ghuss"):
    def _mk(k):
        @field("dr." + k, STOCH + ("ramsey_model", "ramsey_policy"), when=lambda c, k=k: c.order() >= 2)
        def f(c):
            dr = c.drk()
            if k not in dr:
                raise NotProduced(f"k_order did not return {k}")
            return dr[k]
        return f
    _mk(_k)


def finite_moduli(x, lo=0.1, hi=10.0):
    """What is compared of dr.eigval: the number of generalised eigenvalues, the number of stable ones (modulus below the
    default qz_criterium) and the sorted moduli in [lo, hi]. The eigenvalues of a model with long lead/lag chains that are
    infinite or zero in exact arithmetic come out as rounding noise of the Jordan blocks, which differs between programs and
    solver calls (AIM/fs2000x10L9_L: Dynare 0.011 and 73..140, pydyn 0.0086 and 1e22..1e48, for a nilpotent block of order 10:
    noise ~ eps^(1/10)); the cluster values are not comparable, their number is."""
    a = np.sort(np.abs(np.asarray(x).ravel()))
    return np.concatenate([[float(a.size), float(np.sum(a < 1 + 1e-6))], a[(a >= lo) & (a <= hi)]])


@field("dr.eigval", STOCH + ("ramsey_model", "ramsey_policy"), rtol=1e-6)
def f_eig(c):
    other_solver(c, eig=True)
    return finite_moduli(c.sol1().eigenvalues)      # the infinite generalised eigenvalues are numerical noise in both programs


FIELDS["dr.eigval"]["refx"] = lambda ref, c: finite_moduli(ref)


def sorted_abs(x):
    return np.sort(np.abs(np.asarray(x).ravel()))


@field("dr.order_var", STOCH + ("ramsey_model", "ramsey_policy"))
def f_order_var(c):
    return c.sol1().order_var + 1


@field("dr.state_var", STOCH + ("ramsey_model", "ramsey_policy"))
def f_state_var(c):
    return c.sol1().state_var + 1


def other_solver(c, eig=False):
    o = c.stoch_opts()
    dr = str(o.get("dr", "")).lower()
    if "partial_information" in o or dr == "aim" or "aim_solver" in o:
        raise Skip("Dynare solved with partial information / AIM: its rules and eigenvalues have another layout")
    if eig and dr in ("cycle_reduction", "logarithmic_reduction"):
        raise Skip("Dynare's eigenvalues are not computed by the QZ solver here")


def moment_guard(c):
    other_solver(c)
    if int(c.stoch_opts().get("periods", 0) or 0) > 0:
        raise Skip("simulated moments (periods>0, random numbers)")
    if c.has("estimation", "ramsey_model", "ramsey_policy", "osr", "discretionary_policy") and not c.has("stoch_simul"):
        raise Skip("no stoch_simul")


@field("mean", STOCH)
def f_mean(c):
    moment_guard(c)
    return c.moments().mean


@field("var", STOCH)
def f_var(c):
    moment_guard(c)
    return c.moments().var


@field("autocorr", STOCH)
def f_autocorr(c):
    moment_guard(c)
    return list(c.moments().autocorr)


@field("gamma_y", STOCH)
def f_gamma_y(c):
    moment_guard(c)
    ar = int(sget(c.opts, "ar", 5) or 5)
    return list(c.moments().gamma_y)[: ar + 1]     # Dynare appends a further cell (variance decomposition) after these


@field("VarianceDecomposition", STOCH)
def f_vd(c):
    moment_guard(c)
    v = c.moments().variance_decomposition
    if v is None:
        raise NotProduced("no variance decomposition")
    return v


@field("variance_decomposition", STOCH)
def f_vd2(c):
    """Dynare 7.2's oo_.variance_decomposition covers all endogenous variables (declaration order), not the var_list."""
    moment_guard(c)
    from pydyn import moments as mo
    vl = list(c.m.endo_names) if c.m.n_exo == 1 else ([v for v in c.ref_var_list if v in c.m.endo_names] or None)   # one shock: all variables
    mm = mo.theoretical_moments(c.m, c.drk(), var_list=vl, order=c.order())
    if mm.variance_decomposition is None:
        raise NotProduced("no variance decomposition")
    return mm.variance_decomposition


@field("ConditionalVarianceDecomposition", STOCH)
def f_cvd(c):
    moment_guard(c)
    v = c.moments().conditional_variance_decomposition
    if v is None:
        raise NotProduced("no conditional variance decomposition")
    return v


@field("contemporaneous_correlation", STOCH)
def f_corr(c):
    moment_guard(c)
    v = c.moments().corr
    if v is None:
        raise NotProduced("no correlation matrix")
    return v


def irfs_ref(ref, c):
    """oo_.irfs keeps the fields of earlier stoch_simul calls (other var_lists, other parameters): only the current var_list counts."""
    keep = [f for f in fields_of(ref) if not c.ref_var_list or any(f == v + "_" + sh for v in c.ref_var_list for sh in c.m.exo_names)]
    class _S:
        _fieldnames = keep
    o = _S()
    for f in keep:
        setattr(o, f, getattr(ref, f))
    return o


@field("irfs", STOCH)
def f_irfs(c):
    other_solver(c)
    if c.order() > 1:
        raise Skip("IRFs at order >= 2 use random shocks")
    sol = c.sol1()
    nper = int(sget(c.opts, "irf", 40) or 40)
    Sig = np.asarray(c.m.Sigma_e, float)
    ref = sget(c.oo, "irfs")
    out = {}
    for fn in fields_of(ref):
        for j, sh in enumerate(c.m.exo_names):
            if fn.endswith("_" + sh):
                v = fn[: -len(sh) - 1]
                if c.ref_var_list and v not in c.ref_var_list:
                    break           # oo_.irfs keeps the fields of earlier stoch_simul statements (other parameters, other variables)
                if v in c.m.endo_names:
                    if "impulse" not in c.cache:
                        c.cache["impulse"] = {}
                    key = j
                    if key not in c.cache["impulse"]:
                        from pydyn.moments import get_lower_cholesky_covariance
                        shock = np.asarray(get_lower_cholesky_covariance(Sig), float)[:, j]   # Dynare's irf.m: column of the Cholesky factor
                        c.cache["impulse"][key] = sol.irf(shock, nper)
                    out[fn] = c.cache["impulse"][key][:, c.m.endo_names.index(v)]
                    break
    return out


FIELDS["irfs"]["refx"] = irfs_ref


@field("endo_simul", ("perfect_foresight_solver", "perfect_foresight_with_expectation_errors_solver"), rtol=RTOL_PF, rtol_soft=1e-4)
def f_endo_simul(c):
    st = sget(c.oo, "deterministic_simulation.status")
    if st is not None and np.size(st) and not bool(np.all(st)):
        raise Skip("Dynare's perfect-foresight run did not converge")
    if getattr(getattr(c.m, "_matlab", None), "path_edits", False):
        # MATLAB code writes oo_.endo_simul / oo_.exo_simul / oo_.steady_state and may solve again: Dynare's final oo_.endo_simul is
        # that of its last solve, not of the .mod file's solver statement. pydyn does not run those assignments and says so in a
        # warning ("MATLAB code in the .mod file writes simulation paths or results ...").
        raise Skip("MATLAB code in the file writes oo_ simulation paths and solves again: Dynare's final oo_.endo_simul is the last solve's "
                   "(pydyn does not run such assignments and warns)")
    pf = c.pf_result()
    if not pf.status:
        raise RuntimeError(f"pydyn's perfect-foresight solver did not converge where Dynare's did (homotopy completion share "
                           f"{pf.homotopy_completion_share:.3g}, residual {pf.max_residual:.2e})")
    return pf.endo.T



# ---- occbin simulation (occbin_setup / occbin_solver)
@field("occbin.simul", ("occbin_solver",))
def f_occbin_simul(c):
    ob = c.pydyn.Occbin(c.m)
    res = ob.run()
    out = {"piecewise": res.piecewise, "linear": res.linear}
    rh = sget(c.oo, "occbin.simul.regime_history")
    if nonempty(rh):
        out["__regime_history_len"] = len(res.regime_history)
    return out


# ---- estimation: pydyn's Estimation evaluated at Dynare's mode / final parameters
def _estimation(c):
    if "est" not in c.cache:
        if c.has("occbin_setup", "occbin_constraints"):
            raise NotProduced("occbin estimation is not evaluated by the sweep")
        c.cache["est"] = c.pydyn.estimation.Estimation(c.m)
    return c.cache["est"]


def _theta_from_M(c, est):
    """Estimated-parameter vector implied by Dynare's final M_ (params, Sigma_e, H)."""
    M = c.M
    Sig = np.atleast_2d(np.asarray(M.Sigma_e, float))
    H = np.atleast_2d(np.asarray(sget(M, "H", np.zeros((0, 0))), float))
    ref_params = c.ref_params
    out = []
    ix = {v: i for i, v in enumerate(c.m.exo_names)}
    for name in est.names:
        if name.startswith("stderr "):
            v = name[7:]
            if v in ix:
                out.append(np.sqrt(Sig[ix[v], ix[v]]))
            else:
                j = est.varobs.index(v)
                out.append(np.sqrt(H[j, j]) if H.size > 1 else np.sqrt(float(H)))
        elif name.startswith("corr "):
            a, b = [t.strip() for t in name[5:].split(",")]
            if a in ix:
                i, j = ix[a], ix[b]
                out.append(Sig[i, j] / np.sqrt(Sig[i, i] * Sig[j, j]))
            else:
                i, j = est.varobs.index(a), est.varobs.index(b)
                out.append(H[i, j] / np.sqrt(H[i, i] * H[j, j]))
        else:
            out.append(ref_params[c.m._ip[name]])
    return np.array(out, float)


@field("posterior.optimization", ("estimation",), rtol=1e-8, atol=1e-8)
def f_post_opt(c):
    est = _estimation(c)
    mode = sget(c.oo, "posterior.optimization.mode")
    if not nonempty(mode):
        raise Skip("no posterior mode stored")
    theta = np.asarray(mode, float).ravel()
    if theta.size != len(est.names):
        raise NotProduced(f"mode has {theta.size} entries, pydyn's estimated parameters {len(est.names)}")
    out = {}
    if nonempty(sget(c.oo, "posterior.optimization.log_density")):
        out["log_density"] = np.asarray(est.log_posterior(theta))
    else:
        raise Skip("Dynare stored no log_density")
    return out


# ---- smoother (calib_smoother, or estimation(smoother) evaluated at Dynare's final parameters)
def _smooth(c):
    if "smooth" not in c.cache:
        if c.has("calib_smoother") and not c.has("estimation"):
            c.sol1()
            res = c.pydyn.smoother.smooth(c.m, sol=c.sol1())
        elif c.has("estimation"):
            est = _estimation(c)
            theta = _theta_from_M(c, est)
            res = c.pydyn.smoother.smooth(c.m, estimation=est, theta=theta)
        else:
            c.sol1()
            res = c.pydyn.smoother.smooth(c.m, sol=c.sol1())
        c.cache["smooth"] = res
    return c.cache["smooth"]


for _name in ("SmoothedVariables", "UpdatedVariables", "FilteredVariables", "SmoothedShocks", "SmoothedMeasurementErrors"):
    def _mk2(nm):
        @field(nm, ("calib_smoother", "estimation"), rtol=1e-7)
        def f(c):
            d = getattr(_smooth(c), nm)
            return {k: np.asarray(v, float) for k, v in d.items()}
        return f
    _mk2(_name)


@field("Smoother", ("calib_smoother", "estimation"), rtol=1e-7)
def f_smoother_struct(c):
    r = _smooth(c)
    return {"SteadyState": np.asarray(r.SteadyState, float), "Trend": {k: np.asarray(v, float) for k, v in r.Trend.items()} if r.Trend else {},
            "Constant": {k: np.asarray(v, float) for k, v in r.Constant.items()},
            "TrendCoeffs": np.asarray(r.TrendCoeffs, float) if r.TrendCoeffs is not None else np.zeros(c.m.n_endo)}


IGNORE_KEYS = {"occbin", "loglinear", "__regime_history_len", "shocks_sequence", "exo_pos", "error_flag", "regime_history", "ys", "realtime_regime_history", "regime_history_iter", "warning_flag", "mode", "Variance", "HessianMatrix", "hessian"}

# fields of oo_ that are bookkeeping, not results
IGNORE = {"dynare_version", "gui", "exo_det_steady_state", "initial_exo_steady_state", "initial_steady_state", "initval_series",
          "exo_det_simul", "exo_simul", "time", "dr", "irfs", "deterministic_simulation", "endo_simul", "mean", "var", "autocorr",
          "gamma_y", "steady_state", "exo_steady_state", "VarianceDecomposition", "ConditionalVarianceDecomposition",
          "contemporaneous_correlation", "skewness", "kurtosis", "dr_eigval", "endo_simul_orig", "pfm"}

# ------------------------------------------------------------------------------------------ evaluation of one file
REFUSAL_TYPES = ("Unsupported", "NotImplementedError", "ModelError", "ParseError", "TranslationError", "OptimalPolicyError",
                 "DiscretionaryPolicyError", "OSRError", "SamplerError", "NonlinearFilterError", "SmootherError", "EstimationError",
                 "KOrderError", "LoglinearError")
REFUSAL_WORDS = re.compile(r"only has derivatives|no data:|not in the data|not supported|unsupported|not implemented|refus|not handled|cannot handle|is not available|beyond order", re.I)


def is_refusal(e):
    n = type(e).__name__
    if n in ("Unsupported", "NotImplementedError", "ParseError", "TranslationError"):
        return True
    if n == "ModelError":
        return True
    if n in ("SmootherError", "EstimationError", "OptimalPolicyError", "OSRError", "DiscretionaryPolicyError",
             "SamplerError", "NonlinearFilterError", "KOrderError", "OccbinError", "ValueError", "RuntimeError", "LoglinearError"):
        return bool(REFUSAL_WORDS.search(str(e)))
    return False


def compare_value(path, val, ref, spec):
    """List of (name, status, detail) rows for one field."""
    rows = []
    rtol, atol = spec["rtol"], spec["atol"]
    if is_struct(ref):
        keys = fields_of(ref)
        if not isinstance(val, dict):
            raise TypeError("producer must return a dict for a struct field")
        for k in keys:
            rv = getattr(ref, k)
            if k not in val:
                if k not in IGNORE_KEYS and nonempty(rv):
                    rows.append((f"{path}.{k}", "not-produced", "no value from pydyn"))
                continue
            rows.extend(compare_value(f"{path}.{k}", val[k], rv, spec))
        return rows
    if isinstance(ref, np.ndarray) and ref.dtype == object or isinstance(val, list):
        rl = list(np.atleast_1d(ref).ravel()) if isinstance(ref, np.ndarray) else [ref]
        vl = list(val)
        n = min(len(rl), len(vl))
        if path == "gamma_y":
            rl = rl[: len(vl)]
        if len(vl) > len(rl) or len(vl) < len(rl):
            rows.append((path, "wrong", f"{len(vl)} cells vs Dynare {len(rl)}"))   # Dynare's gamma_y has one extra cell (variance decomposition)
        worst = (True, 0.0, "")
        for i in range(n):
            if np.size(rl[i]) == 0:
                continue
            ok, mx, msg = compare(vl[i], rl[i], rtol, atol)
            if not ok and (worst[0] or mx > worst[1]):
                worst = (False, mx, f"cell {i + 1}: {msg or f'max abs diff {mx:.2e}'}")
            elif ok and worst[0]:
                worst = (True, max(worst[1], mx), "")
        if rows:
            return rows
        rows.append((path, "match" if worst[0] else "wrong", f"max abs diff {worst[1]:.2e}" if worst[0] else worst[2]))
        return rows
    ok, mx, msg = compare(val, ref, rtol, atol)
    if not ok and not msg and spec.get("rtol_soft"):
        ok2, mx2, _ = compare(val, ref, spec["rtol_soft"], atol)
        if ok2:
            rows.append((path, "match", f"max abs diff {mx2:.2e} (above {rtol:g} but within the solver tolerance {spec['rtol_soft']:g})"))
            return rows
    rows.append((path, "match" if ok else "wrong", f"max abs diff {mx:.2e}" if (ok or not msg) else msg))
    return rows


def evaluate(mod, entry):
    out = {"file": relid(mod), "rows": [], "notes": [], "kinds": []}
    ref_json = json.loads((REF / (relid(mod) + ".json")).read_text())
    out["dynare_status"] = ref_json["status"]
    if ref_json["status"] not in ("ok",):
        out["verdict"] = {"xfail": "xfail", "xfail-passed": "xfail", "m-runner": "no-comparison"}.get(ref_json["status"], "dynare-failed")
        out["detail"] = ref_json.get("message", "") or ref_json["status"]
        return out
    oo, M, opts = load_ref(REF / (relid(mod) + ".mat"))
    scratch = Path(tempfile.mkdtemp(prefix="accsw_", dir="/var/tmp"))
    try:
        prepare_scratch(entry, SRC, scratch)
        for earlier in [f for f in entry["test"] if f.endswith(".mod")]:
            if earlier == mod:
                break
            copy_artifacts(earlier, scratch)               # files Dynare's run of the earlier file wrote (simulated data ...)
            try:                                       # files written by an earlier file of the same entry (Dynare's driver shares the folder)
                import pydyn as _p
                _p.run_preprocessor(scratch / "tests" / earlier, extra_args=tuple(entry.get("dynare_args", [])))
            except Exception:  # noqa: BLE001
                pass
        copy_artifacts(mod, scratch)       # data a file simulates and then reads itself (randn in Octave: pydyn has Dynare's data)
        c = Ctx(mod.replace("../", ""), entry, scratch, oo, M, opts)
        c.mod = mod
        try:
            c.load()
        except Exception as e:  # noqa: BLE001
            out["verdict"] = "refused" if is_refusal(e) else "pydyn-error"
            out["detail"] = f"load: {type(e).__name__}: {str(e)[:300]}"
            out["trace"] = traceback.format_exc()[-1500:]
            return out
        out["kinds"] = sorted(c.kindset)
        rows = out["rows"]
        # calibration check (files that do not re-estimate parameters)
        if c.ref_params.size == c.calibration_params.size and not c.has("estimation", "osr", "method_of_moments", "identification",
                                                                         "dynare_sensitivity", "ramsey_model", "ramsey_policy"):
            ok, mx, msg = compare(c.calibration_params, c.ref_params, 1e-12, 1e-14)
            rows.append(("M_.params", "match" if ok else "wrong", "calibration after the .mod file's code" if ok else (msg or f"max abs diff {mx:.2e}")))
        if c.has("stoch_simul") and c.ref_var_list and "periods" not in c.stoch_opts():
            try:                                       # pydyn's default variable list against the last stoch_simul statement's
                from pydyn import moments as _mo
                c.sol1()
                got = list(_mo.theoretical_moments(c.m, c.drk(), order=c.order()).var_names)
                rows.append(("var_list", "match" if got == c.ref_var_list else "wrong",
                             "default variable list" if got == c.ref_var_list else f"pydyn's default {got} vs Dynare's {c.ref_var_list}"))
            except Skip as e:
                rows.append(("var_list", "skipped", str(e)))
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ in ("BlanchardKahnError", "SteadyStateError") and c.dynare_unsolved():
                    rows.append(("var_list", "match", f"Dynare did not solve this model either: pydyn raises {type(e).__name__}"))
                else:
                    rows.append(("var_list", "error" if not is_refusal(e) else "refused", f"{type(e).__name__}: {str(e)[:150]}"))
        for path, spec in FIELDS.items():
            ref = sget(oo, path)
            if not nonempty(ref):
                continue
            if spec["needs"] and not c.has(*spec["needs"]):
                continue
            if spec["when"] is not None and not spec["when"](c):
                continue
            try:
                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        val = spec["get"](c)
                except KeyError as ke:
                    if ke.args != ("__import__",):
                        raise
                    # pydyn bug: a numpy RuntimeWarning raised inside the generated model code (builtins removed) fails with
                    # KeyError('__import__') whenever warnings are not ignored; rerun with RuntimeWarning ignored
                    c.notes.append(f"PYDYN BUG: KeyError('__import__') from a RuntimeWarning in generated model code ({path})")
                    c.cache.clear()
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        warnings.simplefilter("ignore", RuntimeWarning)
                        val = spec["get"](c)
                c.warn_notes(w)
                rows.extend(compare_value(path, val, spec["refx"](ref, c) if spec.get("refx") else ref, spec))
            except Skip as e:
                rows.append((path, "skipped", str(e)))
            except SoftMatch as e:
                rows.append((path, "match", str(e)))
            except DifferentRoot as e:
                rows.append((path, "different-root", str(e)))
            except NotProduced as e:
                rows.append((path, "not-produced", str(e)))
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "SteadyStateError" and c.ref_point_inconsistent():
                    rows.append((path, "skipped", "Dynare's final M_.params do not admit the steady state its estimation used (nonstationary model, "
                                                  "optimizer ended at a point with static residual > 1e-4): the reference cannot be reproduced"))
                    continue
                if type(e).__name__ in ("BlanchardKahnError", "SteadyStateError") and c.dynare_unsolved():
                    rows.append((path, "match", f"Dynare did not solve this model either (its oo_.dr has no decision rules; stoch_simul(noprint) "
                                                f"does not report the failure): pydyn raises {type(e).__name__}"))
                    continue
                st = "refused" if is_refusal(e) else "error"
                rows.append((path, st, f"{type(e).__name__}: {str(e)[:250]}"))
                if st == "error":
                    out.setdefault("traces", {})[path] = traceback.format_exc()[-1200:]
        # oo_ fields nobody covers
        for f in fields_of(oo):
            if f in IGNORE or any(r[0].split(".")[0] == f for r in rows) or f in FIELDS:
                continue
            if nonempty(getattr(oo, f)):
                rows.append((f, "not-produced", "no producer for this oo_ field in the sweep"))
        out["notes"] = c.notes
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    out["verdict"], out["detail"] = verdict(out["rows"])
    return out


def verdict(rows):
    st = collections.Counter(r[1] for r in rows)
    if st["wrong"]:
        bad = [r for r in rows if r[1] == "wrong"]
        return "wrong", "; ".join(f"{r[0]}: {r[2]}" for r in bad[:4])
    if st["error"]:
        bad = [r for r in rows if r[1] == "error"]
        return "pydyn-error", "; ".join(f"{r[0]}: {r[2]}" for r in bad[:3])
    nmatch = st["match"]
    if st["different-root"] and not nmatch:
        return "no-comparison", "other root of the static model: " + "; ".join(r[2] for r in rows if r[1] == "different-root")[:200]
    others = st["refused"] + st["not-produced"]
    if nmatch and others:
        bad = [r for r in rows if r[1] in ("refused", "not-produced")]
        return "partial", "; ".join(f"{r[0]}: {r[2]}" for r in bad[:3])
    if nmatch:
        return "match", ""
    if st["refused"]:
        bad = [r for r in rows if r[1] == "refused"]
        return "refused", "; ".join(f"{r[0]}: {r[2]}" for r in bad[:2])
    if st["not-produced"]:
        bad = [r for r in rows if r[1] == "not-produced"]
        return "refused", "not produced: " + "; ".join(f"{r[0]}" for r in bad[:4])
    return "no-comparison", "; ".join(f"{r[0]}: {r[2]}" for r in rows[:2]) if rows else "no comparable oo_ field"


# ------------------------------------------------------------------------------------------ driver
# ------------------------------------------------------------------------------------------ backend "oo": pydyn.dynare()
OO_SKIP_TOP = IGNORE - {"dr", "irfs", "steady_state", "exo_steady_state", "mean", "var", "autocorr", "gamma_y", "endo_simul",
                        "VarianceDecomposition", "ConditionalVarianceDecomposition", "contemporaneous_correlation"}


def _walk(ref, got, path, rows, tol):
    """Compare Dynare's oo_ struct ``ref`` with pydyn's ``got`` field by field (every non-empty field of ref)."""
    for f in fields_of(ref):
        rv = getattr(ref, f)
        p = f"{path}.{f}" if path else f
        if not nonempty(rv) or (not path and f in OO_SKIP_TOP):
            continue
        try:
            gv = getattr(got, f)
        except Exception as e:  # noqa: BLE001  (ResultsError: field pydyn does not fill)
            kind = "refused" if type(e).__name__ == "ResultsError" else "not-produced"
            rows.append((p, kind, f"{type(e).__name__}: {str(e)[:160]}"))
            continue
        if is_struct(rv):
            _walk(rv, gv, p, rows, tol)
            continue
        try:
            if isinstance(rv, np.ndarray) and rv.dtype == object:
                for i, (a, b) in enumerate(zip(list(gv), list(rv.ravel()))):
                    if np.size(b):
                        ok, mx, msg = compare(a, b, tol, ATOL)
                        if not ok:
                            rows.append((p, "wrong", f"cell {i + 1}: {msg or f'max abs diff {mx:.2e}'}"))
                            break
                else:
                    rows.append((p, "match", ""))
                continue
            if isinstance(rv, str):
                rows.append((p, "match" if str(gv) == rv else "wrong", "")); continue
            ok, mx, msg = compare(gv, rv, tol, ATOL)
            if not ok and not msg and p == "endo_simul" and compare(gv, rv, 1e-4, ATOL)[0]:
                ok, msg = True, ""                     # perfect foresight: Dynare stops at tolf
            rows.append((p, "match" if ok else "wrong", f"max abs diff {mx:.2e}" if (ok or not msg) else msg))
        except Exception as e:  # noqa: BLE001
            rows.append((p, "error", f"{type(e).__name__}: {str(e)[:160]}"))


def evaluate_oo(mod, entry):
    """Run the file with ``pydyn.dynare()`` and compare ``res.oo_`` with Dynare's, every field Dynare stored.  Fields that depend on
    random numbers are compared too (the layer replays Octave's generator): a difference there is reported, with that note."""
    out = {"file": relid(mod), "rows": [], "notes": [], "kinds": [], "backend": "oo"}
    ref_json = json.loads((REF / (relid(mod) + ".json")).read_text())
    out["dynare_status"] = ref_json["status"]
    if ref_json["status"] != "ok":
        out["verdict"] = {"xfail": "xfail", "xfail-passed": "xfail", "m-runner": "no-comparison"}.get(ref_json["status"], "dynare-failed")
        out["detail"] = ref_json.get("message", "") or ref_json["status"]
        return out
    oo, M, opts = load_ref(REF / (relid(mod) + ".mat"))
    scratch = Path(tempfile.mkdtemp(prefix="accoo_", dir="/var/tmp"))
    try:
        prepare_scratch(entry, SRC, scratch)
        import pydyn
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = pydyn.dynare(scratch / "tests" / mod)
        except Exception as e:  # noqa: BLE001
            out["verdict"] = "refused" if (type(e).__name__ == "ResultsError" or is_refusal(e)) else "pydyn-error"
            out["detail"] = f"{type(e).__name__}: {str(e)[:300]}"
            return out
        rows = out["rows"]
        _walk(oo, res.oo_, "", rows, 1e-8)
        for r in rows:
            if r[0].split(".")[0] in ("endo_simul", "exo_simul", "irfs") and r[1] == "wrong":
                out["notes"].append(f"{r[0]} depends on random numbers")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    out["verdict"], out["detail"] = verdict(out["rows"])
    return out


def copy_artifacts(mod, scratch):
    """Put the files Dynare's (Octave) run of ``mod`` wrote next to it (and in subfolders) into the scratch tests folder."""
    art = REF / "_files" / relid(mod)
    if not art.is_dir():
        return
    dest = scratch / "tests" / Path(mod).parent
    for f in art.rglob("*"):
        if f.is_file():
            t = dest / f.relative_to(art)
            t.parent.mkdir(parents=True, exist_ok=True)
            if not t.exists() or f.stat().st_size and t.stat().st_mtime < f.stat().st_mtime:
                shutil.copy(f, t)


def entries():
    seen = {}
    for e in load_manifest(SRC):
        for f in e["test"]:
            if f.endswith(".mod"):
                seen[f] = e
    return seen


BACKEND = os.environ.get("PYDYN_ACCEPT_BACKEND", "api")      # "api": the producers above; "oo": pydyn.dynare() and res.oo_


def run_one_to_file(mod, entry, timeout):
    res = REF / ("_pydyn" if BACKEND == "api" else "_pydyn_oo") / (relid(mod) + ".json")
    res.parent.mkdir(parents=True, exist_ok=True)

    def alarm(*_):
        raise TimeoutError("pydyn timeout")
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(timeout)
    t = time.time()
    try:
        out = evaluate(mod, entry) if BACKEND == "api" else evaluate_oo(mod, entry)
    except TimeoutError:
        out = {"file": relid(mod), "verdict": "pydyn-error", "detail": f"timeout after {timeout}s", "rows": []}
    except Exception as e:  # noqa: BLE001
        out = {"file": relid(mod), "verdict": "pydyn-error", "detail": f"sweep: {type(e).__name__}: {str(e)[:300]}",
               "trace": traceback.format_exc()[-1500:], "rows": []}
    finally:
        signal.alarm(0)
    out["seconds"] = round(time.time() - t, 1)
    res.write_text(json.dumps(out, default=str, indent=1))
    return out


def cmd_run(a):
    ents = entries()
    todo = []
    for mod, e in ents.items():
        if a.only and not any(o in mod for o in a.only):
            continue
        res = REF / ("_pydyn" if BACKEND == "api" else "_pydyn_oo") / (relid(mod) + ".json")
        if res.exists() and not a.force:
            continue
        if not (REF / (relid(mod) + ".json")).exists():
            continue                                   # Octave has not got to it yet
        todo.append(mod)
    print(len(todo), "files to run", flush=True)
    procs = []
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    for mod in todo:
        while len([p for p in procs if p.poll() is None]) >= a.jobs:
            time.sleep(0.5)
        cmd = [sys.executable, str(Path(__file__)), "child", mod, str(a.timeout)]
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs[-1].mod = mod
    for p in procs:
        try:
            p.wait(a.timeout + 120)
        except subprocess.TimeoutExpired:
            p.kill()
    print("done", flush=True)


def cmd_one(mod, timeout=600):
    e = entries()[mod]
    out = run_one_to_file(mod, e, timeout)
    print(json.dumps({k: v for k, v in out.items() if k != "trace"}, indent=1, default=str))
    for k, v in out.get("traces", {}).items():
        print(k, v)
    if out.get("trace"):
        print(out["trace"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "one", "child", "report"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("arg2", nargs="?")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--backend", choices=["api", "oo"], help="api (default): producers calling pydyn's functions; oo: pydyn.dynare() + res.oo_")
    a = ap.parse_args()
    if a.backend:
        BACKEND = a.backend
        os.environ["PYDYN_ACCEPT_BACKEND"] = a.backend
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "one":
        cmd_one(a.arg)
    elif a.cmd == "child":
        run_one_to_file(a.arg, entries()[a.arg], int(a.arg2))
    elif a.cmd == "report":
        from accept_report import report
        report(REF, ROOT / "docs" / "ACCEPTANCE.md")
