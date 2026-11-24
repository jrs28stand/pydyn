# This file is part of pydyn.
#
# It follows, and in places is a Python port of, matlab/estimation/{dsge_likelihood, dynare_estimation_init,
# dynare_estimation_1, set_prior, initial_estimation_checks, posterior_sampler_core, posterior_sampler_iteration,
# posterior_sampler_initialization, set_mcmc_prior_bounds, check_bounds_and_definiteness_estimation}.m,
# matlab/optimization/{hessian, dynare_minimize_objective, penalty_objective_function}.m, matlab/ispd.m and the
# data-set utilities from Dynare 7.2 (Copyright (C) 2003-2026 Dynare Team, https://www.dynare.org), modified in 2026.
# The csminwel optimiser is a port of matlab/optimization/{csminwel1, csminit1, bfgsi1, numgrad2, numgrad3, numgrad5,
# numgrad3_, numgrad5_, step_length_correction}.m of Dynare 7.2 (Copyright (C) 1993-2009 Christopher Sims,
# Copyright (C) 2006-2025 Dynare Team; GPL-3.0-or-later; original: http://sims.princeton.edu/yftp/optimize/).
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Likelihood-based estimation of first-order DSGE models, following Dynare 7.2.

Typical use::

    m = pydyn.Model(pydyn.run_preprocessor("model.mod"))
    est = estimation.Estimation(m)                # options and data file taken from the .mod file
    est.names, est.x0                             # ordering of Dynare's bayestopt_.name / xparam1
    est.log_likelihood(theta); est.log_prior(theta); est.log_posterior(theta)
    mode = est.mode()                             # posterior mode, Hessian, Laplace approximation
    mc = est.sample(replic=20000, mode=mode)      # random-walk Metropolis-Hastings
    est.posterior_statistics(mc); est.modified_harmonic_mean(mc); est.convergence_diagnostics(mc)
    est.posterior_irf(mc)                         # see pydyn.posterior and pydyn.posterior_plots
    est.calibration(theta)                        # {"params", "Sigma_e", "H"} implied by a parameter vector

The parameter vector is ordered as in Dynare: standard deviations of structural shocks,
standard deviations of measurement errors, correlations of structural shocks, correlations of
measurement errors, structural parameters; within each group in the order of the
``estimated_params`` block.

The model object is never modified by an evaluation: every call works on copies of the
parameter vector and covariance matrices, recomputes the steady state, and uses the parameters
returned by the steady state (which may have been updated by a ``steady_state_model`` block).

Implemented, with Dynare's defaults: ``datafile`` (``.m``, ``.mat``, ``.csv``), ``first_obs``,
``nobs``, ``loglinear``, ``logdata``, ``prefilter``, ``presample``, ``observation_trends``,
missing observations (NaN), ``lik_init`` 1 (stationary), 2 (large diagonal covariance) and 3
(``diffuse_filter``), ``kalman_algo`` 0-4 (multivariate and univariate filters, including the
automatic switch to the univariate filter when the forecast-error covariance is singular),
``kalman_tol``, ``diffuse_kalman_tol``, ``riccati_tol``, ``qz_criterium``, ``prior_trunc``.
Priors: beta, gamma, normal, inverse gamma 1 and 2, uniform, Weibull; none at all means maximum
likelihood.

Mode (``Estimation.mode``): ``mode_compute`` 4 (csminwel, ported), 5 (newrat, ported), 1, 3, 7, 8, 9 (scipy equivalents),
0 with ``mode_file``; ``optim``, ``additional_optimizer_steps``, ``cova_compute``,
``use_penalized_objective_for_hessian``. When the ``.mod`` file gives no ``mode_compute`` (Dynare's
default is 5, newrat) csminwel is used: see ``mode`` for the reasons.

Sampler (``Estimation.sample``): random-walk Metropolis-Hastings with ``mh_replic``,
``mh_nblocks``, ``mh_jscale`` (and the per-parameter ``jscale``), ``mh_drop``,
``mh_init_scale_factor``, ``mcmc_jumping_covariance``, ``mh_tune_jscale``, ``mh_tune_guess``,
saving and continuing chains (the counterpart of ``load_mh_file``).
Other samplers (slice, TaRB, independent Metropolis-Hastings, the SMC samplers hssmc and dsmh), the
``posterior_sampler_options`` and Dynare's random numbers (``rng='octave'``): :mod:`pydyn.samplers`.

Options used by :mod:`pydyn.posterior` and :mod:`pydyn.posterior_plots` when they are given an
``Estimation``: ``mh_conf_sig``, ``sub_draws``, ``posterior_max_subsample_draws``, ``taper_steps``,
``geweke_interval``, ``raftery_lewis_diagnostics``, ``raftery_lewis_qrs``,
``brooks_gelman_plotrows``, ``bayesian_irf``, ``irf``, ``relative_irf``, ``irf_shocks``,
``mode_check`` and its three settings, ``huge_number``; by :mod:`pydyn.posterior_moments`: ``moments_varendo``, ``ar``,
``nodecomposition``, ``conditional_variance_decomposition``, ``contemporaneous_correlation``, the ``consider_*``
options, ``smoother``, ``filtered_vars``, ``filter_step_ahead``. As everywhere in pydyn a flag such as
``bayesian_irf`` or ``mode_check`` does not make ``Estimation`` compute anything by itself: the
corresponding function does, with the settings of the ``.mod`` file as defaults.

Every other option of the ``estimation`` command is listed in ``unsupported_options`` and reported
once in an ``UnsupportedOptionWarning`` (display-only options excepted).

Not implemented: orders above one, ``lik_init`` 4 and 5, the univariate filters with correlated
measurement errors, calibrated measurement-error covariances, ``estimated_params_bounds``,
endogenous priors, prior restrictions,
``noconstant`` (which Dynare 7.2 itself resets), recursive estimation, the other samplers
(DIME, online), ``mode_compute`` 2, 6, 10 and above, ``mh_recover``,
``load_mh_file`` as a ``.mod`` option (use ``sample(load=...)``), forecasts, posterior distributions of
``filter_covariance`` and ``smoothed_state_uncertainty`` and the smoother for non-stationary models.
"""
from __future__ import annotations

import csv
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import linalg, optimize

from . import kalman
from .model import ModelError
from .perturbation import BlanchardKahnError, dr_ordering, solve_linear_re
from .priors import Prior, PriorError
from .steady import SteadyStateError, steady_state

__all__ = ["Estimation", "EstimationError", "ModeResult", "MCMCResult", "load_data", "read_m_data",
           "hessian", "metropolis_hastings", "csminwel", "Prior", "PriorError"]


class EstimationError(Exception):
    pass


# ====================================================================== data files
class _MFile:
    """Evaluator for the subset of MATLAB found in Dynare data files: matrix literals and simple
    array expressions (``diff``, ``log``, indexing, element-wise arithmetic). Values are 2-D arrays."""

    _TOKEN = re.compile(r"""
        (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?)
      | (?P<id>[A-Za-z_]\w*)
      | (?P<op>\.\*|\./|\.\^|\.'|==|~=|<=|>=|[-+*/^'()\[\],;:=<>])
      | (?P<nl>\n)
      | (?P<ws>[ \t\r]+)
    """, re.X)

    def __init__(self, text):
        text = re.sub(r"\.\.\.[^\n]*\n", " ", text)
        self.tokens = self._tokenise(text)
        self.pos = 0
        self.vars: dict[str, np.ndarray] = {}
        self._end_stack: list = []

    @staticmethod
    def _strip_comment(line):
        out, prev = [], ""
        for ch in line:
            if ch in "%#":
                break
            if ch == "'" and not (prev.isalnum() or prev in "_)].'"):
                raise EstimationError("character strings are not supported in data files")
            out.append(ch)
            if not ch.isspace():
                prev = ch
        return "".join(out)

    def _tokenise(self, text):
        text = "\n".join(self._strip_comment(line) for line in text.split("\n"))
        toks, depth, i, n = [], [], 0, len(text)      # depth: stack of '[' and '('
        operand_end = lambda: bool(toks) and (toks[-1][0] in ("num", "id") or toks[-1][1] in (")", "]", "'", ".'"))  # noqa: E731
        while i < n:
            mt = self._TOKEN.match(text, i)
            if not mt:
                raise EstimationError(f"cannot read data file near {text[i:i + 30]!r}")
            kind, val = mt.lastgroup, mt.group()
            i = mt.end()
            in_matrix = bool(depth) and depth[-1] == "["
            if kind == "ws":
                if in_matrix and operand_end() and i < n:
                    nxt = text[i]
                    if nxt.isalnum() or nxt in "._([" or (nxt in "+-" and i + 1 < n and not text[i + 1].isspace()):
                        toks.append(("sep", ","))
                continue
            if kind == "nl":
                if in_matrix:
                    if toks and toks[-1][1] not in ("[", ";"):
                        toks.append(("op", ";"))
                elif not depth:
                    toks.append(("end", "\n"))
                continue
            if kind == "op":
                if val in "[(":
                    depth.append(val)
                elif val in "])":
                    if not depth:
                        raise EstimationError("unbalanced brackets in data file")
                    depth.pop()
                    if val == "]" and toks and toks[-1] == ("op", ";"):
                        toks.pop()
                elif val == ";" and not depth:
                    toks.append(("end", ";"))
                    continue
                elif val == "," and in_matrix and toks and toks[-1][0] == "sep":
                    toks.pop()
            toks.append((kind, val))
        toks.append(("end", "\n"))
        return toks

    # -- token helpers
    def _peek(self):
        return self.tokens[self.pos]

    def _next(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _accept(self, val):
        if self._peek()[1] == val and self._peek()[0] in ("op", "sep"):
            self.pos += 1
            return True
        return False

    def _expect(self, val):
        if not self._accept(val):
            raise EstimationError(f"data file: expected {val!r}, found {self._peek()[1]!r}")

    # -- statements
    def run(self):
        while self.pos < len(self.tokens):
            if self._peek()[0] == "end":
                self.pos += 1
                continue
            kind, name = self._next()
            if kind != "id" or not self._accept("="):
                raise EstimationError(f"data file: only assignments 'name = expression' are supported (near {name!r})")
            self.vars[name] = self._expr()
            if self._peek()[0] != "end":
                raise EstimationError(f"data file: unexpected {self._peek()[1]!r} after the assignment to {name}")
        return self.vars

    # -- expressions (MATLAB precedence)
    def _expr(self):
        left = self._additive()
        if self._peek() == ("op", ":"):
            self.pos += 1
            mid = self._additive()
            if self._accept(":"):
                stop = self._additive()
                return np.arange(left.item(), stop.item() + mid.item() * 1e-10, mid.item())[None, :]
            return np.arange(left.item(), mid.item() + 1e-10)[None, :]
        return left

    def _additive(self):
        v = self._term()
        while self._peek() in (("op", "+"), ("op", "-")):
            op = self._next()[1]
            w = self._term()
            v = v + w if op == "+" else v - w
        return v

    def _term(self):
        v = self._unary()
        while self._peek()[0] == "op" and self._peek()[1] in ("*", "/", ".*", "./"):
            op = self._next()[1]
            w = self._unary()
            if op == ".*" or (op == "*" and (v.size == 1 or w.size == 1)):
                v = v * w
            elif op == "*":
                v = v @ w
            elif op == "./" or w.size == 1:
                v = v / w
            else:
                raise EstimationError("data file: matrix right division is not supported")
        return v

    def _unary(self):
        if self._accept("-"):
            return -self._unary()
        if self._accept("+"):
            return self._unary()
        return self._power()

    def _power(self):
        v = self._postfix()
        while self._peek()[0] == "op" and self._peek()[1] in ("^", ".^"):
            op = self._next()[1]
            neg = self._accept("-")
            w = self._postfix()
            w = -w if neg else w
            if op == "^" and v.size != 1 and w.size != 1:
                raise EstimationError("data file: matrix power is not supported")
            v = v ** w
        return v

    def _postfix(self):
        v = self._primary()
        while self._peek()[0] == "op" and self._peek()[1] in ("'", ".'"):
            self.pos += 1
            v = v.T
        return v

    def _args(self, shape=None):
        args = []
        if self._accept(")"):
            return args
        while True:
            self._end_stack.append((shape, len(args)))
            if self._peek() == ("op", ":") and self.tokens[self.pos + 1][1] in (",", ")"):
                self.pos += 1
                args.append(slice(None))
            else:
                args.append(self._expr())
            self._end_stack.pop()
            if self._accept(")"):
                return args
            self._expect(",")

    _FUNCS = {"log": np.log, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs, "log10": np.log10,
              "transpose": np.transpose}

    def _primary(self):
        kind, val = self._next()
        if kind == "num":
            return np.array([[float(val.replace("d", "e").replace("D", "e"))]])
        if kind == "op" and val == "(":
            v = self._expr()
            self._expect(")")
            return v
        if kind == "op" and val == "[":
            return self._matrix()
        if kind != "id":
            raise EstimationError(f"data file: unexpected {val!r}")
        if val == "end" and self._end_stack:
            shape, k = self._end_stack[-1]
            if shape is None:
                raise EstimationError("data file: 'end' outside an index")
            return np.array([[float(shape)]])           # size along the dimension being indexed
        if val in self.vars:
            x = self.vars[val]
            if self._accept("("):
                return self._index(x)
            return x
        if val in ("NaN", "nan", "Inf", "inf", "pi"):
            const = {"nan": np.nan, "inf": np.inf, "pi": np.pi}[val.lower()]
            if self._accept("("):
                a = [int(v.item()) for v in self._args()]
                return np.full((a[0], a[-1]) if a else (1, 1), const)
            return np.array([[const]])
        if not self._accept("("):
            raise EstimationError(f"data file: unknown name {val!r}")
        args = self._args()
        return self._call(val, args)

    def _index(self, x):
        # 'end' needs the size along the indexed dimension: parse arguments one at a time
        args, k = [], 0
        start = self.pos
        nargs = self._count_args(start)
        while True:
            dim = x.size if nargs == 1 else x.shape[k]
            self._end_stack.append((dim, k))
            if self._peek() == ("op", ":") and self.tokens[self.pos + 1][1] in (",", ")"):
                self.pos += 1
                args.append(slice(None))
            else:
                args.append(self._expr())
            self._end_stack.pop()
            k += 1
            if self._accept(")"):
                break
            self._expect(",")
        conv = lambda a: a if isinstance(a, slice) else np.asarray(a, float).ravel().astype(int) - 1  # noqa: E731
        if len(args) == 1:
            flat = x.ravel(order="F")[conv(args[0])]
            if min(x.shape) == 1:                 # a vector keeps its orientation
                return flat[None, :] if x.shape[0] == 1 else flat[:, None]
            if isinstance(args[0], slice) or np.shape(args[0])[0] != 1:
                return flat[:, None]              # a(:) and column indices give a column
            return flat[None, :]
        if len(args) == 2:
            r, c = conv(args[0]), conv(args[1])
            r = np.arange(x.shape[0])[r] if isinstance(r, slice) else r
            c = np.arange(x.shape[1])[c] if isinstance(c, slice) else c
            return x[np.ix_(r, c)]
        raise EstimationError("data file: arrays with more than two dimensions are not supported")

    def _count_args(self, pos):
        depth, n = 0, 1
        while True:
            val = self.tokens[pos][1]
            kind = self.tokens[pos][0]
            if kind == "end":
                raise EstimationError("data file: unbalanced parentheses")
            if val in ("(", "["):
                depth += 1
            elif val in (")", "]"):
                if depth == 0:
                    return n
                depth -= 1
            elif val == "," and depth == 0:
                n += 1
            pos += 1

    def _call(self, name, args):
        def first_dim(x):
            return 1 if x.shape[0] == 1 else 0

        if name in self._FUNCS and len(args) == 1:
            with np.errstate(all="ignore"):
                return self._FUNCS[name](args[0])
        x = args[0] if args else None
        if name == "diff" and len(args) == 1:
            return np.diff(x, axis=first_dim(x))
        if name in ("cumsum", "cumprod") and len(args) == 1:
            return getattr(np, name)(x, axis=first_dim(x))
        if name in ("mean", "sum") and len(args) == 1:
            return getattr(np, name)(x, axis=first_dim(x), keepdims=True)
        if name in ("zeros", "ones") and args:
            a = [int(v.item()) for v in args]
            return getattr(np, name)((a[0], a[-1]))
        if name == "length" and len(args) == 1:
            return np.array([[float(max(x.shape))]])
        if name == "numel" and len(args) == 1:
            return np.array([[float(x.size)]])
        if name == "size" and len(args) == 2:
            return np.array([[float(x.shape[int(args[1].item()) - 1])]])
        raise EstimationError(f"data file: function {name!r} is not supported")

    def _matrix(self):
        rows, row = [], []
        while True:
            if self._accept("]"):
                break
            if self._accept(";"):
                rows.append(row)
                row = []
                continue
            if self._accept(","):
                continue
            row.append(self._expr())
        if row:
            rows.append(row)
        rows = [r for r in rows if r]
        if not rows:
            return np.zeros((0, 0))
        try:
            return np.vstack([np.hstack(r) for r in rows])
        except ValueError as exc:
            raise EstimationError(f"data file: inconsistent matrix dimensions ({exc})") from None


def read_m_data(path) -> dict:
    """Read a MATLAB ``.m`` data file of the kind used with Dynare's ``datafile`` option: a script
    that defines one vector per observed variable, possibly through simple expressions.
    Returns ``{name: 1-D array}`` for every vector the script defines."""
    values = _MFile(Path(path).read_text()).run()
    return {k: np.asarray(v, float).ravel(order="F") for k, v in values.items()
            if v.ndim == 2 and min(v.shape) == 1}


def _read_csv(path) -> dict:
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    header, body = [h.strip() for h in rows[0]], rows[1:]

    def num(s):
        s = s.strip()
        if s == "" or s.lower() in ("nan", "na"):
            return np.nan
        return float(s)

    out = {}
    for j, name in enumerate(header):
        try:
            out[name] = np.array([num(r[j]) if j < len(r) else np.nan for r in body])
        except ValueError:
            continue                              # a column of dates or labels
    return out


def load_data(source, varobs, base_dir=None) -> np.ndarray:
    """Return a (periods, len(varobs)) array with the observed variables in ``varobs`` order.

    ``source`` may be a dict ``{name: series}``, an array whose columns follow ``varobs``, or the
    name of a ``.m``, ``.mat`` or ``.csv`` file (extension optional, as in Dynare; relative names
    are looked up in ``base_dir``)."""
    if isinstance(source, dict):
        table = source
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_absolute() and base_dir is not None:
            p = Path(base_dir) / p
        if p.suffix.lower() not in (".m", ".mat", ".csv"):
            found = [q for q in (p.with_name(p.name + ext) for ext in (".m", ".mat", ".csv")) if q.exists()]
            if len(found) != 1:
                raise EstimationError(f"data file {p}: {'no' if not found else 'more than one'} "
                                      "file with extension .m, .mat or .csv")
            p = found[0]
        if not p.exists():
            raise EstimationError(f"data file {p} not found")
        if p.suffix.lower() == ".m":
            table = read_m_data(p)
        elif p.suffix.lower() == ".mat":
            from scipy.io import loadmat
            table = {k: v for k, v in loadmat(p, squeeze_me=True).items() if not k.startswith("__")}
        else:
            table = _read_csv(p)
    else:
        arr = np.asarray(source, float)
        if arr.ndim != 2 or arr.shape[1] != len(varobs):
            raise EstimationError(f"data array must have one column per observed variable ({len(varobs)})")
        return arr.copy()
    missing = [v for v in varobs if v not in table]
    if missing:
        raise EstimationError(f"observed variables not found in the data: {missing}")
    cols = [np.asarray(table[v], float).ravel() for v in varobs]
    if len({len(c) for c in cols}) != 1:
        raise EstimationError("observed variables have different numbers of observations")
    return np.column_stack(cols)


# ====================================================================== numerical Hessian
def hessian(func, x, gstep=(1e-2, 1.0)):
    """Numerical Hessian of a scalar function, the algorithm of Dynare's ``hessian.m``
    (Abramowitz and Stegun 25.3.23 and 25.3.27) with the same step sizes."""
    x = np.asarray(x, float)
    n = x.size
    h1 = np.maximum(np.abs(x), np.sqrt(gstep[0])) * np.finfo(float).eps ** (1 / 6) * gstep[1]
    h_1 = h1.copy()
    h1 = (x + h1) - x
    h_1 = x - (x - h_1)
    f0 = func(x)
    f1, f_1 = np.zeros(n), np.zeros(n)
    for i in range(n):
        xh = x.copy()
        xh[i] = x[i] + h1[i]
        f1[i] = func(xh)
        xh[i] = x[i] - h_1[i]
        f_1[i] = func(xh)
    temp = f1 + f_1 - f0
    hh = np.zeros((n, n))
    for i in range(n):
        hh[i, i] = (f1[i] + f_1[i] - 2 * f0) / (h1[i] * h_1[i])
        for j in range(i + 1, n):
            xp, xm = x.copy(), x.copy()
            xp[i] = x[i] + h1[i]
            xp[j] = x[j] + h_1[j]
            xm[i] = x[i] - h1[i]
            xm[j] = x[j] - h_1[j]
            hh[i, j] = hh[j, i] = -(-func(xp) - func(xm) + temp[i] + temp[j]) / (2 * h1[i] * h_1[j])
    return hh


# ====================================================================== csminwel (mode_compute=4)
# Port of matlab/optimization/{csminwel1, csminit1, bfgsi1, numgrad2, numgrad3, numgrad5, numgrad3_, numgrad5_,
# penalty_objective_function, step_length_correction}.m of Dynare 7.2: Copyright (C) 1993-2009 Christopher Sims,
# Copyright (C) 2006-2025 Dynare Team, GPL-3.0-or-later (original at http://sims.princeton.edu/yftp/optimize/).
def _csm_numgrad(pfun, f0, x, method, epsilon):
    """Numerical gradient by the 2, 3 or 5 point formula (``method`` 2, 3, 5; 13 and 15 are the
    variants that set a component to zero at an apparent local minimum). Returns (g, badg, evaluations)."""
    n = x.size
    g = np.zeros(n)
    badg = False
    x = x.copy()
    h = epsilon
    if method not in (2, 3, 5, 13, 15):
        raise ValueError("csminwel: unknown method for gradient evaluation (use 2, 3, 5, 13 or 15)")
    for i in range(n):
        xiold = x[i]
        if method == 2:
            x[i] = x[i] + h
            g0 = (pfun(x) - f0) / h
        elif method in (3, 13):
            x[i] = xiold + h
            f1 = pfun(x)
            x[i] = xiold - h
            f2 = pfun(x)
            g0 = 0.0 if (method == 13 and f0 < f1 and f0 < f2) else (f1 - f2) / (2 * h)
        else:
            x[i] = xiold + h
            f1 = pfun(x)
            x[i] = xiold - h
            f2 = pfun(x)
            x[i] = xiold + 2 * h
            f3 = pfun(x)
            x[i] = xiold - 2 * h
            f4 = pfun(x)
            if method == 15 and f0 < f1 and f1 < f3 and f0 < f2 and f2 < f4:
                g0 = 0.0
            else:
                g0 = (8 * (f1 - f2) + f4 - f3) / (12 * h)
        if abs(g0) < 1e15:
            g[i] = g0
        else:
            g[i] = 0.0
            badg = True
        if method != 15:
            x[i] = xiold
        # method 15: numgrad5_.m of Dynare 7.2 does not restore x(i), so every later component is
        # differentiated at a point shifted by -2h in the earlier ones; reproduced as is.
    return g, badg, {2: 1, 3: 2, 13: 2, 5: 4, 15: 4}[method] * n


def _csminit(pfun, x0, f0, g0, badg, H0, log):
    """Line search of csminwel (``csminit1.m``). Returns (fhat, xhat, evaluations, retcode)."""
    ANGLE, THETA, FCHANGE, MINLAMB, MINDFAC = 0.005, 0.3, 1000, 1e-9, 0.01
    fcount = 0
    lam = 1.0
    xhat, fhat = x0, f0
    g = g0
    gnorm = np.linalg.norm(g)
    retcode = None
    if gnorm < 1e-12 and not badg:
        return fhat, xhat, fcount, 1
    dx = -H0 @ g
    dxnorm = np.linalg.norm(dx)
    if dxnorm > 1e12:
        log("Near-singular H problem.")
        dx = dx * FCHANGE / dxnorm
    dfhat = dx @ g0
    if not badg:
        a = -dfhat / (gnorm * dxnorm)
        if a < ANGLE:
            dx = dx - (ANGLE * dxnorm / gnorm + dfhat / (gnorm * gnorm)) * g
            dfhat = dx @ g
            dxnorm = np.linalg.norm(dx)
            log(f"Correct for low angle: {a:g}")
    log(f"Predicted improvement: {-dfhat / 2:18.9f}")
    done = False
    factor = 3.0
    shrink = True
    lambda_max = np.inf
    lambda_peak = 0.0
    f_peak = f0
    while not done:
        dxtest = x0 + dx * lam
        f = pfun(dxtest)
        log(f"lambda = {lam:10.5g}; f = {f:20.7f}")
        if f < fhat:
            fhat, xhat = f, dxtest
        fcount += 1
        shrink_signal = ((not badg) and (f0 - f < max(-THETA * dfhat * lam, 0))) or (badg and (f0 - f) < 0)
        grow_signal = (not badg) and (lam > 0) and (f0 - f > -(1 - THETA) * dfhat * lam)
        if shrink_signal and (lam > lambda_peak or lam < 0):
            if lam > 0 and ((not shrink) or lam / factor <= lambda_peak):
                shrink = True
                factor = factor ** 0.6
                while lam / factor <= lambda_peak:
                    factor = factor ** 0.6
                if abs(factor - 1) < MINDFAC:
                    retcode = 2 if abs(lam) < 4 else 7
                    done = True
            if lam < lambda_max and lam > lambda_peak:
                lambda_max = lam
            lam = lam / factor
            if abs(lam) < MINLAMB:
                if lam > 0 and f0 <= fhat:
                    lam = -lam * factor ** 6
                else:
                    retcode = 6 if lam < 0 else 3
                    done = True
        elif (grow_signal and lam > 0) or (shrink_signal and lam <= lambda_peak and lam > 0):
            if shrink:
                shrink = False
                factor = factor ** 0.6
                if abs(factor - 1) < MINDFAC:
                    retcode = 4 if abs(lam) < 4 else 7
                    done = True
            if f < f_peak and lam > 0:
                f_peak = f
                lambda_peak = lam
                if lambda_max <= lambda_peak:
                    lambda_max = lambda_peak * factor * factor
            lam = lam * factor
            if abs(lam) > 1e20:
                retcode = 5
                done = True
        else:
            done = True
            retcode = 7 if factor < 1.2 else 0
    log(f"Norm of dx {dxnorm:10.5g}")
    return fhat, xhat, fcount, retcode


def _bfgsi(H0, dg, dx, log):
    """BFGS update of the inverse Hessian (``bfgsi1.m``)."""
    Hdg = H0 @ dg
    dgdx = dg @ dx
    if abs(dgdx) > 1e-12:
        return H0 + (1 + (dg @ Hdg) / dgdx) * np.outer(dx, dx) / dgdx - (np.outer(dx, Hdg) + np.outer(Hdg, dx)) / dgdx
    log("bfgs update failed.")
    return H0


_CSMINWEL_MESSAGES = {0: "normal step", 1: "zero gradient", 2: "back and forth on step length never finished",
                      3: "smallest step still improving too slow", 4: "back and forth on step length never finished",
                      5: "largest step still improving too fast",
                      6: "smallest step still improving too slow, reversed gradient",
                      7: "warning: possible inaccuracy in H matrix", 8: "warning: iteration count termination",
                      -1: "Bad initial parameter."}


def csminwel(fcn, x0, H0=None, crit=1e-7, nit=1000, method=2, epsilon=1e-6, rand=None, verbose=False):
    """Christopher Sims's quasi-Newton minimiser with BFGS updates of the inverse Hessian, as shipped
    with Dynare 7.2 (``csminwel1.m``, Dynare's ``mode_compute=4``), with numerical gradients.

    ``fcn(x)`` returns ``(value, code, penalty)``: where the function is not defined ``code`` is
    non-zero and the value used is the function value at the start of the current iteration plus
    ``penalty`` (Dynare's ``penalty_objective_function``). ``H0`` is the initial inverse Hessian
    (default ``1e-4 * I``), ``crit`` the convergence criterion on the function value, ``nit`` the
    maximum number of iterations, ``method`` the gradient formula (2, 3 or 5 points; 13, 15) and
    ``epsilon`` its step. When the line search hits a "cliff" the search direction is perturbed with
    uniform random numbers: ``rand(n)`` supplies them (default: a numpy generator seeded with 0, so
    results are reproducible; Dynare uses MATLAB's stream).

    Returns a dict: ``f``, ``x``, ``g`` (gradient), ``H`` (inverse Hessian approximation),
    ``iterations``, ``fcount`` (function evaluations as counted by Dynare), ``retcode``, ``message``.
    The algorithm is deterministic and reproduces Dynare's iterates evaluation by evaluation as
    long as the two objective functions agree to rounding error."""
    x0 = np.array(x0, float).ravel()
    nx = x0.size
    H = 1e-4 * np.eye(nx) if H0 is None else np.array(H0, float)
    if rand is None:
        rand = np.random.default_rng(0).random
    log = print if verbose else (lambda *_: None)
    state = {"penalty": 1e8}

    def pfun(x):
        fval, code, pen = fcn(x)
        return state["penalty"] + pen if code else fval

    def numgrad(f, x):
        return _csm_numgrad(pfun, f, x, method, epsilon)

    out = {"f": None, "x": None, "g": None, "H": None, "iterations": 0, "fcount": 0, "retcode": None, "message": ""}
    f0, code0, _ = fcn(x0)
    fcount = 1
    if code0:
        out.update(retcode=-1, message=_CSMINWEL_MESSAGES[-1], fcount=fcount)
        return out
    g, badg, fc = numgrad(f0, x0)
    fcount += fc
    x, f = x0, f0
    gh = None
    itct = 0
    done = False
    while not done:
        state["penalty"] = f
        g1 = g2 = g3 = None
        log("-----------------")
        log(f"f at the beginning of new iteration, {f:20.10f}")
        itct += 1
        f1, x1, fc, retcode1 = _csminit(pfun, x, f, g, badg, H, log)
        fcount += fc
        badg1 = badg2 = badg3 = None
        x2 = x3 = None
        if retcode1 != 1:
            if retcode1 in (2, 4):
                wall1, badg1 = True, True
            else:
                g1, badg1, fc = numgrad(f1, x1)
                fcount += fc
                wall1 = badg1
            if wall1:
                Hcliff = H + np.diag(np.diag(H) * np.asarray(rand(nx), float))
                log("Cliff.  Perturbing search direction.")
                f2, x2, fc, retcode2 = _csminit(pfun, x, f, g, badg, Hcliff, log)
                fcount += fc
                if f2 < f:
                    if retcode2 in (2, 4):
                        wall2, badg2 = True, True
                    else:
                        g2, badg2, fc = numgrad(f2, x2)
                        fcount += fc
                        wall2 = badg2
                    if wall2:
                        log("Cliff again.  Try traversing")
                        if np.linalg.norm(x2 - x1) < 1e-13:
                            f3, x3, badg3, retcode3 = f, x, True, 101
                        else:
                            gcliff = ((f2 - f1) / (np.linalg.norm(x2 - x1) ** 2)) * (x2 - x1)
                            f3, x3, fc, retcode3 = _csminit(pfun, x, f, gcliff, False, np.eye(nx), log)
                            fcount += fc
                            if retcode3 in (2, 4):
                                badg3 = True
                            else:
                                g3, badg3, fc = numgrad(f3, x3)
                                fcount += fc
                    else:
                        f3, x3, badg3, retcode3 = f, x, True, 101
                else:
                    f3, x3, badg3, retcode3 = f, x, True, 101
            else:
                f2, f3, badg2, badg3, retcode2, retcode3 = f, f, True, True, 101, 101
        else:
            f2, f3, f1, retcode2, retcode3 = f, f, f, retcode1, retcode1
        # how to pick gh and xh
        if f3 < f - crit and badg3 is False and f3 < f2 and f3 < f1:
            fh, xh, gh, badgh, retcodeh = f3, x3, g3, badg3, retcode3
        elif f2 < f - crit and badg2 is False and f2 < f1:
            fh, xh, gh, badgh, retcodeh = f2, x2, g2, badg2, retcode2
        elif f1 < f - crit and badg1 is False:
            fh, xh, gh, badgh, retcodeh = f1, x1, g1, badg1, retcode1
        else:                                     # stuck or bad gradient
            ih = int(np.argmin([f1, f2, f3]))
            fh = (f1, f2, f3)[ih]
            xh = (x1, x2, x3)[ih]
            retcodeh = (retcode1, retcode2, retcode3)[ih]
            badgh = True
            if gh is None:                        # recompute the gradient (only if none was ever accepted)
                gh, badgh, fc = numgrad(fh, xh)
                fcount += fc
        stuck = abs(fh - f) < crit
        if (not badg) and (not badgh) and (not stuck):
            H = _bfgsi(H, gh - g, xh - x, log)
        log("----")
        log(f"Improvement on iteration {itct} = {f - fh:18.9f}")
        if itct > nit:
            log("iteration count termination")
            done = True
            retcodeh = 8
        elif stuck:
            log("improvement < crit termination")
            done = True
        if retcodeh not in _CSMINWEL_MESSAGES or retcodeh == -1:
            raise EstimationError("csminwel: unaccounted case (Dynare stops with the same error)")
        message = _CSMINWEL_MESSAGES[retcodeh]
        if verbose or done:
            log(message)
        f, x, g, badg = fh, xh, gh, badgh
    out.update(f=float(fh), x=np.array(xh), g=np.array(gh), H=H, iterations=itct, fcount=fcount, retcode=int(retcodeh),
               message=message)
    return out


# ====================================================================== results
@dataclass
class ModeResult:
    """Posterior mode (or maximum-likelihood estimate)."""
    x: np.ndarray                     # mode, ordered as Estimation.names
    names: list
    log_posterior: float              # log posterior kernel (log likelihood for ML) at the mode
    hessian: np.ndarray = None        # Hessian of minus the log posterior
    inv_hessian: np.ndarray = None    # posterior covariance approximation
    laplace: float = np.nan           # Laplace approximation of the log marginal data density
    success: bool = True
    message: str = ""
    nfev: int = 0

    @property
    def std(self):
        """Standard deviations from the inverse Hessian (NaN if it could not be computed, for
        instance when the optimum lies on a bound)."""
        if self.inv_hessian is None:
            return np.full(len(self.x), np.nan)
        return np.sqrt(np.diag(self.inv_hessian))

    def as_dict(self):
        return dict(zip(self.names, self.x.tolist()))


@dataclass
class MCMCResult:
    draws: np.ndarray                 # (chains, replic, n_parameters)
    log_posterior: np.ndarray         # (chains, replic)
    acceptance_rate: np.ndarray       # (chains,)
    names: list
    x0: np.ndarray                    # initial point of each chain
    drop: float = 0.5
    accepted: np.ndarray = field(default=None, repr=False)   # (chains, replic) booleans
    jscale: float = None              # scale of the jumping distribution (after tuning, if any)
    jscale_vector: np.ndarray = field(default=None, repr=False)    # scale per parameter (bayestopt_.jscale)
    proposal_chol: np.ndarray = field(default=None, repr=False)    # upper Cholesky factor of the jumping covariance

    def kept(self, drop=None):
        """Draws after discarding the first ``drop`` share of every chain, pooled."""
        drop = self.drop if drop is None else drop
        first = int(np.floor(drop * self.draws.shape[1]))
        return self.draws[:, first:, :].reshape(-1, self.draws.shape[2])

    def mean(self, drop=None):
        return self.kept(drop).mean(axis=0)

    def std(self, drop=None):
        return self.kept(drop).std(axis=0, ddof=1)

    def hpd_interval(self, prob=0.9, drop=None):
        """Shortest interval containing ``prob`` of the kept draws; shape (n_parameters, 2)."""
        d = np.sort(self.kept(drop), axis=0)
        n = d.shape[0]
        m = int(np.floor(prob * n))
        width = d[m:, :] - d[:n - m, :]
        j = np.argmin(width, axis=0)
        cols = np.arange(d.shape[1])
        return np.column_stack([d[j, cols], d[j + m, cols]])


def metropolis_hastings(log_post, x0, logpost0, chol_upper, lb, ub, innovations, uniforms, uniform_stream=False):
    """One random-walk Metropolis-Hastings chain driven by given random numbers.

    ``chol_upper`` is the upper Cholesky factor of the proposal covariance (already scaled);
    the proposal is ``last + innovations[i] @ chol_upper``. As in Dynare a proposal is evaluated
    only if it lies strictly inside the bounds, and accepted if its log posterior is finite and
    ``log(u) < log_post(proposal) - log_post(last)``.

    ``uniforms[i]`` is used for draw ``i``. With ``uniform_stream=True`` the uniforms are instead
    consumed in order, one for every proposal with a finite log posterior: that is how Dynare's
    ``posterior_sampler_iteration.m`` uses its random-number stream (``log(rand)`` is only evaluated
    then), so a chain can be reproduced from Dynare's seeds.
    Returns (draws, log posteriors, accepted flags).
    """
    n_draws, n = innovations.shape
    draws = np.empty((n_draws, n))
    logpo = np.empty(n_draws)
    acc = np.zeros(n_draws, bool)
    last, last_lp = np.array(x0, float), float(logpost0)
    used = 0
    for i in range(n_draws):
        par = last + innovations[i] @ chol_upper
        if np.all(par > lb) and np.all(par < ub):
            try:
                lp = float(log_post(par))
            except Exception:                     # noqa: BLE001 - as Dynare: any failure rejects
                lp = -np.inf
        else:
            lp = -np.inf
        if lp > -np.inf:
            u = uniforms[used] if uniform_stream else uniforms[i]
            used += 1
            if np.log(u) < lp - last_lp:
                last, last_lp = par, lp
                acc[i] = True
        draws[i] = last
        logpo[i] = last_lp
    return draws, logpo, acc


# ====================================================================== the estimation object
_DEFAULTS = dict(
    datafile=None, first_obs=1, nobs=None, loglinear=False, logdata=False, prefilter=0, presample=0,
    lik_init=1, diffuse_filter=False, kalman_algo=0, kalman_tol=1e-10, diffuse_kalman_tol=1e-6,
    riccati_tol=1e-6, qz_criterium=None, prior_trunc=0.0, Harvey_scale_factor=10.0,
    lyapunov_complex_threshold=1e-15, rescale_prediction_error_covariance=False,
    gstep=(1e-2, 1.0), order=1, use_univariate_filters_if_singularity_is_detected=True,
    # posterior mode (Estimation.mode); mode_compute=None means "not given in the .mod file"
    mode_compute=None, mode_file=None, optim=None, additional_optimizer_steps=(), cova_compute=True,
    use_penalized_objective_for_hessian=False, gradient_method=2, gradient_epsilon=1e-6,
    # Metropolis-Hastings (Estimation.sample)
    mh_replic=20000, mh_nblck=2, mh_jscale=None, mh_drop=0.5, mh_init_scale_factor=2.0, mh_init_scale=None,
    mcmc_jumping_covariance="hessian", mh_tune_jscale=None, mh_tune_guess=None,
    posterior_sampling_method="random_walk_metropolis_hastings", posterior_sampler_options=None,
    # posterior analysis (pydyn.posterior, pydyn.posterior_plots)
    mh_conf_sig=0.9, sub_draws=None, posterior_max_subsample_draws=1200,
    taper_steps=(4, 8, 15), geweke_interval=(0.2, 0.5), raftery_lewis_diagnostics=False,
    raftery_lewis_qrs=(0.025, 0.005, 0.95), brooks_gelman_plotrows=3,
    bayesian_irf=False, irf=40, relative_irf=False, irf_shocks=None,
    mode_check=False, mode_check_neighbourhood_size=0.5, mode_check_symmetric_plots=True,
    mode_check_number_of_points=20, huge_number=1e7,
    # posterior distributions of endogenous statistics (pydyn.posterior_moments)
    moments_varendo=False, ar=5, nodecomposition=False, conditional_variance_decomposition=(),
    contemporaneous_correlation=False, endo_vars_for_moment_computations_in_estimation=None,
    smoother=False, filtered_vars=False, filter_step_ahead=(),
    # time-varying shock variances (pydyn.heteroskedastic); endogenous prior (pydyn.endogenous_prior)
    heteroskedastic_filter=False, endogenous_prior=False,
    # DSGE-VAR (pydyn.dsge_var): dsge_var is the calibrated weight, "NaN"/True when it is estimated
    dsge_var=None, dsge_varlag=4,
)
#: names used in the preprocessor's JSON (or old names) -> option names of the manual used here
_ALIASES = {"mh_nblocks": "mh_nblck", "harvey_scale_factor": "Harvey_scale_factor",
            "MCMC_jumping_covariance": "mcmc_jumping_covariance", "optim_opt": "optim",
            "hessian.use_penalized_objective": "use_penalized_objective_for_hessian",
            "convergence.geweke.geweke_interval": "geweke_interval", "convergence.geweke.taper_steps": "taper_steps",
            "convergence.rafterylewis.indicator": "raftery_lewis_diagnostics",
            "convergence.rafterylewis.qrs": "raftery_lewis_qrs",
            "convergence.brooksgelman.plotrows": "brooks_gelman_plotrows",
            "mode_check.status": "mode_check", "mode_check.neighbourhood_size": "mode_check_neighbourhood_size",
            "mode_check.symmetric_plots": "mode_check_symmetric_plots",
            "mode_check.number_of_points": "mode_check_number_of_points",
            "posterior_sampler_options.posterior_sampling_method": "posterior_sampling_method",
            "posterior_sampler_options.sampling_opt": "posterior_sampler_options",
            "mh_tune_jscale.target": "mh_tune_jscale", "mh_tune_jscale.guess": "mh_tune_guess"}
#: options that must keep their list form
_VECTOR_OPTIONS = {"gstep", "additional_optimizer_steps", "taper_steps", "geweke_interval", "raftery_lewis_qrs", "irf_shocks",
                   "conditional_variance_decomposition", "filter_step_ahead"}
#: who acts on an option when it is not Estimation itself (documentation and option_report())
_HONOURED_BY = {
    "mode()": ["mode_compute", "mode_file", "optim", "additional_optimizer_steps", "cova_compute",
               "use_penalized_objective_for_hessian", "gradient_method", "gradient_epsilon", "gstep"],
    "sample()": ["mh_replic", "mh_nblck", "mh_jscale", "mh_drop", "mh_init_scale_factor", "mh_init_scale",
                 "mcmc_jumping_covariance", "mh_tune_jscale", "mh_tune_guess", "posterior_sampling_method",
                 "posterior_sampler_options"],
    "pydyn.posterior": ["mh_conf_sig", "sub_draws", "posterior_max_subsample_draws", "taper_steps", "geweke_interval",
                        "raftery_lewis_diagnostics", "raftery_lewis_qrs", "bayesian_irf", "irf", "relative_irf",
                        "irf_shocks", "mode_check", "mode_check_neighbourhood_size", "mode_check_symmetric_plots",
                        "mode_check_number_of_points", "huge_number"],
    "pydyn.posterior_plots": ["brooks_gelman_plotrows"],
    "pydyn.posterior_moments": ["moments_varendo", "ar", "nodecomposition", "conditional_variance_decomposition",
                                "contemporaneous_correlation", "endo_vars_for_moment_computations_in_estimation",
                                "smoother", "filtered_vars", "filter_step_ahead"],
}
#: mode_compute values: ported algorithm, scipy stand-in, or not available
_MODE_COMPUTE_EQUIVALENTS = {1: ("SLSQP", "fmincon"), 3: ("BFGS", "fminunc"), 7: ("Nelder-Mead", "fminsearch"),
                             8: ("Nelder-Mead", "Dynare's simplex routine"),
                             9: ("differential_evolution", "CMA-ES (cmaes.m)")}


def _parse_optim(spec) -> dict:
    """The ``optim`` option: Dynare's ``('NAME', VALUE, ...)`` string from the JSON, a dict, or a
    sequence of names and values."""
    if not spec:
        return {}
    if isinstance(spec, dict):
        return dict(spec)
    if isinstance(spec, str):
        items = [t.strip() for t in re.split(r",(?![^\[\(]*[\]\)])", spec.replace("''", "'")) if t.strip()]
    else:
        items = list(spec)
    if len(items) % 2:
        raise EstimationError(f"optim: expected pairs of names and values, got {spec!r}")
    out = {}
    for name, val in zip(items[::2], items[1::2]):
        name = str(name).strip().strip("'\"")
        if isinstance(val, str):
            v = val.strip().strip("'\"")
            try:
                val = float(v)
                val = int(val) if val == int(val) and "." not in v and "e" not in v.lower() else val
            except ValueError:
                val = v
        out[name] = val
    return out


class _Rejected(Exception):
    """Raised inside an evaluation when the parameter vector cannot be used. ``code`` and ``penalty``
    are Dynare's ``info(1)`` and ``info(4)`` of ``dsge_likelihood.m`` (the penalty is what csminwel
    adds to the last function value outside the domain)."""

    def __init__(self, message, code=0, penalty=0.1):
        super().__init__(message)
        self.code, self.penalty = int(code), float(penalty)


def _pd_penalty(A):
    """``ispd.m``: minus the sum of the negative eigenvalues."""
    if not np.all(np.isfinite(A)):
        return 1.0
    ev = np.linalg.eigvalsh(A)
    return float(np.sum(-ev[ev < 0]))


class Estimation:
    """Estimation problem for a first-order approximated model.

    Parameters
    ----------
    model : pydyn.Model
        Must contain ``varobs`` and an ``estimated_params`` block.
    data : optional
        Dict ``{name: series}``, array (periods x observed variables, columns in ``varobs`` order)
        or a file name. Default: the ``datafile`` option of the ``estimation`` command, looked up
        next to the ``.mod`` file. NaN marks a missing observation.
    options : keyword arguments overriding those of the ``estimation`` command, with Dynare's
        names: ``first_obs``, ``nobs``, ``loglinear``, ``logdata``, ``prefilter``, ``presample``,
        ``lik_init``, ``diffuse_filter``, ``kalman_tol``, ``riccati_tol``, ``qz_criterium``,
        ``prior_trunc``, ``mh_replic``, ``mh_nblocks``, ``mh_jscale``, ``mh_drop``, ...

    Subclassing
    -----------
    Two methods are documented extension points; subclasses (e.g. ``pydyn.occbin_estimation``) may
    override them, and their names and signatures are kept stable:

    ``_minus_loglik(theta, return_lik=False)``
        minus the log likelihood at ``theta`` (with ``return_lik=True``: ``(minus log likelihood,
        log density of each observation after the presample)``); raise ``_Rejected(message, code,
        penalty)`` where the likelihood is not defined. ``log_likelihood``, ``log_posterior``,
        ``objective``, ``mode`` and ``sample`` all go through it.
    ``smoother(theta)``
        smoothed variables, shocks and measurement errors at ``theta`` (see its docstring).

    A model with ``occbin_constraints`` is refused unless the subclass sets the class attribute
    ``handles_occbin = True``.
    """

    def __init__(self, model, data=None, **options):
        from . import options as _opt
        if not getattr(type(self), "handles_occbin", False):
            _opt.refuse_occbin(model, "Estimation")
        self.model = model
        opts = dict(_DEFAULTS)
        raw = dict(model.options.get("estimation", {}))
        raw.update(options)
        self.ignored_options = []
        tune_status = bool(raw.pop("mh_tune_jscale.status", False))
        for k, v in raw.items():
            k = _ALIASES.get(k, k)
            if isinstance(v, dict) and "symbol_list" in v:
                v = list(v["symbol_list"])
            if isinstance(v, (list, tuple)) and len(v) == 1 and k not in _VECTOR_OPTIONS:
                v = v[0]
            if k in ("nobs", "first_obs") and isinstance(v, (list, tuple)):
                raise NotImplementedError(f"recursive estimation ({k} given as a vector) is not supported")
            if k in opts:
                opts[k] = v
            else:
                self.ignored_options.append(k)
        if opts["mh_tune_jscale"] is True or (tune_status and opts["mh_tune_jscale"] is None):
            opts["mh_tune_jscale"] = 0.33                     # Dynare's default target acceptance rate
        elif opts["mh_tune_jscale"] is False:
            opts["mh_tune_jscale"] = None
        if opts["mh_tune_jscale"] is not None and opts["mh_jscale"] is not None:
            raise EstimationError("the mh_tune_jscale and mh_jscale options are incompatible")
        self.options = opts
        #: variables listed after the estimation command (used for posterior IRFs)
        self.var_list = next((list(st.get("symbol_list", [])) for st in model.statements
                              if st.get("statementName") == "estimation"), [])
        self.unsupported_options = sorted(set(self.ignored_options) - _opt.COSMETIC)
        notes = []
        from . import samplers as _samplers      # samplers other than RWMH, posterior_sampler_options (pydyn.samplers)
        if opts["posterior_sampling_method"] not in _samplers.IMPLEMENTED:
            self.unsupported_options.append("posterior_sampling_method")
            notes.append(f"posterior_sampling_method={opts['posterior_sampling_method']}: not implemented, sample() "
                         "will refuse to run")
        else:
            _samplers.sampler_options(opts["posterior_sampling_method"], opts["posterior_sampler_options"])
            if opts["posterior_sampling_method"] == "slice" and not opts["prior_trunc"]:
                opts["prior_trunc"] = 1e-10       # dynare_estimation_init.m: slice requires bounded support
        if self.unsupported_options:
            _opt.warn_once(model, "Estimation does not act on these estimation options: "
                           + ", ".join(self.unsupported_options)
                           + " (smoothing and forecasts are separate calls or not available"
                           + ("; " + "; ".join(notes) if notes else "") + ")")
        if int(opts["order"]) != 1:
            raise NotImplementedError("estimation is implemented for order=1 only")
        if opts["diffuse_filter"] or opts["kalman_algo"] in (3, 4):
            if opts["lik_init"] == 2:
                raise EstimationError("options diffuse_filter, lik_init and/or kalman_algo have contradictory settings")
            opts["lik_init"] = 3
        if opts["kalman_algo"] not in (0, 1, 2, 3, 4):
            raise NotImplementedError(f"kalman_algo={opts['kalman_algo']} is not implemented")
        if opts["kalman_algo"] in (1, 2) and opts["lik_init"] == 3:
            raise EstimationError("the diffuse filter needs kalman_algo 0, 3 or 4")
        if opts["lik_init"] not in (1, 2, 3):
            raise NotImplementedError(f"lik_init={opts['lik_init']} is not implemented")
        if opts["qz_criterium"] is None:           # select_qz_criterium_value.m
            opts["qz_criterium"] = 1 - 1e-6 if opts["lik_init"] == 1 else 1 + 1e-6
        elif opts["lik_init"] == 1 and opts["qz_criterium"] > 1 - np.finfo(float).eps:
            raise EstimationError("qz_criterium is too large for a stationary model; use diffuse_filter")
        elif opts["lik_init"] != 1 and opts["qz_criterium"] <= 1:
            opts["qz_criterium"] = 1 + 1e-6

        self.varobs = list(model.varobs)
        if not self.varobs:
            raise EstimationError("the model has no varobs statement")
        if len(set(self.varobs)) != len(self.varobs):
            raise EstimationError("a variable is declared twice in varobs")
        self._obs_decl = np.array([model._ie[v] for v in self.varobs])
        self._setup_parameters()
        self._setup_state_space()
        self._setup_trends()
        self._setup_data(data)
        self.last_info, self.last_code, self.last_penalty = "", 0, 0.0
        self.last_filter = ""
        if opts["prefilter"]:
            try:
                centred = self.check_prefilter()
            except (SteadyStateError, BlanchardKahnError):
                centred = True                    # cannot be checked at the initial values
            if not centred:
                raise EstimationError('the option "prefilter" is inconsistent with the non-zero mean '
                                      "measurement equations in the model")

    # ------------------------------------------------------------------ parameters and priors
    def _num(self, s):
        if isinstance(s, (int, float)):
            return float(s)
        return float(self.model.eval_expr(s))

    def _setup_parameters(self):
        m = self.model
        blocks = [st for st in m.statements if st.get("statementName") == "estimated_params"]
        for st in m.statements:
            if st.get("statementName") in ("estimated_params_bounds", "estimated_params_remove", "prior",
                                           "std_prior", "corr_prior"):
                raise NotImplementedError(f"{st['statementName']} blocks are not supported")
        if not blocks:
            raise EstimationError("the model has no estimated_params block")
        if len(blocks) > 1:
            # the JSON output does not say whether a later block has the 'overwrite' option
            raise NotImplementedError("several estimated_params blocks are not supported")
        entries = list(blocks[0].get("params", []))
        if not entries:
            raise EstimationError("the estimated_params block is empty")
        groups = {"vx": [], "vn": [], "cx": [], "cn": [], "p": []}
        for e in entries:
            if "param" in e:
                if e["param"] not in m._ip:
                    raise EstimationError(f"estimated_params: unknown parameter {e['param']}")
                groups["p"].append((e, m._ip[e["param"]], e["param"]))
            elif "var" in e:
                v = e["var"]
                if v in m._ix:
                    groups["vx"].append((e, m._ix[v], f"stderr {v}"))
                elif v in self.varobs:
                    groups["vn"].append((e, self.varobs.index(v), f"stderr {v}"))
                else:
                    raise EstimationError(f"the variable {v} has to be declared as observable since "
                                          "you assume a measurement error on it")
            elif "var1" in e:
                a, b = e["var1"], e["var2"]
                if a in m._ix and b in m._ix:
                    groups["cx"].append((e, (m._ix[a], m._ix[b]), f"corr {a}, {b}"))
                elif a in self.varobs and b in self.varobs:
                    groups["cn"].append((e, (self.varobs.index(a), self.varobs.index(b)), f"corr {a}, {b}"))
                else:
                    raise EstimationError(f"corr {a}, {b}: both must be shocks or both observed variables")
            else:
                raise NotImplementedError(f"estimated_params entry not supported: {e}")
        self._groups = groups
        ordered = groups["vx"] + groups["vn"] + groups["cx"] + groups["cn"] + groups["p"]
        self.names = [name for _, _, name in ordered]
        get = lambda key: np.array([self._num(e[key]) for e, _, _ in ordered])  # noqa: E731
        init, lb, ub = get("init_val"), get("lower_bound"), get("upper_bound")
        ncorr0 = len(groups["vx"]) + len(groups["vn"])
        ncorr1 = ncorr0 + len(groups["cx"]) + len(groups["cn"])
        ub[ncorr0:ncorr1] = np.clip(ub[ncorr0:ncorr1], -1, 1)
        lb[ncorr0:ncorr1] = np.clip(lb[ncorr0:ncorr1], -1, 1)
        shape = np.array([int(e["prior_distribution"]) for e, _, _ in ordered])
        if np.any(shape == 0) and np.any(shape != 0):
            raise EstimationError("estimation must be either fully ML or fully Bayesian")
        self.prior = Prior(shape, get("mean"), get("std"), get("p3"), get("p4"), names=self.names)
        self.jscale_individual = get("jscale")
        self._apply_init_blocks(init)
        k = np.isnan(init)
        if np.any(k) and not self.prior.is_bayesian:
            raise EstimationError("ML estimation requires all estimated parameters to be initialized")
        init[k] = self.prior.p1[k]
        self.x0 = init
        self.user_lb, self.user_ub = lb, ub
        if self.prior.is_bayesian:
            plb, pub = self.prior.bounds(self.options["prior_trunc"])
            self.lb, self.ub = np.maximum(plb, lb), np.minimum(pub, ub)
        else:
            self.lb, self.ub = lb, ub
        self.n = len(self.names)
        # calibrated covariance structure, as Dynare's set_all_parameters uses it: Sigma_e is rebuilt
        # from the standard deviations and M_.Correlation_matrix (set by 'corr' statements only); when
        # correlations are estimated, the other non-zero calibrated covariances keep their value
        # (check_for_calibrated_covariances, which Dynare 7.2 only applies when ncx > 0)
        S = np.array(m.Sigma_e, float)
        C = getattr(m, "Correlation_matrix", None)
        if C is None or np.shape(C) != S.shape:                 # a model object without it
            sd = np.sqrt(np.diag(S))
            with np.errstate(divide="ignore", invalid="ignore"):
                C = S / np.outer(sd, sd)
            C[~np.isfinite(C)] = 0.0
            np.fill_diagonal(C, 1.0)
        C = np.array(C, float)
        self._Sigma_e0, self._corr0 = S, C
        estimated_corr = {frozenset(k) for _, k, _ in groups["cx"]}
        calibrated = [(i, j) for i in range(len(S)) for j in range(i)
                      if S[i, j] != 0 and frozenset((i, j)) not in estimated_corr]
        self._calibrated_cov = calibrated if groups["cx"] else []
        if groups["vx"] and not groups["cx"]:
            sd = np.sqrt(np.diag(S))
            lost = [(i, j) for i, j in calibrated if not np.isclose(S[i, j], C[i, j] * sd[i] * sd[j], rtol=1e-12, atol=0)]
            if lost:
                warnings.warn("calibrated covariances of " + ", ".join(f"{m.exo_names[j]} and {m.exo_names[i]}" for i, j in lost)
                              + " are not kept when standard deviations are estimated: as in Dynare 7.2 (set_all_parameters), "
                              "Sigma_e is rebuilt from the correlations of 'corr' statements (M_.Correlation_matrix), and "
                              "these covariances were not set by one (e.g. 'var e, u = ...'); write them as 'corr' to keep them",
                              UserWarning, stacklevel=3)
        pp = len(self.varobs)
        try:
            H = m.me_H(self.varobs)
        except ModelError as exc:
            raise EstimationError(str(exc)) from None
        self._H0, self._corrME0 = H, m.me_corr_matrix(self.varobs)
        # check_for_calibrated_covariances.m: the calibrated off-diagonal entries of M_.H that are not estimated are kept at
        # their value (whenever any exists, unlike those of Sigma_e, which are only looked for when a shock correlation is
        # estimated)
        estimated_corrn = {frozenset(k) for _, k, _ in groups["cn"]}
        self._calibrated_cov_ME = [(i, j) for i in range(pp) for j in range(i)
                                   if H[i, j] != 0 and frozenset((i, j)) not in estimated_corrn]
        nz = ~np.all(S == 0, axis=0)
        nz[[k for _, k, _ in groups["vx"]]] = True
        self._Q_check = np.flatnonzero(nz)
        nz = ~np.all(H == 0, axis=0)
        nz[[k for _, k, _ in groups["vn"]]] = True
        self._H_check = np.flatnonzero(nz)

    def _apply_init_blocks(self, init):
        """``estimated_params_init`` blocks: explicit initial values; with ``use_calibration`` the estimated parameters
        that have none start at their calibrated values (``do_parameter_initialization.m``: the calibrated standard
        deviation of a shock or measurement error that is 0 with an inverse gamma prior (IG1) is not used)."""
        use_cal = False
        for st in self.model.statements:
            if st.get("statementName") != "estimated_params_init":
                continue
            use_cal = use_cal or bool(st.get("use_calibration_initialization") or st.get("use_calibration"))
            for e in st.get("params", []):
                if "param" in e:
                    name = e["param"]
                elif "var" in e:
                    name = f"stderr {e['var']}"
                else:
                    name = f"corr {e['var1']}, {e['var2']}"
                if name not in self.names:
                    raise EstimationError(f"estimated_params_init: {name} is not estimated")
                init[self.names.index(name)] = self._num(e["init_val"])
        if use_cal:
            m, g = self.model, self._groups
            Q, H = np.array(m.Sigma_e, float), np.array(m.me_H(self.varobs), float)
            C = getattr(m, "Correlation_matrix", None)
            CH = np.array(m.me_corr_matrix(self.varobs), float)
            calib = []
            for _, k, _ in g["vx"]:
                calib.append(np.sqrt(Q[k, k]))
            for _, k, _ in g["vn"]:
                calib.append(np.sqrt(H[k, k]))
            for _, (k1, k2), _ in g["cx"]:
                calib.append(C[k1, k2] if C is not None and np.shape(C) == Q.shape else np.nan)
            for _, (k1, k2), _ in g["cn"]:
                calib.append(CH[k1, k2])
            for _, k, _ in g["p"]:
                calib.append(m.params[k])
            calib = np.array(calib, float)
            shape = np.asarray(self.prior.shape)
            nstd = len(g["vx"]) + len(g["vn"])
            calib[:nstd][(calib[:nstd] == 0) & (shape[:nstd] == 4)] = np.nan   # violates the inverse gamma prior
            if not self.prior.is_bayesian and np.any(np.isnan(init) & np.isnan(calib)):
                raise EstimationError("When using the use_calibration option with ML, the parameters must be properly "
                                      "initialized.")
            fill = np.isnan(init) & ~np.isnan(calib)
            init[fill] = calib[fill]

    def covariances(self, theta):
        """Structural-shock covariance matrix and measurement-error covariance matrix implied by
        ``theta`` (``set_all_parameters.m``)."""
        return self._covariances(theta)[:2]

    def _covariances(self, theta):
        """``(Sigma_e, H, Correlation_matrix_ME)`` implied by ``theta`` (the last as ``set_all_parameters`` stores it)."""
        g = self._groups
        Q, C = self._Sigma_e0.copy(), self._corr0.copy()
        H, CH = self._H0.copy(), self._corrME0.copy()
        o = 0
        for i, (_, k, _) in enumerate(g["vx"]):
            Q[k, k] = theta[o + i] ** 2
        o += len(g["vx"])
        for i, (_, k, _) in enumerate(g["vn"]):
            H[k, k] = theta[o + i] ** 2
        o += len(g["vn"])
        for i, (_, (k1, k2), _) in enumerate(g["cx"]):
            C[k1, k2] = C[k2, k1] = theta[o + i]
        o += len(g["cx"])
        for i, (_, (k1, k2), _) in enumerate(g["cn"]):
            CH[k1, k2] = CH[k2, k1] = theta[o + i]
        if g["vx"] or g["cx"]:
            sd = np.sqrt(np.diag(Q))
            Q = C * np.outer(sd, sd)
            for k1, k2 in self._calibrated_cov:                # estim_params_.calibrated_covariances
                Q[k1, k2] = Q[k2, k1] = self._Sigma_e0[k1, k2]
        if g["vn"] or g["cn"]:
            sd = np.sqrt(np.diag(H))
            H = CH * np.outer(sd, sd)
            for k1, k2 in self._calibrated_cov_ME:             # estim_params_.calibrated_covariances_ME
                H[k1, k2] = H[k2, k1] = self._H0[k1, k2]
        return Q, H, CH

    def model_params(self, theta):
        """Copy of the model's parameter vector with the estimated structural parameters set."""
        p = np.array(self.model.params, float)
        g = self._groups
        o = self.n - len(g["p"])
        for i, (_, k, _) in enumerate(g["p"]):
            p[k] = theta[o + i]
        return p

    def calibration(self, theta, correlation_me=False) -> dict:
        """Calibration implied by ``theta`` as plain arrays, without touching the model:
        ``params`` (the model's parameter vector with the estimated parameters replaced; parameters
        that the steady state computes are as stored in the model: see ``state_space(theta)["params"]``
        for the values after the steady state), ``Sigma_e`` (covariance matrix of the structural
        shocks) and ``H`` (covariance matrix of the measurement errors, in ``varobs`` order); with
        ``correlation_me`` also ``Correlation_matrix_ME`` (Dynare's M_.Correlation_matrix_ME)."""
        theta = np.asarray(theta, float)
        if theta.shape != (self.n,):
            raise EstimationError(f"theta must have {self.n} elements (ordered as Estimation.names)")
        Q, H, CH = self._covariances(theta)
        out = {"params": self.model_params(theta), "Sigma_e": Q, "H": H}
        if correlation_me:
            out["Correlation_matrix_ME"] = CH
        return out

    def _bk_penalty(self, theta):
        """Dynare's info code and penalty when the Blanchard-Kahn conditions fail
        (``dyn_first_order_solver.m``): 3 (no stable solution) or 4 (indeterminacy), with the squared
        distances ``|lambda| - 1 - qz_criterium`` of the eigenvalues on the wrong side."""
        try:
            m = self.model
            ys, params = steady_state(m, self.model_params(theta))
            n = m.n_endo
            J = m.dynamic_jac(np.tile(ys, 3), m.exo_steady(), np.array(params, float), ys)
            fm, f0, fp = J[:, :n], J[:, n:2 * n], J[:, 2 * n:3 * n]
            I, Z0 = np.eye(n), np.zeros((n, n))
            alpha, beta = linalg.eigvals(np.block([[Z0, I], [-fm, Z0]]), np.block([[I, Z0], [f0, fp]]),
                                         homogeneous_eigvals=True)
            qz = self.options["qz_criterium"]
            with np.errstate(divide="ignore", invalid="ignore"):
                mod = np.sort(np.abs(alpha / beta))
            stable = np.abs(alpha) < qz * np.abs(beta)
            ns = int(stable.sum())
            if ns < n:                            # too many unstable roots
                code, temp = 3, mod[ns:ns + (n - ns)] - 1 - qz
            elif ns > n:
                code, temp = 4, mod[n:ns] - 1 - qz
            else:
                return 5, 0.1
            pen = float(temp @ temp)
            return code, (pen if np.isfinite(pen) else 0.1)
        except Exception:                         # noqa: BLE001
            return 3, 0.1

    # ------------------------------------------------------------------ state space
    def _setup_state_space(self):
        m = self.model
        order_var, state_var, counts = dr_ordering(m.lead_lag_incidence)
        inv = np.empty(m.n_endo, int)
        inv[order_var] = np.arange(m.n_endo)
        obs_dr = inv[self._obs_decl]
        state_dr = np.arange(counts["nstatic"], counts["nstatic"] + counts["npred"] + counts["nboth"])
        k2 = np.union1d(obs_dr, state_dr)                       # dr.restrict_var_list
        self._restrict = order_var[k2]                          # declaration indices of the filter states
        self._mf = np.searchsorted(k2, obs_dr)                  # bayestopt_.mf1
        self._restrict_columns = np.searchsorted(k2, state_dr)  # dr.restrict_columns
        self._state_var = state_var
        # variables that are not logged under loglinear: auxiliaries for lagged/leaded exogenous
        skip = [d["endo_index"] - 1 for d in m.raw.get("aux_vars", []) if d.get("type") in (2, 3)]
        self._loggable = np.ones(m.n_endo, bool)
        self._loggable[skip] = False

    def solve(self, theta):
        """Steady state and first-order solution at ``theta``.

        Returns ``(ys, G, H, params)`` with ``y_t - ys = G (y_{t-1} - ys) + H u_t`` in declaration
        order (in logs when ``loglinear`` is set) and the parameter vector actually used. Raises
        ``SteadyStateError`` or ``BlanchardKahnError``."""
        m = self.model
        ys, params = steady_state(m, self.model_params(theta))
        params = np.array(params, float)
        finite = np.isfinite(params)
        if "dsge_prior_weight" in m._ip:                    # dsge_var=Inf: an infinite weight is legitimate
            finite[m._ip["dsge_prior_weight"]] = True
        if not np.all(finite):
            raise SteadyStateError("parameters are NaN or Inf after the steady-state computation")
        n = m.n_endo
        J = m.dynamic_jac(np.tile(ys, 3), m.exo_steady(), params, ys)
        if not np.all(np.isfinite(J)):
            raise BlanchardKahnError("the Jacobian of the dynamic model contains NaN or Inf")
        try:
            G, H, _ = solve_linear_re(J[:, :n], J[:, n:2 * n], J[:, 2 * n:3 * n], J[:, 3 * n:],
                                      qz_criterium=self.options["qz_criterium"])
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise BlanchardKahnError(f"first-order solution failed: {exc}") from None
        mask = np.ones(n, bool)
        mask[self._state_var] = False
        G[:, mask] = 0.0
        if self.options["loglinear"]:
            if np.any(ys[self._loggable] < 1e-16):
                raise SteadyStateError("loglinear: the steady state of some variables is not strictly positive")
            scale = np.where(self._loggable, ys, 1.0)
            G = G * scale[None, :] / scale[:, None]
            H = H / scale[:, None]
        return ys, G, H, params

    def state_space(self, theta):
        """Matrices of the filter at ``theta``: dict with ``T``, ``R``, ``Q``, ``H``, the index
        vector ``z`` of the observed variables in the state vector, the ``constant`` of the
        measurement equation, the steady state ``ys`` and the model parameters ``params``."""
        theta = np.asarray(theta, float)
        Q, Hm = self.covariances(theta)
        ys, G, Hs, params = self.solve(theta)
        r = self._restrict
        const = ys[self._obs_decl]
        if self.options["loglinear"]:
            const = np.log(const)
        return {"T": G[np.ix_(r, r)], "R": Hs[r, :], "Q": Q, "H": Hm, "z": self._mf, "constant": const,
                "ys": ys, "params": params, "G": G, "Hfull": Hs,
                "state_names": [self.model.endo_names[i] for i in r]}

    # ------------------------------------------------------------------ data
    def _setup_trends(self):
        """Linear trends of an ``observation_trends`` block: coefficient expressions per observable."""
        self._trend_expr = None
        for st in self.model.statements:
            if st.get("statementName") == "observation_trends":
                tr = st.get("trends", st.get("vals", []))
                table = {d.get("var", d.get("name")): d.get("trend", d.get("value")) for d in tr} \
                    if isinstance(tr, list) else dict(tr)
                unknown = [v for v in table if v not in self.varobs]
                if unknown:
                    raise EstimationError(f"observation_trends on non-observed variables: {unknown}")
                self._trend_expr = [table.get(v) for v in self.varobs]

    def _trend(self, params, nobs):
        if self._trend_expr is None:
            return None
        coeff = np.array([0.0 if e is None else float(self.model.eval_expr(e, {"params": params}))
                          for e in self._trend_expr])
        first = int(self.options["first_obs"])
        add = np.outer(coeff, np.arange(first, first + nobs))
        if self.options["prefilter"]:
            add = add - add.mean(axis=1, keepdims=True)
        return add

    def _setup_data(self, data):
        o = self.options
        data_command = [st for st in self.model.statements if st.get("statementName") == "estimation_data"]
        if data is None:
            if not o["datafile"] and not data_command:
                raise EstimationError("no data: pass data=... or set the datafile option")
            data = o["datafile"] or data_command[-1].get("options", {}).get("file")
        base = self.model.json_dir.parents[2] if hasattr(self.model, "json_dir") else None
        if not isinstance(data, (dict, np.ndarray)):                   # data command, dates (pydyn.dates)
            from .dates import apply_data_command
            if isinstance(data, (str, Path)) and data != o["datafile"]:
                o = dict(o, datafile=data)
            self.first_date = apply_data_command(self.model, o, base)
            self.options.update({k: o[k] for k in ("datafile", "first_obs", "nobs")})
            o, data = self.options, o["datafile"]
        else:
            self.first_date = None
        raw = load_data(data, self.varobs, base)
        if o["loglinear"] and not o["logdata"]:
            with np.errstate(all="ignore"):
                raw = np.log(raw)
        first = int(o["first_obs"])
        nobs = raw.shape[0] - first + 1 if o["nobs"] is None else int(o["nobs"])
        if first < 1 or nobs < 1 or first - 1 + nobs > raw.shape[0]:
            raise EstimationError(f"first_obs={first} and nobs={nobs} are not consistent with the "
                                  f"{raw.shape[0]} observations in the data")
        self._alldata = raw                              # with the observations before first_obs (DSGE-VAR)
        raw = raw[first - 1:first - 1 + nobs]
        self.rawdata = raw.copy()
        if o["prefilter"]:
            raw = raw - np.nanmean(raw, axis=0)
        self.data = raw
        self.nobs = nobs
        nan = np.isnan(raw)
        self.missing = bool(nan.any())
        if self.missing:
            self._data_index = [np.flatnonzero(~nan[t]) for t in range(nobs)]
            rows = np.flatnonzero(nan.any(axis=1))
            self._no_more_missing = int(rows[-1]) + 1          # 0-based first period without gaps after
        else:
            self._data_index, self._no_more_missing = None, 0
        if o["presample"] >= nobs:
            raise EstimationError("presample must be smaller than the number of observations")

    def check_prefilter(self, theta=None):
        """True if the observed variables have a zero steady state (zero log steady state under
        ``loglinear``); Dynare refuses ``prefilter`` otherwise."""
        ss = self.state_space(self.x0 if theta is None else theta)
        return bool(np.all(np.abs(ss["constant"]) < 1e-9))

    # ------------------------------------------------------------------ likelihood
    def _minus_loglik(self, theta, return_lik=False):
        """Minus the log likelihood; raises _Rejected with Dynare's reason when it is not defined."""
        o = self.options
        theta = np.asarray(theta, float)
        if o["dsge_var"] is not None and o["dsge_var"] is not False:
            return self._dsge_var_minus_loglik(theta, return_lik)
        Q, Hm = self.covariances(theta)
        if self._groups["cx"] and not _ispd(Q[np.ix_(self._Q_check, self._Q_check)]):
            raise _Rejected("covariance matrix of the structural shocks is not positive definite (info 43)",
                            43, _pd_penalty(Q[np.ix_(self._Q_check, self._Q_check)]))
        if (self._groups["cn"] or self._calibrated_cov_ME) and not _ispd(Hm[np.ix_(self._H_check, self._H_check)]):
            raise _Rejected("covariance matrix of the measurement errors is not positive definite (info 44)",
                            44, _pd_penalty(Hm[np.ix_(self._H_check, self._H_check)]))
        if self._calibrated_cov_ME and not _consistent_covariances(Hm):
            pos = [(i, j) for i, j in self._calibrated_cov_ME]
            raise _Rejected("the calibrated covariances of the measurement errors are inconsistent with their variances "
                            "(info 72)", 72, float(sum(Hm[i, j] ** 2 for i, j in pos) * 2))
        try:
            ys, G, Hs, params = self.solve(theta)
        except SteadyStateError as exc:
            raise _Rejected(f"steady state: {exc}", 20) from None
        except BlanchardKahnError as exc:
            code, pen = self._bk_penalty(theta)
            raise _Rejected(f"solution: {exc}", code, pen) from None
        r = self._restrict
        T, R, z = G[np.ix_(r, r)], Hs[r, :], self._mf
        if not (np.all(np.isfinite(T)) and np.all(np.isfinite(R))):
            raise _Rejected("the decision rule contains NaN or Inf", 6)
        const = ys[self._obs_decl]
        if o["loglinear"]:
            const = np.log(const)
        Y = self.data.T - const[:, None]
        trend = self._trend(params, self.nobs)
        if trend is not None:
            Y = Y - trend
        QQ = R @ Q @ R.T
        QQf = self._heteroskedastic_QQ(R, Q)          # QQ, or its (mm, mm, nobs + 1) path
        mm = T.shape[0]
        a = np.zeros(mm)
        pre = int(o["presample"])
        kw = dict(kalman_tol=o["kalman_tol"], riccati_tol=o["riccati_tol"], presample=pre)
        dkw = dict(kalman_tol=o["kalman_tol"], diffuse_kalman_tol=o["diffuse_kalman_tol"], presample=pre,
                   data_index=self._data_index)
        algo = o["kalman_algo"]
        dLIK, dp, dlik = 0.0, 0, np.zeros(0)

        pp_ = len(z)
        correlated = bool(np.any(np.abs(Hm - np.diag(np.diag(Hm))) >= 1e-14))

        def Zmat():
            """Selection matrix and measurement-error variances of the univariate filters.  With correlated measurement
            errors Dynare augments the state vector by the errors (Durbin-Koopman 2012, section 6.4.3): see ``aug_sys``."""
            Zm = np.zeros((pp_, mm))
            Zm[np.arange(pp_), z] = 1.0
            if correlated:
                return np.hstack([Zm, np.eye(pp_)]), np.zeros(pp_)
            return Zm, np.diag(Hm).copy()

        def aug_sys(T_, QQ_):
            """T and QQ = R Q R' of the augmented system (identity when the measurement errors are uncorrelated)."""
            if not correlated:
                return T_, QQ_
            Ta = np.zeros((mm + pp_, mm + pp_))
            Ta[:mm, :mm] = T_
            if QQ_.ndim == 3:
                Qa = np.zeros((mm + pp_, mm + pp_, QQ_.shape[2]))
                Qa[:mm, :mm, :] = QQ_
                Qa[mm:, mm:, :] = Hm[:, :, None]
            else:
                Qa = np.zeros((mm + pp_, mm + pp_))
                Qa[:mm, :mm] = QQ_
                Qa[mm:, mm:] = Hm
            return Ta, Qa

        def aug_P(P_, H_block):
            if not correlated:
                return P_
            Pa = np.zeros((mm + pp_, mm + pp_))
            Pa[:mm, :mm] = P_
            Pa[mm:, mm:] = H_block
            return Pa

        def aug_a(a_):
            return np.concatenate([a_, np.zeros(pp_)]) if correlated else a_

        if o["lik_init"] == 1:
            algo = 2 if algo == 2 else 1
            P = kalman.lyapunov_symm(T, QQ, o["qz_criterium"], o["lyapunov_complex_threshold"])
        elif o["lik_init"] == 2:
            algo = 2 if algo == 2 else 1
            P = o["Harvey_scale_factor"] * np.eye(mm)
            if o["Harvey_scale_factor"] == 0:
                P = T @ P @ T.T + QQ
        else:
            algo = 3 if algo == 0 else algo
            Pstar, Pinf = kalman.compute_Pinf_Pstar(z, T, R, Q, o["qz_criterium"], self._restrict_columns)
            singular = False
            if algo == 3:
                dLIK, dlik, a1, P = kalman.kalman_filter_d(Y, a, Pinf, Pstar, T, QQf, Hm, z, **dkw)
                if np.isinf(dLIK):
                    singular = True               # as Dynare: redo the diffuse stage observation by observation
                else:
                    a = a1
            if singular or algo == 4:
                Zm, Hd = Zmat()
                Tu, QQu = aug_sys(T, QQf)
                dLIK, dlik, a, P = kalman.univariate_kalman_filter_d(
                    Y, np.zeros(mm + (pp_ if correlated else 0)), aug_P(Pinf, np.zeros((pp_, pp_))), aug_P(Pstar, Hm),
                    Tu, QQu, Hd, Zm, **dkw)
            if np.isnan(dLIK):
                raise _Rejected("the diffuse filter did not leave the diffuse stage, or a variance is negative (info 45)", 45)
            dp = len(dlik)
        if not np.all(np.isfinite(P)):
            raise _Rejected("initial state covariance is not finite", 45)
        di = None if self._data_index is None else self._data_index[dp:]
        nmm = max(self._no_more_missing - dp, 0)
        lik = None
        if QQf.ndim == 3:
            QQ = QQf[:, :, dp:]
        if algo in (1, 3):
            out = kalman.kalman_filter(Y[:, dp:], a, P, T, QQ, Hm, z, diffuse_periods=dp, data_index=di,
                                       no_more_missing=nmm, return_lik=True,
                                       rescale=bool(o["rescale_prediction_error_covariance"]), **kw)
            LIK, lik = out
            if np.isinf(LIK):
                if not o["use_univariate_filters_if_singularity_is_detected"]:
                    raise _Rejected("the forecast-error covariance is singular (info 50)", 50)
                algo = 2 if algo == 1 else 4
        if algo in (2, 4):
            Zm, Hd = Zmat()
            Tu, QQu = aug_sys(T, QQ)
            if correlated and P.shape[0] == mm:      # augment the state of the stationary (or large-variance) start
                P, a = aug_P(P, Hm), aug_a(a)
            LIK, lik = kalman.univariate_kalman_filter(Y[:, dp:], a, P, Tu, QQu, Hd, Zm, diffuse_periods=dp,
                                                       data_index=di, no_more_missing=nmm, return_lik=True, **kw)
        self.last_filter = {1: "multivariate", 2: "univariate", 3: "multivariate (diffuse)",
                            4: "univariate (diffuse)"}[algo]
        if np.isinf(LIK):
            raise _Rejected("the forecast-error covariance is singular (info 50)", 50)
        LIK = LIK + dLIK
        if np.isnan(LIK) or np.iscomplexobj(LIK):
            raise _Rejected("the likelihood is NaN or complex (info 45/46)", 45)
        if return_lik:
            return LIK, np.concatenate([dlik, lik])[pre:]
        return LIK

    def _dsge_var_minus_loglik(self, theta, return_lik=False):
        """``dsge_var_likelihood.m`` (pydyn.dsge_var) in place of the Kalman-filter likelihood."""
        from . import dsge_var
        if return_lik:
            raise NotImplementedError("DSGE-VAR: no per-observation likelihood")
        o = self.options
        nlag = int(o["dsge_varlag"])
        if getattr(self, "_dvar_moments", None) is None:
            dsge_var.check_model(self)
            first = int(o["first_obs"])
            if first - nlag < 1:
                raise EstimationError(f"DSGE-VAR: first_obs ({first}) - {nlag} cannot be less than the first "
                                      "observation in the dataset; increase first_obs")
            self._dvar_moments = dsge_var.sample_moments(self._alldata[first - 1 - nlag:first - 1 + self.nobs], nlag)
        Q, _ = self.covariances(theta)
        if not _ispd(Q):
            raise _Rejected("covariance matrix of the structural shocks is not positive definite (info 43)",
                            43, _pd_penalty(Q))
        lam = dsge_var.weight_value(self.model, self.model_params(theta))
        n = len(self.varobs)
        k = n * nlag + 1
        TT = self.nobs + nlag
        if lam < (k + n) / TT:
            raise _Rejected("dsge_prior_weight is below (number of VAR parameters + observables) / T (info 51)",
                            51, abs(TT * lam - (k + n)))
        try:
            ys, G, Hs, params = self.solve(theta)
        except SteadyStateError as exc:
            raise _Rejected(f"steady state: {exc}", 20) from None
        except BlanchardKahnError as exc:
            code, pen = self._bk_penalty(theta)
            raise _Rejected(f"solution: {exc}", code, pen) from None
        r = self._restrict
        const = ys[self._obs_decl]
        if o["loglinear"]:
            const = np.log(const)
        lik, code, pen = dsge_var.minus_loglik(G[np.ix_(r, r)], Hs[r, :], Q, self._mf, const, lam, TT, nlag,
                                               self._dvar_moments, o["qz_criterium"], o["lyapunov_complex_threshold"])
        if code:
            raise _Rejected(f"DSGE-VAR likelihood not defined (info {code})", code, pen)
        self.last_filter = "DSGE-VAR"
        return lik

    def _heteroskedastic_QQ(self, R, Q):
        """``R Q R'`` or, with the heteroskedastic_filter option, its path over the sample (pydyn.heteroskedastic)."""
        if not self.options["heteroskedastic_filter"]:
            return R @ Q @ R.T
        from . import heteroskedastic
        key = (self.nobs, int(self.options["first_obs"]), getattr(self, "first_date", None))
        if getattr(self, "_het_key", None) != key:
            self._het_arrays = heteroskedastic.q_arrays(self.model, *key)
            self._het_key = key
        return heteroskedastic.rqr(R, heteroskedastic.qvec(Q, *self._het_arrays))

    def log_likelihood(self, theta) -> float:
        """Log likelihood of the data at ``theta``; ``-inf`` if the steady state or the solution
        fails, or the filter breaks down (the reason is kept in ``last_info``)."""
        self.last_info, self.last_code, self.last_penalty = "", 0, 0.0
        try:
            with np.errstate(all="ignore"):
                return -float(self._minus_loglik(theta))
        except (_Rejected, FloatingPointError, np.linalg.LinAlgError) as exc:
            self.last_info = str(exc)
            self.last_code, self.last_penalty = getattr(exc, "code", 45), getattr(exc, "penalty", 0.1)
            return -np.inf

    def log_likelihood_contributions(self, theta) -> np.ndarray:
        """Log density of each observation after the presample."""
        with np.errstate(all="ignore"):
            return -self._minus_loglik(theta, return_lik=True)[1]

    def log_prior(self, theta) -> float:
        """Log prior density (0 for maximum likelihood)."""
        with np.errstate(all="ignore"):
            return self.prior.logpdf(theta)

    def log_posterior(self, theta, bounds=None) -> float:
        """Log posterior kernel, minus the value of Dynare's ``dsge_likelihood``: ``-inf`` outside
        the bounds, when the model cannot be solved, or when the prior density is zero.

        ``bounds=(lb, ub)`` replaces ``self.lb``/``self.ub`` (the bounds of the ``estimated_params``
        block intersected with the prior support). The sampler passes the prior support alone,
        because that is what Dynare's Metropolis-Hastings uses."""
        theta = np.asarray(theta, float)
        lb, ub = (self.lb, self.ub) if bounds is None else bounds
        self.last_info, self.last_code, self.last_penalty = "", 0, 0.0
        if np.any(theta < lb):
            k = theta < lb
            self.last_info, self.last_code = "below the lower bound (info 41)", 41
            self.last_penalty = float(np.sum((lb[k] - theta[k]) ** 2))
            return -np.inf
        if np.any(theta > ub):
            k = theta > ub
            self.last_info, self.last_code = "above the upper bound (info 42)", 42
            self.last_penalty = float(np.sum((theta[k] - ub[k]) ** 2))
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            self.last_info = "the prior density is zero, infinite or NaN (info 40/47)"
            self.last_code, self.last_penalty = 40, 0.1
            return -np.inf
        if self.options["endogenous_prior"]:
            lp = lp + self._endogenous_prior(theta)
        return ll + lp

    def _endogenous_prior(self, theta):
        """``lnpriormom`` of ``endogenous_prior.m`` (pydyn.endogenous_prior), added to the log prior."""
        from .endogenous_prior import lnpriormom
        if self.options["lik_init"] != 1:
            raise EstimationError("Endogenous prior not supported with non-stationary models")
        ss = self.state_space(theta)
        T, R, Q = ss["T"], ss["R"], ss["Q"]
        Pstar = kalman.lyapunov_symm(T, R @ Q @ R.T, self.options["qz_criterium"],
                                     self.options["lyapunov_complex_threshold"])
        Y = self.data.T - ss["constant"][:, None]
        trend = self._trend(ss["params"], self.nobs)
        if trend is not None:
            Y = Y - trend
        return lnpriormom(Y[:, self._no_more_missing:], Pstar, ss["z"], ss["H"])

    def objective(self, theta, bounds=None):
        """Dynare's objective function: ``(fval, code, penalty)`` with ``fval`` minus the log posterior
        kernel (``inf`` where it is not defined), and Dynare's ``info(1)`` and ``info(4)``. The
        penalty is exact for bound violations and non-positive-definite covariance matrices and for
        Blanchard-Kahn failures; for a failing steady state Dynare uses the squared residuals, here
        0.1 is returned (as Dynare does for all other failures)."""
        lp = self.log_posterior(theta, bounds)
        if np.isfinite(lp):
            return -lp, 0, 0.0
        return np.inf, (self.last_code or 45), (self.last_penalty if self.last_code else 0.1)

    # ------------------------------------------------------------------ mode, Hessian, Laplace
    def hessian(self, theta, gstep=None):
        """Hessian of minus the log posterior at ``theta`` (Dynare's ``hessian.m``)."""
        return hessian(lambda x: -self.log_posterior(x), theta, gstep or self.options["gstep"])

    @staticmethod
    def inverse_hessian(hh):
        """Inverse computed after scaling to unit diagonal, as in ``dynare_estimation_1.m``."""
        hsd = np.sqrt(np.diag(hh))
        s = np.outer(hsd, hsd)
        return np.linalg.inv(hh / s) / s

    def laplace(self, theta, inv_hessian) -> float:
        """Laplace approximation of the log marginal data density."""
        stdh = np.sqrt(np.diag(inv_hessian))
        if np.any(~np.isfinite(stdh)) or not _ispd(inv_hessian):
            return np.nan
        logdet = np.log(np.linalg.det(inv_hessian / np.outer(stdh, stdh))) + 2 * np.sum(np.log(stdh))
        return 0.5 * self.n * np.log(2 * np.pi) + 0.5 * logdet + self.log_posterior(theta)

    def mode(self, x0=None, method=None, compute_hessian=None, tol=1e-9, maxiter=5000, polish=False,
             mode_compute=None, mode_file=None, optim=None, additional_optimizer_steps=None, rand=None,
             verbose=False):
        """Posterior mode (maximum-likelihood estimate without priors), its Hessian and the Laplace
        approximation, starting from ``x0`` (default: the initial values of the ``estimated_params``
        block, else the prior means).

        Which optimiser runs is decided by ``method``/``mode_compute`` (keyword, else the
        ``mode_compute`` option of the ``estimation`` command if the ``.mod`` file gives one):

        ``4`` (the default when nothing is specified)
            :func:`csminwel`, a port of Dynare's ``csminwel1.m``. It follows Dynare's iterates
            (see :func:`csminwel` for how closely). ``optim`` options: ``MaxIter``, ``TolFun``,
            ``NumgradAlgorithm``, ``NumgradEpsilon``, ``InitialInverseHessian`` (a matrix, or a
            string of the form ``'c*eye(n)'``), ``verbosity``; ``SaveFiles`` has no effect.
            Dynare's own default is ``mode_compute=5`` (newrat, ported, see below); csminwel stays
            pydyn's default (a one-time warning says so) because it is a Dynare algorithm validated
            evaluation by evaluation against Dynare, and because on the two reference models it
            needed 2-3 times fewer evaluations than the ``"scipy"`` routine and stopped within
            1e-4 posterior standard deviations of Dynare's csminwel optimum (``"scipy"``: 3e-3).
            Like Dynare it does not enforce bounds: outside them the objective returns Dynare's
            penalty.
        ``"scipy"``
            Two bounded quasi-Newton searches with numerical gradients, scipy's SLSQP and then
            L-BFGS-B from its solution, keeping the best point: pydyn's own routine (the default of
            earlier versions), not a Dynare algorithm; it keeps every iterate inside the bounds of
            the ``estimated_params`` block. ``polish=True`` adds restarts of a simplex search. The
            old name ``"auto"`` is accepted.
        ``1``, ``3``, ``7``, ``8``, ``9``
            scipy stand-ins for MATLAB's ``fmincon`` (SLSQP with the bounds), ``fminunc`` (BFGS),
            ``fminsearch`` and Dynare's simplex routine (both Nelder-Mead), and Dynare's CMA-ES
            (``differential_evolution``, another population-based global search, followed by
            L-BFGS-B; it needs a box: infinite bounds are replaced by the start plus or minus ten
            prior standard deviations, or ten times the start's magnitude). These are *equivalents,
            not ports*: same class of algorithm, different iterates, and a warning says so once.
            ``optim`` options ``MaxIter``, ``MaxFunEvals``, ``TolFun``, ``TolX`` are passed on
            (for 9: ``MaxIter`` generations; ``seed`` from ``rand`` if it is an integer, else 0).
        ``0``
            No optimisation. With ``mode_file`` the mode (and Hessian, if stored) are read from an
            ``.npz`` file written by :meth:`save_mode` or from a Dynare ``_mode.mat`` file; the
            Hessian is recomputed only if the file has none. Without ``mode_file`` Dynare would
            take the initial values as the mode: ``mode()`` being an explicit request for a mode,
            pydyn then runs csminwel and says so in a warning.
        any other scipy ``minimize`` method name
            used as is, without bounds.

        ``5``
            newrat (Dynare's default optimiser), ported in :mod:`pydyn.newrat` (``optim``: ``TolFun``,
            ``MaxIter``, ``TolGstep``, ``TolGstepRel``, ``robust``; ``Hessian=1`` only).

        Other values of ``mode_compute`` (2, 6, 10, ...) raise ``NotImplementedError``.
        ``additional_optimizer_steps``: further ``mode_compute`` values run one after the other from
        the previous optimum. ``cova_compute=0`` (or ``compute_hessian=False``) skips the Hessian:
        the result then has no standard deviations and no Laplace approximation.
        ``use_penalized_objective_for_hessian`` is honoured. ``rand`` supplies csminwel's random
        numbers (see :func:`csminwel`).
        """
        from . import options as _opt
        o = self.options
        if method == "auto":
            method = "scipy"
        first = method if method is not None else (mode_compute if mode_compute is not None else o["mode_compute"])
        if first is None:
            _opt.warn_once(self.model, "mode_compute not given: Dynare's default is 5 (newrat, available as "
                           "mode_compute=5); Estimation.mode() uses 4 (csminwel) as its default")
            first = 4
        steps = [first]
        extra = additional_optimizer_steps if additional_optimizer_steps is not None else o["additional_optimizer_steps"]
        steps += list(np.atleast_1d(extra)) if extra is not None and np.size(extra) else []
        mode_file = mode_file if mode_file is not None else o["mode_file"]
        opt = _parse_optim(optim if optim is not None else o["optim"])
        if compute_hessian is None:
            compute_hessian = bool(o["cova_compute"])
        x = np.array(self.x0 if x0 is None else x0, float)
        nfev = [0]
        big = 1e12

        def f(v):
            nfev[0] += 1
            val = -self.log_posterior(v)
            return val if np.isfinite(val) else big

        msgs, ok, loaded_hessian = [], True, None
        niter = 0
        for istep, step in enumerate(steps):
            if isinstance(step, (float, np.floating, np.integer)) or (isinstance(step, str) and step.lstrip("-").isdigit()):
                step = int(step)
            if step is None:
                step = 4
            if step == 0:
                if mode_file:
                    x, loaded_hessian = self.load_mode(mode_file)
                    msgs.append(f"mode read from {mode_file}")
                    continue
                if x0 is None and istep == 0:
                    _opt.warn_once(self.model, "mode_compute=0 without mode_file: Dynare would use the initial values "
                                   "as the mode; Estimation.mode() runs its default optimiser (4, csminwel) instead")
                step = 4
            fx = f(x)
            if fx >= big:
                raise EstimationError(f"the log posterior is not finite at the initial values: {self.last_info}")
            if step == "scipy":
                width = np.where(np.isfinite(self.ub - self.lb), self.ub - self.lb, 1.0)
                bnds = [(lo + 1e-10 * w if np.isfinite(lo) else None, hi - 1e-10 * w if np.isfinite(hi) else None)
                        for lo, hi, w in zip(self.lb, self.ub, width)]
                stages = [("SLSQP", {"ftol": 1e-12, "eps": 1e-7, "maxiter": maxiter}),
                          ("L-BFGS-B", {"ftol": 1e-13, "gtol": 1e-7, "eps": 1e-7, "maxiter": maxiter, "maxls": 40})]
                for name, sopt in stages:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            res = optimize.minimize(f, x, method=name, bounds=bnds, options=sopt)
                    except (ValueError, FloatingPointError) as exc:
                        msgs.append(f"{name}: failed ({exc})")
                        continue
                    fr = f(res.x)
                    if fr < fx:
                        x, fx = np.array(res.x), fr
                    msgs.append(f"{name}: {res.message}")
                if polish:
                    for _ in range(8):
                        res = optimize.minimize(f, x, method="Nelder-Mead",
                                                options={"xatol": 1e-10, "fatol": tol, "maxiter": 4 * maxiter,
                                                         "maxfev": 4 * maxiter, "adaptive": True})
                        improved = fx - res.fun
                        if res.fun < fx:
                            x, fx = np.array(res.x), res.fun
                        if improved < tol:
                            break
                    msgs.append(f"Nelder-Mead: {res.message}")
            elif step == 4:
                H0 = opt.get("InitialInverseHessian")
                if isinstance(H0, str):
                    mt = re.fullmatch(r"\s*([-+0-9.eE]+)\s*\*\s*eye\(\s*(\d+)\s*\)\s*", H0)
                    if not mt or int(mt.group(2)) != self.n:
                        raise NotImplementedError(f"optim: InitialInverseHessian={H0!r} must be a matrix or 'c*eye({self.n})'")
                    H0 = float(mt.group(1)) * np.eye(self.n)
                unknown = set(opt) - {"MaxIter", "InitialInverseHessian", "TolFun", "NumgradAlgorithm", "NumgradEpsilon",
                                      "verbosity", "SaveFiles"}
                if unknown:
                    warnings.warn(f"csminwel: unknown option(s) {sorted(unknown)}")      # as Dynare
                if "SaveFiles" in opt and opt["SaveFiles"]:
                    _opt.warn_once(self.model, "optim: SaveFiles has no effect (csminwel writes no files in pydyn)")

                def counted(v):
                    nfev[0] += 1
                    return self.objective(v)
                res = csminwel(counted, x, H0=H0, crit=float(opt.get("TolFun", 1e-7)), nit=int(opt.get("MaxIter", 1000)),
                               method=int(opt.get("NumgradAlgorithm", o["gradient_method"])),
                               epsilon=float(opt.get("NumgradEpsilon", o["gradient_epsilon"])), rand=rand,
                               verbose=verbose)
                if res["retcode"] == -1:
                    raise EstimationError(f"csminwel: bad initial parameter ({self.last_info})")
                x = res["x"]
                niter += res["iterations"]
                self.last_csminwel = res
                msgs.append(f"csminwel: {res['message']} ({res['iterations']} iterations, {res['fcount']} evaluations)")
            elif step == 5:                       # newrat, Dynare's default (pydyn.newrat)
                from . import newrat as _newrat
                res = _newrat.run_mode(self, x, opt, nfev, verbose)
                x = res["x"]
                niter += res["iterations"]
                self.last_newrat = res
                msgs.append(f"newrat: {res['message']} ({res['iterations']} iterations, {res['fcount']} evaluations)")
            elif step in _MODE_COMPUTE_EQUIVALENTS:
                name, matlab = _MODE_COMPUTE_EQUIVALENTS[step]
                _opt.warn_once(self.model, f"mode_compute={step}: Dynare uses {matlab}; pydyn uses scipy's {name}, "
                               "an equivalent method and not a port: the path and possibly the local optimum differ")
                if step == 9:
                    res = self._evolutionary_mode(f, x, opt, rand)
                    if f(res.x) < fx:
                        x = np.array(res.x)
                    ok = ok and bool(res.success)
                    msgs.append(f"differential_evolution: {res.message}")
                    continue
                sopt = {"maxiter": int(opt.get("MaxIter", 6000 if step in (7, 8) else maxiter))}
                kw = {}
                if step == 1:
                    width = np.where(np.isfinite(self.ub - self.lb), self.ub - self.lb, 1.0)
                    kw["bounds"] = [(lo + 1e-10 * w if np.isfinite(lo) else None, hi - 1e-10 * w if np.isfinite(hi) else None)
                                    for lo, hi, w in zip(self.lb, self.ub, width)]
                    sopt.update(ftol=float(opt.get("TolFun", 1e-8)), eps=1e-7)
                elif step == 3:
                    sopt.update(gtol=float(opt.get("TolFun", 1e-8)) * 10, eps=1e-7)
                else:
                    sopt.update(xatol=float(opt.get("TolX", 1e-6)), fatol=float(opt.get("TolFun", 1e-8)),
                                maxfev=int(opt.get("MaxFunEvals", 1000000)), adaptive=(step == 8))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = optimize.minimize(f, x, method=name, options=sopt, **kw)
                    if step == 8:                 # Dynare's routine restarts the simplex; do so until no gain
                        for _ in range(5):
                            res2 = optimize.minimize(f, res.x, method=name, options=sopt)
                            gain = res.fun - res2.fun
                            if res2.fun < res.fun:
                                res = res2
                            if gain < float(opt.get("TolFun", 1e-8)):
                                break
                if f(res.x) < fx:
                    x = np.array(res.x)
                ok = ok and bool(res.success)
                msgs.append(f"{name}: {res.message}")
            elif isinstance(step, str):
                res = optimize.minimize(f, x, method=step, tol=tol, options={"maxiter": maxiter})
                if f(res.x) < fx:
                    x = np.array(res.x)
                ok = ok and bool(res.success)
                msgs.append(str(res.message))
            else:
                raise NotImplementedError(
                    f"mode_compute={step} is not implemented: use 4 (csminwel, ported from Dynare), 1, 3, 7, 8 or 9 "
                    "(scipy equivalents), 0 with mode_file, or method='scipy'")
        out = ModeResult(x=np.array(x), names=list(self.names), log_posterior=self.log_posterior(x),
                         success=ok, message="; ".join(msgs), nfev=nfev[0])
        if loaded_hessian is not None:
            out.hessian = loaded_hessian
            self._finish_hessian(out)
        elif compute_hessian:
            self.add_hessian(out)
        return out

    def _evolutionary_mode(self, f, x, opt, seed):
        """Stand-in for mode_compute=9 (Dynare's CMA-ES): scipy's differential evolution on a box, then
        L-BFGS-B. Finite bounds are those of the estimated_params block and the priors; an infinite bound
        becomes the start plus or minus ten prior standard deviations (ten times max(|start|, 1) without
        a finite prior standard deviation). The start is put in the initial population."""
        p2 = np.asarray(self.prior.p2, float) if self.prior.is_bayesian else np.full(self.n, np.nan)
        half = np.where(np.isfinite(p2) & (p2 > 0), 10 * p2, 10 * np.maximum(np.abs(x), 1.0))
        lo = np.where(np.isfinite(self.lb), self.lb, x - half)
        hi = np.where(np.isfinite(self.ub), self.ub, x + half)
        width = hi - lo
        lo, hi = lo + 1e-10 * width, hi - 1e-10 * width
        rng = np.random.default_rng(int(seed) if isinstance(seed, (int, np.integer)) else 0)
        popsize = 15
        init = lo + (hi - lo) * rng.random((popsize * self.n, self.n))
        init[0] = np.clip(x, lo, hi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return optimize.differential_evolution(
                f, list(zip(lo, hi)), maxiter=int(opt.get("MaxIter", 1000)), tol=float(opt.get("TolFun", 1e-8)),
                init=init, polish=True, seed=rng, updating="immediate")

    def add_hessian(self, mode: ModeResult, penalized=None):
        """Compute the Hessian, its inverse and the Laplace approximation at ``mode.x``.
        ``penalized`` (default: the ``use_penalized_objective_for_hessian`` option): where the
        objective is not defined use its value at the mode plus Dynare's penalty instead of
        infinity."""
        penalized = self.options["use_penalized_objective_for_hessian"] if penalized is None else penalized
        with np.errstate(invalid="ignore"):
            if penalized:
                base = -mode.log_posterior

                def fun(x):
                    fval, code, pen = self.objective(x)
                    return base + pen if code else fval
                mode.hessian = hessian(fun, mode.x, self.options["gstep"])
            else:
                mode.hessian = self.hessian(mode.x)
        return self._finish_hessian(mode)

    def _finish_hessian(self, mode: ModeResult):
        if np.all(np.isfinite(mode.hessian)) and np.all(np.diag(mode.hessian) > 0):
            mode.inv_hessian = self.inverse_hessian(mode.hessian)
            if self.prior.is_bayesian:
                mode.laplace = self.laplace(mode.x, mode.inv_hessian)
        return mode

    def save_mode(self, path, mode: ModeResult):
        """Write a mode to an ``.npz`` file that ``mode(mode_compute=0, mode_file=path)`` reads back:
        arrays ``xparam1``, ``hh`` (Hessian of minus the log posterior; absent if not computed),
        ``fval`` (minus the log posterior) and ``parameter_names``: the names of Dynare's mode files."""
        data = {"xparam1": np.asarray(mode.x, float), "fval": -float(mode.log_posterior),
                "parameter_names": np.array(list(mode.names))}
        if mode.hessian is not None:
            data["hh"] = np.asarray(mode.hessian, float)
        np.savez(path, **data)

    def load_mode(self, path):
        """Mode and Hessian (or None) from an ``.npz`` file written by :meth:`save_mode` or from a
        Dynare mode file (``<model>/Output/<model>_mode.mat``, also ``_mh_mode.mat``). The file is
        looked up as given, with ``.npz`` or ``.mat`` appended, and next to the ``.mod`` file.
        Parameters are matched by name."""
        base = self.model.json_dir.parents[2] if hasattr(self.model, "json_dir") else Path(".")
        cands = [Path(path)] + [Path(str(path) + ext) for ext in (".npz", ".mat")]
        cands += [base / c for c in cands if not c.is_absolute()]
        found = next((c for c in cands if c.is_file()), None)
        if found is None:
            raise EstimationError(f"mode_file {path}: no such .npz or .mat file")
        if found.suffix.lower() == ".npz":
            with np.load(found, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
        else:
            from scipy.io import loadmat
            data = loadmat(found, squeeze_me=True)
        if "xparam1" not in data:
            raise EstimationError(f"mode_file {found}: no xparam1 in the file")
        x = np.atleast_1d(np.asarray(data["xparam1"], float)).ravel()
        hh = np.atleast_2d(np.asarray(data["hh"], float)) if "hh" in data and np.size(data["hh"]) else None
        names = [str(v).strip() for v in np.atleast_1d(data["parameter_names"])] if "parameter_names" in data else None
        if x.size != self.n:
            raise EstimationError(f"mode_file {found}: {x.size} parameters in the file, {self.n} are estimated")
        if names is not None and names != self.names:
            if sorted(names) != sorted(self.names):
                raise EstimationError(f"mode_file {found}: the parameter names do not match the estimated parameters")
            k = [names.index(v) for v in self.names]
            x = x[k]
            hh = None if hh is None else hh[np.ix_(k, k)]
        return x, hh

    # ------------------------------------------------------------------ Metropolis-Hastings
    def jscale(self, mh_jscale=None):
        """Scale of the jumping distribution for every parameter (``bayestopt_.jscale``)."""
        base = mh_jscale if mh_jscale is not None else self.options["mh_jscale"]
        if base is None:
            base = 2.38 / np.sqrt(self.n)
        js = self.jscale_individual.copy()
        js[np.isnan(js)] = base
        return js, float(base)

    def sample(self, replic=None, nblocks=None, mode=None, cov=None, jscale=None, seed=0, x0=None,
               drop=None, init_scale_factor=None, innovations=None, uniforms=None, uniform_stream=False,
               init_innovations=None, jumping_covariance=None, tune_jscale=None, tune_guess=None,
               tune_options=None, load=None, save=None, **sampler_kw) -> MCMCResult:
        """Random-walk Metropolis-Hastings sampler (Dynare's default posterior sampler).

        The proposal is Gaussian, ``last + N(0, J V J)`` with ``J = diag(jscale)`` and ``V`` the
        jumping covariance: ``cov`` if given, else what ``jumping_covariance`` (default: the
        ``mcmc_jumping_covariance`` option) selects: ``'hessian'`` (the inverse Hessian at the mode,
        ``mode.inv_hessian``), ``'prior_variance'``, ``'identity_matrix'``, a matrix, or a file
        holding one (see :func:`pydyn.posterior.jumping_covariance`). As in Dynare the chains are
        confined to the support of the priors (``prior_trunc`` applies) and, as Dynare 7.2's code does
        (its manual says otherwise), proposals outside the bounds of the ``estimated_params`` block are
        rejected too.

        Other samplers (``posterior_sampling_method``: ``'slice'``, ``'tailored_random_block_metropolis_hastings'``,
        ``'independent_metropolis_hastings'``, ``'hssmc'``, ``'dsmh'``), ``posterior_sampler_options`` (e.g. a Student
        proposal) and Dynare's own random numbers (``rng='octave'``) are handled by :func:`pydyn.samplers.sample`;
        its extra keywords (``rng``, ``random``, ``sampler_options``, ``method``, ...) are passed through.

        Starting points, as ``posterior_sampler_initialization.m``: a single chain starts at the
        mode; with several chains each starts from the first draw
        ``N(mode, (mh_init_scale_factor * mh_jscale)^2 V)`` that lies inside the bounds and has a
        finite posterior (after 100 failures the scale is cut by 10 percent per further failure, and
        the search is abandoned after 10 cuts). ``x0`` (chains x parameters) overrides the starting
        points.

        ``tune_jscale`` (default: the ``mh_tune_jscale`` option): target acceptance rate; the scale
        is then calibrated before sampling (:func:`pydyn.posterior.tune_jscale`, ``tune_guess`` and
        ``tune_options`` are passed on) and used for all parameters. The scale used is stored in
        the result (``jscale``).

        ``drop`` (``mh_drop``) is the share of every chain treated as burn-in by the statistics; all
        draws are returned. ``save``: file name; the chains are written with
        :func:`pydyn.posterior.save_chains`. ``load``: file written that way (or an ``MCMCResult``);
        the chains are continued from their last draws with the stored proposal (Dynare's
        ``load_mh_file``) and the result holds the old and the new draws. ``replic=0`` with ``load``
        just returns the stored chains.

        Random numbers: ``seed`` seeds a numpy ``Generator``; every chain gets its own stream
        (a continued run derives new streams from the seed and the number of draws already made).
        ``innovations`` (chains, replic, n), ``uniforms`` (chains, replic) and ``init_innovations``
        (rows of standard normals for the starting points) replace them; with ``uniform_stream=True``
        the uniforms are consumed as Dynare consumes its stream (see :func:`metropolis_hastings`),
        which reproduces a Dynare chain draw for draw from the same random numbers.
        """
        from . import posterior as _post
        from . import samplers as _samplers
        o = self.options
        if not self.prior.is_bayesian:
            raise EstimationError("the sampler needs priors")
        if _samplers.dispatch(self, sampler_kw):  # other samplers, Student proposal, Dynare's random numbers
            return _samplers.sample(self, replic=replic, nblocks=nblocks, mode=mode, cov=cov, jscale=jscale, seed=seed,
                                    x0=x0, drop=drop, init_scale_factor=init_scale_factor,
                                    jumping_covariance=jumping_covariance, load=load, save=save, innovations=innovations,
                                    uniforms=uniforms, init_innovations=init_innovations, tune_jscale=tune_jscale,
                                    **sampler_kw)
        replic = int(o["mh_replic"] if replic is None else replic)
        drop = o["mh_drop"] if drop is None else drop
        previous = None
        if load is not None:
            previous = load if isinstance(load, MCMCResult) else _post.load_chains(load)
            if list(previous.names) != list(self.names):
                raise EstimationError("load: the stored chains are for other parameters")
            if previous.proposal_chol is None:
                raise EstimationError("load: the stored chains carry no proposal density")
            nblocks = previous.draws.shape[0]          # as Dynare: the previous number of chains is kept
            if replic == 0:
                return previous
        nblocks = int(o["mh_nblck"] if nblocks is None else nblocks)
        lb, ub = self.prior.bounds(o["prior_trunc"])              # set_mcmc_prior_bounds.m

        def log_post(x):                          # Dynare evaluates the posterior with these bounds
            return self.log_posterior(x, bounds=(lb, ub))
        ss = np.random.SeedSequence(seed if previous is None else [int(seed), previous.draws.shape[1]])
        children = ss.spawn(nblocks + 2)
        rngs = [np.random.default_rng(c) for c in children[:nblocks]]
        rng_init, rng_tune = np.random.default_rng(children[nblocks]), np.random.default_rng(children[nblocks + 1])
        if previous is not None:
            d, js, base = previous.proposal_chol, np.asarray(previous.jscale_vector, float), previous.jscale
            starts, lp0 = previous.draws[:, -1, :].copy(), previous.log_posterior[:, -1].copy()
        else:
            xmode = np.asarray(mode.x if isinstance(mode, ModeResult) else (self.x0 if mode is None else mode), float)
            if cov is None:
                kind = o["mcmc_jumping_covariance"] if jumping_covariance is None else jumping_covariance
                inv_h = mode.inv_hessian if isinstance(mode, ModeResult) else None
                cov = _post.jumping_covariance(self, kind, inv_h)
            d = np.linalg.cholesky(np.asarray(cov, float)).T          # upper factor, d' d = cov
            if np.any(xmode < lb) or np.any(xmode > ub):
                raise EstimationError("the mode is outside the prior bounds")
            target = o["mh_tune_jscale"] if tune_jscale is None else tune_jscale
            if target is not None and target is not False:
                if jscale is not None:
                    raise EstimationError("the mh_tune_jscale and mh_jscale options are incompatible")
                guess = o["mh_tune_guess"] if tune_guess is None else tune_guess
                tuned = _post.tune_jscale(self, xmode, np.asarray(cov, float), target=float(0.33 if target is True else target),
                                          guess=guess, rng=rng_tune, **(tune_options or {}))
                js, base = np.full(self.n, tuned), float(tuned)   # bayestopt_.jscale(:) = options_.mh_jscale
            else:
                js, base = self.jscale(jscale)
            starts, lp0 = np.empty((nblocks, self.n)), np.empty(nblocks)
            if x0 is not None:
                for j in range(nblocks):
                    starts[j] = np.asarray(x0, float).reshape(nblocks, self.n)[j]
                    lp0[j] = log_post(starts[j])
                    if not np.isfinite(lp0[j]):
                        raise EstimationError(f"chain {j}: the log posterior is not finite at the starting point")
            elif nblocks == 1:
                starts[0], lp0[0] = xmode, log_post(xmode)
                if not np.isfinite(lp0[0]):
                    raise EstimationError("the log posterior is not finite at the mode")
            else:
                factor = o["mh_init_scale_factor"] if init_scale_factor is None else init_scale_factor
                fixed = o["mh_init_scale"] if init_scale_factor is None else None     # deprecated option
                used = 0
                for j in range(nblocks):
                    validate, init_iter, trial = False, 0, 1
                    while not validate and trial <= 10:
                        if init_innovations is not None:
                            if used >= len(init_innovations):
                                raise EstimationError("init_innovations: not enough rows")
                            z = np.asarray(init_innovations, float)[used]
                            used += 1
                        else:
                            z = rng_init.standard_normal(self.n)
                        scale = fixed if fixed is not None else factor * base
                        cand = xmode + z @ (d * scale)
                        if np.all(cand >= lb) and np.all(cand <= ub):
                            lp = log_post(cand)
                            if np.isfinite(lp):
                                starts[j], lp0[j], validate = cand, lp, True
                        init_iter += 1
                        if not validate and init_iter > 100:
                            if fixed is not None:
                                fixed *= 0.9
                            else:
                                factor *= 0.9
                            trial += 1
                    if not validate:
                        raise EstimationError(f"chain {j}: no valid starting point found (reduce mh_init_scale_factor)")
        chol_prop = d * js[None, :]                               # d * diag(jscale)
        draws = np.empty((nblocks, replic, self.n))
        logpo = np.empty((nblocks, replic))
        acc = np.empty((nblocks, replic), bool)
        for j in range(nblocks):
            eps = rngs[j].standard_normal((replic, self.n)) if innovations is None else np.asarray(innovations)[j]
            u = rngs[j].random(replic) if uniforms is None else np.asarray(uniforms)[j]
            # proposals are checked against the bounds of the mode search (self.lb/ub: prior support and
            # estimated_params bounds), the kernel against the prior support: posterior_sampler_iteration.m
            draws[j], logpo[j], acc[j] = metropolis_hastings(log_post, starts[j], lp0[j], chol_prop,
                                                             self.lb, self.ub, eps, u, uniform_stream=uniform_stream)
        x_init = starts
        if previous is not None:
            draws = np.concatenate([previous.draws, draws], axis=1)
            logpo = np.concatenate([previous.log_posterior, logpo], axis=1)
            if previous.accepted is not None:
                acc = np.concatenate([previous.accepted, acc], axis=1)
            x_init = previous.x0
        out = MCMCResult(draws=draws, log_posterior=logpo, acceptance_rate=acc.mean(axis=1),
                         names=list(self.names), x0=x_init, drop=drop, accepted=acc, jscale=float(base),
                         jscale_vector=np.array(js, float), proposal_chol=np.array(d))
        if save is not None:
            _post.save_chains(save, out)
        return out

    # ------------------------------------------------------------------ posterior analysis (pydyn.posterior)
    def posterior_statistics(self, mc, **kw):
        """:func:`pydyn.posterior.posterior_statistics` with this estimation's priors and ``mh_conf_sig``."""
        from . import posterior as _post
        return _post.posterior_statistics(mc, self, **kw)

    def modified_harmonic_mean(self, mc, **kw):
        """Log marginal data density from the draws (:func:`pydyn.posterior.modified_harmonic_mean`)."""
        from . import posterior as _post
        return _post.modified_harmonic_mean(mc, **kw)

    def convergence_diagnostics(self, mc, **kw):
        """:func:`pydyn.posterior.convergence_diagnostics` with the options of the ``estimation`` command."""
        from . import posterior as _post
        return _post.convergence_diagnostics(mc, self, **kw)

    def posterior_irf(self, mc=None, **kw):
        """Posterior impulse responses (:func:`pydyn.posterior.posterior_irf`)."""
        from . import posterior as _post
        return _post.posterior_irf(self, mc, **kw)

    def posterior_moments(self, mc=None, **kw):
        """Posterior distributions of theoretical moments (``moments_varendo``;
        :func:`pydyn.posterior_moments.posterior_theoretical_moments`)."""
        from . import posterior_moments as _pm
        return _pm.posterior_theoretical_moments(self, mc, **kw)

    def posterior_smoother(self, mc=None, **kw):
        """Posterior distributions of smoothed, updated and filtered variables
        (:func:`pydyn.posterior_moments.posterior_smoother_statistics`)."""
        from . import posterior_moments as _pm
        return _pm.posterior_smoother_statistics(self, mc, **kw)

    def mode_check(self, mode, **kw):
        """Data of the mode-check plots (:func:`pydyn.posterior.mode_check`)."""
        from . import posterior as _post
        return _post.mode_check(self, mode, **kw)

    # ------------------------------------------------------------------ smoother
    def smoother(self, theta):
        """Kalman smoother at ``theta`` for a stationary model (``lik_init=1``).

        Returns a dict: ``variables`` {name: smoothed series, steady state included; in logs under
        ``loglinear``}, ``shocks`` {name: smoothed innovations}, ``measurement_errors`` {observed
        variable: series}, ``updated`` {name: E[y_t | data up to t]}."""
        if self.options["lik_init"] != 1:
            raise NotImplementedError("the smoother is implemented for lik_init=1 only")
        if self.options["heteroskedastic_filter"]:
            raise NotImplementedError("Estimation.smoother ignores heteroskedastic_filter; use "
                                      "pydyn.smoother.smooth(model, estimation=est, theta=theta)")
        m = self.model
        ss = self.state_space(theta)
        G, Hs, Q = ss["G"], ss["Hfull"], ss["Q"]
        Y = self.data.T - ss["constant"][:, None]
        trend = self._trend(ss["params"], self.nobs)
        if trend is not None:
            Y = Y - trend
        P = kalman.lyapunov_symm(G, Hs @ Q @ Hs.T, self.options["qz_criterium"],
                                 self.options["lyapunov_complex_threshold"])
        res = kalman.kalman_smoother(Y, np.zeros(m.n_endo), P, G, Hs, Q, ss["H"], self._obs_decl,
                                     data_index=self._data_index)
        level = np.where(self._loggable, np.log(np.abs(ss["ys"]) + (ss["ys"] == 0)), ss["ys"]) \
            if self.options["loglinear"] else ss["ys"]
        return {"variables": {v: res["alpha"][i] + level[i] for i, v in enumerate(m.endo_names)},
                "updated": {v: res["a_upd"][i] + level[i] for i, v in enumerate(m.endo_names)},
                "shocks": {v: res["eps"][i] for i, v in enumerate(m.exo_names)},
                "measurement_errors": {v: res["eta"][i] for i, v in enumerate(self.varobs)},
                "steady_state": ss["ys"]}


def _consistent_covariances(C):
    """check_consistency_covariances.m: no off-diagonal entry exceeds the product of the standard deviations."""
    C = np.asarray(C, float)
    if C.size == 0:
        return True
    sd = np.sqrt(np.diag(C))
    bound = np.outer(sd, sd)
    low = np.tril(C, -1)
    return not (np.any(low > bound) or np.any(low < -bound))


def _ispd(A):
    if A.size == 0:
        return True
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
