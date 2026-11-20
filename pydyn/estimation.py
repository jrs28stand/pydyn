# This file is part of pydyn.
#
# It follows, and in places is a Python port of, matlab/estimation/{dsge_likelihood, dynare_estimation_init, dynare_estimation_1, set_prior, initial_estimation_checks, posterior_sampler_core, set_mcmc_prior_bounds}.m, matlab/optimization/hessian.m and the data-set utilities from Dynare 7.2
# (Copyright (C) 2003-2026 Dynare Team, https://www.dynare.org), modified in 2026.
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
``kalman_tol``, ``diffuse_kalman_tol``, ``riccati_tol``, ``qz_criterium``, ``prior_trunc``,
``mh_replic``, ``mh_nblocks``, ``mh_jscale`` (and the per-parameter ``jscale``), ``mh_drop``,
``mh_init_scale_factor``. Priors: beta, gamma, normal, inverse gamma 1 and 2, uniform, Weibull;
none at all means maximum likelihood.

Not implemented: orders above one, ``lik_init`` 4 and 5, the univariate filters with correlated
measurement errors, calibrated measurement-error covariances, ``estimated_params_bounds``,
``estimated_params_init(use_calibration)``, endogenous priors, prior restrictions,
``noconstant`` (which Dynare 7.2 itself resets), recursive estimation, the other samplers,
the modified harmonic mean estimator and the smoother for non-stationary models.
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
from .perturbation import BlanchardKahnError, dr_ordering, solve_linear_re
from .priors import Prior, PriorError
from .steady import SteadyStateError, steady_state

__all__ = ["Estimation", "EstimationError", "ModeResult", "MCMCResult", "load_data", "read_m_data",
           "hessian", "metropolis_hastings", "Prior", "PriorError"]


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


def metropolis_hastings(log_post, x0, logpost0, chol_upper, lb, ub, innovations, uniforms):
    """One random-walk Metropolis-Hastings chain driven by given random numbers.

    ``chol_upper`` is the upper Cholesky factor of the proposal covariance (already scaled);
    the proposal is ``last + innovations[i] @ chol_upper``. As in Dynare a proposal is evaluated
    only if it lies strictly inside the bounds, and accepted if its log posterior is finite and
    ``log(u) < log_post(proposal) - log_post(last)``.
    Returns (draws, log posteriors, accepted flags).
    """
    n_draws, n = innovations.shape
    draws = np.empty((n_draws, n))
    logpo = np.empty(n_draws)
    acc = np.zeros(n_draws, bool)
    last, last_lp = np.array(x0, float), float(logpost0)
    for i in range(n_draws):
        par = last + innovations[i] @ chol_upper
        if np.all(par > lb) and np.all(par < ub):
            try:
                lp = float(log_post(par))
            except Exception:                     # noqa: BLE001 - as Dynare: any failure rejects
                lp = -np.inf
        else:
            lp = -np.inf
        if lp > -np.inf and np.log(uniforms[i]) < lp - last_lp:
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
    mode_compute=5, mh_replic=20000, mh_nblck=2, mh_jscale=None, mh_drop=0.5, mh_init_scale_factor=2.0,
    gstep=(1e-2, 1.0), order=1, use_univariate_filters_if_singularity_is_detected=True,
)
_ALIASES = {"mh_nblocks": "mh_nblck", "mh_init_scale": None, "harvey_scale_factor": "Harvey_scale_factor"}


class _Rejected(Exception):
    """Raised inside an evaluation when the parameter vector cannot be used (Dynare's info code)."""


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
    """

    def __init__(self, model, data=None, **options):
        self.model = model
        opts = dict(_DEFAULTS)
        raw = dict(model.options.get("estimation", {}))
        raw.update(options)
        self.ignored_options = []
        for k, v in raw.items():
            k = _ALIASES.get(k, k) if k in _ALIASES else k
            if isinstance(v, (list, tuple)) and len(v) == 1 and k != "gstep":
                v = v[0]
            if k in ("nobs", "first_obs") and isinstance(v, (list, tuple)):
                raise NotImplementedError(f"recursive estimation ({k} given as a vector) is not supported")
            if k in opts:
                opts[k] = v
            elif k is not None:
                self.ignored_options.append(k)
        self.options = opts
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
        self.last_info = ""
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
        # calibrated covariance structure
        S = np.array(m.Sigma_e, float)
        sd = np.sqrt(np.diag(S))
        with np.errstate(divide="ignore", invalid="ignore"):
            C = S / np.outer(sd, sd)
        C[~np.isfinite(C)] = 0.0
        np.fill_diagonal(C, 1.0)
        self._Sigma_e0, self._corr0 = S, C
        pp = len(self.varobs)
        H = np.zeros((pp, pp))
        for name, var in m.measurement_errors.items():
            if name not in self.varobs:
                raise EstimationError(f"measurement error on {name}, which is not an observed variable")
            H[self.varobs.index(name), self.varobs.index(name)] = var
        self._H0 = H
        if any("measurement-error covariance" in s for s in m.unhandled_statements):
            warnings.warn("calibrated measurement-error covariances are ignored")
        nz = ~np.all(S == 0, axis=0)
        nz[[k for _, k, _ in groups["vx"]]] = True
        self._Q_check = np.flatnonzero(nz)
        nz = ~np.all(H == 0, axis=0)
        nz[[k for _, k, _ in groups["vn"]]] = True
        self._H_check = np.flatnonzero(nz)

    def _apply_init_blocks(self, init):
        """``estimated_params_init`` blocks: explicit initial values (``use_calibration`` is not supported)."""
        for st in self.model.statements:
            if st.get("statementName") != "estimated_params_init":
                continue
            if st.get("use_calibration_initialization") or st.get("use_calibration"):
                raise NotImplementedError("estimated_params_init(use_calibration) is not supported")
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

    def covariances(self, theta):
        """Structural-shock covariance matrix and measurement-error covariance matrix implied by
        ``theta`` (``set_all_parameters.m``)."""
        g = self._groups
        Q, C = self._Sigma_e0.copy(), self._corr0.copy()
        H, CH = self._H0.copy(), np.eye(len(self.varobs))
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
        if g["vn"] or g["cn"]:
            sd = np.sqrt(np.diag(H))
            H = CH * np.outer(sd, sd)
        return Q, H

    def model_params(self, theta):
        """Copy of the model's parameter vector with the estimated structural parameters set."""
        p = np.array(self.model.params, float)
        g = self._groups
        o = self.n - len(g["p"])
        for i, (_, k, _) in enumerate(g["p"]):
            p[k] = theta[o + i]
        return p

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
        if not np.all(np.isfinite(params)):
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
        if data is None:
            if not o["datafile"]:
                raise EstimationError("no data: pass data=... or set the datafile option")
            data = o["datafile"]
        base = self.model.json_dir.parents[2] if hasattr(self.model, "json_dir") else None
        raw = load_data(data, self.varobs, base)
        if o["loglinear"] and not o["logdata"]:
            with np.errstate(all="ignore"):
                raw = np.log(raw)
        first = int(o["first_obs"])
        nobs = raw.shape[0] - first + 1 if o["nobs"] is None else int(o["nobs"])
        if first < 1 or nobs < 1 or first - 1 + nobs > raw.shape[0]:
            raise EstimationError(f"first_obs={first} and nobs={nobs} are not consistent with the "
                                  f"{raw.shape[0]} observations in the data")
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
        Q, Hm = self.covariances(theta)
        if self._groups["cx"] and not _ispd(Q[np.ix_(self._Q_check, self._Q_check)]):
            raise _Rejected("covariance matrix of the structural shocks is not positive definite (info 43)")
        if self._groups["cn"] and not _ispd(Hm[np.ix_(self._H_check, self._H_check)]):
            raise _Rejected("covariance matrix of the measurement errors is not positive definite (info 44)")
        try:
            ys, G, Hs, params = self.solve(theta)
        except SteadyStateError as exc:
            raise _Rejected(f"steady state: {exc}") from None
        except BlanchardKahnError as exc:
            raise _Rejected(f"solution: {exc}") from None
        r = self._restrict
        T, R, z = G[np.ix_(r, r)], Hs[r, :], self._mf
        if not (np.all(np.isfinite(T)) and np.all(np.isfinite(R))):
            raise _Rejected("the decision rule contains NaN or Inf")
        const = ys[self._obs_decl]
        if o["loglinear"]:
            const = np.log(const)
        Y = self.data.T - const[:, None]
        trend = self._trend(params, self.nobs)
        if trend is not None:
            Y = Y - trend
        QQ = R @ Q @ R.T
        mm = T.shape[0]
        a = np.zeros(mm)
        pre = int(o["presample"])
        kw = dict(kalman_tol=o["kalman_tol"], riccati_tol=o["riccati_tol"], presample=pre)
        dkw = dict(kalman_tol=o["kalman_tol"], diffuse_kalman_tol=o["diffuse_kalman_tol"], presample=pre,
                   data_index=self._data_index)
        algo = o["kalman_algo"]
        dLIK, dp, dlik = 0.0, 0, np.zeros(0)

        def Zmat():
            if np.any(np.abs(Hm - np.diag(np.diag(Hm))) >= 1e-14):
                raise _Rejected("the univariate filter with correlated measurement errors is not implemented")
            Zm = np.zeros((len(z), mm))
            Zm[np.arange(len(z)), z] = 1.0
            return Zm, np.diag(Hm).copy()

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
                dLIK, dlik, a1, P = kalman.kalman_filter_d(Y, a, Pinf, Pstar, T, QQ, Hm, z, **dkw)
                if np.isinf(dLIK):
                    singular = True               # as Dynare: redo the diffuse stage observation by observation
                else:
                    a = a1
            if singular or algo == 4:
                Zm, Hd = Zmat()
                dLIK, dlik, a, P = kalman.univariate_kalman_filter_d(Y, np.zeros(mm), Pinf, Pstar, T, QQ, Hd, Zm, **dkw)
            if np.isnan(dLIK):
                raise _Rejected("the diffuse filter did not leave the diffuse stage, or a variance is negative (info 45)")
            dp = len(dlik)
        if not np.all(np.isfinite(P)):
            raise _Rejected("initial state covariance is not finite")
        di = None if self._data_index is None else self._data_index[dp:]
        nmm = max(self._no_more_missing - dp, 0)
        lik = None
        if algo in (1, 3):
            out = kalman.kalman_filter(Y[:, dp:], a, P, T, QQ, Hm, z, diffuse_periods=dp, data_index=di,
                                       no_more_missing=nmm, return_lik=True,
                                       rescale=bool(o["rescale_prediction_error_covariance"]), **kw)
            LIK, lik = out
            if np.isinf(LIK):
                if not o["use_univariate_filters_if_singularity_is_detected"]:
                    raise _Rejected("the forecast-error covariance is singular (info 50)")
                algo = 2 if algo == 1 else 4
        if algo in (2, 4):
            Zm, Hd = Zmat()
            LIK, lik = kalman.univariate_kalman_filter(Y[:, dp:], a, P, T, QQ, Hd, Zm, diffuse_periods=dp,
                                                       data_index=di, no_more_missing=nmm, return_lik=True, **kw)
        self.last_filter = {1: "multivariate", 2: "univariate", 3: "multivariate (diffuse)",
                            4: "univariate (diffuse)"}[algo]
        if np.isinf(LIK):
            raise _Rejected("the forecast-error covariance is singular (info 50)")
        LIK = LIK + dLIK
        if np.isnan(LIK) or np.iscomplexobj(LIK):
            raise _Rejected("the likelihood is NaN or complex (info 45/46)")
        if return_lik:
            return LIK, np.concatenate([dlik, lik])[pre:]
        return LIK

    def log_likelihood(self, theta) -> float:
        """Log likelihood of the data at ``theta``; ``-inf`` if the steady state or the solution
        fails, or the filter breaks down (the reason is kept in ``last_info``)."""
        self.last_info = ""
        try:
            with np.errstate(all="ignore"):
                return -float(self._minus_loglik(theta))
        except (_Rejected, FloatingPointError, np.linalg.LinAlgError) as exc:
            self.last_info = str(exc)
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
        self.last_info = ""
        if np.any(theta < lb):
            self.last_info = "below the lower bound (info 41)"
            return -np.inf
        if np.any(theta > ub):
            self.last_info = "above the upper bound (info 42)"
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            self.last_info = "the prior density is zero, infinite or NaN (info 40/47)"
            return -np.inf
        return ll + lp

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

    def mode(self, x0=None, method="auto", compute_hessian=True, tol=1e-9, maxiter=5000, polish=False):
        """Maximise the log posterior (the log likelihood for ML) starting from ``x0`` (default:
        the initial values of the ``estimated_params`` block, else the prior means).

        ``method='auto'`` runs two quasi-Newton searches with numerical gradients inside the
        bounds, scipy's SLSQP and then L-BFGS-B from its solution, and keeps the best point. They
        are the counterparts of Dynare's ``mode_compute=1`` (fmincon) and ``mode_compute=4``
        (csminwel, a BFGS method). ``polish=True`` adds restarts of a derivative-free simplex
        search (as ``mode_compute=7``/``8``). Any scipy ``minimize`` method name can be given
        instead of ``'auto'``. The path of the optimiser is not that of any Dynare optimiser; only
        the optimum is comparable.
        """
        x = np.array(self.x0 if x0 is None else x0, float)
        nfev = [0]
        big = 1e12

        def f(v):
            nfev[0] += 1
            val = -self.log_posterior(v)
            return val if np.isfinite(val) else big

        fx = f(x)
        if fx >= big:
            raise EstimationError(f"the log posterior is not finite at the initial values: {self.last_info}")
        msgs, ok = [], True
        if method == "auto":
            width = np.where(np.isfinite(self.ub - self.lb), self.ub - self.lb, 1.0)
            bnds = [(lo + 1e-10 * w if np.isfinite(lo) else None, hi - 1e-10 * w if np.isfinite(hi) else None)
                    for lo, hi, w in zip(self.lb, self.ub, width)]
            stages = [("SLSQP", {"ftol": 1e-12, "eps": 1e-7, "maxiter": maxiter}),
                      ("L-BFGS-B", {"ftol": 1e-13, "gtol": 1e-7, "eps": 1e-7, "maxiter": maxiter, "maxls": 40})]
            for name, opt in stages:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = optimize.minimize(f, x, method=name, bounds=bnds, options=opt)
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
        else:
            res = optimize.minimize(f, x, method=method, tol=tol, options={"maxiter": maxiter})
            if f(res.x) < fx:
                x = np.array(res.x)
            ok = bool(res.success)
            msgs.append(str(res.message))
        out = ModeResult(x=np.array(x), names=list(self.names), log_posterior=self.log_posterior(x),
                         success=ok, message="; ".join(msgs), nfev=nfev[0])
        if compute_hessian:
            self.add_hessian(out)
        return out

    def add_hessian(self, mode: ModeResult):
        """Compute the Hessian, its inverse and the Laplace approximation at ``mode.x``."""
        with np.errstate(invalid="ignore"):
            mode.hessian = self.hessian(mode.x)
        if np.all(np.isfinite(mode.hessian)) and np.all(np.diag(mode.hessian) > 0):
            mode.inv_hessian = self.inverse_hessian(mode.hessian)
            if self.prior.is_bayesian:
                mode.laplace = self.laplace(mode.x, mode.inv_hessian)
        return mode

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
               drop=None, init_scale_factor=None, innovations=None, uniforms=None) -> MCMCResult:
        """Random-walk Metropolis-Hastings sampler (Dynare's default posterior sampler).

        The proposal is Gaussian, ``last + N(0, J V J)`` with ``V`` the inverse Hessian at the
        mode (``cov``, or ``mode.inv_hessian``) and ``J = diag(jscale)``. As in Dynare the chains
        are confined to the support of the priors (``prior_trunc`` applies); the bounds given in
        the ``estimated_params`` block restrict the mode search only. With several chains
        each one starts from a draw ``N(mode, (init_scale_factor * mh_jscale)^2 V)`` inside the
        bounds with a finite posterior; a single chain starts at the mode. ``x0`` (chains x
        parameters) overrides the starting points. ``seed`` seeds a numpy ``Generator``; every
        chain gets its own stream. ``innovations`` (chains, replic, n) and ``uniforms`` (chains,
        replic) replace the random numbers (for tests and replication).
        """
        o = self.options
        if not self.prior.is_bayesian:
            raise EstimationError("the sampler needs priors")
        replic = int(o["mh_replic"] if replic is None else replic)
        nblocks = int(o["mh_nblck"] if nblocks is None else nblocks)
        drop = o["mh_drop"] if drop is None else drop
        xmode = np.asarray(mode.x if isinstance(mode, ModeResult) else (self.x0 if mode is None else mode), float)
        if cov is None:
            if not isinstance(mode, ModeResult) or mode.inv_hessian is None:
                raise EstimationError("pass cov=... or a ModeResult with an inverse Hessian")
            cov = mode.inv_hessian
        js, base = self.jscale(jscale)
        d = np.linalg.cholesky(np.asarray(cov, float)).T          # upper factor, d' d = cov
        chol_prop = d * js[None, :]                               # d * diag(jscale)
        lb, ub = self.prior.bounds(o["prior_trunc"])              # set_mcmc_prior_bounds.m
        if np.any(xmode < lb) or np.any(xmode > ub):
            raise EstimationError("the mode is outside the prior bounds")

        def log_post(x):                          # Dynare evaluates the posterior with these bounds
            return self.log_posterior(x, bounds=(lb, ub))
        rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(nblocks)]
        scale0 = (o["mh_init_scale_factor"] if init_scale_factor is None else init_scale_factor) * base
        starts, lp0 = np.empty((nblocks, self.n)), np.empty(nblocks)
        for j in range(nblocks):
            if x0 is not None:
                starts[j] = np.asarray(x0, float).reshape(nblocks, self.n)[j]
                lp0[j] = log_post(starts[j])
                if not np.isfinite(lp0[j]):
                    raise EstimationError(f"chain {j}: the log posterior is not finite at the starting point")
            elif nblocks == 1:
                starts[j], lp0[j] = xmode, log_post(xmode)
            else:
                scale = scale0
                for trial in range(1, 1001):
                    cand = xmode + rngs[j].standard_normal(self.n) @ (d * scale)
                    if np.all(cand >= lb) and np.all(cand <= ub):
                        lp = log_post(cand)
                        if np.isfinite(lp):
                            starts[j], lp0[j] = cand, lp
                            break
                    if trial % 100 == 0:
                        scale *= 0.9              # Dynare (nointeractive): reduce the scale by 10 percent
                else:
                    raise EstimationError(f"chain {j}: no valid starting point found")
        draws = np.empty((nblocks, replic, self.n))
        logpo = np.empty((nblocks, replic))
        acc = np.empty((nblocks, replic), bool)
        for j in range(nblocks):
            eps = rngs[j].standard_normal((replic, self.n)) if innovations is None else np.asarray(innovations)[j]
            u = rngs[j].random(replic) if uniforms is None else np.asarray(uniforms)[j]
            draws[j], logpo[j], acc[j] = metropolis_hastings(log_post, starts[j], lp0[j], chol_prop,
                                                             lb, ub, eps, u)
        return MCMCResult(draws=draws, log_posterior=logpo, acceptance_rate=acc.mean(axis=1),
                          names=list(self.names), x0=starts, drop=drop, accepted=acc)

    # ------------------------------------------------------------------ smoother
    def smoother(self, theta):
        """Kalman smoother at ``theta`` for a stationary model (``lik_init=1``).

        Returns a dict: ``variables`` {name: smoothed series, steady state included; in logs under
        ``loglinear``}, ``shocks`` {name: smoothed innovations}, ``measurement_errors`` {observed
        variable: series}, ``updated`` {name: E[y_t | data up to t]}."""
        if self.options["lik_init"] != 1:
            raise NotImplementedError("the smoother is implemented for lik_init=1 only")
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


def _ispd(A):
    if A.size == 0:
        return True
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
