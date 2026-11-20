"""Translate Dynare expression strings (as written in the JSON output) into Python source."""
from __future__ import annotations

import keyword
import math
import re

import numpy as np
from scipy import special

_TOKEN = re.compile(
    r"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<id>[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<op>\*\*|==|!=|<=|>=|&&|\|\||[-+*/^()<>,!]))"
)


def _normcdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1.0 + special.erf((x - mu) / (sigma * math.sqrt(2.0))))


def _normpdf(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def _get_power_deriv(x, p, k):
    """k-th derivative of x**p, as Dynare's get_power_deriv helper defines it."""
    if abs(x) < 1e-12 and p >= 0 and k > p and abs(p - round(p)) < 1e-12:
        return 0.0
    dxp = x ** (p - k)
    for i in range(int(k)):
        dxp *= p - i
    return dxp


class _Infix:
    """Numeric comparison written as an infix operator: ``a ^_lt^ b`` or ``a |_eq| b``.

    Dynare (like MATLAB) treats comparisons as numbers (0 or 1) and chains them left to right.
    Python's own comparison operators return booleans, chain differently, and numpy booleans do
    not support arithmetic, so generated code uses these instead. ``^`` and ``|`` bind more
    loosely than arithmetic, and ``^`` (for < > <= >=) more tightly than ``|`` (for == !=),
    which reproduces the precedence in Dynare's grammar.
    """
    __array_ufunc__ = None

    def __init__(self, fn, left=None, bound=False):
        self.fn, self.left, self.bound = fn, left, bound

    def _l(self, other):
        return _Infix(self.fn, other, True)

    def _r(self, other):
        if not self.bound:
            raise TypeError("comparison is missing its left operand")
        return np.asarray(self.fn(self.left, other), dtype=float)[()]

    __rxor__ = __ror__ = _l
    __xor__ = __or__ = _r


def _matlab_round(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


#: names available to generated code
NAMESPACE = {
    "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10, "log2": np.log2,
    "sqrt": np.sqrt, "cbrt": np.cbrt, "abs": np.abs, "sign": np.sign,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos,
    "atan": np.arctan, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh,
    "max": np.maximum, "min": np.minimum, "erf": special.erf, "erfc": special.erfc,
    "get_power_deriv": _get_power_deriv, "normcdf": _normcdf, "normpdf": _normpdf,
    "Inf": np.inf, "NaN": np.nan, "inf": np.inf, "nan": np.nan, "pi": np.pi,
    "round": _matlab_round, "floor": np.floor, "ceil": np.ceil, "mod": np.mod,
    "_lt": _Infix(np.less), "_gt": _Infix(np.greater), "_le": _Infix(np.less_equal),
    "_ge": _Infix(np.greater_equal), "_eq": _Infix(np.equal), "_ne": _Infix(np.not_equal),
}


# ---------------------------------------------------------------- scalar namespace
# The generated model functions are written once and executed in two namespaces. In
# FAST_NAMESPACE their inputs are Python lists of floats and the functions come from ``math``:
# about three times faster than numpy scalars. Python floats raise where numpy returns nan or
# inf (log of a negative number, division by zero, overflow) and a negative base with a
# fractional exponent gives a complex number; the callers catch ArithmeticError, ValueError and
# TypeError and re-evaluate with NAMESPACE, so results are those of numpy in every such case.
# For that to hold no function here may quietly turn a complex number into a real one, hence
# ``math.fabs`` rather than ``abs`` and comparisons that reject complex operands.
class _FastInfix(_Infix):
    __slots__ = ()

    def _l(self, other):
        return _FastInfix(self.fn, other, True)

    def _r(self, other):
        if not self.bound:
            raise TypeError("comparison is missing its left operand")
        return 1.0 if self.fn(float(self.left), float(other)) else 0.0

    __rxor__ = __ror__ = _l
    __xor__ = __or__ = _r


def _fmax(a, b):
    return a if a > b else b if b >= a else math.nan      # nan if either is nan, as np.maximum


def _fmin(a, b):
    return a if a < b else b if b <= a else math.nan


def _fsign(x):
    return 1.0 if x > 0 else -1.0 if x < 0 else 0.0 if x == 0 else math.nan


def _fround(x):
    return math.copysign(math.floor(math.fabs(x) + 0.5), x) if x == x and x not in (math.inf, -math.inf) else x


def _ffloor(x):
    return float(math.floor(x)) if math.isfinite(x) else x


def _fceil(x):
    return float(math.ceil(x)) if math.isfinite(x) else x


def _fnormcdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def _fnormpdf(x, mu=0.0, sigma=1.0):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def _get_power_deriv_fast(x, p, k):
    if k == 1:          # nearly every call in a Jacobian
        if -1e-12 < x < 1e-12 and 0 <= p < 1 and -1e-12 < p - round(p) < 1e-12:
            return 0.0
        return x ** (p - 1) * p
    return _get_power_deriv(x, p, k)


import operator as _op  # noqa: E402

FAST_NAMESPACE = {
    "exp": math.exp, "log": math.log, "ln": math.log, "log10": math.log10, "log2": math.log2,
    "sqrt": math.sqrt, "cbrt": math.cbrt, "abs": math.fabs, "sign": _fsign,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos,
    "atan": math.atan, "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
    "max": _fmax, "min": _fmin, "erf": math.erf, "erfc": math.erfc,
    "get_power_deriv": _get_power_deriv_fast, "normcdf": _fnormcdf, "normpdf": _fnormpdf,
    "Inf": math.inf, "NaN": math.nan, "inf": math.inf, "nan": math.nan, "pi": math.pi,
    "round": _fround, "floor": _ffloor, "ceil": _fceil, "mod": _op.mod,
    "_lt": _FastInfix(_op.lt), "_gt": _FastInfix(_op.gt), "_le": _FastInfix(_op.le),
    "_ge": _FastInfix(_op.ge), "_eq": _FastInfix(_op.eq), "_ne": _FastInfix(_op.ne),
}
#: exceptions after which a scalar evaluation is repeated with numpy semantics
FAST_ERRORS = (ArithmeticError, ValueError, TypeError)

_CMP = {"<": "^_lt^", ">": "^_gt^", "<=": "^_le^", ">=": "^_ge^", "==": "|_eq|", "!=": "|_ne|"}


def safe_name(name: str) -> str:
    """Python-safe spelling of a non-model identifier (local variables called ``lambda`` etc.)."""
    return name + "_" if keyword.iskeyword(name) else name


def temp_name(name: str) -> str:
    return "_tt_" + name


def tokenize(s: str):
    pos, out = 0, []
    s = s.strip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            raise ValueError(f"cannot tokenize {s!r} at {pos}: {s[pos:pos+20]!r}")
        out.append((m.lastgroup, m.group(m.lastgroup)))
        pos = m.end()
    return out


class Translator:
    """Maps model symbols to Python expressions.

    ``resolve(name, shift)`` must return Python source for a model symbol, or ``None``
    if the name is not a model symbol (functions, temporary terms, locals).
    ``steady(name)`` returns source for ``STEADY_STATE(name)``.
    """

    def __init__(self, resolve, steady=None, shiftable=(), temps=()):
        self.resolve = resolve
        self.steady = steady
        self.shiftable = set(shiftable)
        self.temps = set(temps)      # temporary-term names; they take precedence over model symbols

    def __call__(self, s: str) -> str:
        toks = tokenize(s)
        out, i, n = [], 0, len(toks)
        while i < n:
            kind, val = toks[i]
            if kind == "id":
                if val in ("STEADY_STATE", "steady_state") and i + 3 < n and toks[i + 1][1] == "(" \
                        and toks[i + 2][0] == "id" and toks[i + 3][1] == ")":
                    if self.steady is None:
                        raise ValueError("STEADY_STATE not allowed here")
                    out.append(self.steady(toks[i + 2][1]))
                    i += 4
                    continue
                if val in ("STEADY_STATE", "steady_state"):
                    raise NotImplementedError("steady_state() of an expression is not supported")
                if val in self.temps:
                    out.append(temp_name(val))
                    i += 1
                    continue
                shift = 0
                if val in self.shiftable and i + 1 < n and toks[i + 1][1] == "(":
                    # var(k), var(-k), var(+k)
                    j = i + 2
                    sgn = 1
                    if j < n and toks[j][1] in "+-":
                        sgn = -1 if toks[j][1] == "-" else 1
                        j += 1
                    if j + 1 < n and toks[j][0] == "num" and toks[j + 1][1] == ")":
                        shift = sgn * int(toks[j][1])
                        i = j + 1
                src = self.resolve(val, shift)
                out.append(src if src is not None else safe_name(val))
            elif kind == "op":
                out.append(_CMP.get(val) or {"^": "**", "&&": " and ", "||": " or ", "!": " not "}.get(val, val))
            else:
                out.append(val)
            i += 1
        return "".join(out)
