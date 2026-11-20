"""The fast (Python-float) generated model functions must give exactly what the numpy versions give,
including nan/inf where plain floating-point arithmetic would raise or turn complex."""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

import pydyn
import pydyn.model as model_module

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
EXAMPLES = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/dl/src72/examples"))

KITCHEN_SINK = """
var a b c d f; varexo e u; parameters p q;
p = 0.5; q = 2;
model;
a = p*a(-1) + log(b) + e;
b = exp(c)^q + sqrt(d) + max(a, 0) - min(c, 1) + (a > c) + (b >= d)*(c != 0);
c = abs(a)^p + sign(d) + 1/(1 + f) + normcdf(a) + erf(c(+1)) + u;
d = (a/b)^(1/3) + cbrt(d(-1)) + log10(b) + tan(c);
f = a(+1)^2 + b^(-q) + normpdf(c, 0, 1 + p) + asin(d/10) + acosh(1 + b^2);
end;
shocks; var e; stderr 1; var u; stderr 1; end;
stoch_simul(order=2, irf=0);
"""


def models(tmp_path):
    out = {}
    (tmp_path / "sink.mod").write_text(KITCHEN_SINK)
    sources = {"sink": tmp_path / "sink.mod",
               "collard": EXAMPLES / "stochastic_simulations" / "collard_2001_simulated_moments.mod",
               "aux": REF / "aux_a.mod", "sw": REF / "sw_big.mod", "occbin": EXAMPLES / "occbin" / "rbc_occbin.mod"}
    for name, src in sources.items():
        d = tmp_path / name
        d.mkdir()
        shutil.copy(src, d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[name] = pydyn.Model(pydyn.run_preprocessor(d / src.name))
    return out


def both(monkeypatch, fn, *args):
    res = []
    for mode in ("fast", "numpy"):
        monkeypatch.setattr(model_module, "CODEGEN", mode)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res.append(fn(*args))
    return res


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.array_equal(np.isposinf(a), np.isposinf(b)) and np.array_equal(np.isneginf(a), np.isneginf(b))
    ok = np.isfinite(a)
    scale = max(1.0, np.max(np.abs(b[ok]))) if ok.any() else 1.0       # last-bit differences in cancellations
    np.testing.assert_allclose(a[ok], b[ok], rtol=1e-12, atol=1e-13 * scale)


def test_fast_and_numpy_code_agree_everywhere(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for name, m in models(tmp_path).items():
        n, k = m.n_endo, m.n_exo
        points = [rng.uniform(0.2, 2.0, 3 * n),                   # ordinary
                  rng.uniform(-2.0, 2.0, 3 * n),                  # negative values: log, sqrt, fractional powers
                  np.zeros(3 * n),                                # division by zero, log(0)
                  rng.uniform(500, 900, 3 * n),                   # overflow in exp
                  np.where(rng.random(3 * n) < 0.3, np.nan, 1.0),  # nan inputs
                  rng.choice([-np.inf, np.inf, 1.0, -1.0, 0.0], 3 * n)]
        for yy in points:
            x = rng.standard_normal(k)
            ss = yy[n:2 * n]
            same(*both(monkeypatch, m.dynamic_resid, yy, x, None, ss))
            same(*both(monkeypatch, m.dynamic_jac, yy, x, None, ss))
            same(*both(monkeypatch, m.static_resid, yy[:n], x))
            same(*both(monkeypatch, m.static_jac, yy[:n], x))
            for order in range(2, m.max_deriv_order + 1):
                (r1, c1, v1), (r2, c2, v2) = both(monkeypatch, m.dynamic_sparse_derivs, order, yy, x, None, ss)
                assert np.array_equal(r1, r2) and np.array_equal(c1, c2)
                same(v1, v2)


def test_fallback_is_actually_exercised(tmp_path, monkeypatch):
    """At a point where plain floats raise (log of a negative number) the result is numpy's nan."""
    m = models(tmp_path)["sink"]
    yy = -np.ones(3 * m.n_endo)
    fast, ref = both(monkeypatch, m.dynamic_resid, yy, np.zeros(m.n_exo), None, yy[:m.n_endo])
    assert np.isnan(ref).any()
    same(fast, ref)


def test_solutions_do_not_depend_on_the_code_path(tmp_path, monkeypatch):
    for name, m in models(tmp_path).items():
        if name == "sink":
            continue
        a, b = both(monkeypatch, lambda: pydyn.first_order(m))
        np.testing.assert_allclose(a.ys, b.ys, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(a.G, b.G, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(a.H, b.H, rtol=1e-9, atol=1e-11)
