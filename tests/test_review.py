"""Regression tests for the issues found in the independent review.

Expected values marked "Dynare" were obtained by the reviewer from Dynare 7.2 under Octave."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import pydyn

EXAMPLES = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/dl/src72/examples"))


def build(tmp_path, text, name="m", **kw):
    f = tmp_path / f"{name}.mod"
    f.write_text(text)
    return pydyn.Model(pydyn.run_preprocessor(f), **kw)


def fd_jac(m, y, h=1e-6):
    n = m.n_endo
    J = np.zeros((n, n))
    for j in range(n):
        d = np.zeros(n); d[j] = h
        J[:, j] = (m.static_resid(y + d) - m.static_resid(y - d)) / (2 * h)
    return J


TWO = "var c k; varexo e; parameters a; a=0.5;\n"


def test_comparisons_are_numeric(tmp_path):
    m = build(tmp_path, TWO + "model; c = (c > a) + (k > a); k = 0.5*k(-1)+0.5*k(+1) + e; end;")
    np.testing.assert_allclose(m.static_resid([1.0, 1.0]), [-1.0, 0.0])          # Dynare


def test_chained_comparisons(tmp_path):
    m = build(tmp_path, "var c k; varexo e; parameters a b; a=3; b=1;\n"
                        "model; c = (b < a < 2) + e; k = (c == 1 == 0); end;")
    np.testing.assert_allclose(m.static_resid([2.0, 0.0]), [1.0, -1.0])          # Dynare


@pytest.mark.parametrize("expr", ["max(k, 0)", "min(k, 1)", "max(0, k)"])
def test_max_min_derivatives(tmp_path, expr):
    m = build(tmp_path, TWO + f"model; c = {expr} + a*c(-1); k = 0.5*k(-1) + e; end;")
    y = np.array([0.3, 0.4])
    np.testing.assert_allclose(m.static_jac(y), fd_jac(m, y), atol=1e-7)


SH = ("var x1 x2; varexo e1 e2; model; x1 = 0.5*x1(-1)+e1; x2 = 0.5*x2(-1)+e2; end;\n"
      "shocks; var e1; stderr 1; var e2; stderr 1; corr e1,e2=0.5; end;\n")


@pytest.mark.parametrize("second,expected", [
    ("shocks; var e1; stderr 2; end;", [[4, 0.5], [0.5, 1]]),
    ("shocks(overwrite); var e1; stderr 2; var e2; stderr 3; end;", [[4, 0], [0, 9]]),
    ("shocks; var e1,e2 = 0.1; end;", [[1, 0.1], [0.1, 1]]),
])
def test_shock_blocks_follow_dynare_order(tmp_path, second, expected):
    m = build(tmp_path, SH + second)
    np.testing.assert_allclose(m.Sigma_e, expected)                              # Dynare


def test_native_parameter_updates(tmp_path):
    m = build(tmp_path, "var c; varexo e; parameters rho s; rho=0.5; s=2;\n"
                        "lambda = 0.25;\nscale = lambda*2;\nset_param_value('rho', 0.9);\nM_.params(2) = scale;\n"
                        "model; c = rho*c(-1) + s*e; end;")
    assert m.get_param("rho") == 0.9 and m.get_param("s") == 0.5


@pytest.mark.parametrize("eqs,G00", [
    ("c = a*c(-1) + e; k = 0.9*k(-1) + c;", 0.5),          # no leads
    ("c = a*c(+1) + e; k = 0.9*k(+1) + c;", 0.0),          # no lags
])
def test_purely_backward_and_forward_models(tmp_path, eqs, G00):
    m = build(tmp_path, TWO + f"model; {eqs} end;")
    sol = pydyn.first_order(m)
    np.testing.assert_allclose(sol.G[0, 0], G00, atol=1e-12)
    np.testing.assert_allclose(sol.H[0, 0], 1.0, atol=1e-12)


def test_numerical_steady_state_with_auxiliary_variables(tmp_path):
    src = EXAMPLES / "macroprocessor" / "bkk_1992.mod"
    shutil.copy(src, tmp_path)
    m = pydyn.Model(pydyn.run_preprocessor(tmp_path / src.name))
    assert m.raw["aux_vars"]
    ys, params = pydyn.steady_state(m)
    assert np.max(np.abs(m.static_resid(ys, params=params))) < 1e-10
    pydyn.first_order(m)


def test_initval_may_refer_to_earlier_values(tmp_path):
    m = build(tmp_path, "var a b; varexo e; parameters rho; rho=0.5;\n"
                        "model; a = 1 + rho*(a(-1)-1) + e; b = 2*a + rho; end;\n"
                        "initval; a=1; b=2*a+rho; end;")
    assert m.initval == {"a": 1.0, "b": 2.5}


def test_linear_model_at_second_order(tmp_path):
    m = build(tmp_path, TWO + "model; c = a*c(-1) + 0.1*c(+1) + e + k; k = 0.5*k(-1)+e; end;\n"
                              "shocks; var e; stderr 1; end;\nstoch_simul(order=2, irf=0);")
    try:
        pydyn.korder.load_library()
    except FileNotFoundError:
        pytest.skip("libpydyn_korder.so not built")
    dr = pydyn.k_order(m, order=2)
    assert np.max(np.abs(dr["ghxx"])) < 1e-12
    np.testing.assert_allclose(dr["ghx"], pydyn.first_order(m).ghx, atol=1e-12)


def test_model_without_statements(tmp_path):
    m = build(tmp_path, "var a b; varexo e1; model; a=0.5*a(-1)+e1; b=0.5*b(+1)+a; end;")
    pydyn.first_order(m)


def test_variable_named_like_a_temporary_term(tmp_path):
    m = build(tmp_path, "var c k T34; varexo e; parameters alpha; alpha=0.5;\n"
                        "model; c = 0.3*c(-1) + e; k = alpha*k(-1) + exp(c)*exp(c)+exp(c)^2;\n"
                        "T34 = 0.5*T34(-1) + exp(c)*exp(c) + e; end;")
    y = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(m.static_jac(y), fd_jac(m, y), atol=1e-6)
