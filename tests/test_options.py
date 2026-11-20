"""Options written in the .mod file are either acted on or reported: nothing that changes results is
dropped silently. Expected values marked "Dynare" were checked against Dynare 7.2 in the audit."""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

import pydyn
from pydyn import smoother
from pydyn.estimation import Estimation
from pydyn.options import UnsupportedOptionWarning

EXAMPLES = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/dl/src72/examples"))
REF = Path(__file__).resolve().parent.parent / "reference"


def build(tmp_path, text, name="m"):
    f = tmp_path / f"{name}.mod"
    f.write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(f))


RBC = """var c k a; varexo e; parameters beta alpha delta rho;
beta=0.99; alpha=0.33; delta=0.025; rho=0.9;
model;
1/c = beta/c(+1)*(alpha*exp(a(+1))*k^(alpha-1) + 1 - delta);
c + k = exp(a)*k(-1)^alpha + (1-delta)*k(-1);
a = rho*a(-1) + e;
end;
steady_state_model;
a = 0; k = ((1/beta - 1 + delta)/alpha)^(1/(alpha-1)) %s; c = k^alpha - delta*k;
end;
shocks; var e; stderr 0.01; end;
%s
"""


def test_qz_criterium_from_the_mod_file(tmp_path):
    text = "var x; varexo e; model; x = 1.02*x(-1) + e; end;\nshocks; var e; stderr 1; end;\n%s"
    m = build(tmp_path, text % "stoch_simul(order=1, qz_criterium=1.05, irf=0);", "a")
    np.testing.assert_allclose(pydyn.first_order(m).ghx, [[1.02]])                      # Dynare
    m = build(tmp_path, text % "stoch_simul(order=1, irf=0);", "b")
    with pytest.raises(pydyn.perturbation.BlanchardKahnError):
        pydyn.first_order(m)
    np.testing.assert_allclose(pydyn.first_order(m, qz_criterium=1.05).ghx, [[1.02]])


def test_order_and_pruning_default_to_the_mod_file(tmp_path):
    try:
        pydyn.korder.load_library()
    except FileNotFoundError:
        pytest.skip("libpydyn_korder.so not built")
    m = build(tmp_path, RBC % ("", "stoch_simul(order=2, pruning, irf=0);"))
    dr = pydyn.k_order(m)
    assert dr["order"] == 2 and dr["pruning"] is True
    shocks = np.random.default_rng(0).standard_normal((200, 1)) * 0.05
    pruned, unpruned = pydyn.simulate(dr, shocks, pruning=True), pydyn.simulate(dr, shocks, pruning=False)
    assert np.max(np.abs(pruned - unpruned)) > 1e-6
    np.testing.assert_array_equal(pydyn.simulate(dr, shocks), pruned)                  # follows the solved rule
    assert pydyn.k_order(m, pruning=False)["pruning"] is False


def test_loglinear_is_reported_by_the_level_solver(tmp_path):
    m = build(tmp_path, RBC % ("", "stoch_simul(order=1, loglinear, irf=0);"))
    with pytest.warns(UnsupportedOptionWarning, match="loglinear"):
        pydyn.first_order(m)


def test_steady_state_tolerance_follows_dynare(tmp_path):
    exact = build(tmp_path, RBC % ("", "steady;"), "exact")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ys, _ = pydyn.steady_state(exact)
    rough = build(tmp_path, RBC % ("+ 1e-4", "steady;"), "rough")          # residual between 1e-8 and Dynare's 6e-6
    with pytest.warns(UnsupportedOptionWarning, match="accepted with max"):
        pydyn.steady_state(rough)
    with pytest.raises(pydyn.SteadyStateError):
        pydyn.steady_state(rough, tol=1e-12)
    wrong = build(tmp_path, RBC % ("+ 0.5", "steady;"), "wrong")
    with pytest.raises(pydyn.SteadyStateError):
        pydyn.steady_state(wrong)
    tight = build(tmp_path, RBC % ("+ 1e-6", "steady(tolf=1e-13);"), "tight")
    with pytest.raises(pydyn.SteadyStateError):
        pydyn.steady_state(tight)
    nocheck = build(tmp_path, RBC % ("+ 0.5", "steady(nocheck);"), "nocheck")
    pydyn.steady_state(nocheck)


def test_linear_estimation_and_smoother_refuse_occbin_models(tmp_path):
    shutil.copy(EXAMPLES / "occbin" / "rbc_occbin.mod", tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "rbc_occbin.mod"))
    with pytest.raises(NotImplementedError, match="occbin_constraints"):
        smoother.smooth(m, data=np.zeros((10, 1)), varobs=["log_C"])
    with pytest.raises(NotImplementedError, match="occbin_constraints"):
        Estimation(m)


def test_perfect_foresight_options(tmp_path):
    base = (EXAMPLES / "perfect_foresight" / "perfect_foresight_rbc.mod").read_text()
    i = base.index("perfect_foresight_solver")
    j = base.index(";", i)
    m = build(tmp_path, base[:i] + "perfect_foresight_solver(tolf=1e-4, maxit=1, stack_solve_algo=0)" + base[j:], "a")
    with pytest.warns(UnsupportedOptionWarning, match="stack_solve_algo"):
        with pytest.raises(pydyn.PerfectForesightError):       # maxit=1 from the file is honoured
            pydyn.perfect_foresight(m)
    assert pydyn.perfect_foresight(m, maxit=50).max_residual < 1e-4


def test_estimation_reports_options_it_does_not_act_on(tmp_path):
    for f in REF.glob("est_nk*"):
        if f.suffix in (".mod", ".m", ".csv"):
            shutil.copy(f, tmp_path)
    text = (tmp_path / "est_nk.mod").read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "est_nk.mod"))
    with pytest.warns(UnsupportedOptionWarning, match="posterior_sampling_method"):
        Estimation(m, posterior_sampling_method="slice")
