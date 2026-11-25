# This file is part of pydyn. GPL-3.0-or-later (see LICENSE).
"""No Dynare option is silently ignored: pydyn.option_policy classifies every option of the handled statements."""
import json
import os
import shutil
import warnings
from pathlib import Path

import pytest

import pydyn
from pydyn import ResultsError, option_policy as P

SRC = Path(os.environ.get("DYNARE_SRC", "/home/claude/dl/src72"))

MOD = """var y c k a; varexo e;
parameters alpha beta delta rho;
alpha=0.36;beta=0.99;delta=0.025;rho=0.9;
model;
c^(-1)=beta*c(+1)^(-1)*(alpha*a(+1)*k^(alpha-1)+1-delta);
k=a*k(-1)^alpha+(1-delta)*k(-1)-c;
y=a*k(-1)^alpha;
log(a)=rho*log(a(-1))+e;
end;
initval; k=30;c=2.5;a=1;y=3;e=0;end;
steady;
shocks; var e; stderr 0.01; end;
"""


def _run(tmp_path, stmt):
    (tmp_path / "m.mod").write_text(MOD + stmt + "\n")
    return pydyn.dynare(tmp_path / "m.mod", nosave=True)


def test_tables_are_disjoint_and_cover_the_grammar():
    """Each option is in exactly one set; with the preprocessor sources, the tables list exactly the options of Dynare 7.2."""
    for st, s in P.SPECS.items():
        assert not (s["honoured"] & s["harmless"]) and not (s["honoured"] & s["refused"]) and not (s["harmless"] & s["refused"]), st
    if not (SRC / "preprocessor/src/DynareBison.yy").exists():
        pytest.skip("Dynare sources not available")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import options_audit
    gt = options_audit.statements(SRC)
    for st, s in P.SPECS.items():
        assert s["honoured"] | s["harmless"] | s["refused"] == set(gt[st]), st


@pytest.fixture
def refused_simul_replic(monkeypatch):
    """A permanently refused option is needed to test the mechanism: put simul_replic (implemented since) back for the test."""
    for st in ("stoch_simul", "osr", "ramsey_policy", "discretionary_policy"):
        spec = P.SPECS[st]
        monkeypatch.setitem(spec, "honoured", spec["honoured"] - {"simul_replic"})
        monkeypatch.setitem(spec, "refused", spec["refused"] | {"simul_replic"})


def test_classify(refused_simul_replic):
    assert P.classify("stoch_simul", "order") == "honoured"
    assert P.classify("stoch_simul", "nograph") == "harmless"
    assert P.classify("stoch_simul", "dr_cycle_reduction") == "harmless"
    assert P.classify("stoch_simul", "simul_replic") == "refused"
    assert P.classify("stoch_simul", "an_option_of_a_later_dynare") == "refused"      # unknown options are refused
    assert P.classify("plot_shock_decomposition", "plot_shock_decomp.colormap") == "harmless"
    assert P.classify("sensitivity", "rmse") is None                                   # not a statement pydyn runs
    why = P.check_statement("estimation", {"skewed_kalman.prune_tol": 1, "order": 1})
    assert "skewed_kalman.prune_tol" in why and "skewed Kalman" in why and "order" not in why.split(";")[0]


def test_refused_option_stops_the_statement(tmp_path, refused_simul_replic):
    with pytest.raises(ResultsError, match=r"<stoch_simul>: pydyn does not act on simul_replic"):
        _run(tmp_path, "stoch_simul(order=1,periods=50,simul_replic=3,nograph);")


def test_harmless_and_honoured_options_run(tmp_path):
    r = _run(tmp_path, "stoch_simul(order=1,irf=0,nograph,nomodelsummary,dr=cycle_reduction,dr_cycle_reduction_tol=1e-9);")
    assert r.oo_["dr"]["ghx"] is not None


def test_plain_api_warns_about_refused_options(tmp_path, refused_simul_replic):
    (tmp_path / "m.mod").write_text(MOD + "stoch_simul(order=1,periods=50,simul_replic=3,nograph);\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pydyn.Model(pydyn.run_preprocessor(tmp_path / "m.mod"))
    assert any("simul_replic" in str(x.message) for x in w)


def test_fsolve_options_are_refused_unless_they_cannot_move_the_solution():
    """steady_state/ramst_solve_algo_0 passes Display and AutoScaling only; TolFun or MaxIter would change fsolve's end point."""
    assert not P.refused_options("steady", {"fsolve_options": "''Display'',''iter''"})
    assert not P.refused_options("steady", {"fsolve_options": "''AutoScaling'',''on'',''Display'',''off''"})
    assert P.refused_options("steady", {"fsolve_options": "''TolFun'',1e-3"})
    assert P.refused_options("check", {"fsolve_options": "''Display'',''iter'',''MaxIter'',3"})


def test_realtime_shock_decomposition_options_are_honoured():
    """They reach pydyn.smoother.realtime_shock_decomposition through the statement's options (reference/oo/nd_oo.mod)."""
    for k in ("forecast", "save_realtime", "presample", "fast_realtime"):
        assert P.classify("realtime_shock_decomposition", "shock_decomp." + k) == "honoured"


def test_algorithm_tolerances_are_harmless_only_at_dynares_default_or_tighter():
    """decision_rules/first_order/walsh: cycle reduction at dr_cycle_reduction_tol=1e-1 differs from QZ (Dynare's stored rules are loose)."""
    ok = {"dr_cycle_reduction": "true", "dr_cycle_reduction_tol": "1e-7", "dr_cycle_reduction_maxiter": 300,
          "dr_logarithmic_reduction": "true", "dr_logarithmic_reduction_tol": "1e-12"}
    assert not P.refused_options("stoch_simul", ok)
    for k, v in {"dr_cycle_reduction_tol": "1e-6", "dr_cycle_reduction_maxiter": 50, "dr_logarithmic_reduction_tol": "1e-8"}.items():
        assert [x[0] for x in P.refused_options("stoch_simul", {k: v})] == [k]
    assert not P.refused_options("estimation", {"lyapunov_db": "true"})
    assert not P.refused_options("estimation", {"lyapunov_srs": "true"})
    assert P.refused_options("estimation", {"lyapunov_fp": "true"})          # 1.6e-8 off at the default tolerance
    assert P.refused_options("estimation", {"lyapunov_fixed_point_tol": "1e-10"})


def test_shock_decomposition_smoother_options_reach_the_smoother(tmp_path):
    """tools/options_probe.py found shock_decomposition(nobs=..., first_obs=..., lik_init=...) ignored: Dynare writes them into
    options_ before evaluate_smoother.  Sums of oo_.shock_decomposition below are from Dynare 7.2 under Octave."""
    ref = Path(__file__).resolve().parent.parent / "reference" / "oo"
    full = (ref / "est_conv.mod").read_text()
    head = full.split("estimated_params;")[0]
    tail = full.split("end;\n\nvarobs")[1].split("options_.solve_tolf")[0]
    shutil.copy(ref / "fsdat_simul.m", tmp_path)
    base = head + "varobs" + tail + "calib_smoother(datafile=fsdat_simul,nobs=150,first_obs=3,loglinear) gp_obs;\n"
    expected = {"": -4.983154700884333, ",nobs=100": -1.7931448682171935, ",first_obs=8": -5.050645165604415}
    for opt, total in expected.items():
        (tmp_path / "m.mod").write_text(base + f"shock_decomposition(parameter_set=calibration,nograph{opt}) y;\n")
        z = pydyn.dynare(tmp_path / "m.mod", nosave=True).oo_["shock_decomposition"]
        assert float(z.sum()) == pytest.approx(total, rel=1e-9)
