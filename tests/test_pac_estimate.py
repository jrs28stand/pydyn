"""pac.estimate.nls / iterative_ols of the results layer: the optimisers Octave cannot run (lsqnonlin), the refusals, and the
messages of Dynare's checks.  The estimates themselves are checked against Octave in tests/test_results.py (cases pe_*)."""
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

import pydyn
from pydyn import ResultsError

REF = Path(__file__).resolve().parent.parent / "reference" / "oo"


def run(tmp_path, case, edit=lambda s: s):
    text = edit((REF / f"{case}.mod").read_text())
    f = tmp_path / f"{case}.mod"
    f.write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.dynare(f)


def test_lsqnonlin_finds_the_csminwel_minimum(tmp_path):
    def edit(s):
        k = s.index("e_c_m_lsqnonlin =")
        call = "pac.estimate.nls('zpac', eparams, edata, 2005Q1:2005Q1+200, 'lsqnonlin', 'Algorithm', 'levenberg-marquardt');\n"
        return s[:k] + call + s[k:]
    ws = run(tmp_path, "pe_t18", edit)
    for p in ("e_c_m", "c_z_1", "c_z_2", "lambda"):
        assert abs(float(ws.vars[f"{p}_nls"]) - float(ws.vars[f"{p}_lsqnonlin"])) < 1e-4, p


@pytest.mark.parametrize("opt, kind", [("'annealing'", "random search"), ("'particleswarm'", "random search")])
def test_random_search_is_refused(tmp_path, opt, kind):
    with pytest.raises(ResultsError, match=kind):
        run(tmp_path, "pe_t6", lambda s: s.replace("2005Q1+400);", f"2005Q1+400, {opt});"))


def test_messages(tmp_path):
    with pytest.raises(ResultsError, match="Options must come by key/value pairs"):
        run(tmp_path, "pe_t6", lambda s: s.replace("2005Q1+400);", "2005Q1+400, 'csminwel', 'MaxIter');"))
    with pytest.raises(ResultsError, match="is not an implemented optimization routine"):
        run(tmp_path, "pe_t6", lambda s: s.replace("2005Q1+400);", "2005Q1+400, 'nosuch');"))
    with pytest.raises(ResultsError, match="This is not a PAC equation"):
        run(tmp_path, "pe_t6", lambda s: s.replace("pac.estimate.nls('zpac'", "pac.estimate.nls('eq:y'"))
    with pytest.raises(ResultsError, match="must have NaN values"):
        run(tmp_path, "pe_t6", lambda s: s.replace("edata.ez = dseries(NaN(TrueData.nobs, 1), 200Q1, 'ez');",
                                                   "edata.ez = dseries(ones(TrueData.nobs, 1), 2000Q1, 'ez');"))
    with pytest.raises(ResultsError, match="must be used in equation"):
        run(tmp_path, "pe_t6", lambda s: s.replace("eparams.gamma = .1;", "eparams.gamma = .1; eparams.rho_1 = .5;"))


def test_oo_pac_fields(tmp_path):
    ws = run(tmp_path, "pe_t6")
    o = ws.oo_["pac"]["pacman"]
    assert set(o.keys()) >= {"lhs", "fit", "residual", "ssr", "s2", "R2", "parnames", "estimator", "covariance", "student"}
    est = np.asarray(o["estimator"]).ravel()
    assert est.size == 4 and np.allclose(est[:3], [0.475, 0.206, -0.081], atol=1e-3)
    p = np.asarray(ws.M_["params"]).ravel()
    names = list(ws.model.param_names)
    assert abs(p[names.index("e_c_m")] - est[0]) < 1e-14
