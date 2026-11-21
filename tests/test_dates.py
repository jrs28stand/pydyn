"""Dates (pydyn.dates) against Dynare 7.2: arithmetic of the dseries dates class (reference/dates/dates_ops.mat
from reference/dates/r_dates.m), perfect foresight with dated shocks and first/last_simulation_period
(reference/dates/pf_dates*.{mod,mat}, saved with reference/pf_save.m), and estimation with the data command
(first_obs/last_obs/nobs as dates, a .csv file with a date column) and dated heteroskedastic_shocks
(reference/dates/est_dated.mod, est_dated_v<V>.mat from reference/dates/r_dated_estimation.m)."""
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import estimation, heteroskedastic, smoother
from pydyn.dates import Date, DateError

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference" / "dates"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_arithmetic_matches_dynare():
    r = loadmat(REF / "dates_ops.mat", squeeze_me=True)
    s = str(r["strings"]).split(",")
    k = np.atleast_1d(r["k"]).astype(int)
    plus = np.array(str(r["plus"]).split(",")).reshape(len(k), len(s)).T           # MATLAB column-major
    for i, x in enumerate(s):
        d = Date.parse(x)
        assert d.freq == int(r["freq"][i])
        for j, kk in enumerate(k):
            assert str(d + int(kk)) == plus[i, j], (x, kk)
            assert (d + int(kk)) - d == kk
    pairs = np.array(str(r["pairs"]).split(",")).reshape(2, -1).T
    for (a, b), dd in zip(pairs, np.atleast_1d(r["diff"])):
        assert Date.parse(a) - Date.parse(b) == dd
    assert Date.parse("dates('2000Q1')") == Date.parse("2000q1") < Date.parse("2000Q2")
    assert [str(x) for x in Date.parse("1999Q4").range_to(Date.parse("2000Q2"))] == ["1999Q4", "2000Q1", "2000Q2"]
    with pytest.raises(DateError):
        Date.parse("2000Q1") - Date.parse("2000M1")
    with pytest.raises(DateError):
        Date.parse("2001W53")                     # 2001 has 52 ISO weeks


@pytest.mark.parametrize("name", ["pf_dates1", "pf_dates2"])
def test_perfect_foresight_with_dated_shocks(tmp_path, name):
    """1: first and last_simulation_period without periods; 2: periods and last_simulation_period."""
    from test_perfect_foresight import check, solve
    shutil.copy(REF / f"{name}.mod", tmp_path)
    m = pydyn.Model(pydyn.run_preprocessor(tmp_path / f"{name}.mod"))
    r = loadmat(REF / f"{name}.mat", squeeze_me=True)
    n = len(str(r["endo_names"]).split(","))
    r["endo"] = np.asarray(r["endo_simul"], float).reshape(n, -1).T
    r["exo"] = np.asarray(r["exo_simul"], float).reshape(r["endo"].shape[0], -1)
    check(solve(m), r)


def test_perfect_foresight_date_errors(tmp_path):
    from test_perfect_foresight import solve
    text = (REF / "pf_dates1.mod").read_text()
    bad = text.replace("periods 2000Q1 2000Q3:2001Q2 2003Q4;", "periods 1999Q4 2000Q3:2001Q2 2003Q4;")
    (tmp_path / "a.mod").write_text(bad)
    m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "a.mod"))
    with pytest.raises(pydyn.PerfectForesightError, match="before the first simulation period"):
        solve(m)
    (tmp_path / "b.mod").write_text(text.replace(
        "perfect_foresight_setup(first_simulation_period=2000Q1, last_simulation_period=2049Q4);",
        "perfect_foresight_setup(periods=200);"))
    m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "b.mod"))
    with pytest.raises(pydyn.PerfectForesightError, match="neither first_simulation_period"):
        solve(m)
    with pytest.raises(ValueError, match="inconsistent"):
        solve(pydyn.Model(pydyn.run_preprocessor(tmp_path / "a.mod")), periods=10)


@pytest.mark.parametrize("v", [0, 1, 2])
def test_estimation_with_the_data_command(tmp_path, v):
    """0: first_obs/last_obs dates and dated heteroskedastic periods; 1: first_obs date and nobs; 2: smoother."""
    for f in ("est_dated.mod", "est_nk_dated.csv"):
        shutil.copy(REF / f, tmp_path)
    m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "est_dated.mod", extra_args=(f"-DVARIANT={v}",)))
    assert "estimation_data" not in m.unhandled_statements
    r = loadmat(REF / f"est_dated_v{v}.mat", squeeze_me=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = estimation.Estimation(m)
    assert (est.options["first_obs"], est.nobs, str(est.first_date)) == \
        ((11, 94, "1992Q3") if v != 1 else (5, 80, "1991Q1"))
    for k in range(r["X"].shape[1]):
        if r["info"][k] == 0:
            np.testing.assert_allclose(est.log_likelihood(r["X"][:, k]), r["loglik"][k], rtol=1e-12)
    if "Qvalue" in r:
        qv, qs = heteroskedastic.q_arrays(m, est.nobs, est.options["first_obs"], est.first_date)
        for mine, theirs in ((qv, r["Qvalue"]), (qs, r["Qscale"])):
            assert np.array_equal(np.isnan(mine), np.isnan(theirs))
            np.testing.assert_allclose(mine[~np.isnan(mine)], theirs[~np.isnan(theirs)], rtol=1e-15)
    if v == 2:
        res = smoother.smooth(m, estimation=est, theta=est.x0)
        np.testing.assert_allclose(res.shocks, r["shocks"], rtol=1e-9, atol=1e-10)
