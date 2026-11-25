"""Why a news inversion-filter trial point fails (cause: horizon_too_short / no_solution / undecided / milp_timeout / numerical), the failure
info of the estimation (``last_failure``, ``last_info``), and the opt-in ``obc_horizon_max`` (retry with a doubled horizon).
Found by the OBC soak test: at extreme NKM vectors the news filter at horizon 40 returned the generic code 304 where OccBin has a finite
value and horizon 80 reproduces it (docs/OBC_SOAK.md, soak/repro/finite_vs_inf_news_304.py)."""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn.obc import OBCError
from pydyn.obc_filter import NewsFilterSolver, RegimeFilterSolver
from pydyn.occbin import Occbin
from pydyn.occbin_estimation import OccbinEstimation

REF = Path(os.environ.get("PYDYN_REF", str(Path(__file__).resolve().parent.parent / "reference")))
# an extreme NKM vector: OccBin's inversion filter gives -210551.277947312 (10 s); news at horizon 40 fails (code 304), horizon 80 gives the same value
TH = np.array([114.66637644899114, 1.0892489266590764, 0.19766855658573093, 0.6874612608464392, 0.6985799052891387,
               0.46604253593794787, 0.005397842157745668, 0.0016956255944929521, 0.0021002683787342213])
OCCBIN_LP = -210551.277947312


def _model(tmp, name):
    shutil.copy(REF / name, tmp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(tmp / name))


@pytest.fixture(scope="module")
def nkm_est(tmp_path_factory):
    m = _model(tmp_path_factory.mktemp("nkm"), "occest_nkm.mod")
    r = loadmat(REF / "occest_nkm_lik.mat", squeeze_me=True)
    data = np.atleast_2d(r["data"].T).T
    X = np.atleast_2d(r["X"])
    X = X if X.shape[0] == 9 else X.T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = OccbinEstimation(m, data={v: data[:, j] for j, v in enumerate(m.varobs)}, nobs=data.shape[0], first_obs=1,
                             occbin={"likelihood_inversion_filter": True}, obc_method="news")
    return e, X


def test_default_failure_keeps_code_and_adds_cause(nkm_est):
    e, X = nkm_est
    e.obc_horizon_max = None
    assert e.log_posterior(TH) == -np.inf and e.last_code == 304
    f = e.last_failure
    assert f["period"] == 66 and f["horizon"] == 40 and f["primary"] in ("horizon_too_short", "undecided")
    assert "obc_horizon" in f["advice"] and "period 66" in e.last_info and f["primary"] in e.last_info
    s = e.last_news_solver
    assert sum(s.causes.values()) == s.errors[313] and s.retries == 0
    assert sum(f["counts"].values()) > 0


def test_horizon_max_gives_occbin_value(nkm_est):
    e, X = nkm_est
    try:
        # the LCP needs horizon 80 here; the regime formulation ('auto' for this approx-class model) needs 160
        for form, hmax, big in (("lcp", 80, {80}), ("auto", 160, {80, 160})):
            e.obc_formulation, e.obc_horizon_max = form, hmax
            lp = e.log_posterior(TH)
            assert e.last_code == 0 and e.last_failure is None
            assert lp == pytest.approx(OCCBIN_LP, abs=1e-6)
            s = e.last_news_solver
            assert s.retries >= s.retried_ok > 0 and sum(s.causes.values()) == 0 and set(s._big) == big
        e.obc_formulation, e.obc_horizon_max = "auto", 80
        assert e.log_posterior(TH) == -np.inf and e.last_failure["primary"] == "horizon_too_short"
    finally:
        e.obc_horizon_max = None
        e.obc_formulation = "auto"


def test_horizon_max_changes_nothing_where_the_default_succeeds(nkm_est):
    e, X = nkm_est
    th = X[:, 3]
    e.obc_horizon_max = None
    a = e.log_posterior(th)
    e.obc_horizon_max = 160
    try:
        b = e.log_posterior(th)
        assert a == b and e.last_news_solver.retried_ok == 0          # trial points that fail are retried (and fail again)
    finally:
        e.obc_horizon_max = None


def test_horizon_max_below_horizon_is_an_error(tmp_path):
    ob = Occbin(_model(tmp_path, "obc_zlb.mod"))
    with pytest.raises(ValueError, match="obc_horizon_max"):
        NewsFilterSolver(ob, horizon=40, horizon_max=20)


@pytest.fixture(scope="module")
def borrcon(tmp_path_factory):
    return Occbin(_model(tmp_path_factory.mktemp("bc"), "occ_borrcon.mod"))


def _scan(ob, **kw):
    s = NewsFilterSolver(ob, horizon=20, **kw)
    rng = np.random.default_rng(1)
    for _ in range(400):
        sh = rng.standard_normal((1, ob.n_exo)) * float(rng.choice([0.05, 0.2, 0.6, 1.5]))
        init = rng.standard_normal(ob.n) * float(rng.choice([0, 0.05, 0.2]))
        s.solve(sh, init, {})
    return s


def test_lcp_solver_classifies_horizon_failures_and_retries(borrcon):
    a = _scan(borrcon)
    assert a.errors[313] > 50 and a.causes == {"horizon_too_short": a.errors[313]}          # class (b): the check-ahead rows
    b = _scan(borrcon, horizon_max=80)
    assert b.retried_ok > 0 and b.errors.get(313, 0) < a.errors[313] and sum(b.causes.values()) == b.errors.get(313, 0)


def test_regime_solver_maps_status_to_cause(tmp_path, monkeypatch):
    ob = Occbin(_model(tmp_path, "obc_zlb.mod"))
    s = RegimeFilterSolver(ob, horizon=10, check_ahead=10)          # forced class: the stub below replaces its step
    for status, cause in (("horizon_too_short", "horizon_too_short"), ("no_solution", "no_solution"), ("not_converged", "milp_timeout")):
        def step(*a, _st=status, **k):
            raise OBCError("x", _st)
        monkeypatch.setattr(s.rb, "_step", step)
        flag, res = s.solve(np.zeros((1, ob.n_exo)), np.zeros(ob.n), {})
        assert flag == 313 and res is None and s.last_cause == cause
    assert s.causes == {"horizon_too_short": 1, "no_solution": 1, "milp_timeout": 1}
