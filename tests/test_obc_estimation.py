"""Estimation with the news-shock solver: the ``obc_method`` option of OccbinEstimation, the general (several
equations) filter solver, and the fail-fast handling of trial points.  Tolerances are stated where they are set.

References: reference/occest_nkm*_lik.mat (Dynare 7.2, see test_occbin_estimation.py); reference/occest_bf_data.mat is
the data of Dynare's tests/estimation/slice/datafile.mat (identical to tests/occbin/ppf/datafile.mat; a v7.3 file,
converted to v7 with Octave) for bf.mod and bf_ppf.mod, whose likelihood is compared with pydyn's OccBin solver.
"""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn.estimation import EstimationError
from pydyn.obc_filter import LCPFilterSolver, NewsFilterSolver, RegimeFilterSolver
from pydyn.occbin_estimation import OccbinEstimation

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
EXAMPLES = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/dl/src72/examples"))
DYNARE_TESTS = Path(os.environ.get("PYDYN_DYNARE_TESTS", str(EXAMPLES.parent / "tests")))
needs_dynare = pytest.mark.skipif(not (DYNARE_TESTS / "occbin").exists(),
                                  reason="Dynare's tests folder not found (PYDYN_DYNARE_TESTS)")
IVF = {"likelihood_inversion_filter": True}


def _model(tmp, name, src=REF, text=None):
    p = tmp / name
    if text is None:
        shutil.copy(Path(src) / name, p)
    else:
        p.write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(p))


def _est(m, data, occbin=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OccbinEstimation(m, data={v: data[:, j] for j, v in enumerate(m.varobs)}, nobs=data.shape[0], first_obs=1,
                                occbin=occbin, **kw)


@pytest.fixture(scope="module")
def nkms(tmp_path_factory):
    m = _model(tmp_path_factory.mktemp("nkms"), "occest_nkms.mod")
    r = loadmat(REF / "occest_nkms_lik.mat", squeeze_me=True)
    data = np.atleast_2d(r["data"].T).T
    X = np.atleast_2d(r["X"])
    return m, data, r, X


# ------------------------------------------------------------------------------------------------ the option
def test_obc_method_option_default_and_errors(nkms):
    m, data, r, X = nkms
    e = _est(m, data, IVF)
    assert (e.obc_method, e.obc_horizon, e.obc_selection) == ("occbin", 40, "occbin")
    e.obc_method = "news"                                    # assignment keeps working
    assert e.obc_method == "news"
    for bad in ("minnorm", "holden", "", None, 3):
        with pytest.raises(ValueError, match="unknown obc_method"):
            e.obc_method = bad
        if bad is not None:                                  # None in the constructor means "not given"
            with pytest.raises(ValueError, match="unknown obc_method"):
                _est(m, data, IVF, obc_method=bad)
    with pytest.raises(ValueError, match="unknown obc_selection"):
        e.obc_selection = "median"
    for bad in (0, -3, 2.5, "x"):
        with pytest.raises(ValueError, match="obc_horizon"):
            e.obc_horizon = bad
    e2 = _est(m, data, IVF, obc_method="news", obc_horizon=25)
    assert (e2.obc_method, e2.obc_horizon) == ("news", 25)


def test_obc_option_from_native_mod_lines(tmp_path, nkms):
    m0, data, *_ = nkms
    src = (REF / "occest_nkms.mod").read_text()
    m = _model(tmp_path, "occest_nkms_opt.mod",
               text="options_.occbin.obc_method = 'news';\noptions_.occbin.obc_horizon = 30;\n" + src)
    e = _est(m, data, IVF)
    assert (e.obc_method, e.obc_horizon) == ("news", 30)
    assert _est(m, data, IVF, obc_method="occbin", obc_horizon=20).obc_method == "occbin"      # keyword wins
    with pytest.raises(EstimationError, match="inversion filter only"):
        _est(m, data)                                         # the .mod file says news, the piecewise Kalman filter is on


def test_piecewise_kalman_filter_refuses_news(nkms):
    m, data, r, X = nkms
    with pytest.raises(EstimationError, match="inversion filter only"):
        _est(m, data, obc_method="news")                       # default filter of an OccBin model is the PKF
    e = _est(m, data)
    with pytest.raises(EstimationError, match="inversion filter only"):
        e.obc_method = "news"
    e.obc_method = "occbin"                                    # still allowed
    e.log_posterior(X[:, 0] if X.shape[0] == e.n else X[0])
    ei = _est(m, data, IVF, obc_method="news")
    ei.inversion_filter = False                                # defeat the set-up checks: run_filter must still refuse
    with pytest.raises(EstimationError, match="inversion filter only"):
        ei.run_filter(ei.x0)
    with pytest.raises(EstimationError, match="inversion filter only"):
        ei.occbin_smoother(ei.x0)


def test_default_results_unchanged(nkms):
    m, data, r, X = nkms
    a, b = _est(m, data, IVF), _est(m, data, IVF, obc_method="occbin")
    X = X if X.shape[0] == a.n else X.T
    for k in range(3):
        assert a.log_posterior(X[:, k]) == b.log_posterior(X[:, k])                    # bit for bit
        assert a.log_posterior(X[:, k]) == pytest.approx(-r["ivf_fval"][k], rel=1e-12, abs=1e-8)
    assert a.last_news_solver is None if hasattr(a, "last_news_solver") else True


def test_news_solver_class_choice(nkms):
    m, data, r, X = nkms
    e = _est(m, data, IVF, obc_method="news")
    e.log_posterior(X[:, 0] if X.shape[0] == e.n else X[0])
    assert isinstance(e.last_news_solver, LCPFilterSolver) and isinstance(e.last_news_solver, NewsFilterSolver)


# ------------------------------------------------------------------------------ several equations (bf, bf_ppf)
def _bf(tmp, rel):
    src = DYNARE_TESTS / rel
    m = _model(tmp, src.name, src=src.parent)
    d = loadmat(REF / "occest_bf_data.mat", squeeze_me=True)
    data = np.column_stack([d["q"], d["r"]])
    kw = dict(posterior_sampling_method="random_walk_metropolis_hastings", posterior_sampler_options=())
    return m, data, kw


@needs_dynare
@pytest.mark.parametrize("rel", ["estimation/slice/bf.mod", "occbin/ppf/bf_ppf.mod"])
def test_inversion_filter_news_equals_occbin_several_equations(tmp_path, rel):
    """Two equations switch.  Same regimes and shocks as OccBin's solver at 8 parameter vectors (the mod file's initial
    values and 7 perturbed by about 5%, seeded); log likelihood equal to 1e-9 (measured: 3.4e-13, rounding of the
    sums); shocks 1e-12 (measured 5e-16); one of the vectors makes the filter fail in both."""
    m, data, kw = _bf(tmp_path, rel)
    e_occ = _est(m, data, IVF, **kw)
    e_news = _est(m, data, IVF, obc_method="news", **kw)
    rng = np.random.default_rng(1)
    x0 = np.array(e_occ.x0)
    nfail = 0
    for k in range(8):
        th = x0 if k == 0 else x0 * np.exp(0.05 * rng.standard_normal(5))
        a, b = e_occ.log_posterior(th), e_news.log_posterior(th)
        assert isinstance(e_news.last_news_solver, RegimeFilterSolver)
        if not np.isfinite(a):
            nfail += 1
            assert not np.isfinite(b) and e_occ.last_code == e_news.last_code
            continue
        assert b == pytest.approx(a, abs=1e-9)
        ra, rb = e_occ.last_result, e_news.last_result
        assert ra.regime_history == rb.regime_history
        np.testing.assert_allclose(rb.filtered_errs, ra.filtered_errs, atol=1e-12)
        np.testing.assert_allclose(rb.stateval, ra.stateval, atol=1e-12)
        assert sum(reg[0][0][0] for reg in rb.regime_history) >= 5            # the sample reaches the bound
    assert nfail <= 2


@needs_dynare
def test_regime_filter_rule_matches_occbin_rule(tmp_path):
    """The first-period rule (T, R, C) of the regime solver equals OccBin's, in a binding period."""
    m, data, kw = _bf(tmp_path, "estimation/slice/bf.mod")
    e = _est(m, data, IVF, obc_method="news", **kw)
    ob = e.occbin_at(e.x0)
    sv = NewsFilterSolver(ob)
    from pydyn.occbin_filter import _Solver
    ref = _Solver(ob)
    sim = dict(periods=3, maxit=30, check_ahead_periods=200, max_check_ahead_periods=np.inf)
    rng = np.random.default_rng(2)
    sd = np.sqrt(np.diag(m.Sigma_e))
    nbind = 0
    for _ in range(40):
        init = 0.05 * rng.standard_normal(ob.n) * np.array([1, 1, 1, 1, 1.0])
        shock = rng.standard_normal(ob.n_exo) * sd * 2
        f1, r1 = sv.solve(shock, init, sim)
        f2, r2 = ref.solve(shock, init, sim)
        assert f1 == f2 == 0
        assert r1.regimes == r2.regimes
        nbind += r1.regimes[0][0][0][0]
        for u, v in zip(r1.ss[0], r2.ss[0]):
            np.testing.assert_allclose(u, v, atol=1e-11)
        np.testing.assert_allclose(r1.piecewise[0], r2.piecewise[0], atol=1e-11)
    assert nbind >= 5


# ------------------------------------------------------------------------------------------------- fail fast
def test_failed_trial_point_fails_fast(tmp_path):
    """NKM, 4th stored parameter vector: one trial shock of the root finder has no solution; the active-set iteration
    fails and the linear programme is feasible, so only an integer programme can decide (it needs 1.5 s to prove
    that there is none).  With the default (no integer programme: a failed trial point after the cheap checks) the
    likelihood is Dynare's and the time is that of the other points; OccBin fails at the same trial point."""
    m = _model(tmp_path, "occest_nkm.mod")
    r = loadmat(REF / "occest_nkm_lik.mat", squeeze_me=True)
    data = np.atleast_2d(r["data"].T).T
    e = _est(m, data, IVF, obc_method="news")
    X = np.atleast_2d(r["X"])
    X = X if X.shape[0] == e.n else X.T
    lp = e.log_posterior(X[:, 3])
    assert lp == pytest.approx(-r["ivf_fval"][3], rel=1e-11, abs=1e-8)
    sv = e.last_news_solver
    assert sv.fallbacks == 2 and sv.proved_infeasible == 1 and sv.skipped == 1 and sv.timeouts == 0 and sv.rescued == 0
    assert sv.errors == {313: 2}
    # with the integer programme switched on the answer is the same (it finds no solution either)
    e.obc_milp_time_limit = 5.0
    assert e.log_posterior(X[:, 3]) == lp
    assert e.last_news_solver.skipped == 0 and e.last_news_solver.rescued == 0
