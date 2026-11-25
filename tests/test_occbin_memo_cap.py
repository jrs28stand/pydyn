"""The memo of Occbin.rules (the rule trie) is bounded by ``Occbin.rule_memo_bytes`` and the bound never changes a result:
simulations and inversion-filter likelihoods with a tiny budget or no memo are bit for bit those of the default budget.
(Found by the OBC soak test: an extreme NKM parameter vector built 232,740 nodes, 1.3 GB, in one OccBin filter evaluation; docs/OBC_SOAK.md.)"""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn.occbin import Occbin
from pydyn.occbin_estimation import OccbinEstimation

REF = Path(os.environ.get("PYDYN_REF", str(Path(__file__).resolve().parent.parent / "reference")))


def _model(tmp, name):
    shutil.copy(REF / name, tmp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(tmp / name))


@pytest.fixture
def cap_restore():
    old = Occbin.rule_memo_bytes
    yield
    Occbin.rule_memo_bytes = old


def test_simulation_identical_with_any_budget(tmp_path, cap_restore):
    m = _model(tmp_path, "obc_zlb2.mod")
    sd = np.sqrt(np.diag(np.array(m.Sigma_e, float)))
    S = np.random.default_rng(3).standard_normal((300, m.n_exo)) * sd * 3
    res = {}
    for cap in (None, 0, 3000, 20000):
        if cap is not None:
            Occbin.rule_memo_bytes = cap
        ob = Occbin(m)
        r = ob.simulate(S, periods=300)
        res[cap] = (r, ob)
        if cap is not None:
            assert ob.__dict__.get("_rule_bytes", 0) <= cap
            assert sum(a.nbytes for nd in ob._rule_node[1:] for a in nd if a is not None) <= cap
    base = res[None][0]
    assert sum(1 for h in base.regime_history if any(c[0][0] for c in h)) > 20           # the constraint binds
    for cap in (0, 3000, 20000):
        r = res[cap][0]
        assert np.array_equal(r.piecewise, base.piecewise)
        assert r.regime_history == base.regime_history
        for (T1, R1, C1), (T2, R2, C2) in zip(r.ss, base.ss):
            assert np.array_equal(T1, T2) and np.array_equal(R1, R2) and np.array_equal(C1, C2)
    assert len(res[0][1].__dict__.get("_rule_node", [0])) == 1                           # budget 0: nothing stored


def test_inversion_filter_identical_with_small_budget(tmp_path, cap_restore):
    m = _model(tmp_path, "occest_nkm.mod")
    r = loadmat(REF / "occest_nkm_lik.mat", squeeze_me=True)
    data = np.atleast_2d(r["data"].T).T
    X = np.atleast_2d(r["X"])
    X = X if X.shape[0] == 9 else X.T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = OccbinEstimation(m, data={v: data[:, j] for j, v in enumerate(m.varobs)}, nobs=data.shape[0], first_obs=1,
                             occbin={"likelihood_inversion_filter": True})
    th = X[:, 3]
    base = e.log_posterior(th)
    assert np.isfinite(base)
    sizes = {}
    for cap in (0, 20000, 400000):
        Occbin.rule_memo_bytes = cap
        assert e.log_posterior(th) == base
        ob = e.last_result.occbin
        sizes[cap] = len(ob.__dict__.get("_rule_node", [0])) - 1
        assert ob.__dict__.get("_rule_bytes", 0) <= cap
    assert sizes[0] == 0 and 0 < sizes[20000] < 200


def test_default_budget_is_large():
    assert Occbin.rule_memo_bytes >= 256 * 2 ** 20              # ordinary runs need at most ~15 MB (2,600 nodes)
