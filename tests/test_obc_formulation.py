"""``obc_formulation`` (auto / lcp / regime / hybrid) of the news method in simulation and in the inversion filter (docs/OBC_ESTIMATION.md,
'Approx-class constraints').  Models: the zlb model with a policy shock in the rate equation, exact (additive shock, d = 0) and approx (d = 1%)."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "soak" / "approx"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "soak"))
gen = pytest.importorskip("gen")
from pydyn.obc import constraint_objects  # noqa: E402
from pydyn.occbin import Occbin  # noqa: E402


@pytest.fixture(scope="module")
def exact():
    return gen.build("tf_exact", gen.zlbm(0))


@pytest.fixture(scope="module")
def approx():
    return gen.build("tf_approx", gen.zlbm(0.01))


def _shocks(m, T=300, seed=3, scale=1.0):
    return np.random.default_rng(seed).standard_normal((T, m.n_exo)) * scale


def test_classes(exact, approx):
    assert [c.cls for c in constraint_objects(Occbin(exact))] == ["complementarity"]
    c = constraint_objects(Occbin(approx))[0]
    assert c.cls == "complementarity" and 0.005 < c.approx < 0.02


def test_exact_auto_is_lcp_bitwise(exact):
    S = _shocks(exact)
    ob = Occbin(exact)
    a = ob.simulate(S, periods=300, obc_method="news")
    b = ob.simulate(S, periods=300, obc_method="news", obc_formulation="lcp")
    assert np.array_equal(a.piecewise, b.piecewise) and a.regime_history == b.regime_history


def test_exact_all_formulations_agree_with_occbin(exact):
    S = _shocks(exact)
    ob = Occbin(exact)
    ref = ob.simulate(S, periods=300)
    for f in ("auto", "lcp", "regime", "hybrid"):
        r = ob.simulate(S, periods=300, obc_method="news", obc_formulation=f)
        assert np.abs(r.piecewise - ref.piecewise).max() < 1e-8, f


def test_approx_regime_is_occbin_lcp_is_not(approx):
    S = _shocks(approx, scale=1.0)
    ob = Occbin(approx)
    ref = ob.simulate(S, periods=300)
    assert sum(1 for h in ref.regime_history if any(c[0][0] for c in h)) > 3          # the constraint binds in the sample
    reg = ob.simulate(S, periods=300, obc_method="news", obc_formulation="regime")
    assert np.abs(reg.piecewise - ref.piecewise).max() < 1e-8
    hyb = ob.simulate(S, periods=300, obc_method="news", obc_formulation="hybrid")
    assert np.abs(hyb.piecewise - ref.piecewise).max() < 1e-2                          # multiplicity can move a few periods (see the docs)
    assert hyb.news["formulation"] == "hybrid" and reg.news["formulation"] == "regime"
    lcp = ob.simulate(S, periods=300, obc_method="news", obc_formulation="lcp")
    assert np.isfinite(lcp.piecewise).all()


def test_invalid_formulation(exact):
    ob = Occbin(exact)
    with pytest.raises(ValueError, match="obc_formulation"):
        ob.simulate(_shocks(exact, 20), periods=20, obc_method="news", obc_formulation="bogus")


def test_estimation_formulation(exact, approx):
    import models  # soak/approx/models.py: builds the estimation object the same way
    e0 = models.get("zlbm_0")
    e1 = models.get("zlbm_1e-2")
    th0, th1 = e0.x0v.copy(), e1.x0v.copy()
    for e in (e0, e1):
        e.obc_method = "news"
    # exact class: the default 'auto' is the LCP, bit for bit
    e0.obc_formulation = "auto"; a = e0.log_posterior(th0)
    e0.obc_formulation = "lcp"; b = e0.log_posterior(th0)
    assert a == b
    # approx class: the regime formulation reproduces OccBin's inversion filter
    e1.obc_method = "occbin"; ref = e1.log_posterior(th1)
    e1.obc_method = "news"
    e1.obc_formulation = "regime"; r = e1.log_posterior(th1)
    assert abs(r - ref) < 1e-6 * max(1, abs(ref))
    e1.obc_formulation = "hybrid"; h = e1.log_posterior(th1)
    assert np.isfinite(h)
    e1.obc_formulation = "bogus"
    with pytest.raises(ValueError):
        e1.log_posterior(th1)
    e1.obc_formulation = "auto"
