"""The speed-ups of pydyn.occbin.Occbin (memoised rules, single evaluation of the constraint sides, allocation-free
path) leave every result bit for bit as the frozen code of bench/occbin_before.py: paths, regime histories, iteration
counts, the rules of every shock period, diagnostics and errors, on one- and two-constraint models, and the
likelihoods, filtered states and regimes of both filters on the estimation examples."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))
import obc_bench as B  # noqa: E402
import occbin_before as OB  # noqa: E402
import pydyn.occbin_estimation as oe  # noqa: E402
from pydyn.occbin import Occbin  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_occbin_estimation as te  # noqa: E402,F401  (fixtures and helpers of the estimation tests)
from test_occbin_estimation import models  # noqa: E402,F401


def _run(obj, shocks, periods):
    try:
        return obj.simulate(shocks, periods=periods, check_ahead_periods=B.OCCBIN_CHECK_AHEAD), None
    except Exception as err:  # noqa: BLE001 - the frozen module has its own error classes
        if not hasattr(err, "error_flag"):
            raise
        return None, (type(err).__name__, err.error_flag, err.period, str(err))


def _same(a, b):
    if a is None or b is None:
        return a is b
    ok = all(np.array_equal(getattr(a, f), getattr(b, f)) for f in ("piecewise", "linear", "shocks", "ys"))
    ok &= a.regime_history == b.regime_history and a.iterations == b.iterations and a.periodic == b.periodic
    ok &= a.error_flag == b.error_flag and a.exo_pos == b.exo_pos
    ok &= len(a.ss) == len(b.ss) and all(np.array_equal(x, y) for sa, sb in zip(a.ss, b.ss) for x, y in zip(sa, sb))
    ok &= len(a.diagnostics) == len(b.diagnostics) and all(
        all(np.array_equal(da[k], db[k]) or (k == "selected" and da[k] == db[k]) for k in da)
        for da, db in zip(a.diagnostics, b.diagnostics))
    return bool(ok)


@pytest.mark.parametrize("name", ["zlb", "zlb2", "nkm", "sw"])
def test_simulation_identical_to_frozen_code(name):
    m = B.load_model(name)
    cfg = B.MODELS[name]
    sets = [None] + B.perturbation_sets(m, cfg["params"], 5, B.SEED, size=0.02)
    n_bind = 0
    for scale in (1.0, 2.0):                      # twice the scale produces failures on sw: errors must agree too
        shocks = B.draw_shocks(m, 120, cfg["s"] * scale, B.SEED)
        for p in sets:
            new = Occbin(m, p)
            old = OB.clone(new)
            ra, ea = _run(new, shocks, 120)
            rb, eb = _run(old, shocks, 120)
            assert ea == eb
            assert _same(ra, rb)
            n_bind += ra is not None and bool(B.binding_flags(ra).any())
    assert n_bind > 0


def _frozen_class(model, params):
    return OB.clone(Occbin(model, params))


@pytest.mark.parametrize("case,ks,inversion", [("nkm", [0, 1], True), ("nkm", [0, 1], False),
                                                ("rbc", [0], True), ("rbc", [0], False)])
def test_filters_identical_to_frozen_code(models, monkeypatch, case, ks, inversion):    # noqa: F811
    r = te.ref(f"occest_{case}_lik")
    kw = {"occbin": {"likelihood_inversion_filter": True}} if inversion else {}
    est = te.estimation(models[f"occest_{case}"], r, **kw)
    X = te.points(r, est.n)
    got = []
    for frozen in (False, True):
        with monkeypatch.context() as mp:
            if frozen:
                mp.setattr(oe, "Occbin", _frozen_class)
            row = []
            for k in ks:
                lp = est.log_posterior(X[:, k])
                res = est.last_result
                row.append((lp, res.regime_history if inversion else res.regimes,
                            np.asarray(res.stateval if inversion else res.a0)))
            got.append(row)
    for (lp1, rg1, s1), (lp2, rg2, s2) in zip(*got):
        assert lp1 == lp2 and rg1 == rg2 and np.array_equal(s1, s2)
