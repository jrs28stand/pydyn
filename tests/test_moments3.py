"""Third-order pruned-state-space theoretical moments against Dynare 7.2 (reference/mom3_*.mat, produced by
reference/r_moments3.m from reference/mom3_*.mod, which are reference/mom_*.mod with order=3)."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn import moments

ROOT = Path(__file__).resolve().parent.parent
REF = Path(os.environ.get("PYDYN_REF", str(ROOT / "reference")))
MODELS = ["mom3_collard", "mom3_aux", "mom3_schorfheide"]
SUBSET = {"mom3_collard": ["c", "y", "a"], "mom3_aux": ["z", "c", "AUX_ENDO_LAG_2_1", "AUX_EXO_LAG_6_1"],
          "mom3_schorfheide": ["y", "R", "gy_obs", "AUX_ENDO_LEAD_65"]}
CASES = {"th3p": (dict(order=3, pruning=True, ar=3), False),
         "th3p_sub": (dict(order=3, pruning=True, ar=2, steps=[2, 5]), True)}

try:
    pydyn.korder.load_library()
    HAVE_LIB = True
except FileNotFoundError:
    HAVE_LIB = False
pytestmark = pytest.mark.skipif(not HAVE_LIB, reason="libpydyn_korder.so not built")


class Case:
    def __init__(self, name, tmp):
        shutil.copy(REF / f"{name}.mod", tmp)
        self.m = pydyn.Model(pydyn.run_preprocessor(Path(tmp) / f"{name}.mod"))
        self.r = loadmat(REF / f"{name}.mat", squeeze_me=True)
        self.dr = moments.solve(self.m, 3)

    def dynare_dr(self):
        d = {k[3:]: self.r[k] for k in self.r if k.startswith("dr_")}
        for k in ("ghx", "ghu", "ghxx", "ghxu", "ghuu", "ghxxx", "ghxxu", "ghxuu", "ghuuu", "ghxss", "ghuss"):
            d[k] = np.atleast_2d(d[k])
        d["ghs2"] = np.atleast_1d(d["ghs2"]).astype(float)
        d["order_var"] = d["order_var"].astype(int) - 1
        d["state_var"] = self.dr["state_var"]
        return d


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    return {n: Case(n, tmp_path_factory.mktemp(n)) for n in MODELS}


def close(a, b, what, rtol=1e-9, atol=1e-11):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol * np.max(np.abs(b)), err_msg=what)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("rules", ["pydyn", "dynare"])
def test_third_order_pruned_moments_against_dynare(cases, name, case, rules):
    cs = cases[name]
    kw, sub = CASES[case]
    dr = cs.dr if rules == "pydyn" else cs.dynare_dr()
    mo = moments.theoretical_moments(cs.m, dr, SUBSET[name] if sub else None, **kw)
    r = cs.r
    assert list(mo.var_index) == list(np.atleast_1d(r[f"{case}_ivar"]) - 1)
    assert mo.kind == "pruned" and mo.order == 3
    close(mo.mean, r[f"{case}_mean"], "mean")
    close(mo.var, r[f"{case}_var"], "var")
    close(mo.corr, r[f"{case}_contemporaneous_correlation"], "corr")
    close(np.stack(mo.autocorr, axis=2), r[f"{case}_autocorr"], "autocorr")
    # Dynare computes no (conditional) variance decomposition at third order
    assert f"{case}_variance_decomposition" not in r and mo.variance_decomposition is None
    assert f"{case}_conditional_variance_decomposition" not in r and mo.conditional_variance_decomposition is None


def test_decision_rules_match_dynare(cases):
    for cs in cases.values():
        d = cs.dynare_dr()
        for k in ("ghxxx", "ghxxu", "ghxuu", "ghuuu", "ghxss", "ghuss"):
            close(np.atleast_2d(cs.dr[k]), d[k], k, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("name", MODELS)
def test_third_order_text_output(cases, name):
    cs = cases[name]
    mo = moments.stoch_simul_moments(cs.m, None, cs.dr, **CASES["th3p"][0])
    mine = [ln.rstrip() for ln in mo.format().splitlines() if ln.strip()]
    theirs = [ln.rstrip() for ln in str(cs.r["th3p_text"]).splitlines() if ln.strip()]
    if mine != theirs:      # signed numerical zeros, see tests/test_moments.py
        mine, theirs = ([[x.lstrip("-") if set(x) <= set("-0.") else x for x in ln.split()] for ln in side]
                        for side in (mine, theirs))
    assert mine == theirs


def test_third_order_without_pruning_is_refused(cases):
    cs = cases["mom3_collard"]
    assert not cs.r["th3_has_mean"]        # Dynare prints and stores nothing in this case
    with pytest.raises(NotImplementedError, match="without pruning"):
        moments.theoretical_moments(cs.m, cs.dr, order=3, pruning=False)
    with pytest.raises(NotImplementedError, match="filtering at order 3"):
        moments.theoretical_moments(cs.m, cs.dr, order=3, pruning=True, hp_filter=1600)
