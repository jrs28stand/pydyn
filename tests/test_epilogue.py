"""epilogue block against Dynare 7.2: reference/epilogue/epi_example.mod (Dynare's tests/epilogue/example1.mod
with more definitions); reference/epilogue/epi_example.mat from reference/epilogue/r_epilogue.m: the generated
epilogue_dynamic.m applied to a simulated dseries (with NaNs at the start of h and a user-supplied start of x)
and epilogue_static.m on the steady state."""
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn.epilogue import Epilogue, parse

REF = Path(__file__).resolve().parent.parent / "reference" / "epilogue"


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    d = tmp_path_factory.mktemp("epi")
    shutil.copy(REF / "epi_example.mod", d)
    m = pydyn.Model(pydyn.run_preprocessor(d / "epi_example.mod"))
    r = loadmat(REF / "epi_example.mat", squeeze_me=True)
    S = {v: r["endo_in"][:, i] for i, v in enumerate(m.endo_names)}
    S.update({v: r["exo_in"][:, i] for i, v in enumerate(m.exo_names)})
    x = np.full(len(r["endo_in"]), np.nan)
    x[:7] = r["x_in"]
    S["x"] = x
    return m, r, S


@pytest.mark.parametrize("rounding", [True, False])
def test_dynamic_epilogue_matches_dynare(case, rounding):
    """With dynare_rounding the series are identical (Dynare's dseries 'from' loop writes each epilogue
    variable back through sprintf('%f'), i.e. rounded to 6 decimals, and the rounded values feed the next
    definitions); without it pydyn keeps full precision and differs from Dynare by that rounding only."""
    m, r, S = case
    ep = Epilogue(m)
    assert ep.names == str(r["names"]).split(",")
    out = ep.dynamic(S, r["params"], dynare_rounding=rounding)
    for n in ep.names:
        a, b = out[n], r[f"dyn_{n}"]
        assert np.array_equal(np.isnan(a), np.isnan(b)), n
        if rounding:
            np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
        elif (~np.isnan(b)).any():
            assert np.nanmax(np.abs(a - b)) < 1e-6, n
    assert np.all(np.isnan(out["z"]))           # z(-1) is never observed: Dynare leaves z NaN
    assert np.all(np.isnan(out["v"]))           # a lead: the loop would run past the sample
    if not rounding:                            # given values before the first computed period (Dynare rounds them too)
        np.testing.assert_array_equal(out["x"][:5], S["x"][:5])


def test_static_epilogue_matches_dynare(case):
    m, r, _ = case
    ep = Epilogue(m)
    ss = {v: np.full(3, r["steady_state"][i]) for i, v in enumerate(m.endo_names)}
    ss.update({v: np.zeros(3) for v in m.exo_names})
    ss.update(x=np.array([1.0, 2, 3]), z=np.array([0.1, 0.2, 0.3]))
    st = ep.static(ss, r["params"])
    for n in ep.names:
        np.testing.assert_allclose(st[n], r[f"sta_{n}"], rtol=1e-15, atol=1e-15)


def test_parse_and_paths(case):
    m, r, S = case
    eq = parse("var y; model; y=1; end;\nepilogue; // comment\n a = y(-1)/2; /* block */ b = a + 1;\nend;\n")
    assert eq == [("a", "y(-1)/2"), ("b", "a + 1")]
    ep = Epilogue(m)
    endo = np.column_stack([S[v] for v in m.endo_names])
    out = ep.from_paths(endo, params=r["params"])
    np.testing.assert_array_equal(out["g"], ep.dynamic(S, r["params"])["g"])
