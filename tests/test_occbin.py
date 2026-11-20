"""Compare the piecewise-linear OccBin simulation with Dynare 7.2 (rbc_occbin example)."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import dynpy

REF = Path(os.environ.get("DYNPY_REF", str(Path(__file__).resolve().parent.parent / "reference")))
EXAMPLES = Path(os.environ.get("DYNPY_EXAMPLES", "/home/claude/dl/src72/examples"))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    work = tmp_path_factory.mktemp("occbin")
    shutil.copy(EXAMPLES / "occbin" / "rbc_occbin.mod", work)
    m = dynpy.Model(dynpy.run_preprocessor(work / "rbc_occbin.mod"))
    ob = dynpy.Occbin(m)
    ref = loadmat(REF / "clean_rbc_occbin.mat", squeeze_me=True)
    opts = m.options["occbin_solver"]
    res = ob.simulate(ob.surprise_shocks(), periods=opts["simul.periods"],
                      check_ahead_periods=opts["simul.check_ahead_periods"])
    return ob, res, ref


def test_shock_sequence(setup):
    _, res, ref = setup
    np.testing.assert_array_equal(res.shocks.ravel(), np.asarray(ref["occ_shocks"]).ravel())


def test_linear_path(setup):
    _, res, ref = setup
    np.testing.assert_allclose(res.linear, ref["occ_linear"], rtol=1e-10, atol=1e-12)


def test_piecewise_path(setup):
    _, res, ref = setup
    assert np.max(np.abs(ref["occ_piecewise"] - ref["occ_linear"])) > 1e-3   # constraints do bind
    print("max |piecewise - Dynare| =", np.max(np.abs(res.piecewise - ref["occ_piecewise"])))
    np.testing.assert_allclose(res.piecewise, ref["occ_piecewise"], rtol=1e-10, atol=1e-12)


def test_regime_history(setup):
    _, res, ref = setup
    rh = ref["occ_regime"]          # rows: regime1, regimestart1, regime2, regimestart2
    assert rh.shape[1] == len(res.regime_history)
    for t, per in enumerate(res.regime_history):
        for j, (regime, start) in enumerate(per):
            assert list(np.atleast_1d(rh[2 * j][t])) == regime
            assert list(np.atleast_1d(rh[2 * j + 1][t])) == start


def test_irf_is_state_dependent(setup):
    ob, _, _ = setup
    # a positive TFP shock raises investment: neither constraint binds, response is linear
    up = ob.irf([0.01], periods=30)
    np.testing.assert_allclose(up, ob.ref.irf([0.01], periods=30), atol=1e-12)
    # a large negative shock hits the investment floor: response departs from the linear one
    down = ob.irf([-0.05], periods=30)
    assert np.max(np.abs(down - ob.ref.irf([-0.05], periods=30))) > 1e-4
