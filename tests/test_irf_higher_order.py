"""Impulse responses at orders 2 and 3 against Dynare 7.2 (reference/r_irf_order3.m).

Two kinds of reference: (a) the algorithm of irf.m written out in Octave with saved draws, and
(b) Dynare's own irf.m run with a fixed seed, whose random draws are regenerated and saved, so
that the port of the routine itself is checked and not only its building blocks."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn

REF = Path(os.environ.get("PYDYN_REF", str(Path(__file__).resolve().parent.parent / "reference")))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    if os.environ.get("PYDYN_REQUIRE_KORDER", "0") != "1":
        try:
            pydyn.korder.load_library()
        except FileNotFoundError:
            pytest.skip("libpydyn_korder.so not built")
    work = tmp_path_factory.mktemp("o3irf")
    shutil.copy(REF / "collard_o3.mod", work)
    m = pydyn.Model(pydyn.run_preprocessor(work / "collard_o3.mod"))
    r = loadmat(REF / "clean_irf_order3.mat", squeeze_me=True)
    dr = pydyn.k_order(m, order=3, ys=r["ys"].astype(float), params=r["params"].astype(float))
    return m, dr, r


@pytest.mark.parametrize("key,pruning", [("irf3_unpruned", False), ("irf3_pruned", True)])
def test_order3_irf_with_saved_draws(setup, key, pruning):
    m, dr, r = setup
    chol = np.linalg.cholesky(m.Sigma_e)
    for i in range(m.n_exo):
        y = pydyn.irf(dr, chol[:, i], m.Sigma_e, periods=int(r["long"]), order=3, pruning=pruning,
                      drop=int(r["drop"]), draws=r["draws"])
        np.testing.assert_allclose(y, r[key][:, :, i].T, rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("key,order,pruning,shock", [("irf_dynare", 3, True, 0), ("irf_dynare_order2", 2, True, 0),
                                                     ("irf_dynare_unpruned", 3, False, 1)])
def test_against_dynares_own_irf_routine(setup, key, order, pruning, shock):
    m, dr, r = setup
    chol = np.linalg.cholesky(m.Sigma_e)
    y = pydyn.irf(dr, chol[:, shock], m.Sigma_e, periods=int(r["long"]), order=order, pruning=pruning,
                  drop=30, draws=r["dynare_draws"])
    assert np.max(np.abs(r[key])) > 1e-4
    np.testing.assert_allclose(y, np.asarray(r[key]).T, rtol=1e-9, atol=1e-13)
