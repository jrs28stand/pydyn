"""pydyn's news-shock OBC method against the real DynareOBC (run under Octave 8.4 with Dynare 4.5.4; recipe and
reference files in reference/dynareobc_octave). Model: a small NK model with a lower bound on the interest rate,
written with max() for DynareOBC and with an occbin_constraints block for pydyn."""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn.obc_ho import HigherOrderNews

REF = Path(os.environ.get("PYDYN_REF", str(Path(__file__).resolve().parent.parent / "reference"))) / "dynareobc_octave"


@pytest.fixture(scope="module")
def nk(tmp_path_factory):
    d = tmp_path_factory.mktemp("dynareobc")
    shutil.copy(REF / "nk_occ.mod", d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(d / "nk_occ.mod"))
    return m, loadmat(REF / "shocks.mat", squeeze_me=True)["ShockSequence"].reshape(-1, 1)


def _ref(m, name):
    r = loadmat(REF / name, squeeze_me=True)
    names = [str(n).strip() for n in r["names"]]
    return r, r["endo"][[names.index(n) for n in m.endo_names]].T


@pytest.mark.parametrize("order", [1, 2])
def test_simulation_matches_dynareobc(nk, order):
    m, S = nk
    r, E = _ref(m, f"ref_o{order}.mat")
    h = HigherOrderNews(m, order=order, horizon=32, check_ahead=32)
    np.testing.assert_allclose(h.news.Mo[:, :32], r["M"][:h.news.Mo.shape[0]], atol=1e-13)
    np.testing.assert_allclose(h.simulate(S)["with_bound"], E, atol=1e-13)
    assert (E[:, list(m.endo_names).index("R")] < 1 + 1e-9).sum() > 50


def test_orders_differ(nk):
    m, _ = nk
    assert np.abs(_ref(m, "ref_o1.mat")[1] - _ref(m, "ref_o2.mat")[1]).max() > 1e-4


def test_cubature_matches_dynareobc(nk):
    m, S = nk
    r, E = _ref(m, "ref_o2cub.mat")
    assert np.abs(E - _ref(m, "ref_o2.mat")[1]).max() > 1e-5          # cubature changes the answer
    h = HigherOrderNews(m, order=2, horizon=32, check_ahead=32, cubature=True, qmc_level=9, regions=1)
    Rd = np.real(r["Rcc"])
    np.testing.assert_allclose(h.R @ h.R.T, Rd @ Rd.T, atol=1e-16)    # conditional covariance, pruned
    own = h.simulate(S)["with_bound"]
    # Own root and SciPy's Sobol points. The root's column signs are arbitrary and change the 1,023-point
    # estimate by about as much as the cubature effect itself (5e-5), so this is a sanity bound only.
    assert np.abs(own - E).max() < 2e-4
    h.R, h.xi = Rd, loadmat(REF / "sob_o2cub.mat")["x"]               # DynareOBC's own root and points
    np.testing.assert_allclose(h.simulate(S)["with_bound"], E, atol=1e-13)


def test_four_region_cubature_matches_dynareobc(nk):
    """DynareOBC's region splitting (CubatureRegions=4). With its root and points, 294 of 300 periods agree to
    rounding; in the others a near-tie in the split falls differently (Octave's pca/eigs against LAPACK here)."""
    m, S = nk
    r, E = _ref(m, "ref_o2cub4.mat")
    assert np.abs(E - _ref(m, "ref_o2.mat")[1]).max() > 0.05          # several regions matter a great deal
    h = HigherOrderNews(m, order=2, horizon=32, check_ahead=32, cubature=True, qmc_level=9, regions=4)
    h.R, h.xi = np.real(_ref(m, "ref_o2cub.mat")[0]["Rcc"]), loadmat(REF / "sob_o2cub.mat")["x"]
    d = np.abs(h.simulate(S)["with_bound"] - E).max(axis=1)
    assert (d < 1e-12).sum() >= 290 and d.max() < 0.01
