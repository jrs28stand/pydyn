# This file is part of pydyn. Copyright (C) 2026; GPL-3.0-or-later (see LICENSE).
"""Correlated measurement errors (shocks-block ``corr``/``covar`` of observed variables) against Dynare 7.2 / Octave 8.4.

reference/me_cov/*.mod are fs2000 runs with a calibrated measurement-error correlation (gy_obs, gp_obs: 0.5) and
standard deviations on both observables; res_*.mat hold what Octave saved (likelihood, M_.H, M_.Correlation_matrix_ME,
smoothed/filtered/updated series).  The three Kalman variants (kalman_algo 1: multivariate, 2: univariate, 3: diffuse
multivariate), lik_init=2 and calib_smoother are compared; the diffuse univariate smoother (kalman_algo=4) is refused.
Differences are rounding (observed 2.5e-12 absolute on a likelihood of 7e2)."""
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn

REF = Path(__file__).resolve().parent.parent / "reference" / "me_cov"
CASES = [("me_alg1", "res_alg1.mat"), ("me_alg2", "res_alg2.mat"), ("me_alg3", "res_alg3.mat"),
         ("me_cal1", "res_cal1.mat"), ("me_cal3", "res_cal3.mat"), ("me_d3", "res_me_d3.mat"),
         ("me_lik2", "res_me_lik2.mat")]


def _run(tmp_path, name):
    for f in REF.iterdir():
        if f.suffix in (".mod", ".inc", ".m"):
            shutil.copy(f, tmp_path / f.name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pydyn.dynare(tmp_path / f"{name}.mod")


@pytest.mark.parametrize("name,ref", CASES)
def test_measurement_error_covariance_matches_octave(tmp_path, name, ref):
    _run(tmp_path, name)
    got = loadmat(tmp_path / ref)["rr"].ravel()
    want = loadmat(REF / ref)["rr"].ravel()
    assert got.shape == want.shape
    assert np.max(np.abs(got - want)) < 2e-10
    if name == "me_alg1":      # rr = [lik; H(:); ...]: corr 0.5 declared -> H12 = 0.5*sd1*sd2
        assert abs(got[2] - 0.5 * 0.05 * 0.03) < 1e-18


def test_diffuse_univariate_smoother_with_correlated_errors_is_refused(tmp_path):
    with pytest.raises(pydyn.ResultsError, match="kalman_algo=4"):
        _run(tmp_path, "me_alg4")
