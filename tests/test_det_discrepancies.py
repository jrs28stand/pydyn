# This file is part of pydyn. GPL-3.0-or-later (see LICENSE).
"""Fixes found while classifying the 'other' refusals: .m data files with statements after the matrices (smoother),
``resol`` under loglinear (dr.ys at the new parameters), and ``forecast`` at order 2 through ``pydyn.dynare()``."""
import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn.smoother import load_data


def test_smoother_data_file_runs_statements_after_the_matrices(tmp_path):
    (tmp_path / "d.m").write_text("a = [1; 2; 4];\nb = [2; 2; 2];\na = log(a);\nb = b * 3;\n")
    got = load_data(str(tmp_path / "d"), ["a", "b"])
    assert np.allclose(got[:, 0], np.log([1, 2, 4])) and np.allclose(got[:, 1], 6.0)


MOD_RESOL = """
var y;
varexo e;
parameters rho ybar;
rho = 0.9; ybar = 2;
model;
log(y) - log(ybar) = rho*(log(y(-1)) - log(ybar)) + e;
end;
steady_state_model;
y = ybar;
end;
shocks;
var e; stderr 0.01;
end;
steady;
stoch_simul(order=1, loglinear, irf=0, noprint);
M_.params(2) = 3;
[dr, info, params] = resol(0, M_, options_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state);
ys_new = dr.ys;
ys_old = oo_.steady_state;
save('out.mat', 'ys_new', 'ys_old');
"""


def test_resol_loglinear_returns_the_steady_state_at_the_new_parameters(tmp_path):
    f = tmp_path / "rl.mod"
    f.write_text(MOD_RESOL)
    cwd = __import__("os").getcwd()
    __import__("os").chdir(tmp_path)
    try:
        pydyn.dynare(str(f))
    finally:
        __import__("os").chdir(cwd)
    d = loadmat(tmp_path / "out.mat")
    assert d["ys_new"].ravel()[0] == pytest.approx(3.0)


MOD_FC = """
var y;
varexo e;
parameters rho;
rho = 0.9;
model;
y = rho*y(-1) + e + 0.5*y(-1)^2;
end;
initval;
y = 0;
end;
steady;
shocks;
var e; stderr 0.05;
end;
stoch_simul(order=2, irf=0, noprint);
options_.forecast_replic = 300;
forecast(periods=6);
m = oo_.forecast.Mean.y; lo = oo_.forecast.HPDinf.y; hi = oo_.forecast.HPDsup.y;
save('fc.mat', 'm', 'lo', 'hi');
"""


def test_forecast_at_order_2_runs_and_replays_the_stream(tmp_path):
    import os
    cwd = os.getcwd()
    out = []
    for tag in ("a", "b"):
        d = tmp_path / tag
        d.mkdir()
        (d / "fc.mod").write_text(MOD_FC)
        os.chdir(d)
        try:
            pydyn.dynare(str(d / "fc.mod"))
        finally:
            os.chdir(cwd)
        out.append(loadmat(d / "fc.mat"))
    for k in ("m", "lo", "hi"):
        assert np.array_equal(out[0][k], out[1][k])         # the same stream each run
    assert np.all(out[0]["hi"] > out[0]["lo"]) and out[0]["hi"].shape[0] == 6
