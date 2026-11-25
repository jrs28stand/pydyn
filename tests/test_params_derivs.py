"""pydyn.params_derivs (get_perturbation_params_derivs.m, order 1) against Dynare 7.2 under Octave 8.4 (reference/pd/*_derivs.mat,
made by calling identification.get_perturbation_params_derivs with analytic_derivation_mode=0 on the calibrated model: all model
parameters, the standard deviation of every shock and every non-zero correlation)."""
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import pydyn
from pydyn.params_derivs import ParamsDerivsError, perturbation_params_derivs

REF = Path(__file__).resolve().parent.parent / "reference" / "pd"
CASES = ["BrockMirman", "fs2000_corr", "forward_looking_varobs_x", "as2007_order3_no_stoch_simul"]


@pytest.mark.parametrize("case", CASES)
def test_against_octave(case, tmp_path):
    for f in REF.glob("*.inc"):
        shutil.copy(f, tmp_path / f.name)
    shutil.copy(REF / f"{case}.mod", tmp_path / f"{case}.mod")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(tmp_path / f"{case}.mod"))
        R = loadmat(REF / f"{case}_derivs.mat", struct_as_record=False, squeeze_me=True)
        ic = np.atleast_2d(R["indpcorr"]).astype(int) - 1
        if ic.size == 0:
            ic = np.zeros((0, 2), int)
        d = perturbation_params_derivs(m, np.arange(len(m.param_names)), np.arange(m.n_exo), ic)
    ref = R["DERIVS"]
    for k in ("dYss", "dSigma_e", "dCorrelation_matrix", "dghx", "dghu", "dOm"):
        a = getattr(ref, k)
        a = a.toarray() if hasattr(a, "toarray") else np.asarray(a, float)
        b = d[k]
        a = a.reshape(b.shape)
        scale = max(1.0, np.max(np.abs(a))) if a.size else 1.0
        assert np.max(np.abs(a - b), initial=0.0) <= 1e-10 * scale, k


def test_unsupported_operator_is_refused(tmp_path):
    (tmp_path / "m.mod").write_text("var y; varexo e; parameters a; a = 0.5;\nmodel; y = a*y(-1) + abs(e); end;\n"
                                    "initval; y = 0; e = 0; end; shocks; var e; stderr 0.1; end; steady;\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "m.mod"))
        with pytest.raises(ParamsDerivsError):
            perturbation_params_derivs(m, [0], [0])


def test_orders_above_one_are_refused(tmp_path):
    (tmp_path / "m.mod").write_text("var y; varexo e; parameters a; a = 0.5;\nmodel; y = a*y(-1) + e; end;\n"
                                    "initval; y = 0; e = 0; end; shocks; var e; stderr 0.1; end; steady;\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "m.mod"))
        with pytest.raises(NotImplementedError):
            perturbation_params_derivs(m, [0], [0], order=2)


def test_osr_analytic_gradient_equals_central_differences(tmp_path):
    from pydyn import optimal_policy as op
    (tmp_path / "m.mod").write_text("""var y inflation r; varexo y_ inf_; parameters delta sigma alpha kappa gx gc gy gi;
delta = 0.44; kappa = 0.18; alpha = 0.48; sigma = -0.06; gx = 0.3; gc = 1.5; gy = 8; gi = 3;
model(linear);
y = delta*y(-1) + (1-delta)*y(+1) + sigma*(r - inflation(+1)) + y_;
inflation = alpha*inflation(-1) + (1-alpha)*inflation(+1) + kappa*y + inf_;
r = gx*y(-1) + gc*inflation(-1) + gy*y_ + gi*inf_;
end;
shocks; var y_; stderr 0.63; var inf_; stderr 0.4; end;
optim_weights; inflation 1; y 1; y,inflation 0.5; end;
osr_params gx gc gy gi;
""")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pydyn.Model(pydyn.run_preprocessor(tmp_path / "m.mod"))
        pr = op._OSRProblem(m, ["gx", "gc", "gy", "gi"], 1, None)
        for x in (np.array([0.3, 1.5, 8.0, 3.0]), np.array([0.5, 1.2, 5.0, 2.0])):
            g, bad = pr.gradient(x)
            fd = [(pr(x + 1e-6 * np.eye(4)[i])[0] - pr(x - 1e-6 * np.eye(4)[i])[0]) / 2e-6 for i in range(4)]
            assert not bad
            np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-8)
