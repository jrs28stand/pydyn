# This file is part of pydyn. GPL-3.0-or-later (see LICENSE).
"""Options that pydyn used to refuse and now implements; the numbers are from Dynare 7.2 under Octave (see the comments)."""
import numpy as np
import pytest

import pydyn

RBC = """var y c k a; varexo e u;
parameters alpha beta delta rho;
alpha=0.36;beta=0.99;delta=0.025;rho=0.9;
model;
c^(-1)=beta*c(+1)^(-1)*(alpha*a(+1)*k^(alpha-1)+1-delta);
k=a*k(-1)^alpha+(1-delta)*k(-1)-c;
y=a*k(-1)^alpha;
log(a)=rho*log(a(-1))+e+u;
end;
steady_state_model; a=1; k=(alpha*beta/(1-beta*(1-delta)))^(1/(1-alpha)); y=k^alpha; c=y-delta*k; end;
steady;
shocks; var e; stderr 0.01; var u; stderr 0.007; end;
"""


def test_simul_replic(tmp_path):
    """simult.m: replications drawn one after the other from the stream, oo_.endo_simul is the first, oo_.exo_simul the last,
    <name>/Output/<name>_simul holds all of them as float64; the option stays in options_ for the next stoch_simul
    (Octave: sums below, file of 3 x 100 x 4 values written by the second stoch_simul)."""
    (tmp_path / "sr.mod").write_text(RBC + "stoch_simul(order=1,periods=200,drop=20,irf=0,nograph,noprint,simul_replic=3);\n"
                                     "e1=oo_.exo_simul; s1=oo_.endo_simul;\n"
                                     "stoch_simul(order=2,periods=100,drop=10,irf=0,nograph,noprint);\n")
    r = pydyn.dynare(tmp_path / "sr.mod", nosave=True)
    assert float(np.sum(r.vars["e1"])) == pytest.approx(-0.06563940739826754, abs=1e-12)
    assert float(np.sum(r.vars["s1"])) == pytest.approx(9114.162245673795, rel=1e-11)
    assert float(np.sum(r.oo_["endo_simul"])) == pytest.approx(4532.508334242881, rel=1e-11)
    f = np.fromfile(tmp_path / "sr" / "Output" / "sr_simul")
    assert f.size == 1200 and f.sum() == pytest.approx(13780.917830513608, rel=1e-11)
    assert f[-400:].sum() == pytest.approx(4579.720900930178, rel=1e-11)


def test_qz_zero_threshold():
    """Octave (build-octave mjdgges.mex) on E=diag(1.8e-8,1), D=diag(1e-8,2): info -30 (0/0) at threshold 1e-6, 0 at 1e-12;
    the same pencil as a 1-variable model with a 1e-8 scaled forward equation."""
    from pydyn.perturbation import BlanchardKahnError, solve_linear_re
    fm, f0, fp, fu = np.zeros((1, 1)), np.array([[-2e-8]]), np.array([[1e-8]]), np.array([[1.0]])
    with pytest.raises(BlanchardKahnError, match="0/0"):
        solve_linear_re(fm, f0, fp, fu)
    G, H, eig = solve_linear_re(fm, f0, fp, fu, qz_zero_threshold=1e-12)
    assert abs(eig[0]) == pytest.approx(2.0)


@pytest.mark.parametrize("extra,total,col5", [("", 140.34628975308527, 10.102195794639695),
                                              (",hp_filter=1600", 2.4612677006488286, 2.0552798681879868e-05),
                                              (",bandpass_filter", 1.896156861944084, 0.0)])
def test_spectral_density(tmp_path, extra, total, col5):
    """UnivariateSpectralDensity.m; sums of oo_.SpectralDensity.density (2 x 512) from Dynare 7.2 under Octave."""
    (tmp_path / "sd.mod").write_text(RBC + f"stoch_simul(order=1,irf=0,nograph,noprint,spectral_density{extra}) y k;\n")
    r = pydyn.dynare(tmp_path / "sd.mod", nosave=True)
    d = np.asarray(r.oo_["SpectralDensity"]["density"])
    assert d.shape == (2, 512) and np.asarray(r.oo_["SpectralDensity"]["freqs"])[-1, 0] == pytest.approx(np.pi, abs=0)
    assert d.sum() == pytest.approx(total, rel=1e-10) and d[:, 5].sum() == pytest.approx(col5, rel=1e-8, abs=1e-12)


NUMSS = """var y c k a; varexo e u;
parameters alpha beta delta rho;
alpha=0.36;beta=0.99;delta=0.025;rho=0.9;
model;
c^(-1)=beta*c(+1)^(-1)*(alpha*a(+1)*k^(alpha-1)+1-delta);
k=a*k(-1)^alpha+(1-delta)*k(-1)-c;
y=a*k(-1)^alpha;
log(a)=rho*log(a(-1))+e+u;
end;
initval; k=30;c=2.5;a=1;y=3;e=0;u=0;end;
shocks; var e; stderr 0.01; var u; stderr 0.007; end;
"""


@pytest.mark.parametrize("pre", ["steady;\n", ""])
def test_decision_rule_steady_state_is_the_live_one(tmp_path, pre):
    """resol.m evaluate_steady_state starts at oo_.steady_state (initval without `steady`) and dynare_solve returns a point that
    satisfies tolf unchanged: Dynare 7.2 (Octave) has oo_.dr.ys = [3.70405173 2.7543283 37.98905167 1] with and without
    `steady` (k 2e-4 from the exact 37.98925354), equal to oo_.steady_state after `steady`; pydyn used to polish it to
    machine precision in stoch_simul."""
    (tmp_path / "nss.mod").write_text(NUMSS + pre + "stoch_simul(order=1,irf=0,nograph,noprint);\n")
    r = pydyn.dynare(tmp_path / "nss.mod", nosave=True)
    ys = np.ravel(r.oo_["dr"]["ys"])
    assert ys == pytest.approx([3.70405173, 2.7543283, 37.98905167, 1.0], abs=1e-8)
    if pre:
        assert np.ravel(r.oo_["steady_state"]) == pytest.approx(ys, abs=1e-14)
