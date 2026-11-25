# This file is part of pydyn. GPL-3.0-or-later (see LICENSE).
"""Options that pydyn used to refuse and now implements; the numbers are from Dynare 7.2 under Octave (see the comments)."""
import shutil
from pathlib import Path

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


def test_selected_variables_only(tmp_path):
    """store_smoother_results.m: with selected_variables_only the smoother results hold the listed variables only (the
    constants also those of the observed ones).  Octave sums of the smoothed, updated, filtered and one-/three-step-ahead
    results of est_conv at its posterior mode, and the field lists below."""
    ref = Path(__file__).resolve().parent.parent / "reference" / "oo"
    lines = (ref / "est_conv.mod").read_text().split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("estimation("))
    est = ("estimation(order=1,datafile=fsdat_simul,nobs=150,first_obs=3,loglinear,mode_compute=0,mode_file=est_conv_mode,"
           "mh_replic=0,silent_optimizer,nograph,smoother,filtered_vars,filter_step_ahead=[1,3],selected_variables_only) gp_obs y m;\n")
    for f in ("fsdat_simul.m", "est_conv_mode.mat"):
        shutil.copy(ref / f, tmp_path)
    (tmp_path / "t.mod").write_text("\n".join(lines[:i]) + "\n" + est)
    oo = pydyn.dynare(tmp_path / "t.mod", nosave=True).oo_
    tot = lambda k: sum(float(np.sum(v)) for v in oo[k].values())
    assert sorted(oo["SmoothedVariables"].keys()) == ["gp_obs", "m", "y"]
    assert sorted(oo["Smoother"]["Constant"].keys()) == ["gp_obs", "gy_obs", "m", "y"]
    assert tot("SmoothedVariables") == pytest.approx(58.4305374365, rel=1e-8)
    assert tot("UpdatedVariables") == pytest.approx(58.4813842537, rel=1e-8)
    assert tot("FilteredVariables") == pytest.approx(58.3419463987, rel=1e-8)
    fk = np.asarray(oo["FilteredVariablesKStepAhead"])
    assert fk.shape == (2, 3, 153)
    assert fk[:, 0, :].sum() == pytest.approx(3.31407962041, rel=1e-8)
    assert fk[:, 1, :].sum() == pytest.approx(2.97211311556, rel=1e-8)


@pytest.mark.parametrize("extra,laplace", [("", 909.219367804493), (",fast_kalman_filter", 909.219367821956)])
def test_fast_kalman_filter(tmp_path, extra, laplace):
    """kalman_filter_fast.m (Chandrasekhar recursions): Dynare 7.2 under Octave gives the Laplace approximation of est_conv
    at its posterior mode as below; the fast filter differs from the standard one by 1.7e-8 (no steady-state switch)."""
    ref = Path(__file__).resolve().parent.parent / "reference" / "oo"
    lines = (ref / "est_conv.mod").read_text().split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("estimation("))
    est = ("estimation(order=1,datafile=fsdat_simul,nobs=150,first_obs=3,loglinear,mode_compute=0,mode_file=est_conv_mode,"
           f"mh_replic=0,silent_optimizer,nograph,smoother{extra}) gp_obs;\n")
    for f in ("fsdat_simul.m", "est_conv_mode.mat"):
        shutil.copy(ref / f, tmp_path)
    (tmp_path / "t.mod").write_text("\n".join(lines[:i]) + "\n" + est)
    oo = pydyn.dynare(tmp_path / "t.mod", nosave=True).oo_
    assert float(oo["MarginalDensity"]["LaplaceApproximation"]) == pytest.approx(laplace, abs=1e-9)


def test_conditional_likelihood(tmp_path):
    """dsge_conditional_likelihood_1.m: Octave (Dynare 7.2) on fs2000, mode_compute=4 with conditional_likelihood gives the mode in
    reference/oo/cond_lik_mode.mat; its Laplace approximation at that mode is 1172.75469599273 (the exact-likelihood objective
    gives another value)."""
    ref = Path(__file__).resolve().parent.parent / "reference" / "oo"
    for f, g in (("fsdat_simul.m", "fsdat_simul.m"), ("cond_lik_mode.mat", "cond_lik_mode.mat")):
        shutil.copy(ref / f, tmp_path / g)
    est = ("estimation(conditional_likelihood,silent_optimizer,order=1,datafile=fsdat_simul,nobs=192,mode_compute=0,"
           "mode_file=cond_lik_mode,loglinear,mh_replic=0,nograph);\n")
    (tmp_path / "t.mod").write_text((ref / "cond_lik_base.mod").read_text() + est)
    oo = pydyn.dynare(tmp_path / "t.mod", nosave=True).oo_
    assert float(oo["MarginalDensity"]["LaplaceApproximation"]) == pytest.approx(1172.75469599273, abs=1e-8)


def test_particle_filter_estimation_through_the_results_layer(tmp_path):
    """estimation(order=2, filter_algorithm=sis, ...) uses the nonlinear likelihood (options_.particle.status = true is what the
    preprocessor writes for order >= 2).  Dynare 7.2 under Octave, RBC model of tests/particle, 50 observations, 300 particles,
    no mode search: log density 497.833067201875 (pydyn agrees to ~1e-9 relative: tiny measurement errors magnify rounding),
    oo_.dr = order_var, inv_order_var, restrict_var_list, restrict_columns only."""
    ref = Path(__file__).resolve().parent.parent / "reference" / "oo"
    for f in ("pf_sample.m", "pf_base.inc"):
        shutil.copy(ref / f, tmp_path)
    (tmp_path / "t.mod").write_text('@#include "pf_base.inc"\n\nestimation(order=2,nograph,filter_algorithm=sis,number_of_particles=300,'
                                    "mh_replic=0,mode_compute=8,optim=('MaxIter',15),silent_optimizer,cova_compute=0);\n")
    oo = pydyn.dynare(tmp_path / "t.mod", nosave=True).oo_
    assert float(oo["posterior"]["optimization"]["log_density"]) == pytest.approx(497.833067201875, rel=1e-8)
    assert set(oo["dr"].keys()) == {"order_var", "inv_order_var", "restrict_var_list", "restrict_columns"}


def test_particle_options_with_first_order_status_and_unsupported_filters_are_refused(tmp_path):
    ref = Path(__file__).resolve().parent.parent / "reference" / "oo"
    for f in ("pf_sample.m", "pf_base.inc"):
        shutil.copy(ref / f, tmp_path)
    (tmp_path / "t.mod").write_text('@#include "pf_base.inc"\n\nestimation(order=2,nograph,filter_algorithm=gmf,number_of_particles=300,'
                                    "mh_replic=0,mode_compute=0);\n")
    with pytest.raises(Exception, match="gmf"):
        pydyn.dynare(tmp_path / "t.mod", nosave=True)


def test_dr_restrict_fields_and_fuu(tmp_path):
    """dynare_estimation_init.m: oo_.dr.restrict_var_list = observed + predetermined variables in decision-rule order,
    restrict_columns their places for the states (Octave: [7 8 10 11 12 13] and [3 4 5 6] on est_conv).
    dyn_second_order_solver.m: oo_.dr.fuu at order 2 (Octave on the RBC model with correlated shocks: 1.7157260641558e-06 in the
    Euler equation, zero elsewhere)."""
    ref = Path(__file__).resolve().parent.parent / "reference" / "oo"
    lines = (ref / "est_conv.mod").read_text().split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("estimation("))
    est = ("estimation(order=1,datafile=fsdat_simul,nobs=150,first_obs=3,loglinear,mode_compute=0,mode_file=est_conv_mode,"
           "mh_replic=0,silent_optimizer,nograph,smoother) gp_obs;\n")
    for f in ("fsdat_simul.m", "est_conv_mode.mat"):
        shutil.copy(ref / f, tmp_path)
    (tmp_path / "t.mod").write_text("\n".join(lines[:i]) + "\n" + est)
    dr = pydyn.dynare(tmp_path / "t.mod", nosave=True).oo_["dr"]
    assert np.ravel(dr["restrict_var_list"]).tolist() == [7, 8, 10, 11, 12, 13]
    assert np.ravel(dr["restrict_columns"]).tolist() == [3, 4, 5, 6]
    (tmp_path / "f.mod").write_text(RBC + "shocks; corr e,u=0.3; end;\nstoch_simul(order=2,irf=0,nograph,noprint);\n")
    fuu = np.ravel(pydyn.dynare(tmp_path / "f.mod", nosave=True).oo_["dr"]["fuu"])
    assert fuu[0] == pytest.approx(1.7157260641558e-06, rel=1e-9)
    assert np.all(fuu[1:] == 0)


def test_data_command_dates_of_a_file_without_init(tmp_path):
    """load_m_file_data.m: a .m data file without FREQ__ / INIT__ starts at dates(1,1) = 1Y, so data(first_obs=3Y) is
    observation 3 (Dynare runs the data(first_obs=801Y) of tests/particle on such a file)."""
    ref = Path(__file__).resolve().parent.parent / "reference" / "oo"
    lines = (ref / "est_conv.mod").read_text().split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("estimation("))
    for f in ("fsdat_simul.m", "est_conv_mode.mat"):
        shutil.copy(ref / f, tmp_path)
    est = ("estimation(order=1,loglinear,mode_compute=0,mode_file=est_conv_mode,mh_replic=0,silent_optimizer,nograph) gp_obs;\n")
    base = "\n".join(lines[:i]) + "\n"
    (tmp_path / "a.mod").write_text(base + "data(file=fsdat_simul,first_obs=3Y,nobs=150);\n" + est)
    (tmp_path / "b.mod").write_text(base + "estimation(order=1,datafile=fsdat_simul,first_obs=3,nobs=150,loglinear,mode_compute=0,"
                                    "mode_file=est_conv_mode,mh_replic=0,silent_optimizer,nograph) gp_obs;\n")
    la = pydyn.dynare(tmp_path / "a.mod", nosave=True).oo_["MarginalDensity"]["LaplaceApproximation"]
    lb = pydyn.dynare(tmp_path / "b.mod", nosave=True).oo_["MarginalDensity"]["LaplaceApproximation"]
    assert float(la) == float(lb)
