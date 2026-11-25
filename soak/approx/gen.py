"""Approx-class test models for the formulation study (docs/OBC_ESTIMATION.md, 'Approx-class constraints').  Built from the exact test models
zlb / zlb2 / nk_occ by adding a multiplicative policy shock m to the rate equation of both regimes (as NKM does: inom = inomlb*exp(mp), reference
inom = inomnot*exp(mp)), while the bind/relax condition is written on the notional rate alone ('bind rn < rbar').  The two linearised versions
differ by (rn_ss - rbar) * km * m, which the condition lacks: relative discrepancy d = gap * km (checked with constraint_objects).  The standardised policy shock
m has a fixed sd (SIGM), so the mismatch in levels, gap*km*sigm = d*sigm, scales with the discrepancy d."""
import sys, tempfile, warnings
from pathlib import Path
sys.path.insert(0, "soak")
import numpy as np, lib, pydyn
from pydyn.occbin import Occbin
from pydyn.obc import constraint_objects

TAIL = """
varobs {obs};
estimated_params;
phi,,,,NORMAL_PDF,1.5,0.25;
rho,,,,BETA_PDF,0.7,0.1;
sige,,,,INV_GAMMA_PDF,{sige},{sige};
sigm,,,,INV_GAMMA_PDF,{sigm},{sigm2};
end;
occbin_setup(likelihood_inversion_filter);
estimation(datafile=dummy, mode_compute=4, nobs=150, first_obs=1, mh_replic=0, plot_priors=0, nograph);
"""
GAP_Z, GAP_N = 0.008, 0.008        # steady-state gaps rn_ss - bound
SIGM = {"zlbm": 0.05, "zlb2m": 0.05, "nkocc": 0.003}     # sd of the standardised policy shock m (fixed: the physical mismatch gap*km*sigm then scales with d)


def zlbm(d, two=False):
    additive = d == 0                      # exact control: the policy shock enters additively (r = rn + km*m), discrepancy 0
    km = 0.0015 if additive else d / GAP_Z
    sigm = 1.0 if additive else SIGM["zlb2m" if two else "zlbm"]
    s = "var y pi r rn a m" + (" s sn" if two else "") + ";\nvarexo e em;\nparameters beta kappa phi rho rbar rss sige sigm km" + (" sbar" if two else "") + ";\n"
    s += f"beta=0.99; kappa=0.1; phi=1.5; rho=0.7; rbar=0.002; rss=0.01; sige=0.003; sigm={sigm}; km={km};" + (" sbar=-0.006;" if two else "") + "\n"
    s += "model;\ny = y(+1) - (r - rss - pi(+1))" + (" + 0.3*s" if two else "") + " + a;\npi = beta*pi(+1) + kappa*y;\nrn = rss + phi*pi" + (" + 0.2*y" if two else "") + ";\n"
    s += "[name='zlb', relax='zlb']\nr = rn" + ("+ km*m" if additive else "*(1 + km*m)") + ";\n[name='zlb', bind='zlb']\nr = rbar" + ("+ km*m" if additive else "*(1 + km*m)") + ";\n"
    if two:
        s += "sn = 0.5*y;\n[name='floor', relax='floor']\ns = sn;\n[name='floor', bind='floor']\ns = sbar;\n"
    s += "a = rho*a(-1) + sige*e;\nm = sigm*em;\nend;\noccbin_constraints;\nname 'zlb'; bind rn < rbar; relax rn >= rbar;\n"
    if two:
        s += "name 'floor'; bind sn < sbar;\n"
    s += "end;\nshocks; var e = 1; var em = 1; end;\nsteady_state_model;\n    y = 0; pi = 0; rn = rss; r = rss; a = 0; m = 0;" + (" s = 0; sn = 0;" if two else "") + "\nend;\nsteady; check;\n"
    return s + TAIL.format(obs="pi r", sige=0.003, sigm=sigm, sigm2=sigm / 2)


def nkocc(d):
    km = d / GAP_N; sigm = SIGM["nkocc"]
    s = f"""var c pi R Rn a m;
varexo e em;
parameters beta kappa phi phiy rho Rlb sige sigm km;
beta = 0.995; kappa = 0.1; phi = 1.5; phiy = 0.25; rho = 0.8; Rlb = 0.997; sige = 0.004; sigm = {sigm}; km = {km};
model;
    1 / c = beta * exp( a ) * R / ( c(+1) * pi(+1) );
    ( pi - 1 ) * pi = beta * ( pi(+1) - 1 ) * pi(+1) + kappa * ( c - 1 );
    Rn = ( 1 / beta ) * pi ^ phi * c ^ phiy;
    [name='zlb', relax='zlb']
    R = Rn * exp(km*m);
    [name='zlb', bind='zlb']
    R = Rlb * exp(km*m);
    a = rho * a(-1) + sige * e;
    m = sigm * em;
end;
occbin_constraints;
name 'zlb'; bind Rn < Rlb; relax Rn >= Rlb;
end;
steady_state_model;
    c = 1; pi = 1; Rn = 1 / beta; R = 1 / beta; a = 0; m = 0;
end;
shocks; var e = 1; var em = 1; end;
steady; check;
"""
    return s + TAIL.format(obs="pi R", sige=0.004, sigm=sigm, sigm2=sigm / 2)


FAMILIES = {"zlbm": lambda d: zlbm(d), "zlb2m": lambda d: zlbm(d, two=True), "nkocc": nkocc}


def build(name, text):
    d = Path(tempfile.mkdtemp(prefix="apx_")); p = d / f"{name}.mod"; p.write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(p))


if __name__ == "__main__":
    for fam, f in FAMILIES.items():
        for dd in (0.001, 0.01, 0.05):
            try:
                m = build(fam, f(dd)); ob = Occbin(m); cs = constraint_objects(ob)
                print(fam, dd, [(c.cls, round(c.approx, 5)) for c in cs])
            except Exception as e: print(fam, dd, type(e).__name__, str(e)[:150])
