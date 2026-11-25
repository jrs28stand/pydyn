# This file is part of pydyn. GPL-3.0-or-later (see LICENSE).
"""Differential probe for the options audit: run a statement of a small model with and without an option and report
whether pydyn's oo_ changes (``differs``), does not change (``SAME``) or the run stops (``ERR``).

    DYNARE_PREPROCESSOR=... python tools/options_probe.py [cases.json]

``cases.json`` is a list of [base_statements, [variant_statements, ...]]; without it the built-in stoch_simul cases run.
SAME is not an error: an option may legitimately not matter in the configuration probed (``replic`` at order 1, ``drop``
without simulated paths).  What it is for: an option that Dynare acts on and that is SAME in a configuration where it must
matter is silently ignored, and belongs in the ``refused`` set of pydyn/option_policy.py (this is how ``simul_replic`` was found).
"""
import sys,os,shutil,tempfile,warnings,itertools,json
import numpy as np
sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import pydyn
from pydyn.results import Struct
warnings.simplefilter("ignore")
MOD="""var y c k a; varexo e u;
parameters alpha beta delta rho sig;
alpha=0.36;beta=0.99;delta=0.025;rho=0.9;sig=0.01;
model;
c^(-1)=beta*c(+1)^(-1)*(alpha*a(+1)*k^(alpha-1)+1-delta);
k=a*k(-1)^alpha+(1-delta)*k(-1)-c;
y=a*k(-1)^alpha;
log(a)=rho*log(a(-1))+e+u;
end;
initval; k=30;c=2.5;a=1;y=3;e=0;u=0;end;
steady;
SHOCKSLINE
"""
def flat(o,prefix=""):
    out={}
    if isinstance(o,Struct):
        for k in o: out.update(flat(o[k],prefix+"."+k))
    elif isinstance(o,(list,tuple)):
        for i,x in enumerate(o): out.update(flat(x,prefix+f"[{i}]"))
    else:
        try: out[prefix]=np.asarray(o,float)
        except Exception: out[prefix]=str(o)
    return out
def run(stmt, tail=""):
    d=tempfile.mkdtemp()
    mod=MOD.replace("SHOCKSLINE","" if "shocks;" in stmt else "shocks; var e; stderr 0.01; var u; stderr 0.007; end;")
    open(d+"/m.mod","w").write(mod+stmt+"\n"+tail+"\n")
    try:
        r=pydyn.dynare(d+"/m.mod",nosave=True)
    except Exception as e:
        return ("ERR",str(e)[:100])
    finally: pass
    f=flat(r.oo_,"oo_")
    return f
def same(a,b):
    if set(a)!=set(b): return False
    for k in a:
        x,y=a[k],b[k]
        if isinstance(x,str) or isinstance(y,str):
            if x!=y: return False
        elif x.shape!=y.shape or not np.allclose(x,y,equal_nan=True,rtol=0,atol=0): return False
    return True
BUILTIN = [["stoch_simul(order=1,irf=0,nograph);", ["stoch_simul(order=1,irf=0,nograph,hp_filter=1600);", "stoch_simul(order=1,irf=0,nograph,hp_filter=1600,filtered_theoretical_moments_grid=128);", "stoch_simul(order=1,irf=0,nograph,bandpass_filter);", "stoch_simul(order=1,irf=0,nograph,ar=2);", "stoch_simul(order=1,irf=0,nograph,nocorr);", "stoch_simul(order=1,irf=0,nograph,nodecomposition);", "stoch_simul(order=1,irf=0,nograph,nomoments);", "stoch_simul(order=1,irf=0,nograph,conditional_variance_decomposition=[1,5]);", "stoch_simul(order=1,irf=0,nograph,contemporaneous_correlation);", "stoch_simul(order=1,irf=0,nograph,one_sided_hp_filter=1600);", "stoch_simul(order=1,irf=0,nograph,qz_criterium=1.5);", "stoch_simul(order=1,irf=0,nograph,solve_algo=1);"]], ["stoch_simul(order=2,irf=0,nograph);", ["stoch_simul(order=2,irf=0,nograph,hp_filter=1600);", "stoch_simul(order=2,irf=0,nograph,hp_filter=1600,filtered_theoretical_moments_grid=128);", "stoch_simul(order=2,irf=0,nograph,bandpass_filter);", "stoch_simul(order=2,irf=0,nograph,ar=2);", "stoch_simul(order=2,irf=0,nograph,nocorr);", "stoch_simul(order=2,irf=0,nograph,nodecomposition);", "stoch_simul(order=2,irf=0,nograph,conditional_variance_decomposition=[1,5]);", "stoch_simul(order=2,irf=0,nograph,pruning);", "stoch_simul(order=2,irf=0,nograph,pruning,k_order_solver);", "stoch_simul(order=2,irf=0,nograph,k_order_solver);"]], ["stoch_simul(order=1,irf=10,nograph);", ["stoch_simul(order=1,irf=10,nograph,irf_shocks=(e));", "stoch_simul(order=1,irf=10,nograph,relative_irf);", "stoch_simul(order=1,irf=5,nograph);", "stoch_simul(order=1,irf=10,nograph,replic=3);", "stoch_simul(order=1,irf=10,nograph,loglinear);", "stoch_simul(order=1,irf=10,nograph,drop=10);"]], ["stoch_simul(order=2,irf=10,nograph);", ["stoch_simul(order=2,irf=10,nograph,irf_shocks=(e));", "stoch_simul(order=2,irf=10,nograph,relative_irf);", "stoch_simul(order=2,irf=10,nograph,replic=3);", "stoch_simul(order=2,irf=10,nograph,drop=10);", "stoch_simul(order=2,irf=10,nograph,pruning,replic=3);"]], ["stoch_simul(order=1,periods=200,irf=0,nograph);", ["stoch_simul(order=1,periods=200,irf=0,nograph,drop=10);", "stoch_simul(order=1,periods=200,irf=0,nograph,hp_filter=1600);", "stoch_simul(order=1,periods=200,irf=0,nograph,ar=2);", "stoch_simul(order=1,periods=200,irf=0,nograph,bandpass_filter);", "stoch_simul(order=1,periods=200,irf=0,nograph,nocorr);", "stoch_simul(order=1,periods=200,irf=0,nograph,nodecomposition);", "stoch_simul(order=1,periods=200,irf=0,nograph,contemporaneous_correlation);", "stoch_simul(order=1,periods=200,irf=0,nograph,simul_replic=3);"]]]


if __name__=="__main__":
    cases=json.load(open(sys.argv[1])) if len(sys.argv)>1 else BUILTIN
    for base,variants in cases:
        b=run(base)
        if isinstance(b,tuple): print("BASE ERR",base,b); continue
        for v in variants:
            r=run(v)
            if isinstance(r,tuple): print("ERR     ",v,r[1]); continue
            print("SAME    " if same(b,r) else "differs ", v)
