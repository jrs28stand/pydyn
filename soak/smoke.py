import sys, time, warnings, resource
sys.path.insert(0,'.')
import numpy as np, tempfile, shutil
from pathlib import Path
from scipy.io import loadmat
import pydyn
from pydyn.occbin_estimation import OccbinEstimation
REF=Path('reference'); T=Path('/home/claude/ref/dynare-7.2/tests')
IVF={"likelihood_inversion_filter": True}
def mk(case, method='occbin', **kw):
    tmp=Path(tempfile.mkdtemp())
    if case=='bf':
        shutil.copy(T/'estimation/slice/bf.mod', tmp); p=tmp/'bf.mod'
        d=loadmat(REF/'occest_bf_data.mat',squeeze_me=True); data=np.column_stack([d['q'],d['r']])
        kw.update(posterior_sampling_method="random_walk_metropolis_hastings", posterior_sampler_options=())
    else:
        shutil.copy(REF/f'occest_{case}.mod',tmp); p=tmp/f'occest_{case}.mod'
        r=loadmat(REF/f'occest_{case}_lik.mat',squeeze_me=True); data=np.atleast_2d(r['data'].T).T
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m=pydyn.Model(pydyn.run_preprocessor(p))
        e=OccbinEstimation(m,data={v:data[:,j] for j,v in enumerate(m.varobs)},nobs=data.shape[0],first_obs=1,occbin=IVF,obc_method=method,**kw)
    return e
for c in ['nkms','nkm','bf']:
    for meth in ['occbin','news']:
        e=mk(c,meth); x=np.array(e.x0); t=time.time(); lp=e.log_posterior(x); print(c,meth,lp,e.last_code,round(time.time()-t,3),len(x), e.lb[:3], e.ub[:3])
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
e=mk('nkm'); print([ (n, ) for n in e.model.param_names[:0]]); print(type(e.prior), e.prior.shape, e.prior.p6, e.prior.p7, e.prior.p3, e.prior.p4); print(getattr(e,'param_names',None), e.names if hasattr(e,'names') else '')
