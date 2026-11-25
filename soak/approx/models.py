"""The model set of the approx-class study, with simulated data and OccbinEstimation objects."""
import sys, warnings
sys.path.insert(0, "soak"); sys.path.insert(0, "soak/approx")
import numpy as np, lib, gen, pydyn
from pydyn.occbin import Occbin
from pydyn.obc import constraint_objects
from pydyn.occbin_estimation import OccbinEstimation

IVF = {"likelihood_inversion_filter": True}
# name -> (family, requested discrepancy d)
SPEC = {"zlbm_0": ("zlbm", 0), "zlbm_1e-3": ("zlbm", 1e-3), "zlbm_1e-2": ("zlbm", 1e-2), "zlbm_5e-2": ("zlbm", 5e-2),
        "zlb2m_0": ("zlb2m", 0), "zlb2m_1e-3": ("zlb2m", 1e-3), "zlb2m_1e-2": ("zlb2m", 1e-2), "zlb2m_5e-2": ("zlb2m", 5e-2),
        "nkocc_1e-3": ("nkocc", 1e-3), "nkocc_5e-3": ("nkocc", 5e-3), "nkocc_1.5e-2": ("nkocc", 0.02)}   # last: km as large as 1/0.015*... see approx printed
_cache = {}


def get(name, T=150, seed=7):
    if name in _cache:
        return _cache[name]
    fam, d = SPEC[name]
    m = gen.build(name, gen.FAMILIES[fam](d))
    ob = Occbin(m)
    cs = constraint_objects(ob)
    for sd_ in range(seed, seed + 40):           # first data seed for which OccBin itself solves every period
        S = np.random.default_rng(sd_).standard_normal((T, m.n_exo))
        try:
            r = ob.simulate(S, periods=T)
            break
        except Exception:                        # noqa: BLE001  (OccbinError: cycling guesses)
            continue
    else:
        raise RuntimeError("no data seed")
    e_seed = sd_
    obs = [m._ie[v] for v in m.varobs]
    data = r.piecewise[:, obs]
    nb = sum(1 for h in r.regime_history if any(c[0][0] for c in h))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = OccbinEstimation(m, data={v: data[:, j] for j, v in enumerate(m.varobs)}, nobs=T, first_obs=1, occbin=IVF)
    e.x0v = np.array(e.x0, float); e.stored = np.array([[]]); e.case = name
    e.info = dict(seed=e_seed, name=name, cls=[(c.cls, c.approx) for c in cs], binding=nb, T=T, true=np.array(e.x0, float))
    _cache[name] = e
    return e


if __name__ == "__main__":
    for nm in SPEC:
        try:
            e = get(nm); out = []
            for meth in ("occbin", "news"):
                e.obc_method = meth; lp = e.log_posterior(e.x0v); out.append((round(lp, 4), e.last_code))
            print(nm, e.info["cls"], "binding", e.info["binding"], "/", e.info["T"], out, flush=True)
        except Exception as ex:
            print(nm, type(ex).__name__, str(ex)[:150])
