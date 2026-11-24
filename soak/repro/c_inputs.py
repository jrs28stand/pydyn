"""Rebuild the inputs of a part-C task: python3 -c 'from c_inputs import *' (mirrors part_cd.task_C's random draws)."""
import sys; sys.path.insert(0, "soak")
import numpy as np, lib, part_cd
from pydyn.occbin import Occbin
def inputs(i, seg):
    rng = lib.rng_for("C", i); name = part_cd.MODELS_C[i % len(part_cd.MODELS_C)]; m = lib.load_model(name)
    pscale = float(rng.choice([0.0, 0.0, 0.02, 0.1, 0.3])); params = lib.perturb_params(m, rng, pscale)
    sscale = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0)))) * part_cd.BASE_C.get(name, 1.0)
    target = int(rng.choice([1000, 3000, 10000, 30000])); h = int(rng.choice([40, 40, 40, 60, 80]))
    sd = np.sqrt(np.maximum(np.diag(np.array(m.Sigma_e, float)), 0.0))
    for s in range(seg + 1):
        S = rng.standard_normal((part_cd.SEG_LEN, m.n_exo)) * sd * sscale
    return m, Occbin(m, params), params, S, h
