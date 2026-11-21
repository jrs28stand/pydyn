"""Statistical comparison of pydyn's other samplers with Dynare's posterior (post_schorfheide).

    DYNARE_PREPROCESSOR=... python reference/smp_compare.py SAMPLER [SAMPLER ...]

SAMPLER: slice, imh, student, tarb, hssmc, hssmc_big (one run of 5000 particles and 25 stages). Each runs independent chains of pydyn's sampler with numpy random numbers
(rng=None) on reference/post_schorfheide.mod (-DPOST=3: Dynare's mode file post_schorfheide_mode0.mat, mh_jscale 0.8),
and stores per chain the mean and standard deviation of every parameter over the second half of the chain (hssmc: per
independent run, over the particles) in smp_compare_<sampler>.npz. tests/test_samplers.py compares them with Dynare's
random-walk runs of r_posterior.m stages 3 and 4 (post_schorfheide_long.mat, post_schorfheide_seed*.mat: 8 chains of
20000 draws; same posterior, another sampler) with Welch t statistics: chains (runs) are independent replicates, so
the statistics hold whatever the autocorrelation within chains.
"""
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

REF = Path(__file__).resolve().parent
sys.path.insert(0, str(REF.parent))

SETTINGS = {  # sampler: (method, chains, draws, sampler options, extra keywords)
    "slice": ("slice", 6, 150, None, {}),
    "imh": ("independent_metropolis_hastings", 8, 4000, None, {"jscale": 1.1}),
    "student": ("random_walk_metropolis_hastings", 8, 4000,
                "'proposal_distribution','rand_multivariate_student','student_degrees_of_freedom',5", {}),
    "tarb": ("tailored_random_block_metropolis_hastings", 4, 150, None, {}),
}


def run(name):
    import pydyn
    from pydyn import estimation, samplers
    work = Path(tempfile.mkdtemp())
    for f in ("post_schorfheide.mod", "est_schorfheide_data.m", "post_schorfheide_mode0.mat"):
        shutil.copy(REF / f, work)
    m = pydyn.Model(pydyn.run_preprocessor(work / "post_schorfheide.mod", extra_args=("-DPOST=3",)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = estimation.Estimation(m)
    mode = est.mode()
    t0 = time.time()
    if name in ("hssmc", "hssmc_big"):
        means, stds = [], []
        runs, spec = (4, "'particles',1000,'steps',12") if name == "hssmc" else (1, "'particles',5000,'steps',25")
        for k in range(runs):
            opts = samplers.sampler_options("hssmc", spec)
            res = samplers.hssmc(est, opts=opts, seed=100 + k)
            means.append(res.particles.mean(axis=0))
            stds.append(res.particles.std(axis=0))
        out = dict(mean=np.array(means), std=np.array(stds))
    else:
        method, nb, T, spec, kw = SETTINGS[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = estimation.Estimation(m, posterior_sampling_method=method) if method == "slice" else est
        mc = e.sample(replic=T, nblocks=nb, mode=mode, seed=12345, method=method,
                      sampler_options=spec if spec is not None else "", **kw)
        half = mc.draws[:, T // 2:, :]
        out = dict(mean=half.mean(axis=1), std=half.std(axis=1), acceptance=mc.acceptance_rate,
                   neval=mc.neval.mean(axis=1))
    out["elapsed"] = time.time() - t0
    out["names"] = np.array(est.names)
    np.savez(REF / f"smp_compare_{name}.npz", **out)
    print(name, "done in", round(out["elapsed"]), "s")


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        run(nm)
