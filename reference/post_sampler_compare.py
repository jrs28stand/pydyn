"""Statistical comparison of pydyn's Metropolis-Hastings sampler with Dynare's.

    DYNARE_PREPROCESSOR=... python reference/post_sampler_compare.py [post_nk] [post_schorfheide]

For each model it reads post_<model>_long.mat (r_posterior.m, POST_STAGES=3: batch statistics of Dynare's 2 x 20000
draws with the default seed) and post_<model>_seed<k>.mat (POST_STAGES=4: the same run with other seeds), runs pydyn
with the same mode, proposal covariance and scale (only the random numbers differ; SEEDS below, POST_CHAINS chains per
seed, default 8),
writes pydyn's batch statistics to post_<model>_long_pydyn.npz and prints the tables. tests/test_posterior.py
recomputes everything from the stored files with the functions below.

Batch statistics (batch_stats, as in r_posterior.m): the second half of every chain is cut into 20 batches of 500
draws; per batch and parameter the mean, the mean of squares, the 5% and 95% quantiles and the share of draws that
moved.

Two comparisons:

* compare(): the batch statistics of all chains pooled, standard errors from the dispersion of the batch values (a
  jackknife over batches for the standard deviation). This treats batches as independent and understates the
  Monte Carlo error: Dynare's default-seed run against its three runs with other seeds gives z-scores with a root
  mean square of 1.42 and z = 3.45 for the posterior mean of stderr e_m (post_schorfheide); post_nk, default seed
  against seed 1000: root mean square 1.17. Kept for the record.
* compare_chains(): every chain is one replicate (its mean, standard deviation and average batch quantiles over the
  second half); two-sample t statistics with the between-chain variance pooled over both samplers. Chains are
  independent, so under the hypothesis that both samplers have the same distribution the statistics are Student t
  with (chains - 2) degrees of freedom whatever the autocorrelation within chains.
"""
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REF = Path(__file__).resolve().parent
sys.path.insert(0, str(REF.parent))
NBATCH = 20


def batch_stats(x, nbatch=NBATCH):
    """(nbatch, n, 5) for one chain x (T, n): as batch_stats in r_posterior.m."""
    T, n = x.shape
    first = T // 2
    L = (T - first) // nbatch
    out = np.zeros((nbatch, n, 5))
    for k in range(nbatch):
        sl = slice(first + k * L, first + (k + 1) * L)
        seg = x[sl]
        s = np.sort(seg, axis=0)
        out[k, :, 0] = seg.mean(axis=0)
        out[k, :, 1] = (seg ** 2).mean(axis=0)
        out[k, :, 2] = s[int(np.floor(0.05 * L + 0.5)) - 1]
        out[k, :, 3] = s[int(np.floor(0.95 * L + 0.5)) - 1]
        out[k, :, 4] = np.mean(np.any(seg != x[sl.start - 1:sl.stop - 1], axis=1))
    return out


def estimates(batch):
    """batch: (batches in all chains, n, 5) -> dict name -> (estimate, standard error), each (n,)."""
    B = batch.shape[0]
    out = {}
    for name, col in (("mean", 0), ("q05", 2), ("q95", 3), ("acceptance", 4)):
        out[name] = (batch[:, :, col].mean(axis=0), batch[:, :, col].std(axis=0, ddof=1) / np.sqrt(B))
    sd = lambda b: np.sqrt(b[:, :, 1].mean(axis=0) - b[:, :, 0].mean(axis=0) ** 2)  # noqa: E731
    full = sd(batch)
    jack = np.array([sd(np.delete(batch, k, axis=0)) for k in range(B)])
    out["std"] = (full, np.sqrt((B - 1) / B * np.sum((jack - jack.mean(axis=0)) ** 2, axis=0)))
    return out


def compare(dyn_batch, py_batch):
    """z-scores {statistic: (n,)} and the estimates of both."""
    d, p = estimates(dyn_batch), estimates(py_batch)
    z = {k: (p[k][0] - d[k][0]) / np.sqrt(p[k][1] ** 2 + d[k][1] ** 2) for k in d}
    return z, d, p


def chain_stats(batch, nbatch=NBATCH):
    """Per-chain statistics from batch statistics shaped (nbatch * chains, n, 5): dict name -> (chains, n)."""
    b = batch.reshape(-1, nbatch, batch.shape[1], batch.shape[2])
    return {"mean": b[:, :, :, 0].mean(axis=1), "std": np.sqrt(b[:, :, :, 1].mean(axis=1) - b[:, :, :, 0].mean(axis=1) ** 2),
            "q05": b[:, :, :, 2].mean(axis=1), "q95": b[:, :, :, 3].mean(axis=1), "acceptance": b[:, :, :1, 4].mean(axis=1)}


def compare_chains(dyn_batch, py_batch):
    """t statistics {statistic: (n,)} comparing the per-chain statistics of two runs (pooled between-chain variance),
    the per-chain statistics of both and the degrees of freedom."""
    d, p = chain_stats(dyn_batch), chain_stats(py_batch)
    kd, kp = d["mean"].shape[0], p["mean"].shape[0]
    t = {}
    for k in d:
        s2 = ((kd - 1) * d[k].var(axis=0, ddof=1) + (kp - 1) * p[k].var(axis=0, ddof=1)) / (kd + kp - 2)
        t[k] = (p[k].mean(axis=0) - d[k].mean(axis=0)) / np.sqrt(s2 * (1 / kd + 1 / kp))
    return t, d, p, kd + kp - 2


def _t_from_stats(x, first):
    """Pooled-variance t statistics (statistics x parameters) for chains ``first`` (boolean mask) against the rest;
    ``x``: (chains, statistics x parameters)."""
    a, b = x[first], x[~first]
    ka, kb = a.shape[0], b.shape[0]
    s2 = ((ka - 1) * a.var(axis=0, ddof=1) + (kb - 1) * b.var(axis=0, ddof=1)) / (ka + kb - 2)
    return (b.mean(axis=0) - a.mean(axis=0)) / np.sqrt(s2 * (1 / ka + 1 / kb))


def permutation_test(dyn_batch, py_batch, n_perm=4000, seed=0):
    """Permutation test of "Dynare's and pydyn's chains have the same distribution": the chains of both samplers are
    pooled and relabelled at random ``n_perm`` times (Dynare's number of chains kept); the statistics are the root mean
    square and the largest absolute value of the t statistics of compare_chains() (mean, std, 5% and 95% quantiles of
    every parameter). Returns {"rms": (observed, p-value), "max": (observed, p-value), "null_rms": ..., "null_max": ...};
    the p-value counts the observed labelling as one of the relabellings. Chains are independent and, under the
    hypothesis, exchangeable, so the test is exact whatever the autocorrelation within chains."""
    d, p = chain_stats(dyn_batch), chain_stats(py_batch)
    keys = ("mean", "std", "q05", "q95")
    x = np.concatenate([np.concatenate([d[k], p[k]], axis=0) for k in keys], axis=1)
    kd = d["mean"].shape[0]
    first = np.zeros(x.shape[0], bool)
    first[:kd] = True
    t = _t_from_stats(x, first)
    obs = np.sqrt(np.mean(t ** 2)), np.abs(t).max()
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, 2))
    for i in range(n_perm):
        f = np.zeros(x.shape[0], bool)
        f[rng.choice(x.shape[0], kd, replace=False)] = True
        tt = _t_from_stats(x, f)
        null[i] = np.sqrt(np.mean(tt ** 2)), np.abs(tt).max()
    return {"rms": (obs[0], (1 + np.sum(null[:, 0] >= obs[0])) / (n_perm + 1)),
            "max": (obs[1], (1 + np.sum(null[:, 1] >= obs[1])) / (n_perm + 1)), "null_rms": null[:, 0], "null_max": null[:, 1]}


def stack(batch4):
    """(nbatch, n, 5, chains) as saved by Octave -> (nbatch * chains, n, 5)."""
    return np.concatenate([batch4[:, :, :, b] for b in range(batch4.shape[3])], axis=0)


def dynare_runs(mod):
    """[(label, batch statistics (nbatch * chains, n, 5), acceptance per chain)] of Dynare's default run and of the
    runs with other seeds found next to this file."""
    L = loadmat(REF / f"{mod}_long.mat", squeeze_me=True)
    out = [("seed 0", stack(L["batch"]), np.atleast_1d(L["acceptance"]))]
    for f in sorted(REF.glob(f"{mod}_seed*.mat"), key=lambda f: int(f.stem.split("_seed")[1])):
        S = loadmat(f, squeeze_me=True)
        out.append((f"seed {int(S['seed'])}", stack(S["batch"]), np.atleast_1d(S["acceptance"])))
    return out


def pydyn_runs(mod, seeds, chains=8, draws=None):
    """Batch statistics of pydyn's chains, one run of ``chains`` chains per seed; ``draws`` {seed: array (chains, T, n)}
    supplies the chains of a run made earlier with the same settings instead of running it again."""
    import pydyn
    from pydyn import estimation
    tmp = Path(tempfile.mkdtemp())
    for f in list(REF.glob("est_*_data.m")) + [REF / f"{mod}.mod", REF / f"{mod}_mode0.mat"]:
        shutil.copy(f, tmp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = estimation.Estimation(pydyn.Model(pydyn.run_preprocessor(tmp / f"{mod}.mod", extra_args=("-DPOST=3",))))
    L = loadmat(REF / f"{mod}_long.mat", squeeze_me=True)
    mode = est.mode()                                     # Dynare's mode file
    assert np.array_equal(mode.x, L["xparam1"]) and est.options["mh_replic"] == L["replic"]
    assert np.allclose(np.linalg.cholesky(mode.inv_hessian).T, L["proposal_chol"], rtol=1e-10, atol=1e-14)
    batch, acc, elapsed = [], [], 0.0
    for seed in seeds:
        if draws is not None and seed in draws:
            d = np.asarray(draws[seed])
            acc.append(np.mean(np.any(np.diff(d, axis=1) != 0, axis=2), axis=1))
        else:
            t = time.time()
            mc = est.sample(nblocks=chains, mode=mode, seed=seed)
            elapsed += time.time() - t
            d = mc.draws
            acc.append(mc.acceptance_rate)
        batch.append(np.concatenate([batch_stats(d[b]) for b in range(d.shape[0])], axis=0))
    return np.concatenate(batch, axis=0), np.concatenate(acc), est.names, elapsed


def report(mod, dyn, py_batch, names):
    """Print the comparison tables."""
    dyn_batch = np.concatenate([b for _, b, _ in dyn], axis=0)
    t, d, p, dof = compare_chains(dyn_batch, py_batch)
    print(f"\n{mod}: Dynare {', '.join(lab for lab, _, _ in dyn)} ({d['mean'].shape[0]} chains), pydyn {p['mean'].shape[0]} "
          f"chains; per-chain statistics over the second halves, t statistics with {dof} degrees of freedom")
    print(f"{'parameter':14s}" + "".join(f"{h:>32s}" for h in ("mean", "std", "5% quantile", "95% quantile")))
    print(f"{'':14s}" + f"{'Dynare':>12s}{'pydyn':>12s}{'t':>8s}" * 4)
    for j, name in enumerate(names):
        print(f"{name:14s}" + "".join(f"{d[k].mean(0)[j]:12.6f}{p[k].mean(0)[j]:12.6f}{t[k][j]:8.2f}" for k in ("mean", "std", "q05", "q95")))
    print(f"acceptance (second halves): Dynare {d['acceptance'].mean():.4f}, pydyn {p['acceptance'].mean():.4f}, "
          f"t = {t['acceptance'][0]:.2f}")
    allt = np.concatenate([t[k] for k in ("mean", "std", "q05", "q95")])
    print(f"{allt.size} t statistics: max |t| = {np.abs(allt).max():.2f}, root mean square = {np.sqrt(np.mean(allt ** 2)):.2f}, "
          f"|t| > 2: {np.sum(np.abs(allt) > 2)}")
    pt = permutation_test(dyn_batch, py_batch)
    print(f"permutation test (chains relabelled at random): p-value {pt['rms'][1]:.3f} for the root mean square, "
          f"{pt['max'][1]:.3f} for the largest |t|")
    for i, (lab, b, _) in enumerate(dyn):
        others = np.concatenate([x for k, (_, x, _) in enumerate(dyn) if k != i], axis=0)
        z, *_ = compare(others, b)
        az = np.concatenate([z[k] for k in ("mean", "std", "q05", "q95")])
        print(f"  Dynare {lab} against the other Dynare runs: batch-means z max |z| {np.abs(az).max():.2f}, root mean square "
              f"{np.sqrt(np.mean(az ** 2)):.2f}; {names[1]}: mean z {z['mean'][1]:.2f}, 5% quantile z {z['q05'][1]:.2f}")
    z, *_ = compare(dyn[0][1], py_batch)
    az = np.concatenate([z[k] for k in ("mean", "std", "q05", "q95")])
    print(f"  pydyn against Dynare seed 0 alone: batch-means z max |z| {np.abs(az).max():.2f}, root mean square "
          f"{np.sqrt(np.mean(az ** 2)):.2f}; {names[1]}: mean z {z['mean'][1]:.2f}, 5% quantile z {z['q05'][1]:.2f}")


#: seeds of pydyn's runs stored in post_<model>_long_pydyn.npz (8 chains each); post_schorfheide has two more runs
#: because the first two looked high in one statistic (95% quantile of logmst) against Dynare and against pydyn driven
#: by Octave's random numbers; the two later runs did not repeat it (t = 1.6 against both Octave-driven samples)
SEEDS = {"post_nk": (7001, 7002), "post_schorfheide": (7001, 7002, 7003, 7004)}


def main(models, chains=8):
    for mod in models:
        seeds = SEEDS[mod]
        batch, acc, names, elapsed = pydyn_runs(mod, seeds, chains)
        np.savez_compressed(REF / f"{mod}_long_pydyn.npz", batch=batch, acceptance=acc, seeds=np.array(seeds), chains=chains,
                            names=np.array(names))
        print(f"{mod}: pydyn {len(seeds)} x {chains} chains in {elapsed:.0f} s")
        report(mod, dynare_runs(mod), batch, names)


if __name__ == "__main__":
    main(sys.argv[1:] or ["post_nk", "post_schorfheide"], chains=int(os.environ.get("POST_CHAINS", 8)))
