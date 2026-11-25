"""Infrastructure of the OBC soak test (see docs/OBC_SOAK.md): state, result files, anomaly log, timeouts, models.

Every task is a pure function of (part, index): its random numbers come from default_rng([SEED, part, index]), so a
re-run of an index reproduces it.  State lives in soak/state/<part>.json (next index, counters); results are appended to
soak/results/<part>.jsonl and anomalies to soak/results/anomalies.jsonl, flushed at once.
"""
from __future__ import annotations

import collections
import contextlib
import json
import os
import re
import resource
import shutil
import signal
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOAK = ROOT / "soak"
RES, STATE = SOAK / "results", SOAK / "state"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
SEED = 20261124
PARTS = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
MAX_DETAIL = 20                     # detailed anomaly lines kept per kind (counts are exact)
TESTS = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/ref/dynare-7.2/examples")).parent / "tests"
REF = ROOT / "reference"


def rng_for(part, i):
    return np.random.default_rng([SEED, PARTS[part], int(i)])


def rss_mb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except OSError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ------------------------------------------------------------------------------------------------ timeouts
class Alarm(BaseException):
    """Raised by the per-evaluation timer (BaseException: no `except Exception` in the library may swallow it)."""


@contextlib.contextmanager
def alarm(seconds):
    def handler(signum, frame):
        raise Alarm(f"timeout after {seconds}s")
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ------------------------------------------------------------------------------------------------ files
def load_state(part):
    p = STATE / f"{part}.json"
    if p.exists():
        return json.loads(p.read_text())
    return dict(next=0, seconds=0.0, kinds={}, warnings={}, counts={}, medians={}, notes={})


def save_state(part, st):
    p = STATE / f"{part}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, separators=(",", ":")))
    os.replace(tmp, p)


def _clean(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


NO_WRITE = False


def append(name, rec):
    if NO_WRITE:
        print("ANOMALY", json.dumps(rec, default=_clean)[:1500])
        return
    with open(RES / name, "a") as f:
        f.write(json.dumps(rec, default=_clean, separators=(",", ":")) + "\n")
        f.flush()


class Anom:
    """Anomaly logger: counts every occurrence of a kind in the state, keeps MAX_DETAIL detailed lines per kind."""

    def __init__(self, part, st):
        self.part, self.st, self.new = part, st, []

    def __call__(self, kind, **info):
        k = self.st["kinds"]
        k[kind] = k.get(kind, 0) + 1
        self.new.append(kind)
        if k[kind] <= MAX_DETAIL:
            info = dict(info)
            if "tb" in info:
                info["tb"] = info["tb"][-1800:]
            append("anomalies.jsonl", dict(kind=kind, part=self.part, n=k[kind], **info))


def norm_msg(msg):
    """A message with numbers removed, to count distinct kinds of an error."""
    return re.sub(r"[-+]?\d+\.?\d*(e[-+]?\d+)?", "#", str(msg))[:90]


def note_time(st, key, t, lim=300):
    """Running reservoir of the last `lim` times of a configuration; returns the median before adding t (or None)."""
    r = st["medians"].setdefault(key, [])
    med = float(np.median(r)) if len(r) >= 20 else None
    r.append(round(float(t), 5))
    if len(r) > lim:
        del r[0]
    return med


def count(st, key, n=1):
    st["counts"][key] = st["counts"].get(key, 0) + n


# ------------------------------------------------------------------------------------------------ models
def expected_error_types():
    from pydyn.obc import OBCClassError, OBCError
    from pydyn.obc_ho import CubatureFailure
    from pydyn.occbin import OccbinError
    from pydyn.perturbation import BlanchardKahnError
    from pydyn.steady import SteadyStateError
    return (OBCError, OBCClassError, CubatureFailure, OccbinError, BlanchardKahnError, SteadyStateError,
            np.linalg.LinAlgError)


_MODELS = {}
SRC = {
    "zlb": REF / "obc_zlb.mod", "zlb2": REF / "obc_zlb2.mod", "sw": REF / "obc_sw_zlb.mod",
    "nkm": TESTS / "occbin/filter/NKM_0_std_shocks.mod", "nkm_est": REF / "occest_nkm.mod",
    "nkms_est": REF / "occest_nkms.mod",
    "bf": TESTS / "estimation/slice/bf.mod", "borrcon_d": TESTS / "occbin/model_borrcon/borrcon.mod",
    "borrcon": REF / "occ_borrcon.mod", "nk": REF / "dynareobc_octave/nk_occ.mod",
}


def load_model(name):
    if name in _MODELS:
        return _MODELS[name]
    import pydyn
    if name == "sw_cp":
        import obc_ho_scale as sc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sc.sw_cp_model()
    else:
        path = SRC[name]
        d = Path(tempfile.mkdtemp(prefix="soak_"))
        for f in path.parent.iterdir():
            if f.is_file() and f.suffix in (".mod", ".inc", ".mat", ".m", ".csv"):
                shutil.copy(f, d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = pydyn.Model(pydyn.run_preprocessor(d / path.name))
    _MODELS[name] = m
    return m


def perturb_params(m, rng, scale):
    """A parameter vector: every parameter except the occbin switches perturbed (logit for values in (0,1), log for
    positive, additive in |p| otherwise) with standard deviation `scale`."""
    base = np.array(m.params, float)
    skip = {m._ip[n] for n in m._ip if n.startswith("occbin_") and n.endswith("_bind")}
    out = base.copy()
    if scale <= 0:
        return out
    for j, p in enumerate(base):
        if j in skip or p == 0 or not np.isfinite(p):
            continue
        z = rng.standard_normal() * scale
        if 0 < p < 1:
            lg = np.log(p / (1 - p)) + z
            out[j] = 1 / (1 + np.exp(-lg))
        elif p > 0:
            out[j] = p * np.exp(z)
        else:
            out[j] = p * np.exp(z)
    return out


# ------------------------------------------------------------------------------------------------ verification hooks
VERIFY = collections.Counter()
VERIFY_FAILS = []


def install_verifiers():
    """Wrap solve_active_set / solve_news (as imported by obc, obc_filter, obc_ho): every returned solution is checked
    with verify_lcp (tolerance 1e-6); failures are queued in VERIFY_FAILS."""
    import importlib
    from pydyn import obc
    for modname in ("pydyn.obc", "pydyn.obc_filter", "pydyn.obc_ho"):
        mod = importlib.import_module(modname)
        for fn in ("solve_active_set", "solve_news"):
            orig = getattr(mod, fn, None)
            if orig is None or getattr(orig, "_soak", False):
                continue

            def wrap(*a, _orig=orig, _fn=fn, **k):
                r = _orig(*a, **k)
                y = r[0] if _fn == "solve_news" else r
                if y is not None:
                    try:
                        M, q, n_s = a[0], a[1], a[2]
                        VERIFY[_fn] += 1
                        if np.asarray(M).shape[1] == np.size(y) and not obc.verify_lcp(M, q, y, n_s, tol=1e-6):
                            g = q + M @ y
                            VERIFY_FAILS.append(dict(fn=_fn, miny=float(np.min(y)), ming=float(np.min(g)),
                                                     comp=float(np.max(np.minimum(np.abs(y[:n_s]), np.abs(g[:n_s]))))))
                    except Exception as exc:                       # the checker itself must not break the run
                        VERIFY["checker_error"] += 1
                return r
            wrap._soak = True
            setattr(mod, fn, wrap)


@contextlib.contextmanager
def record_warnings(st, anom, where):
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter("always")
        try:
            yield W
        finally:
            for w in W:
                key = f"{w.category.__name__}: {norm_msg(w.message)}"
                st["warnings"][key] = st["warnings"].get(key, 0) + 1
                if issubclass(w.category, RuntimeWarning):
                    anom("warning:" + key, **where, file=f"{Path(w.filename).name}:{w.lineno}")
            if VERIFY_FAILS:
                n = len(VERIFY_FAILS)
                for v in VERIFY_FAILS[:3]:
                    anom("verify_lcp_fail", **where, **v)
                VERIFY_FAILS.clear()


def exc_info(e):
    return dict(type=type(e).__name__, msg=str(e)[:300], tb="".join(traceback.format_exception(type(e), e, e.__traceback__)))
