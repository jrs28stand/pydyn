"""Scaling / multi-constraint measurements for HigherOrderNews (docs/OBC_HO_PERF.md, section 'Scaling').
Usage: python3 bench/obc_ho_scale.py probe            which models HigherOrderNews accepts
"""
from __future__ import annotations
import os, shutil, sys, tempfile, time, warnings
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
import pydyn
from pydyn.obc_ho import HigherOrderNews

TESTS = Path(os.environ.get("PYDYN_EXAMPLES", "/home/claude/ref/dynare-7.2/examples")).parent / "tests"
MODS = {
    "nk": REPO / "reference/dynareobc_octave/nk_occ.mod",
    "nkm_est": REPO / "reference/occest_nkm.mod",
    "nkm": TESTS / "occbin/filter/NKM_0_std_shocks.mod",
    "zlb": REPO / "reference/obc_zlb.mod",
    "zlb2": REPO / "reference/obc_zlb2.mod",
    "sw": REPO / "reference/obc_sw_zlb.mod",
    "rbc_est": REPO / "reference/occest_rbc.mod",
    "borrcon": REPO / "reference/occ_borrcon.mod",
    "rbc1": REPO / "reference/occ_rbc1.mod",
}

def load(name, path=None):
    path = Path(path or MODS[name])
    d = Path(tempfile.mkdtemp(prefix="obchos_"))
    if path.parent.name == "filter" or path.parent.is_dir():
        for f in path.parent.iterdir():
            if f.is_file():
                shutil.copy(f, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydyn.Model(pydyn.run_preprocessor(d / path.name))

def probe():
    for name in MODS:
        try:
            m = load(name)
        except Exception as e:
            print(f"{name:8s} LOAD FAIL {type(e).__name__}: {str(e)[:100]}"); continue
        for kw in (dict(order=1), dict(order=2), dict(order=2, cubature=True, qmc_level=7)):
            try:
                t = time.perf_counter()
                h = HigherOrderNews(m, horizon=32, check_ahead=32, **kw)
                print(f"{name:8s} n={m.n_endo:3d} nc={h.news.n_c} {kw}: ok {1e3*(time.perf_counter()-t):.0f} ms, ns={len(h.ist)}")
            except Exception as e:
                print(f"{name:8s} n={m.n_endo:3d} {kw}: REFUSED {type(e).__name__}: {str(e)[:140]}")

if __name__ == "__main__":
    {"probe": probe}[sys.argv[1]]()
