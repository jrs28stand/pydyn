"""Fast subset of the acceptance sweep (tools/accept_pydyn.py + tools/accept_compare.py) against Dynare 7.2 results stored by
tools/accept_octave.py.  Skipped when the reference folder (env PYDYN_ACCEPT_REF, default /home/claude/accept_ref) or a file's result
is absent."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

REF = Path(os.environ.get("PYDYN_ACCEPT_REF", "/home/claude/accept_ref"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
FILES = ["stochastic_simulations/example1", "external_function/benchmark", "deterministic_simulations/rbc_det6",
         "occbin/model_borrcon/borrcon", "irfs/example1_unit_std"]


@pytest.mark.parametrize("name", FILES)
def test_against_dynare(name):
    if not (REF / f"{name}.mat").exists():
        pytest.skip("no Dynare reference for " + name)
    if not os.environ.get("DYNARE_PREPROCESSOR") and not Path("/home/claude/dl/src72/build-octave/preprocessor/src/dynare-preprocessor").exists():
        pytest.skip("Dynare 7.2 preprocessor not available")
    import accept_sweep as s
    with tempfile.TemporaryDirectory() as out:
        res = s.run_tests([name], out=out, ref=REF, timeout=300)
    assert res[name]["verdict"] in ("match", "partial", "refused"), res[name]["detail"]


# ---------------------------------------------------------------------------------- comparator rules (tools/accept_compare.py)
def _mk(tmp_path, tid, ref_oo, port_oo, ref_opts=None, rng="octave", ref_status="ok", log=""):
    import json

    from scipy.io import savemat
    ref, port = tmp_path / "ref", tmp_path / "port"
    for d, oo in ((ref, ref_oo), (port, port_oo)):
        (d / tid).parent.mkdir(parents=True, exist_ok=True)
        if oo is not None:
            savemat(str(d / f"{tid}.mat"), {"oo_": oo, "M_": {"x": 1.0}, "options_": ref_opts if d is ref and ref_opts else {"periods": 0.0}})
    (ref / f"{tid}.json").write_text(json.dumps({"status": ref_status}))
    (ref / f"{tid}.log").write_text(log)
    (port / f"{tid}.json").write_text(json.dumps({"status": "ok", "rng": rng}))
    return ref, port


def _run(tmp_path, tid, ref_oo, port_oo, **kw):
    import accept_compare as ac
    ref, port = _mk(tmp_path, tid, ref_oo, port_oo, **kw)
    return ac.compare_one(tid, port, ref)


def test_cmp_struct_arrays_and_cells(tmp_path):
    import numpy as np

    def cell(v):
        c = np.empty(2, dtype=object)
        c[0] = {"a": np.array([1.0, 2.0]), "tag": "x"}
        c[1] = {"a": np.array([v]), "tag": "y"}
        return c
    r = _run(tmp_path, "t/a", {"occbin": {"simul": {"regime_history": cell(3.0)}}}, {"occbin": {"simul": {"regime_history": cell(3.0)}}})
    assert r["verdict"] == "match", r
    r = _run(tmp_path / "b", "t/a", {"occbin": {"simul": {"regime_history": cell(3.0)}}}, {"occbin": {"simul": {"regime_history": cell(4.0)}}})
    assert r["verdict"] == "wrong" and "{2}" in r["detail"], r


def test_cmp_timing_is_bookkeeping(tmp_path):
    r = _run(tmp_path, "t/a", {"steady_state": [1.0, 2.0], "posterior": {"optimization": {"optimization_info": {"runtime": 3.0, "iterations": 5.0}}}},
             {"steady_state": [1.0, 2.0], "posterior": {"optimization": {"optimization_info": {"runtime": 99.0, "iterations": 7.0}}}})
    assert r["verdict"] == "match", r


def test_cmp_heterogeneity_excluded(tmp_path):
    r = _run(tmp_path, "heterogeneity/ks", {"steady_state": [1.0]}, None, ref_status="ok")
    assert r["verdict"] == "excluded"
    r = _run(tmp_path / "b", "examples/heterogeneity/ks", {"steady_state": [1.0]}, {"steady_state": [1.0]}, ref_status="timeout")
    assert r["verdict"] == "excluded"


def test_cmp_simulated_moments_need_rng(tmp_path):
    import numpy as np
    paths = np.cumsum(np.sin(np.arange(1, 40.0))[None, :] * np.array([[1.0], [2.0]]), axis=1)
    m = paths.mean(axis=1)
    oo = {"endo_simul": paths, "mean": m, "var": np.eye(2), "dr": {"ghx": np.ones((2, 1))}, "steady_state": np.ones(2)}
    opts = {"periods": 39.0, "drop": 0.0}
    r = _run(tmp_path, "t/a", oo, oo, ref_opts=opts, rng="julia")
    rows = {a: b for a, b, _ in r["rows"]}
    assert rows["mean"] == "skipped" and rows["var"] == "skipped" and rows["endo_simul"] == "skipped" and rows["steady_state"] == "match"
    r = _run(tmp_path / "b", "t/a", oo, oo, ref_opts=opts, rng="octave")
    assert r["verdict"] == "match", r
    # periods == 0 and no evidence of simulation: theoretical moments, compared even without the generator
    oo2 = {"mean": m, "var": np.eye(2), "steady_state": np.ones(2)}
    r = _run(tmp_path / "c", "t/a", oo2, oo2, ref_opts={"periods": 0.0}, rng="julia")
    assert r["verdict"] == "match", r


def test_cmp_tolerance_classes(tmp_path):
    import numpy as np
    ss = np.array([1.0, 2.0, 3.0])
    ref = {"steady_state": ss, "dr": {"ys": ss, "ghx": np.array([[0.5, 0.2]])}}
    port = {"steady_state": ss * (1 + 1e-6), "dr": {"ys": ss * (1 + 1e-6), "ghx": np.array([[0.5, 0.2]]) * (1 + 3e-6)}}
    r = _run(tmp_path, "t/a", ref, port, ref_opts={"steadystate_flag": 0.0, "periods": 0.0})
    assert r["verdict"] == "match" and len(r["relaxed"]) == 3, r                   # numerically solved: ss class, ghx inherits
    r = _run(tmp_path / "b", "t/a", ref, port, ref_opts={"steadystate_flag": 2.0, "periods": 0.0})
    assert r["verdict"] == "wrong", r                                               # analytic steady state: 1e-8
    port2 = {"steady_state": ss, "dr": {"ys": ss, "ghx": np.array([[0.5, 0.2]]) * (1 + 3e-6)}}
    r = _run(tmp_path / "c", "t/a", ref, port2, ref_opts={"steadystate_flag": 0.0, "periods": 0.0})
    assert r["verdict"] == "wrong", r                                               # steady state exact: nothing downstream is relaxed
    pf = {"deterministic_simulation": {"status": 1.0}, "endo_simul": np.ones((2, 5)) * 3.0 + np.arange(5)}
    pfp = {"deterministic_simulation": {"status": 1.0}, "endo_simul": (np.ones((2, 5)) * 3.0 + np.arange(5)) * (1 + 5e-7)}
    r = _run(tmp_path / "d", "t/a", pf, pfp, ref_opts={"steadystate_flag": 0.0, "periods": 5.0}, rng="julia")
    assert r["verdict"] == "match", r                                               # perfect-foresight class 1e-6, no generator needed


def test_cmp_errors_are_not_port_errors(tmp_path, monkeypatch):
    import accept_compare as ac

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(ac.Walker, "one", boom)
    r = _run(tmp_path, "t/a", {"steady_state": [1.0]}, {"steady_state": [1.0]})
    assert r["verdict"] == "compare-error" and "boom" in r["detail"], r
