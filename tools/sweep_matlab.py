"""Corpus harness for pydyn.matlab.

    python tools/sweep_matlab.py collect <dynare-src> <scratch>   copy tests/ and examples/, run the preprocessor
    python tools/sweep_matlab.py stats   <scratch>                classify native statements and steady-state files
    python tools/sweep_matlab.py load    <scratch>                load every model with pydyn, tabulate outcomes
    python tools/sweep_matlab.py octave  <dynare-src> <scratch> [list.json]   reference runs (Dynare under Octave)
    python tools/sweep_matlab.py compare <scratch>                compare calibration and steady state with Octave
    python tools/sweep_matlab.py refs    <scratch> <out.mat>      store Dynare's results of the matching files for the tests

<scratch> must be a directory outside the repository.  Needs DYNARE_PREPROCESSOR.
"""
import collections
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pydyn  # noqa: E402
from pydyn import matlab as M  # noqa: E402

#: statements after which Dynare's M_.params no longer is "the calibration" (or that pydyn does not model)
NOT_COMPARABLE = {"estimation", "osr", "method_of_moments", "ramsey_model", "ramsey_policy", "discretionary_policy",
                  "identification", "sensitivity", "pac_model", "var_model", "trend_component_model",
                  "var_expectation_model", "sbvar", "ms_sbvar_estimation", "heterogeneity_solve", "heterogeneity_simulate",
                  "heterogeneity_load_steady_state", "heterogeneity_compute_steady_state", "extended_path",
                  "calib_smoother", "svar_identification", "dynare_sensitivity", "prior_posterior_function",
                  "bvar_density", "bvar_forecast", "bvar_irf", "ms_sbvar_simulation", "conditional_forecast",
                  "realtime_shock_decomposition", "shock_decomposition", "matched_irfs", "estimated_params"}
EXTRA = ["examples/optimal_policy/nk_ramsey_steady_file.mod", "tests/ep/rbcii.mod",
         "tests/estimation/method_of_moments/CET/cet_imh.mod",
         "tests/estimation/method_of_moments/RBC/RBC_MoM_SMM_ME.mod",
         "tests/estimation/method_of_moments/AFVRR/AFVRR_M0.mod"]


def read_modfile(p):
    text = p.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(re.sub(r'("period2":\s*)\}(\d+)', r"\1\2", text))


# ------------------------------------------------------------------ collect
def _preprocess(mod):
    exe = os.environ["DYNARE_PREPROCESSOR"]
    try:
        r = subprocess.run([exe, mod.name, "json=compute", "onlyjson"], cwd=mod.parent, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return str(mod), "timeout"
    out = mod.parent / mod.stem / "model" / "json" / "modfile.json"
    if r.returncode != 0 or not out.exists():
        return str(mod), (r.stderr or r.stdout)[-300:]
    try:
        read_modfile(out)
    except json.JSONDecodeError as exc:
        return str(mod), f"invalid JSON: {exc}"
    return str(mod), None


def collect(src, scratch):
    corpus = scratch / "corpus"
    if not corpus.exists():
        corpus.mkdir(parents=True)
        shutil.copytree(src / "tests", corpus / "tests")
        shutil.copytree(src / "examples", corpus / "examples")
    mods = sorted(corpus.rglob("*.mod"))
    with ThreadPoolExecutor(3) as ex:
        res = dict(ex.map(_preprocess, mods))
    (scratch / "preprocess.json").write_text(json.dumps({str(Path(k).relative_to(corpus)): v for k, v in res.items()}))
    print(f"{len(res)} .mod files, {sum(v is None for v in res.values())} preprocessed")


def ok_mods(scratch):
    res = json.loads((scratch / "preprocess.json").read_text())
    return [k for k, v in sorted(res.items()) if v is None]


# ------------------------------------------------------------------ stats
CONTROL = (M.If, M.For, M.While, M.Switch, M.Try, M.Break, M.Continue, M.Return)
DYNARE_INTERNAL = {"get_param_by_name", "set_param_value", "evaluate_steady_state", "resol", "set_shock_stderr_value",
                   "stoch_simul", "simult_", "dynare_resolve", "set_all_parameters", "perfect_foresight_solver",
                   "simul_backward_model", "extended_path", "dynare_estimation_init", "initial_estimation_checks",
                   "lyapunov_symm", "get_companion_matrix", "model_inversion", "dyn_ols", "sur", "k_order_simul",
                   "backward_model_irf", "set_parameters_locally", "kalman_filter", "dsge_likelihood", "steady_",
                   "send_endogenous_variables_to_workspace", "get_mean", "make_ex_", "make_y_", "disp_th_moments"}
SIDE_EFFECT = {"load", "save", "disp", "fprintf", "plot", "figure", "set_dynare_seed", "datatomfile", "delete",
               "copyfile", "system", "addpath", "close", "clear", "subplot", "title", "legend", "hold", "error",
               "assert", "warning", "skipline", "dprintf", "tic", "toc", "profile", "pause", "rmfield", "histogram",
               "line", "axis", "box", "xlabel", "ylabel", "print", "clc", "format", "dynare", "randn", "rand"}


def classify_stmt(s):
    """One class per simple statement."""
    t = type(s)
    if t is M.Unparsed:
        return "not parsed by pydyn (dates, command syntax, objects)"
    if t is M.Command:
        return f"side-effect call: {s.name}" if s.name in SIDE_EFFECT else "other command"
    if t is M.Global:
        return "global declaration"
    names = [n.base.name for n in M.walk(s) if type(n) is M.Index and type(n.base) is M.Id] + \
            [n.name for n in M.walk(s) if type(n) is M.Id]
    roots = {n.name for n in M.walk(s) if type(n) is M.Id}
    if t is M.Assign:
        tg = [x for x in s.targets if x is not None]
        root = {M._root_name(x) for x in tg}
        if "M_" in root:
            first = next((f.name for x in tg for f in M.walk(x) if type(f) is M.Field and type(f.base) is M.Id and f.base.name == "M_"), "")
            return f"assignment to M_.{first}" if first in ("params", "Sigma_e") else "assignment to M_ (other field or whole struct)"
        if "options_" in root:
            return "assignment to options_.x"
        if "oo_" in root:
            return "assignment to oo_ (results/simulation paths)"
        if "oo_" in roots:
            return "post-processing: assignment reading oo_"
        internal = sorted(set(names) & DYNARE_INTERNAL)
        if internal:
            return f"assignment from Dynare internal: {internal[0]}"
        if any(type(x) is M.Field for x in tg):
            return "assignment to a struct field (workspace)"
        if any(type(x) is M.Index for x in tg):
            return "indexed assignment (workspace)"
        r = s.rhs
        if type(r) in (M.Matrix, M.Range):
            return "assignment of a vector/matrix literal or range"
        if type(r) in (M.AnonFunc, M.FuncHandle):
            return "assignment of a function handle"
        if type(r) is M.Str:
            return "assignment of a string"
        if type(r) is M.CellLit:
            return "assignment of a cell array"
        if type(r) is M.Num or (type(r) is M.Unop and type(r.e) is M.Num):
            return "assignment of a scalar literal"
        if "M_" in roots:
            return "assignment reading M_"
        if "load" in names:
            return "side-effect call: load"
        if any(n in ("rand", "randn") for n in names):
            return "assignment using random numbers"
        return "assignment of an expression (workspace)"
    # expression statement
    e = s.e
    head = e.base.name if type(e) is M.Index and type(e.base) is M.Id else (e.name if type(e) is M.Id else None)
    if head in DYNARE_INTERNAL:
        return f"call into Dynare internals: {head}"
    if head in SIDE_EFFECT or head in M.PLOT_FUNCS:
        return f"side-effect call: {head if head in SIDE_EFFECT else 'plot'}"
    if head is not None:
        return "other call or bare identifier (display of a variable, script, user function)"
    return "other expression statement (display)"


def stats(scratch):
    corpus = scratch / "corpus"
    cls, flow, per_file = collections.Counter(), collections.Counter(), {}
    nfiles = nnative = parse_fail = flow_with_dyn = 0
    for k in ok_mods(scratch):
        mod = corpus / k
        sts = read_modfile(mod.parent / mod.stem / "model" / "json" / "modfile.json").get("statements", [])
        lines = [st["string"] if st["statementName"] in ("native", "verbatim") else ("dyn", i) for i, st in enumerate(sts)]
        n = sum(isinstance(x, str) for x in lines)
        if not n:
            continue
        nfiles += 1
        nnative += n
        try:
            body, funcs = M.parse(lines)
        except M.ParseError:
            parse_fail += 1
            continue
        has = False

        def visit(stmts, depth):
            nonlocal has
            for s in stmts:
                t = type(s)
                if t is M.Dyn:
                    has = has or depth > 0
                elif t in CONTROL:
                    flow[t.__name__.lower()] += 1
                    for f in ("body", "orelse", "handler", "default"):
                        if getattr(s, f, None):
                            visit(getattr(s, f), depth + 1)
                    for _, b in (getattr(s, "clauses", None) or []) + (getattr(s, "cases", None) or []):
                        visit(b, depth + 1)
                else:
                    cls[classify_stmt(s)] += 1
        visit(body, 0)
        flow_with_dyn += has
    print(f"files with native/verbatim lines: {nfiles}; native lines: {nnative}; block structure not parseable: {parse_fail}")
    print(f"files with Dynare statements inside MATLAB control flow: {flow_with_dyn}")
    print("control-flow statements:", dict(flow.most_common()))
    merged = collections.Counter()
    for c, v in cls.items():
        merged[c] += v
    for c, v in merged.most_common():
        print(f"{v:6d}  {c}")
    # steady-state files
    print("\nsteady-state files and helpers:")
    for p in sorted(corpus.rglob("*.m")):
        if not re.search(r"_steadystate\.m$|steady.*helper\.m$", p.name):
            continue
        try:
            body, funcs = M.parse(p.read_text(errors="replace"))
        except M.ParseError as exc:
            print(f"  {p.relative_to(corpus)}: PARSE ERROR {exc}")
            continue
        nodes = [n for f in funcs.values() for n in M.walk(f.body)]
        calls = {n.base.name for n in nodes if type(n) is M.Index and type(n.base) is M.Id}
        main = next(iter(funcs.values()))
        feats = []
        if any(type(n) is M.For and any(type(x) is M.Index and type(x.base) is M.Id and x.base.name == "eval" for x in M.walk(n.body)) for n in nodes):
            feats.append("eval loops over M_.param_names")
        if calls & {"fsolve", "fzero", "csolve"}:
            feats.append("/".join(sorted(calls & {"fsolve", "fzero", "csolve"})))
        if len(funcs) > 1:
            feats.append(f"{len(funcs) - 1} local function(s)")
        if any(type(n) is M.Return for n in nodes):
            feats.append("early return")
        if any(type(n) in (M.While, M.For) for n in nodes) and "eval loops over M_.param_names" not in feats:
            feats.append("loops")
        if any(type(n) is M.Field and type(n.base) is M.Id and n.base.name == "options_" for n in nodes):
            feats.append("reads options_")
        print(f"  {p.relative_to(corpus)}: {len(main.ins)} in / {len(main.outs)} out; {', '.join(feats) or 'pure assignments'}")


# ------------------------------------------------------------------ load
class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout


def load_one(args):
    scratch, k, isoct = args
    M.ISOCTAVE = isoct
    mod = Path(scratch) / "corpus" / k
    out = {"name": k}
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(120)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = pydyn.Model(mod.parent / mod.stem / "model" / "json")
        out.update(status="loaded", params=m.params.tolist(), param_names=m.param_names, Sigma_e=m.Sigma_e.tolist(),
                   skipped=len(m.matlab_skipped), skipped_kinds=dict(collections.Counter(x[1] for x in m.matlab_skipped)),
                   kinds=sorted({st["statementName"] for st in m.statements}),
                   workspace={a: float(b) for a, b in (m._matlab.sc.vars.items() if m._matlab else m.workspace.items())
                              if isinstance(b, (float, np.floating, np.bool_))},
                   ws_arrays={a: np.asarray(b, float).tolist() for a, b in (m._matlab.sc.vars.items() if m._matlab else [])
                              if isinstance(b, np.ndarray) and b.size <= 400})
        try:
            ys, p = pydyn.steady_state(m)
            out.update(ys=ys.tolist(), ss_params=p.tolist(), endo_names=m.endo_names)
        except _Timeout:
            raise
        except Exception as exc:  # noqa: BLE001
            out["ss_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    except pydyn.ModelError as exc:
        out.update(status="refused", detail=str(exc)[:300])
    except _Timeout:
        out.update(status="timeout")
    except NotImplementedError as exc:
        out.update(status="model not supported", detail=str(exc)[:200])
    except Exception as exc:  # noqa: BLE001
        import traceback
        tb = traceback.extract_tb(exc.__traceback__)
        # failures while compiling the model equations happen before any statement is run
        early = any(f.name in ("_compile", "_build", "__init__") and f.filename.endswith("model.py") for f in tb[-6:]) \
            and not any(f.filename.endswith("pydyn/matlab.py") for f in tb)
        out.update(status="model not supported" if early else "crash", detail=f"{type(exc).__name__}: {str(exc)[:300]}")
    finally:
        signal.alarm(0)
    return out


def load(scratch, isoct=True, names=None):
    names = names or ok_mods(scratch)
    with ProcessPoolExecutor(2) as ex:
        rows = list(ex.map(load_one, [(str(scratch), k, isoct) for k in names], chunksize=4))
    (scratch / "load.json").write_text(json.dumps(rows))
    c = collections.Counter(r["status"] for r in rows)
    print(dict(c))
    return rows


# ------------------------------------------------------------------ octave
OCT_TEMPLATE = """addpath %(src)s/matlab
addpath %(stubs)s
warning('off','all');
cd('%(dir)s');
try
  dynare %(name)s nograph
  global M_ oo_ options_
  matref__ = struct();
  matref__.params = M_.params; matref__.ys = oo_.steady_state; matref__.Sigma_e = M_.Sigma_e; matref__.exo_ss = oo_.exo_steady_state;
  matref__.param_names = strjoin(cellstr(M_.param_names)',',');
  matref__.endo_names = strjoin(cellstr(M_.endo_names)',',');
  matref_ws__ = struct(); matref_who__ = who;
  for matref_i__ = 1:numel(matref_who__)
    matref_n__ = matref_who__{matref_i__};
    if any(strcmp(matref_n__, {'M_','oo_','options_','ans'})) || ~isempty(strfind(matref_n__, 'matref_')), continue, end
    matref_v__ = eval(matref_n__);
    if (isnumeric(matref_v__) || islogical(matref_v__)) && isreal(matref_v__) && ~issparse(matref_v__) && numel(matref_v__) <= 400 && ndims(matref_v__) <= 2
      matref_ws__.(matref_n__) = double(matref_v__);
    end
  end
  matref__.ws = matref_ws__;
  save('-mat7-binary', '%(out)s', '-struct', 'matref__');
  disp('MATREF_OK');
catch matref_e__
  disp(['MATREF_FAIL ' matref_e__.message]);
end
"""
PLOT_STUBS = ("figure plot subplot hold title legend xlabel ylabel zlabel axis close saveas drawnow line bar histogram hist "
              "grid box plot3 surf mesh colorbar suptitle sgtitle xlim ylim annotation orient hgsave savefig fill area "
              "scatter stairs semilogy semilogx loglog clf pause").split()


def candidates(scratch):
    corpus = scratch / "corpus"
    out = []
    for k in ok_mods(scratch):
        mod = corpus / k
        sts = read_modfile(mod.parent / mod.stem / "model" / "json" / "modfile.json").get("statements", [])
        kinds = {s["statementName"] for s in sts}
        if kinds & NOT_COMPARABLE:
            continue
        if kinds & {"native", "verbatim"} or (mod.parent / f"{mod.stem}_steadystate.m").exists():
            out.append(k)
    return out + EXTRA


def octave(src, scratch, names=None, budget=500, per_file=75):
    """Run Dynare under Octave in a copy of the corpus; plotting functions are replaced by no-ops."""
    stubs, work, ref, scr = scratch / "stubs", scratch / "corpus_oct", scratch / "oct_ref", scratch / "oct_scripts"
    for d in (stubs, ref, scr):
        d.mkdir(exist_ok=True)
    for f in PLOT_STUBS:
        (stubs / f"{f}.m").write_text(f"function varargout = {f}(varargin)\n  varargout = cell(1, nargout);\nend\n")
    if not work.exists():
        shutil.copytree(scratch / "corpus", work)
    groups = collections.defaultdict(list)
    for k in names or candidates(scratch):
        groups[str(Path(k).parent)].append(k)
    t0 = time.time()

    def rungroup(ks):
        for k in ks:
            key = k.replace("/", "__")[:-4]
            log = ref / (key + ".log")
            if log.exists() or time.time() - t0 > budget:
                continue
            s = scr / ("r_" + re.sub(r"\W", "_", key) + ".m")
            s.write_text(OCT_TEMPLATE % dict(src=src, stubs=stubs, dir=str(work / Path(k).parent), name=Path(k).stem,
                                             out=str(ref / (key + ".mat"))))
            t = time.time()
            try:
                r = subprocess.run(["octave", "--no-gui", "--quiet", str(s)], capture_output=True, text=True, timeout=per_file,
                                   env={**os.environ, "GLIBC_TUNABLES": "glibc.rtld.execstack=2"}, cwd=str(scr))
                txt = r.stdout[-3000:] + "\nSTDERR\n" + r.stderr[-1500:]
            except subprocess.TimeoutExpired:
                txt = "MATREF_TIMEOUT"
            log.write_text(txt + f"\nELAPSED {time.time() - t:.1f}\n")
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(rungroup, groups.values()))


# ------------------------------------------------------------------ compare
def _close(a, b, rtol, atol=0.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all((np.abs(a - b) <= atol + rtol * np.abs(b)) | (np.isnan(a) & np.isnan(b))))


def compare(scratch, rtol=1e-10):
    from scipy.io import loadmat
    ref_dir = scratch / "oct_ref"
    names = sorted(p.stem.replace("__", "/") + ".mod" for p in ref_dir.glob("*.mat"))
    rows = {r["name"]: r for r in load(scratch, True, names)}
    table, counts, details = [], collections.Counter(), {}
    for k in names:
        ref = loadmat(ref_dir / (k[:-4].replace("/", "__") + ".mat"), squeeze_me=True, struct_as_record=False)
        r = rows[k]
        rp = np.atleast_1d(np.asarray(ref["params"], float))
        note = []
        if r["status"] != "loaded":
            verdict = {"refused": "refused (ModelError)", "model not supported": "model not supported by pydyn",
                       "timeout": "timeout", "crash": "CRASH"}[r["status"]]
            note.append(r.get("detail", "")[:150])
        else:
            p = np.asarray(r["params"], float)
            ok = _close(p, rp, rtol)
            if not ok and "ss_params" in r and _close(r["ss_params"], rp, rtol):
                ok = True
                note.append("parameters after the steady-state computation")
            if not ok and "ss_params" not in r and p.shape == rp.shape:
                # the steady state (which updates parameters) is outside pydyn's scope for this model: compare the rest
                upd = (np.abs(p - rp) > rtol * np.abs(rp)) | (np.isnan(p) != np.isnan(rp))
                if r.get("ss_updates") is not None and set(np.flatnonzero(upd)) <= set(r["ss_updates"]):
                    ok = True
                    note.append("except parameters set by the steady state, which pydyn cannot compute here")
            sig = _close(r["Sigma_e"], np.atleast_2d(np.asarray(ref["Sigma_e"], float)).reshape(np.shape(r["Sigma_e"])), rtol) \
                if np.size(ref["Sigma_e"]) == np.size(r["Sigma_e"]) else False
            ws = ref["ws"]
            nws = bad_ws = 0
            for name in getattr(ws, "_fieldnames", []):
                mine = r["workspace"].get(name, r["ws_arrays"].get(name))
                if mine is None:
                    continue
                nws += 1
                theirs = np.asarray(getattr(ws, name), float)
                if not _close(np.asarray(mine, float).reshape(theirs.shape) if np.size(mine) == theirs.size else mine, theirs, rtol):
                    bad_ws += 1
                    note.append(f"workspace variable {name} differs")
            note.append(f"{nws} workspace variables checked")
            if ok and sig and not bad_ws:
                verdict = "identical calibration"
            else:
                verdict = "WRONG"
                if not ok:
                    d = [n for n, a, b in zip(r["param_names"], p, rp) if not _close(a, b, rtol)]
                    note.append(f"params differ: {d[:6]}")
                if not sig:
                    note.append("Sigma_e differs")
            comparable = "steady" in r["kinds"] and "endval" not in r["kinds"]
            if "ys" in r and not comparable:
                note.append("steady state not compared (no 'steady' command, or endval: oo_.steady_state is not the initial steady state)")
            elif "ys" in r:
                ry = np.atleast_1d(np.asarray(ref["ys"], float))
                ys = np.asarray(r["ys"], float)
                if ry.shape == ys.shape and np.all(np.isfinite(ry)):
                    d = float(np.max(np.abs(ys - ry) / np.maximum(1.0, np.abs(ry)))) if ys.size else 0.0
                    note.append(f"steady state max rel diff {d:.1e}")
                    r["ys_diff"] = d
            elif "ss_error" in r:
                note.append("steady state: " + r["ss_error"][:90])
            if r["skipped"]:
                note.append(f"{r['skipped']} MATLAB statements skipped")
        counts[verdict] += 1
        table.append((k, verdict, "; ".join(note)))
        details[k] = dict(verdict=verdict, after_steady=any("after the steady-state" in n for n in note),
                          ys_diff=r.get("ys_diff"))
    for row in table:
        print(" | ".join(row))
    print(dict(counts))
    (scratch / "compare.json").write_text(json.dumps(table))
    (scratch / "compare_details.json").write_text(json.dumps(details))
    return table


def refs(scratch, out):
    """Write the Dynare results of the files with identical calibration to one small MAT-file for the tests."""
    from scipy.io import loadmat, savemat
    details = json.loads((scratch / "compare_details.json").read_text())
    files, store = [], {}
    for k, d in sorted(details.items()):
        if d["verdict"] != "identical calibration":
            continue
        ref = loadmat(scratch / "oct_ref" / (k[:-4].replace("/", "__") + ".mat"), squeeze_me=True)
        params, ys = np.atleast_1d(np.asarray(ref["params"], float)), np.atleast_1d(np.asarray(ref["ys"], float))
        if params.size > 400 or ys.size > 400:
            continue
        diff = d["ys_diff"]
        # steady states found numerically agree with Dynare only up to Dynare's own tolerance on the
        # static residuals (solve_tolf = eps^(1/3)); analytical ones agree to rounding
        check_ys = diff is not None and diff < 1e-5
        store[f"f{len(files)}"] = dict(params=params, Sigma_e=np.atleast_2d(np.asarray(ref["Sigma_e"], float)), ys=ys,
                                       after_steady=int(d["after_steady"]), check_ys=int(check_ys),
                                       ys_rtol=1e-10 if (diff or 0) <= 1e-11 else 1e-5)
        files.append(k)
    savemat(out, {"files": np.array(files, dtype=object), "refs": store}, do_compression=True)
    print(f"{len(files)} files -> {out} ({Path(out).stat().st_size} bytes)")


def mutators(src):
    """Scan Dynare's matlab/ tree for functions that change the calibration behind the caller's back:
    functions that declare ``global M_`` or return M_, and assign M_.params, M_.Sigma_e, M_.H,
    M_.Correlation_matrix(_ME) or M_.Skew_e or replace M_ as a whole; then the functions calling one
    of the global writers (transitively, matched as ``name(``).  pydyn.matlab.MUTATORS and
    MUTATOR_DOTTED must contain every name printed (set_param_value and set_shock_stderr_value are
    implemented; steady and stoch_simul are handled as SOLVE_FUNCS)."""
    codes = {}
    for root, _, files in os.walk(src / "matlab"):
        for f in files:
            if f.endswith(".m"):
                pk = [x[1:] for x in Path(root).parts if x.startswith("+")]
                codes[".".join(pk + [f[:-2]])] = re.sub(r"%.*", "", (Path(root) / f).read_text(errors="replace"))
    direct, global_writers = {}, set()
    for name, code in codes.items():
        writes = re.search(r"\bM_\.(params|Sigma_e|H|Correlation_matrix|Correlation_matrix_ME|Skew_e)\b\s*(\([^=\n]*\))?\s*=(?!=)", code)
        whole = re.search(r"^\s*M_\s*=(?!=)|\[[^\]\n]*\bM_\b(?!\.)[^\]\n]*\]\s*=(?!=)", code, re.M)
        head = re.search(r"^\s*function\s+(?:(\[[^\]]*\]|\w+)\s*=\s*)?([\w.]+)", code, re.M)
        if not (writes or whole) or not head:
            continue
        has_global = bool(re.search(r"^\s*global\b[^\n]*\bM_\b", code, re.M))
        returns = bool(re.search(r"\bM_\b", head.group(1) or ""))
        if has_global or returns:
            direct[name] = "global" if has_global else "returns M_"
            if has_global:
                global_writers.add(name)
    closure = set(global_writers) - {"steady", "prior", "global_initialization", "gui.stochastic-simulation.read"}
    changed = True
    while changed:
        changed = False
        for name, code in codes.items():
            if name not in closure and any(re.search(r"(?<![\w.])" + re.escape(w) + r"\s*\(", code) for w in closure):
                closure.add(name)
                changed = True
    names = sorted(set(direct) | closure)
    known = M.MUTATORS | M.MUTATOR_DOTTED | M.SOLVE_FUNCS | {"set_param_value", "set_shock_stderr_value", "steady", "gui.stochastic-simulation.read"}
    for n in names:
        print(f"{n:45s} {direct.get(n, 'calls a global writer'):22s} {'' if n in known else 'MISSING from pydyn.matlab'}")
    print(f"{len(names)} functions, {sum(n not in known for n in names)} missing")
    return [n for n in names if n not in known]


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "collect":
        collect(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "stats":
        stats(Path(sys.argv[2]))
    elif cmd == "load":
        rows = load(Path(sys.argv[2]), isoct=False)
        for r in rows:
            if r["status"] != "loaded":
                print(f"{r['name']} | {r['status']} | {r.get('detail', '')[:200]}")
    elif cmd == "octave":
        octave(Path(sys.argv[2]), Path(sys.argv[3]), json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else None)
    elif cmd == "compare":
        compare(Path(sys.argv[2]))
    elif cmd == "mutators":
        mutators(Path(sys.argv[2]))
    elif cmd == "refs":
        refs(Path(sys.argv[2]), Path(sys.argv[3]))
