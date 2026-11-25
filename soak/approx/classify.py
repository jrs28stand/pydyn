"""Class of every OccBin model that can be loaded (step 1a of the approx-class study)."""
import sys, shutil, tempfile, warnings, subprocess
from pathlib import Path
sys.path.insert(0, "soak")
import lib, pydyn
from pydyn.occbin import Occbin
from pydyn.obc import constraint_objects
roots = ["reference", "/home/claude/ref/dynare-7.2/examples", "/home/claude/ref/dynare-7.2/tests"]
files = subprocess.run("grep -rl occbin_constraints " + " ".join(roots) + " --include=*.mod", shell=True, capture_output=True, text=True).stdout.split()
for f in sorted(files):
    f = Path(f); d = Path(tempfile.mkdtemp())
    for g in f.parent.iterdir():
        if g.is_file() and g.suffix in (".mod", ".inc", ".mat", ".m", ".csv"): shutil.copy(g, d)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = pydyn.Model(pydyn.run_preprocessor(d / f.name)); ob = Occbin(m)
            cs = constraint_objects(ob)
        print(f"{str(f).replace('/home/claude/ref/dynare-7.2/','')}: " + "; ".join(f"{c.cls}" + (f" approx={c.approx:.4g}" if c.approx else "") for c in cs))
    except Exception as e:
        print(f"{f.name}: {type(e).__name__}: {str(e)[:90]}")
