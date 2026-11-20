"""Run the Dynare preprocessor and locate its JSON output."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def find_preprocessor(explicit: str | None = None) -> str:
    for cand in (explicit, os.environ.get("DYNARE_PREPROCESSOR"), shutil.which("dynare-preprocessor")):
        if cand and Path(cand).exists():
            return str(cand)
    raise FileNotFoundError(
        "dynare-preprocessor not found; pass its path or set DYNARE_PREPROCESSOR"
    )


def run_preprocessor(mod_file: str | os.PathLike, preprocessor: str | None = None,
                     extra_args: tuple[str, ...] = ()) -> Path:
    """Run the preprocessor on ``mod_file`` and return the directory holding the JSON files."""
    mod_file = Path(mod_file).resolve()
    exe = find_preprocessor(preprocessor)
    cmd = [exe, mod_file.name, "json=compute", "onlyjson", *extra_args]
    res = subprocess.run(cmd, cwd=mod_file.parent, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"preprocessor failed:\n{res.stdout}\n{res.stderr}")
    out = mod_file.parent / mod_file.stem / "model" / "json"
    if not (out / "modfile.json").exists():
        raise RuntimeError(f"no JSON output in {out}")
    return out
