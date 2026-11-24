# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Acceptance sweep of pydyn against Dynare 7.2's own test suite = tools/accept_pydyn.py (run) + tools/accept_compare.py (judge).

    python tools/accept_sweep.py run     [accept_pydyn options: --jobs 2 --only substr --force --timeout 600 --out DIR]
    python tools/accept_sweep.py report  [--out DIR] [--ref DIR]      # compares and writes docs/ACCEPTANCE.md
    python tools/accept_sweep.py all     [...]                         # run, then report

The references are those of tools/accept_octave.py (default /home/claude/accept_ref); pydyn's results go to /home/claude/accept_pydyn
(``<id>.mat`` + ``<id>.json``, the format every port writes: see accept_compare.py).
"""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))


def run_tests(only, out="/home/claude/accept_pydyn", ref="/home/claude/accept_ref", timeout=600, force=True):
    """Run and judge the tests whose id contains one of ``only``; returns {id: verdict dict} (used by tests/test_acceptance.py)."""
    import accept_compare
    import accept_pydyn
    args = ["--out", str(out), "--ref", str(ref), "--jobs", "1", "--timeout", str(timeout), "--only", *[o + ".mod" for o in only]]
    accept_pydyn.main(args + (["--force"] if force else []))
    res = accept_compare.main([str(out), "--ref", str(ref), "--only", *only, "--name", "pydyn"])
    return {r["file"]: r for r in res}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    rest = sys.argv[2:]
    out = rest[rest.index("--out") + 1] if "--out" in rest else "/home/claude/accept_pydyn"
    ref = rest[rest.index("--ref") + 1] if "--ref" in rest else "/home/claude/accept_ref"
    if cmd in ("run", "all"):
        import accept_pydyn
        accept_pydyn.main(rest)
    if cmd in ("report", "all"):
        import accept_compare
        accept_compare.main([out, "--ref", ref, "--name", "pydyn", "--report", str(TOOLS.parent / "docs" / "ACCEPTANCE.md")])


if __name__ == "__main__":
    main()
