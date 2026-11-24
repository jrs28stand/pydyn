# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Read the Dynare test list (``mod_and_m_tests`` in Dynare's meson.build) into a list of dicts.

Each entry: {"test": [paths relative to tests/], "extra": [...], "should_fail": bool, "dynare_args": [...], "env": {...}}.
Paths starting with ../ refer to the examples folder.  ``python tools/accept_manifest.py <dynare-src>`` prints a summary.
"""
import ast
import re
import sys
from pathlib import Path


def load_manifest(src):
    text = (Path(src) / "meson.build").read_text()
    start = text.index("mod_and_m_tests = [")
    i = text.index("[", start)
    depth, j = 0, i
    in_str = None
    while True:                                   # find the matching bracket, ignoring strings and comments
        c = text[j]
        if in_str:
            if c == in_str:
                in_str = None
        elif c == "'":
            in_str = c
        elif c == "#":
            j = text.index("\n", j)
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                break
        j += 1
    body = text[i:j + 1]
    body = re.sub(r"(?m)^\s*#.*$", "", body)                      # whole-line comments
    body = re.sub(r"'env'\s*:\s*\{.*?\}\s*\}", "'env' : {} }", body, flags=re.S)           # build-option dependent
    body = re.sub(r"\btrue\b", "True", body)
    body = re.sub(r"\bfalse\b", "False", body)
    return ast.literal_eval(body)


def entry_id(path):
    return path[:-4] if path.endswith(".mod") else path


if __name__ == "__main__":
    ms = load_manifest(sys.argv[1])
    mods = [f for e in ms for f in e["test"] if f.endswith(".mod")]
    print(len(ms), "entries;", len(mods), ".mod files;", sum(f.endswith(".m") for e in ms for f in e["test"]), ".m runners;",
          sum(bool(e.get("should_fail")) for e in ms), "xfail entries")
