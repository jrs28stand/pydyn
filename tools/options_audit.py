# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Ground truth for the options audit: every option the Dynare preprocessor can emit per statement, read from its grammar.

    python tools/options_audit.py [--src /home/claude/dl/src72] [--json out.json]

Parses preprocessor/src/DynareBison.yy: the ``o_<name>`` rules (each sets one or more option keys through
``driver.option_*("key", ...)``), the ``*_options`` lists that combine them, and the statements that take such a list in
parentheses.  Output: {statement rule name: sorted option keys}.  The keys are the ones the preprocessor writes into the JSON
(``options`` of the statement), e.g. ``mh_replic`` or ``nodiagnostic``."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def strip_actions(text):
    out, depth = [], 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def parse(src):
    text = (Path(src) / "preprocessor/src/DynareBison.yy").read_text()
    body = text.split("\n%%\n", 2)[1]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    rules = {}                               # name -> raw text (alternatives, with actions)
    cur, depth, i = [], 0, 0
    chunks = []
    while i < len(body):
        ch = body[i]
        if ch in "\"'":                       # a string or character literal (token such as ';' or text in an action)
            j = i + 1
            while j < len(body) and body[j] != ch:
                j += 2 if body[j] == "\\" else 1
            cur.append(body[i:j + 1])
            i = j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == ";" and depth == 0:
            chunks.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    pieces = []
    for c in chunks:                          # a rule may lack its closing ';': also split at headers at line start
        pieces.extend(re.split(r"\n(?=[a-z_0-9]+\s*:)", c))
    for c in pieces:
        m = re.match(r"\s*([a-z_0-9]+)\s*:(.*)$", c, re.S)
        if m:
            rules[m.group(1)] = m.group(2)
    keys = {}
    for name, raw in rules.items():
        ks = re.findall(r'driver\.option_\w+\(\s*"([^"]+)"', raw)
        if name == "o_graph_format":
            ks.append("graph_format")
        if ks:
            keys[name] = ks
    alts = {}
    for name, raw in rules.items():
        s = strip_actions(raw)
        alts[name] = [a.split() for a in s.split("|")]
    return rules, keys, alts


def expand(name, keys, alts, seen=None):
    """Option keys reachable from ``name`` through option-ish nonterminals (``o_*`` and names containing 'option')."""
    seen = seen if seen is not None else set()
    if name in seen:
        return set()
    seen.add(name)
    out = set(keys.get(name, [])) if name.startswith("o_") else set()
    for alt in alts.get(name, []):
        for sym in alt:
            if sym in alts and sym != name and (sym.startswith("o_") or "option" in sym):
                out |= expand(sym, keys, alts, seen)
    return out


#: grammar rule name -> statementName of the JSON, where they differ
JSON_NAME = {"smoother2histval": "smoother_2_histval", "occbin_write_regimes": "occbin_write_regimes_xls",
             "prior_function": "prior_posterior_function", "posterior_function": "prior_posterior_function"}


def statements(src):
    rules, keys, alts = parse(src)
    res = {}
    for name, a in alts.items():
        if name.startswith("o_") or "option" in name:
            continue
        syms = {sym for alt in a for sym in alt if sym in alts and (sym.startswith("o_") or "option" in sym)}
        if syms and any("(" in "".join(alt) or "'('" in alt for alt in a):
            ks = set()
            for sy in syms:
                ks |= expand(sy, keys, alts)
            res[name] = sorted(ks)
    out = {}
    for k, v in res.items():
        out.setdefault(JSON_NAME.get(k, k), set()).update(v)
    return {k: sorted(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/claude/dl/src72")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = statements(a.src)
    for k, v in sorted(res.items()):
        print(f"{k}: {len(v)}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
