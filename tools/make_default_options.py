# This file is part of pydyn.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see the
# GNU General Public License (file LICENSE) for details.
"""Writes pydyn/data/default_options.json: Dynare's ``options_`` as the driver leaves it for a model
without commands (the defaults of matlab/default_option_values.m), taken from a Dynare 7.2 run.

    octave: dynare e nograph noclearall; save('-mat7-binary','opt.mat','options_')   (e.mod: a one-line model)
    python tools/make_default_options.py opt.mat pydyn/data/default_options.json

Encoding (see pydyn.results): a JSON object is a struct, a number/bool/string is that scalar, and
{"$": "d"|"b"|"c", "shape": [r, c], "data": [...column-major...]} an array (double, logical, cell)."""
import json
import sys

import numpy as np
from scipy.io import loadmat


def enc(v):
    if isinstance(v, np.ndarray) and v.dtype.names:
        if v.size != 1:
            return {"$": "x", "shape": list(v.shape)}        # struct arrays are not represented
        return {n: enc(v[n][0, 0]) for n in v.dtype.names}
    if isinstance(v, np.ndarray) and v.dtype == object:
        return {"$": "c", "shape": list(v.shape), "data": [enc(x) for x in v.ravel(order="F")]}
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "U":
            s = "".join(v.ravel())
            return {"$": "s0"} if v.size == 0 else s
        shape = list(v.shape) if v.ndim == 2 else [1, v.size]
        if v.dtype.kind in "ub":
            if v.size == 1:
                return bool(v.ravel()[0])
            return {"$": "b", "shape": shape, "data": [bool(x) for x in v.ravel(order="F")]}
        a = np.asarray(v, float)
        if a.size == 1:
            x = float(a.ravel()[0])
            return x if np.isfinite(x) else {"$": "d", "shape": [1, 1], "data": [repr(x)]}
        return {"$": "d", "shape": shape, "data": [x if np.isfinite(x) else repr(float(x)) for x in a.ravel(order="F").tolist()]}
    raise TypeError(type(v))


def main(src, dst):
    d = loadmat(src, struct_as_record=True)
    out = enc(d["options_"])
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=0, separators=(",", ":"))
    print("written", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
