# This file is part of pydyn. OctaveRandomStream reproduces the uniform and normal generators of GNU
# Octave (liboctave/numeric/randmtzig.cc and oct-rand.cc: Mersenne Twister MT19937 of M. Matsumoto
# and T. Nishimura, BSD-3-Clause, with the Ziggurat method of Marsaglia and Tsang as implemented by
# D. Bateman; Copyright (C) 2006-2024 The Octave Project Developers, GPL-3.0-or-later), rewritten in
# Python/numpy in 2026.
# Released under the GNU General Public License, version 3 or later; see LICENSE. No warranty.
"""Random-number streams for the particle filters.

Dynare's nonlinear filters draw with MATLAB/Octave's ``randn(r, c)`` and ``rand(r, c)``. A stream
here offers the same two calls (``randn(r, c)``, ``rand(r, c)``), filling arrays in column-major
order as MATLAB does, so that a filter consumes numbers in exactly Dynare's order.

* :class:`OctaveRandomStream` gives the numbers Octave gives after ``rand('state', seed)`` and
  ``randn('state', seed)``: Dynare's filters call ``set_dynare_seed_local_options([], false,
  'default')`` (seed 0) at the start of each likelihood evaluation, so with this stream a pydyn
  likelihood equals Dynare's under Octave bit for bit (MATLAB's ``RandStream`` gives other numbers).
  As in Octave, uniform and normal numbers come from two independent generators.
* :class:`NumpyRandomStream` uses numpy's PCG64 (any seed; faster, not Dynare's numbers).
* :class:`ReplayStream` replays recorded numbers (e.g. captured from MATLAB) and fails if more are
  requested than were recorded.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["OctaveRandomStream", "NumpyRandomStream", "ReplayStream", "make_stream"]

_MASK = 0xFFFFFFFF


def _init_by_array(key) -> np.ndarray:
    """MT19937 state (624 words) of ``init_by_array(key)`` (Matsumoto and Nishimura, 2002)."""
    mt = [0] * 624
    mt[0] = 19650218
    for i in range(1, 624):
        mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & _MASK
    key = [int(k) & _MASK for k in key]
    i, j = 1, 0
    for _ in range(max(624, len(key))):
        mt[i] = ((mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525)) + key[j] + j) & _MASK
        i += 1
        j += 1
        if i >= 624:
            mt[0] = mt[623]
            i = 1
        if j >= len(key):
            j = 0
    for _ in range(623):
        mt[i] = ((mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941)) - i) & _MASK
        i += 1
        if i >= 624:
            mt[0] = mt[623]
            i = 1
    mt[0] = 0x80000000
    return np.array(mt, dtype=np.uint32)


def _double2uint32(d) -> int:
    """Octave's conversion of a seed value to a 32-bit word (oct-rand.cc)."""
    if not math.isfinite(d):
        return 0
    top = float(_MASK)
    d = math.fmod(float(d), top)
    if d < 0:
        d += top
    return int(d)


class _MTWords:
    """Stream of 32-bit words of an MT19937 generator, read through a buffer."""

    def __init__(self, seed_key):
        self.bg = np.random.MT19937(0)
        self.bg.state = {"bit_generator": "MT19937",
                         "state": {"key": _init_by_array(seed_key), "pos": 624}}
        self.buf = np.zeros(0, dtype=np.uint64)
        self.pos = 0

    def take(self, k):
        if self.pos + k > len(self.buf):
            need = self.pos + k - len(self.buf)
            fresh = self.bg.random_raw(max(need, 65536))
            self.buf = np.concatenate([self.buf[self.pos:], fresh])
            self.pos = 0
        out = self.buf[self.pos:self.pos + k]
        self.pos += k
        return out

    def give_back(self, k):
        self.pos -= k


# ---- Ziggurat tables of randmtzig.cc (normal distribution)
_ZR = 3.6541528853610088
_ZIR = 0.27366123732975828
_AREA = 0.00492867323399
_NM = 9007199254740992.0           # 2^53


def _ziggurat_tables():
    ki = [0] * 256
    wi = [0.0] * 256
    fi = [0.0] * 256
    x1 = _ZR
    wi[255] = x1 / _NM
    fi[255] = math.exp(-0.5 * x1 * x1)
    ki[0] = int(x1 * fi[255] / _AREA * _NM)
    wi[0] = _AREA / fi[255] / _NM
    fi[0] = 1.0
    for i in range(254, 0, -1):
        x = math.sqrt(-2.0 * math.log(_AREA / x1 + fi[i + 1]))
        ki[i + 1] = int(x / x1 * _NM)
        wi[i] = x / _NM
        fi[i] = math.exp(-0.5 * x * x)
        x1 = x
    ki[1] = 0
    return np.array(ki, dtype=np.int64), np.array(wi), np.array(fi)


_KI, _WI, _FI = _ziggurat_tables()


class OctaveRandomStream:
    """The numbers of Octave's ``rand`` and ``randn`` after ``rand('state', seed)`` and
    ``randn('state', seed)`` (two independent Mersenne Twisters; ``rand`` is ``randu53``, ``randn``
    the Ziggurat method of randmtzig.cc)."""

    def __init__(self, seed=0):
        key = [_double2uint32(seed)]
        self._u = _MTWords(key)
        self._n = _MTWords(key)

    # -- uniform
    def _uniform(self, k):
        out = np.empty(k)
        got = 0
        while got < k:
            m = k - got
            w = self._u.take(2 * m)
            a = (w[0::2] >> np.uint64(5)).astype(np.float64)
            b = (w[1::2] >> np.uint64(6)).astype(np.float64)
            bad = (a == 0) & (b == 0)
            if bad.any():                       # randu53 rejects a zero (probability 2^-53)
                f = int(np.argmax(bad))
                self._u.give_back(2 * (m - f) - 2)
                out[got:got + f] = (a[:f] * 67108864.0 + b[:f]) / _NM
                got += f
                continue
            out[got:] = (a * 67108864.0 + b) / _NM
            got = k
        return out

    def _randu53_scalar(self, words):
        while True:
            w = words.take(2)
            a, b = int(w[0]) >> 5, int(w[1]) >> 6
            if a or b:
                return (a * 67108864.0 + b) / _NM

    # -- normal
    def _normal_scalar(self):
        words = self._n
        while True:
            w = words.take(2)
            r = ((int(w[1]) & 0x3FFFFF) << 32) | int(w[0])
            rabs = r >> 1
            idx = rabs & 0xFF
            x = (-rabs if (r & 1) else rabs) * _WI[idx]
            if rabs < _KI[idx]:
                return x
            if idx == 0:
                while True:
                    xx = -_ZIR * math.log(self._randu53_scalar(words))
                    yy = -math.log(self._randu53_scalar(words))
                    if yy + yy > xx * xx:
                        break
                return (-_ZR - xx) if (rabs & 0x100) else (_ZR + xx)
            if (_FI[idx - 1] - _FI[idx]) * self._randu53_scalar(words) + _FI[idx] < math.exp(-0.5 * x * x):
                return x

    def _normal(self, k):
        out = np.empty(k)
        got = 0
        while got < k:
            m = min(k - got, 4096)
            w = self._n.take(2 * m)
            lo = w[0::2]
            hi = w[1::2] & np.uint64(0x3FFFFF)
            r = (hi << np.uint64(32)) | lo
            rabs = (r >> np.uint64(1)).astype(np.int64)
            idx = rabs & 0xFF
            x = np.where((r & np.uint64(1)) == 1, -rabs, rabs).astype(np.float64) * _WI[idx]
            ok = rabs < _KI[idx]
            if ok.all():
                out[got:got + m] = x
                got += m
                continue
            f = int(np.argmin(ok))               # first number that needs the slow path
            out[got:got + f] = x[:f]
            got += f
            self._n.give_back(2 * (m - f))
            out[got] = self._normal_scalar()
            got += 1
        return out

    def randn(self, r, c=1):
        return self._normal(int(r) * int(c)).reshape((int(r), int(c)), order="F")

    def rand(self, r, c=1):
        return self._uniform(int(r) * int(c)).reshape((int(r), int(c)), order="F")


class NumpyRandomStream:
    """numpy's PCG64 generators (one for uniform, one for normal numbers), column-major fill."""

    def __init__(self, seed=0):
        su, sn = np.random.SeedSequence(seed).spawn(2)
        self._u = np.random.default_rng(su)
        self._n = np.random.default_rng(sn)

    def randn(self, r, c=1):
        return self._n.standard_normal((int(c), int(r))).T

    def rand(self, r, c=1):
        return self._u.random((int(c), int(r))).T


class ReplayStream:
    """Replays recorded normal and uniform numbers in order (column-major fill)."""

    def __init__(self, normals=(), uniforms=()):
        self.normals = np.asarray(normals, float).ravel(order="F")
        self.uniforms = np.asarray(uniforms, float).ravel(order="F")
        self.i_n = self.i_u = 0

    def randn(self, r, c=1):
        k = int(r) * int(c)
        if self.i_n + k > self.normals.size:
            raise RuntimeError("ReplayStream: not enough recorded normal numbers")
        out = self.normals[self.i_n:self.i_n + k]
        self.i_n += k
        return out.reshape((int(r), int(c)), order="F")

    def rand(self, r, c=1):
        k = int(r) * int(c)
        if self.i_u + k > self.uniforms.size:
            raise RuntimeError("ReplayStream: not enough recorded uniform numbers")
        out = self.uniforms[self.i_u:self.i_u + k]
        self.i_u += k
        return out.reshape((int(r), int(c)), order="F")


def make_stream(spec, seed=0):
    """A fresh stream from ``spec``: ``'octave'`` (default), ``'numpy'``, a callable returning a stream
    (called with no argument), or a stream instance (returned as is)."""
    if spec is None or spec == "octave":
        return OctaveRandomStream(seed)
    if spec == "numpy":
        return NumpyRandomStream(seed)
    if callable(spec) and not hasattr(spec, "randn"):
        return spec()
    if hasattr(spec, "randn") and hasattr(spec, "rand"):
        return spec
    raise ValueError(f"unknown random stream {spec!r}: use 'octave', 'numpy', a stream or a factory")
