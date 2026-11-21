# This file is part of pydyn.
# Released under the GNU General Public License, version 3 or later; see LICENSE. No warranty.
"""Random-number streams for the particle filters (compatibility module).

The streams now live in :mod:`pydyn.octave_rng`, pydyn's single port of Octave's generators; this module re-exports
them under their old import path:

* :class:`~pydyn.octave_rng.OctaveRandomStream`: the numbers Octave gives after ``rand('state', seed)`` and
  ``randn('state', seed)``, via MATLAB-style ``randn(r, c)``/``rand(r, c)`` calls (column-major fill);
* :class:`~pydyn.octave_rng.NumpyRandomStream`: numpy's PCG64 (faster, not Dynare's numbers);
* :class:`~pydyn.octave_rng.ReplayStream`: recorded numbers;
* :func:`~pydyn.octave_rng.make_stream`.
"""
from .octave_rng import NumpyRandomStream, OctaveRandomStream, ReplayStream, make_stream

__all__ = ["OctaveRandomStream", "NumpyRandomStream", "ReplayStream", "make_stream"]
