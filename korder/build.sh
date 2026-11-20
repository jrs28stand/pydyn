#!/bin/sh
# Build Dynare's k-order perturbation library as a standalone shared library (no MATLAB/Octave).
# Usage: korder/build.sh /path/to/dynare-7.2   (needs g++ >= 13, BLAS and LAPACK)
set -e
DYNARE=${1:?path to the Dynare 7.2 source tree}
HERE=$(cd "$(dirname "$0")" && pwd)
S=$DYNARE/mex/sources; L=$S/libkorder
FLAGS="-std=gnu++20 -O2 -fPIC -fopenmp -DOCTAVE_MEX_FILE -DPACKAGE_VERSION=\"7.2\" -I$HERE/stub -I$S -I$L -I$L/kord -I$L/sylv -I$L/tl -I$L/utils"
mkdir -p "$HERE/obj"
ls $L/kord/*.cc $L/sylv/*.cc $L/tl/*.cc $L/utils/*.cc $L/k_ord_dynare.cc "$HERE/korder_capi.cc" \
  | xargs -P "$(nproc)" -I{} sh -c "g++ $FLAGS -c {} -o $HERE/obj/\$(basename {} .cc).o"
g++ -shared -fopenmp "$HERE"/obj/*.o -o "$HERE/libdynpy_korder.so" -llapack -lblas -lpthread
echo "built $HERE/libdynpy_korder.so"
