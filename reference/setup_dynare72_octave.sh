#!/bin/sh
# Rebuild the Dynare 7.2 + Octave reference environment on Ubuntu 24.04 (as used to make the .mat files here).
set -e
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  octave octave-dev octave-statistics octave-io octave-optim octave-control \
  gcc g++ gfortran libboost-dev libslicot-dev libslicot-pic libsuitesparse-dev \
  flex libfl-dev bison meson pkgconf make ninja-build libmatio-dev curl xz-utils
curl -fLO http://archive.ubuntu.com/ubuntu/pool/universe/d/dynare/dynare_7.2.orig.tar.xz
mkdir -p dynare-7.2 && tar -xf dynare_7.2.orig.tar.xz -C dynare-7.2 --strip-components=1
cd dynare-7.2
meson setup -Dbuild_for=octave --buildtype=release build-octave
meson compile -C build-octave
# Octave 8.4 on Ubuntu 24.04 has no "datatypes" package; Dynare 7.2 refuses to start without it.
# It is only used for date handling, so downgrade the check to a warning:
sed -i "s/^    error('You must install the \"datatypes\" package.*/    warning('datatypes package missing: datetime-dependent features unavailable')/" matlab/dynare_config.m
echo "Preprocessor: $(pwd)/build-octave/preprocessor/src/dynare-preprocessor"
echo "Run Octave with: GLIBC_TUNABLES=glibc.rtld.execstack=2 octave --no-gui, then addpath $(pwd)/matlab"
