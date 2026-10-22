#!/usr/bin/env bash
# Set up everything shakti-ai-sdk needs (also run by images that bundle the SDK):
#   ../.venv-ai             the conversion environment: Python 3.12 with env/requirements-convert.lock
#   frameworks/             the submodules at their pins (tools/frameworks.py fetch), patches applied by the builds
#   build/tvm, build/mlir   TVM (tools/tvm_build.py) and LLVM + MLIR + torch-mlir (tools/mlir_build.py)
#   download cache          the sources in third_party.lock (tools/fetch.py; models are fetched when used)
#
# usage: env/setup.sh [--clean]
#   --clean   keep only what the tools use: MLIR's opt / llc / mlir-opt / mlir-translate / torch-mlir-opt and its
#             Python package, clang with its headers (the vector extension), TVM's libraries; the llvm-project
#             working tree, build trees and pip caches go
# needs: git, Ninja, a C++ compiler, lld; PYTHON (default python3.12) with its headers; LLVM_CONFIG for TVM (default
# llvm-config-17); JOBS (default: all cores) for the LLVM build.  CMake comes from pip (TVM's tvm-ffi wants 3.26 or
# later).  Safe to re-run: what is there is kept.
set -euo pipefail
SDK=$(cd "$(dirname "$0")/.." && pwd)
VENV=${PYTHON_AI_VENV:-$(dirname "$SDK")/.venv-ai}
PY=${PYTHON:-python3.12}
LLVM_CONFIG=${LLVM_CONFIG:-llvm-config-17}
JOBS=${JOBS:-$(nproc)}
CLEAN=0
for a in "$@"; do
  case "$a" in
    --clean) CLEAN=1 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done
cd "$SDK"

[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --no-cache-dir --upgrade pip
# torch / torchvision are the CPU builds (+cpu), from PyTorch's index
"$VENV/bin/pip" install -q --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
  -r env/requirements-convert.lock
"$VENV/bin/pip" install -q --no-cache-dir cmake==4.1.2
PYAI="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"

for f in tflite-micro tvm torch-mlir llvm-project; do
  [ -e "frameworks/$f/.git" ] || [ -x build/mlir/bin/opt -a "$f" = llvm-project ] || "$PYAI" tools/frameworks.py fetch "$f"
done
"$PYAI" tools/fetch.py $(sed -n 's/^\([A-Za-z0-9_.-]*\):$/\1/p' third_party.lock) > /dev/null

[ -f build/tvm/lib/libtvm_compiler.so ] || "$PYAI" tools/tvm_build.py build --llvm-config "$(command -v "$LLVM_CONFIG")"
[ -x build/mlir/bin/opt ] || "$PYAI" tools/mlir_build.py build -j "$JOBS"

if [ "$CLEAN" = 1 ]; then
  ( cd build/tvm && find . -mindepth 1 -maxdepth 1 ! -name lib ! -name 3rdparty ! -name config.cmake -exec rm -rf {} + )
  ( cd build/mlir
    pkg=tools/torch-mlir/python_packages
    if [ -d "$pkg" ] && [ ! -d keep ]; then
      mkdir -p keep/bin keep/lib keep/$pkg
      cp -rL "$pkg/torch_mlir" "keep/$pkg/"      # the package links into the sources: copy the files themselves
      cp bin/opt bin/llc bin/mlir-opt bin/mlir-translate bin/torch-mlir-opt keep/bin/
      # clang for the vector extension (mk/common.mk): the versioned binary, its clang / clang++ links and its
      # resource directory (riscv_vector.h and the other builtin headers)
      if [ -e bin/clang ]; then
        cp "bin/$(readlink bin/clang)" keep/bin/ && cp -P bin/clang bin/clang++ keep/bin/ && cp -r lib/clang keep/lib/
      fi
      find . -mindepth 1 -maxdepth 1 ! -name keep -exec rm -rf {} +
      mv keep/* . && rmdir keep
    fi )
  git submodule deinit -q -f frameworks/llvm-project               # 2.7 GB of sources, only needed to build
  rm -rf .git/modules/frameworks/llvm-project
  rm -rf ~/.cache/pip
fi
echo "shakti-ai-sdk set up: conversion environment in $VENV, TVM in build/tvm, MLIR in build/mlir"
