#!/bin/bash
# Quick check of a set-up SDK (env/setup.sh; images that bundle the SDK run it): TFLite Micro's
# hello_world test and mlp_sine on TFLite Micro, TVM and MLIR (against the host reference), on this machine
# (SOC=host) and on Spike, RV64 and RV32 with and without FP, and with the vector extension (once clang is
# built).  What it builds is removed afterwards: build/ is left as it was.
set -u
cd "$(dirname "$0")/.."
mkdir -p build
before=$(ls build)
fail=0
PROJECTS=hello_world tests/phase1.sh || fail=1
for fw in tflm tvm mlir; do tests/examples.sh "$fw" mlp_sine || fail=1; done
./aisdk run mlp_sine --host --fw tvm --precision int8 | grep -q "PASS" && echo "PASS aisdk run on this machine" \
  || { echo "FAIL aisdk run on this machine"; fail=1; }
# the vector extension (clang compiles the C, LLVM vectorises MLIR's code): every framework on Spike rv64imafdcv
if [ -x build/mlir/bin/clang ]; then
  out=$(./aisdk run mlp_sine --isa rv64imafdcv); echo "$out" | grep -E "^  "
  echo "$out" | grep -q FAIL && { echo "FAIL vector (rv64imafdcv)"; fail=1; } || echo "PASS vector (rv64imafdcv)"
else
  echo "SKIP vector (rv64imafdcv): no clang in build/mlir/bin"
fi
for f in build/*; do grep -qx "$(basename "$f")" <<< "$before" || rm -rf "$f"; done
exit $fail
