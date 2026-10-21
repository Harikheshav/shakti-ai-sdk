#!/bin/bash
# Quick check of a set-up SDK (env/setup.sh; images that bundle the SDK run it): TFLite Micro's
# hello_world test and mlp_sine on TFLite Micro, TVM and MLIR (against the host reference), on this machine
# (SOC=host) and on Spike, RV64 and RV32 with and without FP.  What it builds is removed afterwards: build/ is left as it was.
set -u
cd "$(dirname "$0")/.."
mkdir -p build
before=$(ls build)
fail=0
PROJECTS=hello_world tests/phase1.sh || fail=1
for fw in tflm tvm mlir; do tests/examples.sh "$fw" mlp_sine || fail=1; done
./aisdk run mlp_sine --host --fw tvm --precision int8 | grep -q "PASS" && echo "PASS aisdk run on this machine" \
  || { echo "FAIL aisdk run on this machine"; fail=1; }
for f in build/*; do grep -qx "$(basename "$f")" <<< "$before" || rm -rf "$f"; done
exit $fail
