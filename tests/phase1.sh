#!/bin/bash
# Phase 1 checks (TensorFlow Lite Micro): our patches apply on the pinned commit; TFLM's own tests of its examples
# (projects/tflite-micro: hello_world, micro_speech, person_detection) pass on Spike (RV64 first, then RV32); for
# target directories (target.json) they are built only (the SoC's own tool runs them).  PROJECTS: the projects
# (default all three).  A project too big for a target's memory is skipped (SKIP, with the linker's overflow).
# usage: tests/phase1.sh [TARGET_DIR ...]
set -u
cd "$(dirname "$0")/.."
fail=0
mkdir -p build
check() {
  local what=$1; shift
  if "$@" > build/phase1.log 2>&1; then echo "PASS $what"
  elif grep -q "overflowed by" build/phase1.log; then echo "SKIP $what: $(grep -o "region .* overflowed by [0-9]* bytes" build/phase1.log | head -1)"
  else echo "FAIL $what"; tail -5 build/phase1.log; fail=1; fi
}
check "tflite-micro patches apply" python3 tools/frameworks.py check tflite-micro
rm -f build/tflite-micro.patches                    # check left the submodule at the pin: apply again on build
projects=${PROJECTS:-hello_world micro_speech person_detection}
project() {   # project NAME SOC
  make -s -j16 -C "projects/tflite-micro/$1" SOC="$2" && make -s -C "projects/tflite-micro/$1" run SOC="$2" | grep -q '~~~ALL TESTS PASSED~~~'
}
for t in spike:rv64imafdc_zicsr_zifencei/lp64d spike:rv64imac_zicsr_zifencei/lp64 \
         spike:rv32imafc_zicsr_zifencei/ilp32f spike:rv32imac_zicsr_zifencei/ilp32; do
  for a in $projects; do check "tflm $a on $t" project "$a" "$t"; done
done
for p in "$@"; do
  p=$(cd "$p" && pwd)
  for a in $projects; do check "tflm $a built for $(basename "$p")" make -s -j16 -C "projects/tflite-micro/$a" SOC="$p"; done
done
exit $fail
