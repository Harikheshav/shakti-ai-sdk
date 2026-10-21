#!/bin/bash
# Phase 0 checks: the framework pins and patch tool, the runtime's hello on Spike and on this machine (SOC=host),
# and hello built for target directories (target.json: built only, the SoC's own tool runs it).
# usage: tests/phase0.sh [TARGET_DIR ...]
set -u
cd "$(dirname "$0")/.."
fail=0
check() { local what=$1; shift; if "$@" > build/phase0.log 2>&1; then echo "PASS $what"; else echo "FAIL $what"; tail -5 build/phase0.log; fail=1; fi; }
mkdir -p build
check "framework pins (committed)" bash -c '[ "$(git ls-tree HEAD frameworks/ | grep -c "^160000 commit")" = 4 ]'
for t in spike:rv64imafdc_zicsr_zifencei/lp64d spike:rv64imac_zicsr_zifencei/lp64 \
         spike:rv32imafc_zicsr_zifencei/ilp32f spike:rv32imac_zicsr_zifencei/ilp32 host; do
  check "hello on $t" make -s -C examples/hello run SOC="$t"
done
for p in "$@"; do
  check "hello built for $(basename "$p")" make -s -C examples/hello all SOC="$(cd "$p" && pwd)"
done
exit $fail
