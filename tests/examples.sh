#!/bin/bash
# The common examples on one framework, checked against the host reference: on this machine (SOC=host) and on Spike
# (RV64 first, then RV32, with and without FP); for target directories (target.json) the small models are built
# only, with SOC_TESTS test inputs (the SoC's own tool runs them).
# usage: tests/examples.sh FRAMEWORK [MODEL ...] [-- TARGET_DIR ...]     (FRAMEWORK: tflm, tvm, mlir; default
#        models: all four).  TFLite Micro checks int8 and float in one program; TVM and MLIR build one per CHECK.
#        CHECKS="float int8" (default) selects the TVM / MLIR variants.
set -u
cd "$(dirname "$0")/.."
fw=${1:?usage: tests/examples.sh FRAMEWORK [MODEL ...] [-- TARGET_DIR ...]}; shift
models=(); socs=(); seen=0
for a in "$@"; do if [ "$a" = "--" ]; then seen=1; elif [ $seen = 1 ]; then socs+=("$(cd "$a" && pwd)"); else models+=("$a"); fi; done
[ ${#models[@]} = 0 ] && models=(mlp_sine lenet5 mobilenet_v2 resnet18)
SMALL="mlp_sine lenet5"
SOC_TESTS=2
fail=0
mkdir -p build
if [ "$fw" = tflm ]; then variants=(both); else read -ra variants <<< "${CHECKS:-float int8}"; fi
run1() {  # model target variant [make variables]
  local v=() log=build/ex_${fw}_$1_$(echo "$2" | tr '/:' '__' | xargs basename)_$3.log
  [ "$3" = both ] || v=(CHECK="$3")
  if make -s -j16 -C examples/$1/$fw SOC="$2" "${v[@]}" "${@:4}" > "$log" 2>&1 &&
     timeout 7200 make -s -C examples/$1/$fw run SOC="$2" "${v[@]}" "${@:4}" >> "$log" 2>&1; then
    echo "PASS $fw $1 on $(basename "$2")  $(grep -E '^(int8|float):' "$log" | sed 's/ tests, 0 outputs off, max |diff| [0-9.]*,//' | tr '\n' ' ')"
  else
    echo "FAIL $fw $1 ${3/both/} on $(basename "$2")"; grep -E "^(int8|float):|trap|rror|TVM " "$log" | head -4; fail=1
  fi
}
build1() {  # model target variant [make variables]: built only (a target directory)
  local v=() log=build/ex_${fw}_$1_$(basename "$2")_$3.log
  [ "$3" = both ] || v=(CHECK="$3")
  if make -s -j16 -C examples/$1/$fw SOC="$2" "${v[@]}" "${@:4}" > "$log" 2>&1; then echo "BUILT $fw $1 ${3/both/} for $(basename "$2")"
  elif grep -q "overflowed by\|do not fit" "$log"; then echo "SKIP $fw $1 ${3/both/} for $(basename "$2"): too big for its memory"
  else echo "FAIL $fw $1 ${3/both/} for $(basename "$2")"; grep -E "rror" "$log" | head -4; fail=1; fi
}
run() {  # model target [make variables]
  local vs=("${variants[@]}")
  # TFLite Micro on a SoC: int8 and float as two programs (both models in one may not fit a BRAM SoC)
  [ "$fw" = tflm ] && [ "${2#spike:}" = "$2" ] && [ "$2" != host ] && vs=(int8 float)
  for v in "${vs[@]}"; do run1 "$1" "$2" "$v" "${@:3}"; done
}
for m in "${models[@]}"; do
  [ -d examples/$m/$fw ] || { echo "SKIP $fw $m: no examples/$m/$fw"; continue; }
  for t in host spike:rv64imafdc_zicsr_zifencei/lp64d spike:rv64imac_zicsr_zifencei/lp64 \
           spike:rv32imafc_zicsr_zifencei/ilp32f spike:rv32imac_zicsr_zifencei/ilp32; do run "$m" "$t"; done
  case " $SMALL " in *" $m "*) for p in "${socs[@]}"; do
    if [ "$fw" = tflm ]; then vs=(int8 float); else vs=("${variants[@]}"); fi
    for v in "${vs[@]}"; do build1 "$m" "$p" "$v" TESTS=$SOC_TESTS; done; done;; esac
done
exit $fail
