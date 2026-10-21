# shakti-ai-sdk

Machine-learning frameworks on SHAKTI RISC-V SoCs: TensorFlow Lite Micro, TVM and MLIR (through torch-mlir). A model
(PyTorch or Keras) becomes a RISC-V ELF for any ISA, with one bare-metal runtime and one test harness that checks
its outputs against each framework's host reference; programs run on Spike or on this machine's CPU. A SoC is
described to the SDK by a target directory (`target.json`); running on the SoC itself (RTL simulation, board) is
the job of the tool that describes it.

## Rules

- The frameworks are git submodules under `frameworks/`, pinned to one commit each and never edited.
- Every change to a framework is a patch file in `patches/<framework>/`, applied in order on the pinned commit
  (`tools/frameworks.py`).
- Nothing about a particular SoC is written here: the memory map, console UART, ISA and ABI come from its target
  directory (`target.json`, `tools/soc.py`).
- Models, datasets, framework downloads and build output are never committed: `tools/fetch.py` fetches them by
  sha256 into a cache; `build/` is ignored.

## Layout

```
frameworks/      tflite-micro, tvm, torch-mlir, llvm-project   (submodules, pinned, fetched per phase)
patches/         one directory per framework: NNNN-what.patch
runtime/shakti/  start-up, console, exit, traps, newlib system calls, cycle counter; link.ld template
runtime/host/    the same calls for SOC=host (this machine's C library)
runtime/tvm/     what TVM's C kernels call at run time (workspace, errors)
runtime/mlir/    what MLIR's lowered code calls at run time (strided copy, fmaximum / fminimum, roundeven)
mk/              common.mk: build rules for an app on a target (SOC=...); tflm.mk, tvm.mk, mlir.mk: the frameworks
examples/        common examples: the same models on every framework (<model>/<framework>/)
projects/        framework-specific programs (e.g. TFLite Micro's micro_speech)
models/          the common examples' models: <name>/model.py (PyTorch: build, calibration and test inputs);
                 manifest.yaml: model files to fetch (URL + sha256)
templates/       model files to start from, PyTorch and Keras (templates/README.md)
third_party.lock what the frameworks download while building (URL + sha256)
tools/           aisdk.py (the `aisdk` command), frameworks.py, soc.py, run.py, fetch.py; export.py (a model for
                 every framework, with host references), onnx_int8.py, tflm.py / tflm_host.py, tvm_build.py /
                 tvm_aot.py / tvm_int8.py, mlir_build.py / mlir_aot.py; model_info.py (a model file's summary without training),
                 completions.py (names for an editor's completion), tflm_stages.py (what TFLM runs, step by step);
                 helpers: aot_tests.py (test data for the AOT compilers), aot_host_main.c (their host reference),
                 tflm_host_runner.cc (TFLM's host reference)
tests/           the checks of each phase; smoke.sh: a quick check of a set-up SDK; examples.sh: the common examples
                 against the host reference; test_aot.py, test_frameworks.py (pytest)
env/             toolchain and Python environment notes; setup.sh: everything the SDK needs, in one go
```

| Framework | Pinned | Why |
|---|---|---|
| TensorFlow Lite Micro | main `5736ea15` (2026-10-16) | no release tags upstream |
| TVM | `v0.27.0.post1` (`7b5acb32`) | the latest stable release |
| torch-mlir | main `4f21fc26` (2026-10-13) | no release tags upstream |
| llvm-project | `2fc57768` | the commit torch-mlir pins |

## Quick start

```sh
export PATH=<toolchain>/bin:$PATH                 # riscv64-unknown-elf- 13.2 multilib (env/README.md)
make -C examples/hello run                        # Spike, RV64 (default SOC=spike:rv64imafdc_zicsr_zifencei/lp64d)
make -C examples/hello run SOC=host               # this machine: the same program built with gcc, run directly
make -C examples/hello SOC=<target dir>           # built for a SoC described by <target dir>/target.json
tools/frameworks.py status                        # pins, fetched or not, patches
tests/phase0.sh [<target dir> ...]                # the phase 0 checks
```

The same from a model, with one command (`./aisdk -h`):

```sh
./aisdk run lenet5 --host                                   # every framework and precision on this machine
./aisdk run my_model.py --isa rv32imac --fw tvm             # a model file on Spike, any ISA (ABI from the ISA)
./aisdk build my_model.py --isa rv64gc --precision int8 -o out/      # ELFs only
./aisdk build my_model.py --target <target dir> -o out/     # for a SoC: its own tool runs them
```

A target directory's `target.json` names the SoC (`name`), its ISA and ABI (`march`, `mabi`), main memory
(`memory.base`, `memory.size`), `clock_hz`, `bus_width`, header directories (`include`: a `soc.h` there defines
`UART0_BASE`, the console, a uart_v3) and extra C sources linked into every program (`sources`, with `board_init`
when they define `board_init()`); paths are relative to the directory (`tools/soc.py`).

`env/setup.sh [--clean]` sets up the rest in one go: the conversion environment (`../.venv-ai`), the frameworks at
their pins, the download cache, TVM and MLIR (LLVM is built: a few hours); `tests/smoke.sh` then checks it on
Spike and on this machine.

`SOC=host` builds for the machine the SDK runs on (its gcc, C library and start-up; TFLite Micro's generic
platform files instead of the SHAKTI target's; MLIR's LLVM IR compiled by llc for the host): a quick check of an
app before Spike or a SoC.  The check programs' "cycles" are nanoseconds there.

## Common examples

`models/<name>/model.py` is a PyTorch model; `tools/export.py` (the conversion environment,
`env/requirements-convert.lock`) exports it once for every framework, with test inputs and host references:

- ONNX (float) and a QDQ int8 ONNX (`tools/onnx_int8.py`: onnxruntime's static quantisation, int8 activations,
  per-tensor symmetric int8 weights) for TVM and MLIR;
- float and full-integer TFLite for TensorFlow Lite Micro; its references come from TFLite Micro itself built
  for the host (bit-exact for int8).

A model file anywhere works the same way (`tools/export.py path/to/model.py OUTDIR`; the make variables
`MODEL_SRC`, `MODEL_DIR`), also written in Keras (`MODEL_KIND=keras`: TFLite from TensorFlow's own converter, ONNX
for TVM and MLIR through tf2onnx); `templates/README.md` lists the functions a model file defines.  With
`TVM_STAGES=DIR` / `MLIR_STAGES=DIR` the ahead-of-time compilers also write the model at each lowering step
(Relax imported, integer, TIR, C; linalg, bufferized, loops, LLVM dialect, LLVM IR before and after opt), and
`tools/tflm_stages.py` lists what TFLite Micro runs (operators in order, tensors).

| Model | | TFLite Micro | TVM | MLIR |
|---|---|---|---|---|
| mlp_sine | sine regression, 3 dense layers | int8, float | int8, float | int8, float |
| lenet5 | MNIST digits (trained, 97.2 %) | int8, float | int8, float | int8, float |
| mlp_mnist | MNIST digits, dense layers + softmax (trained, 93.9 %) | int8, float | int8, float | int8, float |
| mobilenet_v2 | ImageNet, torchvision weights | int8, float | int8, float | int8, float |
| resnet18 | ImageNet, torchvision weights | int8, float | int8, float | int8, float |

Each framework runs the model on the target and compares: float against the host reference within
1e-4 + 1e-3 |ref|; int8 exactly, against the same framework on the host (TFLite Micro built for the host; TVM's and
MLIR's generated code built for the host, without floating-point contraction on either side).

```sh
make -C examples/lenet5/tflm run SOC=<target>              # TFLite Micro: int8 and float
make -C examples/lenet5/tvm run SOC=<target> CHECK=int8    # TVM, compiled ahead of time to C (CHECK=float|int8)
make -C examples/lenet5/mlir run SOC=<target> CHECK=int8   # MLIR: torch-mlir, MLIR, llc
tests/examples.sh FRAMEWORK [MODEL ...] [-- <target dir> ...]   # every model on this machine and Spike (RV64,
                                                           # RV32, with and without FP); built for target dirs
```

- TVM (`tools/tvm_aot.py`): the Relax ONNX front end, operators lowered to TIR, constants folded, operators
  fused; TVM's C back end writes the kernels and the tool writes the rest as static C (the kernel calls in order,
  one arena planned from the tensors' lifetimes, the weights linked in) - no TVM runtime on the target.  int8:
  `tools/tvm_int8.py` turns each quantised convolution / matrix product into int8 x int8 -> int32 with one
  requantisation.  Built by `tools/tvm_build.py build` (LLVM on the host only, for constant folding).
- MLIR (`tools/mlir_aot.py`): float from the PyTorch model (torch-mlir's FX importer), int8 from the QDQ ONNX
  (torch-mlir's ONNX importer and QDQ fusion: integer `linalg.conv_2d_nchw_fchw_q` / `quantized_matmul`), then
  upstream MLIR (one-shot bufferization, linalg to loops, LLVM dialect), then LLVM's opt and llc for the target's
  ISA and ABI.  Built by `tools/mlir_build.py build` (llvm-project with MLIR, torch-mlir in tree).

## Projects

Programs that only one framework can run.  TensorFlow Lite Micro's own examples, built from its sources unchanged
and checked by its own tests ("~~~ALL TESTS PASSED~~~"):

| Project | |
|---|---|
| projects/tflite-micro/hello_world | sine model, float and int8 |
| projects/tflite-micro/micro_speech | "yes" / "no" in 1 s recordings, through TFLM's audio front end |
| projects/tflite-micro/person_detection | person / no person in a 96x96 greyscale image (int8 MobileNet) |

```sh
make -C projects/tflite-micro/micro_speech run SOC=<target>
tests/phase1.sh [<target dir> ...]                         # all three on Spike (RV64, RV32, with and without
                                                           # FP); built for target dirs
```

## Phases

0. Skeleton: pinned submodules, patch tool, runtime, `hello` on Spike (RV64, then RV32).
1. TensorFlow Lite Micro. 2. TVM. 3. MLIR (torch-mlir). 4. Model files in PyTorch or Keras (templates, summary,
editor completions, the lowering stages), `SOC=host`, target directories and the `aisdk` command.
Each phase is done for RV64 first, then RV32. Next: the vector extension (RVV 1.0), then the draft matrix
extensions (Zvvmm, Zvvfmm).
