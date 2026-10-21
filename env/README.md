# Toolchain and host tools

- RISC-V GCC: `riscv64-unknown-elf-` 13.2 with multilib (every RV32 / RV64 ISA a SHAKTI SoC can have, newlib and
  newlib-nano, libstdc++). Put its `bin` on PATH or pass `CROSS=<prefix>`.
- Spike (for `SOC=spike:...`): on PATH or `SPIKE=<path>`.
- A SoC: a target directory (`target.json`, see the README); programs are only built for it, its own tool runs them.
- Python for the tools (`PYTHON`): the conversion environment below when it is there, else `python3` (3.10 or
  later, with numpy, Pillow and PyYAML).
- The conversion environment (`tools/export.py`, `tools/tvm_aot.py`, `tools/mlir_aot.py`, `tools/model_info.py`,
  `tools/completions.py`, `tools/tflm_stages.py`): Python 3.12 with `requirements-convert.lock` (PyTorch,
  torchvision, ONNX, onnx2tf, tf2onnx, TensorFlow, Keras and tf_keras, ai-edge-litert); the examples look for it in
  `../.venv-ai` (else `PYTHON_AI=<python>`).
- `SOC=host` (`mk/common.mk`, `tools/soc.py`): no cross toolchain; the machine's own gcc / g++ and C library.
- TVM: CMake, Ninja and an LLVM TVM supports (`llvm-config`, e.g. Ubuntu's `llvm-17-dev`) for
  `tools/tvm_build.py build`.
- `env/setup.sh [--clean]` does all of the above that is Python or built from source (conversion environment,
  frameworks, download cache, TVM, MLIR); it needs git, Ninja, a C++ compiler, lld, Python 3.12 with its headers and
  `llvm-config-17` (CMake comes from pip).
