#!/usr/bin/env python3
"""MLIR for shakti-ai-sdk: build LLVM, MLIR and torch-mlir from frameworks/, and set up Python to use them.

usage: tools/mlir_build.py build [-j N]       (run with the conversion environment: env/requirements-convert.lock)

One in-tree build (build/mlir, CMake + Ninja): frameworks/llvm-project with MLIR and its Python bindings, and
frameworks/torch-mlir as an external project (its FX importer: PyTorch -> MLIR, in Python, with the installed
PyTorch; no PyTorch C++ extensions, no StableHLO).  LLVM's targets: the host and RISC-V (llc writes the objects).
Release, without tests, examples or benchmarks: about 15 GB.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLVM = os.path.join(ROOT, "frameworks", "llvm-project")
TORCH_MLIR = os.path.join(ROOT, "frameworks", "torch-mlir")
OUT = os.path.join(ROOT, "build", "mlir")


def use():
    """Make `import torch_mlir` / `import mlir` load the build, and return its bin directory (mlir-opt, llc ...)."""
    pkg = os.path.join(OUT, "tools", "torch-mlir", "python_packages", "torch_mlir")
    if not os.path.isdir(pkg):
        sys.exit("MLIR is not built: run tools/mlir_build.py build")
    sys.path.insert(0, pkg)
    return os.path.join(OUT, "bin")


def build(jobs):
    for d, name in ((LLVM, "llvm-project"), (TORCH_MLIR, "torch-mlir")):
        if not os.path.exists(os.path.join(d, "CMakeLists.txt") if d == TORCH_MLIR else os.path.join(d, "llvm")):
            sys.exit(f"frameworks/{name} is not fetched: run tools/frameworks.py fetch {name}")
    import frameworks
    for name in ("llvm-project", "torch-mlir"):
        frameworks.ensure_patched(name)                 # patches/<name>, if any
    os.makedirs(OUT, exist_ok=True)
    subprocess.run([
        "cmake", "-G", "Ninja", os.path.join(LLVM, "llvm"),
        "-DCMAKE_BUILD_TYPE=Release", "-DLLVM_ENABLE_ASSERTIONS=OFF",
        "-DLLVM_ENABLE_PROJECTS=mlir", "-DLLVM_TARGETS_TO_BUILD=host;RISCV",
        "-DLLVM_EXTERNAL_PROJECTS=torch-mlir", f"-DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR={TORCH_MLIR}",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON", f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DPython_EXECUTABLE={sys.executable}",
        "-DTORCH_MLIR_ENABLE_STABLEHLO=OFF", "-DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=OFF",
        "-DLLVM_INCLUDE_TESTS=OFF", "-DLLVM_INCLUDE_EXAMPLES=OFF", "-DLLVM_INCLUDE_BENCHMARKS=OFF",
        "-DMLIR_INCLUDE_TESTS=OFF", "-DLLVM_ENABLE_ZSTD=OFF", "-DLLVM_USE_LINKER=lld",
        "-DLLVM_PARALLEL_LINK_JOBS=4"], cwd=OUT, check=True)
    subprocess.run(["ninja"] + (["-j", str(jobs)] if jobs else []), cwd=OUT, check=True)
    print(f"MLIR built in {OUT}")


def main(argv):
    if not argv or argv[0] != "build":
        sys.exit(__doc__)
    jobs = None
    if argv[1:2] == ["-j"] and len(argv) == 3:
        jobs = int(argv[2])
    elif len(argv) != 1:
        sys.exit(__doc__)
    build(jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
