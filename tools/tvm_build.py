#!/usr/bin/env python3
"""Apache TVM for shakti-ai-sdk: build the compiler from frameworks/tvm, and set up Python to use it.

usage: tools/tvm_build.py build [--llvm-config PATH]     (run with the conversion environment: env/requirements-convert.lock)

The build (build/tvm, CMake + Ninja) is the compiler only: target code is TVM's C back end, compiled by the
RISC-V GCC (tools/tvm_aot.py).  LLVM is used on the host only, for what TVM evaluates while compiling (constant
folding); any LLVM TVM supports will do (llvm-config: --llvm-config, $LLVM_CONFIG, else llvm-config on PATH).
TVM's Python FFI package (frameworks/tvm/3rdparty/tvm-ffi) is installed into the running Python if missing.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "frameworks", "tvm")
OUT = os.path.join(ROOT, "build", "tvm")

CONFIG = """
# shakti-ai-sdk (tools/tvm_build.py): the compiler for C code generation; LLVM for host-side evaluation only
set(CMAKE_BUILD_TYPE Release)
set(USE_LLVM {llvm})
set(HIDE_PRIVATE_SYMBOLS ON)
set(USE_CUDA OFF)
set(USE_METAL OFF)
set(USE_VULKAN OFF)
set(USE_OPENCL OFF)
set(USE_CUBLAS OFF)
set(USE_CUDNN OFF)
set(USE_CUTLASS OFF)
"""


def use():
    """Make `import tvm` load frameworks/tvm with the library from build/tvm."""
    lib = os.path.join(OUT, "lib")
    if not os.path.exists(os.path.join(lib, "libtvm_compiler.so")):
        sys.exit("TVM is not built: run tools/tvm_build.py build")
    os.environ["TVM_LIBRARY_PATH"] = lib
    sys.path.insert(0, os.path.join(SRC, "python"))


def build(llvm_config):
    if not os.path.exists(os.path.join(SRC, "CMakeLists.txt")):
        sys.exit("frameworks/tvm is not fetched: run tools/frameworks.py fetch tvm")
    llvm = llvm_config or os.environ.get("LLVM_CONFIG") or shutil.which("llvm-config")
    if not llvm:
        sys.exit("no llvm-config: give --llvm-config (e.g. /usr/bin/llvm-config-17)")
    import frameworks
    for name in ("tvm",):
        frameworks.ensure_patched(name)                 # patches/<name>, if any
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(SRC, "cmake", "config.cmake")) as f:
        config = f.read() + CONFIG.format(llvm=llvm)
    with open(os.path.join(OUT, "config.cmake"), "w") as f:
        f.write(config)
    subprocess.run(["cmake", "-G", "Ninja", SRC], cwd=OUT, check=True)
    subprocess.run(["ninja"], cwd=OUT, check=True)
    try:
        import tvm_ffi  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", os.path.join(SRC, "3rdparty", "tvm-ffi")], check=True)
    print(f"TVM built in {OUT}")


def main(argv):
    if not argv or argv[0] != "build":
        sys.exit(__doc__)
    llvm = None
    if argv[1:2] == ["--llvm-config"] and len(argv) == 3:
        llvm = argv[2]
    elif len(argv) != 1:
        sys.exit(__doc__)
    build(llvm)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
