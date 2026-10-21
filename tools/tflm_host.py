#!/usr/bin/env python3
"""TensorFlow Lite Micro on the host: the same sources and flags as the target library (tools/tflm.py), built with
the host C++ compiler, and a runner for one model (tools/tflm_host_runner.cc) - the reference the common examples'
target outputs are compared with (integer results are bit-exact between host and target).

usage: tools/tflm_host.py MODEL_DIR ARENA_KB     (writes MODEL_DIR/output_ref_{int8,float}.csv from input_*.csv)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "build", "host-tflm")
CXX = os.environ.get("HOST_CXX", "g++")
CC = os.environ.get("HOST_CC", "gcc")


def mkvars():
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "tflm.py"), "prepare", os.path.join(ROOT, "build", "tflm")],
                   check=True, stdout=subprocess.DEVNULL)
    v = {}
    cur = None
    for line in open(os.path.join(ROOT, "build", "tflm", "sources.mk")):
        line = line.rstrip("\n")
        if " := " in line:
            cur, val = line.split(" := ", 1)
            v[cur] = val.rstrip("\\").split()
        elif cur and line.startswith("  "):
            v[cur] += line.rstrip("\\").split()
    return v


def library(v):
    os.makedirs(os.path.join(OUT, "obj"), exist_ok=True)
    objs, procs = [], []
    shakti = "/tensorflow/lite/micro/shakti/"           # the SHAKTI target's files -> TFLM's generic ones
    for s in [x.replace(shakti, "/tensorflow/lite/micro/") for x in v["TFLM_SRCS"]] + v["TFLM_THIRD_PARTY_SRCS"]:
        o = os.path.join(OUT, "obj", s.replace(ROOT + "/", "").replace("/", "_") + ".o")
        objs.append(o)
        if os.path.exists(o) and os.path.getmtime(o) >= os.path.getmtime(s):
            continue
        cmd = ([CC, "-O2"] + v["TFLM_CCFLAGS"] if s.endswith(".c") else [CXX, "-O2"] + v["TFLM_CXXFLAGS"]) + \
            HOST + v["TFLM_INCLUDES"] + ["-c", s, "-o", o]
        procs.append(subprocess.Popen(cmd))
        if len(procs) >= os.cpu_count():
            [p.wait() for p in procs]
            procs = []
    if any(p.wait() for p in procs):
        sys.exit("host TFLM build failed")
    lib = os.path.join(OUT, "libtflm.a")
    if not os.path.exists(lib) or any(os.path.getmtime(o) > os.path.getmtime(lib) for o in objs):
        if os.path.exists(lib):
            os.remove(lib)
        subprocess.run(["ar", "rcs", lib] + objs, check=True)
    return lib


# on the host a failed TFLM check must end the run (the target build's TFLITE_ABORT prints HALTED and spins)
HOST = ["-DTFLITE_ABORT=__builtin_abort();"]


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)
    mdir, arena = os.path.abspath(argv[0]), argv[1]
    v = mkvars()
    lib = library(v)
    exe = os.path.join(mdir, "tflm_host_runner")
    subprocess.run([CXX, "-O2"] + v["TFLM_CXXFLAGS"] + HOST + v["TFLM_INCLUDES"] +
                   ["-include", os.path.join(mdir, "ops.h"), os.path.join(ROOT, "tools", "tflm_host_runner.cc"), lib,
                    "-o", exe], check=True)
    for kind in ("int8", "float"):
        subprocess.run([exe, os.path.join(mdir, f"model_{kind}.tflite"), kind, os.path.join(mdir, f"input_{kind}.csv"),
                        os.path.join(mdir, f"output_ref_{kind}.csv"), arena], check=True)
    os.remove(exe)
    print(f"{os.path.basename(mdir)}: host TFLM references written")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
