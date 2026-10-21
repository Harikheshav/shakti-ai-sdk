#!/usr/bin/env python3
"""shakti-ai-sdk from the command line: a model through TensorFlow Lite Micro, TVM and MLIR to a RISC-V ELF, run on
Spike or on this machine's CPU and checked against each framework's host reference.

  aisdk build MODEL [options] (--isa MARCH [--abi MABI] | --host | --target DIR) [-o OUTDIR]
  aisdk run   MODEL [options] (--isa MARCH [--abi MABI] | --host)

MODEL         a model file (PyTorch or Keras: INPUT_SHAPE, model(), build(), calibration(), test_inputs(); see
              templates/README.md) or the name of a model in models/ (mlp_sine, lenet5, ...)
--fw          tflm | tvm | mlir | all (default all)
--precision   int8 | float | all (default all)
--isa MARCH   any RISC-V ISA the toolchain has libraries for (riscv64-unknown-elf-gcc -print-multi-lib), e.g.
              rv32imac or rv64imafdc_zicsr_zifencei (_zicsr_zifencei is added when missing); the ELF is linked
              for Spike's memory map; --abi defaults from the ISA (lp64d, lp64f, lp64, ilp32d, ilp32f, ilp32)
--host        this machine's gcc and C library (SOC=host): quick, its cycles are nanoseconds
--target DIR  a SoC described by DIR/target.json (tools/soc.py): built only; that SoC's own tool runs it
--tests N     only the first N test inputs (default all)
--model-dir   where the model's exports go (default: build/models/<name> for models/, <file's dir>/<stem>_export)
-o OUTDIR     build: copy the ELFs there

Exit status: 0 when every build (and every run's check) passed.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMEWORKS = ("tflm", "tvm", "mlir")
PRECISIONS = ("int8", "float")
RESULT_RE = re.compile(r"^(int8|float): (\d+) tests, (\d+) outputs off, max \|diff\| ([0-9.]+), (\S+) cycles per "
                       r"inference(.*)$", re.M)


def soc_spec(a):
    if a.host:
        return "host"
    if a.target:
        d = os.path.abspath(a.target)
        if not os.path.isfile(os.path.join(d, "target.json")):
            sys.exit(f"aisdk: {d} has no target.json")
        return d
    isa = a.isa.lower()
    if not re.match(r"^rv(32|64)[a-z]+(_[a-z0-9]+)*$", isa):
        sys.exit(f"aisdk: --isa {a.isa}: not a RISC-V ISA string (e.g. rv64imac, rv32imafc_zicsr_zifencei)")
    for ext in ("zicsr", "zifencei"):                  # the runtime reads CSRs; GCC 12+ wants them named
        if f"_{ext}" not in isa:
            isa += f"_{ext}"
    base = isa.split("_")[0]
    abi = a.abi or (("lp64" if base.startswith("rv64") else "ilp32")
                    + ("d" if "d" in base[4:] or "g" in base[4:] else "f" if "f" in base[4:] else ""))
    return f"spike:{isa}/{abi}"


def model_vars(model, model_dir):
    """make variables naming the model (a built-in name or a file) and where its exports go."""
    if os.path.isfile(os.path.join(ROOT, "models", model, "model.py")):
        name, src = model, os.path.join(ROOT, "models", model, "model.py")
        out = model_dir or os.path.join(ROOT, "build", "models", model)
    elif os.path.isfile(model) and model.endswith(".py"):
        src = os.path.abspath(model)
        name = re.sub(r"\W", "_", os.path.splitext(os.path.basename(src))[0])
        out = model_dir or os.path.join(os.path.dirname(src), f"{name}_export")
    else:
        sys.exit(f"aisdk: {model}: neither a model file (.py) nor a model of models/ "
                 f"({', '.join(sorted(os.listdir(os.path.join(ROOT, 'models'))))})")
    return name, [f"MODEL={name}", f"MODEL_SRC={src}", f"MODEL_DIR={os.path.abspath(out)}"], os.path.abspath(out)


def make(fw, target, args):
    cmd = ["make", "--no-print-directory", "-s", f"-j{os.cpu_count() or 4}", "-f",
           os.path.join(ROOT, "examples", "common", f"{fw}_example.mk"), target] + args
    r = subprocess.run(cmd, cwd=os.path.join(ROOT, "examples", "common"), capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write((r.stdout + r.stderr)[-4000:])
    return r


def main(argv):
    ap = argparse.ArgumentParser(prog="aisdk", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("command", choices=("build", "run"))
    ap.add_argument("model")
    ap.add_argument("--fw", default="all", choices=FRAMEWORKS + ("all",))
    ap.add_argument("--precision", default="all", choices=PRECISIONS + ("all",))
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--isa")
    where.add_argument("--host", action="store_true")
    where.add_argument("--target")
    ap.add_argument("--abi")
    ap.add_argument("--tests", type=int)
    ap.add_argument("--model-dir")
    ap.add_argument("-o", dest="out")
    a = ap.parse_args(argv)
    if a.command == "run" and a.target:
        sys.exit("aisdk: a program for a target directory runs through that SoC's own tool (RTL simulation or "
                 "board); here: aisdk build ... --target DIR")
    soc = soc_spec(a)
    name, mvars, mdir = model_vars(a.model, a.model_dir)
    # the export first (once for every framework); its model.json says PyTorch or Keras
    r = make("tvm", os.path.join(mdir, "model.json"), mvars + [f"SOC={soc}"])       # make: only when out of date
    if r.returncode:
        sys.exit(f"aisdk: exporting {name} failed")
    kind = json.load(open(os.path.join(mdir, "model.json"))).get("source", "pytorch")
    fws = FRAMEWORKS if a.fw == "all" else (a.fw,)
    precs = PRECISIONS if a.precision == "all" else (a.precision,)
    print(f"{name} ({kind}) for {soc}")
    ok = True
    for fw in fws:
        for p in precs:
            app = f"{name}_{fw}_{p}" + (f"_t{a.tests}" if a.tests else "")
            args = mvars + [f"MODEL_KIND={kind}", f"CHECK={p}", f"APP={app}", f"SOC={soc}"] + \
                ([f"TESTS={a.tests}"] if a.tests else [])
            r = make(fw, "all" if a.command == "build" else "run", args)
            out = (r.stdout or "") + (r.stderr or "")
            if a.command == "build":
                path = elf_path(fw, args)
                good = r.returncode == 0 and path and os.path.isfile(path)
                line = f"  {fw:4} {p:5}  " + (path if good else "FAILED" + (": does not fit the target's memory"
                                                                              if "do not fit" in out else ""))
                if good and a.out:
                    os.makedirs(a.out, exist_ok=True)
                    shutil.copy(path, os.path.join(a.out, os.path.basename(path)))
            else:
                m = RESULT_RE.search(out)
                good = r.returncode == 0 and "RESULT PASS" in out
                line = f"  {fw:4} {p:5}  " + ("PASS" if good else "FAIL") + (
                    f"  {m.group(2)} tests, {m.group(3)} off, max |diff| {m.group(4)}, {m.group(5)} "
                    f"{'ns' if soc == 'host' else 'cycles'} per inference{m.group(6)}" if m else "")
            print(line, flush=True)
            ok &= bool(good)
    return 0 if ok else 1


def elf_path(fw, args):
    """$(OUT)/$(APP).elf as the SDK's make sets it."""
    r = subprocess.run(["make", "--no-print-directory", "-s", "-f", os.path.join(ROOT, "examples", "common",
                        f"{fw}_example.mk"), "-f", "-", "aisdk-elf"] + args, input="aisdk-elf:\n\t@echo $(OUT)/$(APP).elf\n",
                       cwd=os.path.join(ROOT, "examples", "common"), capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    return lines[-1] if r.returncode == 0 and lines else None


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
