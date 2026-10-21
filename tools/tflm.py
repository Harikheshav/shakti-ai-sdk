#!/usr/bin/env python3
"""TensorFlow Lite Micro for mk/tflm.mk: our patches applied, its downloads unpacked, its source lists.

usage: tools/tflm.py prepare OUTDIR

  1. frameworks/tflite-micro carries patches/tflite-micro (applied again only when the patches change)
  2. the sources TFLM downloads while building (third_party.lock tflm-*: flatbuffers, kissfft, eyalroz_printf),
     from the cache, unpacked into OUTDIR/downloads/<name> and patched as TFLM's download_dependencies.py does
  3. OUTDIR/sources.mk: TFLM_SRCS (its own `make list_library_sources TARGET=shakti`, so the SHAKTI target's
     debug_log / micro_time / system_setup replace the generic ones) and TFLM_THIRD_PARTY_SRCS
"""
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFLM = os.path.join(ROOT, "frameworks", "tflite-micro")
MK = "tensorflow/lite/micro/tools/make/Makefile"
DEPS = {                                     # name -> TFLM's own patch for it (download_dependencies.py)
    "flatbuffers": "tensorflow/lite/micro/tools/make/flatbuffers.patch",
    "kissfft": "third_party/kissfft/kissfft.patch",
    "eyalroz_printf": None,
}
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fetch  # noqa: E402


def digest(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(open(p, "rb").read())
    return h.hexdigest()


def patched():
    import frameworks
    frameworks.ensure_patched("tflite-micro")


def unpack(out):
    entries = fetch.entries()
    for name, patch in DEPS.items():
        e = entries[f"tflm-{name}"]
        dest = os.path.join(out, "downloads", name)
        stamp = dest + ".stamp"
        want = e["sha256"] + (digest([os.path.join(TFLM, patch)]) if patch else "")
        if os.path.isfile(stamp) and open(stamp).read() == want:
            continue
        shutil.rmtree(dest, ignore_errors=True)
        tmp = dest + ".unpack"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(fetch.fetch(f"tflm-{name}", e)) as z:
            z.extractall(tmp)
        top = [p for p in os.listdir(tmp) if p != "__MACOSX"]
        shutil.move(os.path.join(tmp, top[0]) if len(top) == 1 else tmp, dest)
        shutil.rmtree(tmp, ignore_errors=True)
        if patch:
            with open(os.path.join(TFLM, patch), "rb") as f:
                subprocess.run(["patch", "-s", "-p1", "-d", dest], stdin=f, check=True)
        open(stamp, "w").write(want)


def make_env():
    """TFLM's Makefile runs `python3` (generate_cc_arrays.py needs numpy and Pillow): this Python's, first on PATH."""
    return {**os.environ, "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")}


def make_list(target):
    r = subprocess.run(["make", "-s", "--no-print-directory", "-f", MK, "TARGET=shakti", "SKIP_DOWNLOADS=1", target], cwd=TFLM,
                       capture_output=True, text=True, env=make_env())
    if r.returncode:
        sys.exit(f"make {target}: {r.stderr.strip()}")
    return r.stdout.split()


def flags(out):
    """TFLM's own compile flags and include paths for TARGET=shakti (its Makefile evaluated), without the ISA, ABI
    and code-model flags, which come from the SoC; downloads/ points at OUTDIR/downloads."""
    wrapper = os.path.join(out, "print_flags.mk")
    with open(wrapper, "w") as f:
        f.write(f"include {MK}\nshakti_print_flags:\n\t@echo \"$(CXXFLAGS)\" ; echo \"$(CCFLAGS)\" ; echo \"$(INCLUDES)\"\n")
    r = subprocess.run(["make", "-s", "--no-print-directory", "-f", wrapper, "TARGET=shakti", "SKIP_DOWNLOADS=1", "shakti_print_flags"],
                       cwd=TFLM, capture_output=True, text=True, env=make_env())
    if r.returncode:
        sys.exit(f"TFLM flags: {r.stderr.strip()}")
    cxx, cc, inc = (r.stdout.splitlines() + ["", "", ""])[:3]
    dl = os.path.join(TFLM, "tensorflow/lite/micro/tools/make/downloads")

    def clean(v):
        out_ = []
        for w in v.split():
            if w.startswith(("-march=", "-mabi=", "-mcmodel=")) or w.startswith("-W"):
                continue                               # the SoC's ISA; warnings are the app's choice
            if w.startswith("-I"):
                p = w[2:]
                p = p if os.path.isabs(p) else os.path.normpath(os.path.join(TFLM, p))
                if p.startswith(os.path.join(TFLM, "gen")):
                    continue                           # TFLM's own build output: not used here
                w = "-I" + p.replace(dl, os.path.join(out, "downloads"))
            out_.append(w)
        return " ".join(dict.fromkeys(out_))
    return clean(cxx), clean(cc), clean(inc)


def sources(out):
    dl = "tensorflow/lite/micro/tools/make/downloads/"
    lib = [os.path.join(TFLM, s) for s in make_list("list_library_sources")]
    third = [os.path.join(out, "downloads", s.split(dl, 1)[1]) for s in make_list("list_third_party_sources")
             if s.endswith((".c", ".cc", ".cpp"))]
    with open(os.path.join(out, "sources.mk"), "w") as f:
        f.write("# generated by tools/tflm.py: TFLM's own source lists for TARGET=shakti\n")
        f.write("TFLM_SRCS := " + " \\\n  ".join(lib) + "\n")
        f.write("TFLM_THIRD_PARTY_SRCS := " + " \\\n  ".join(third) + "\n")
        cxx, cc, inc = flags(out)
        f.write(f"TFLM_CXXFLAGS := {cxx}\nTFLM_CCFLAGS := {cc}\nTFLM_INCLUDES := {inc}\n")
    return len(lib), len(third)


def main(argv):
    if len(argv) != 2 or argv[0] != "prepare":
        sys.exit(__doc__)
    out = os.path.abspath(argv[1])
    os.makedirs(out, exist_ok=True)
    patched()
    unpack(out)
    n, t = sources(out)
    shutil.rmtree(os.path.join(TFLM, "gen"), ignore_errors=True)   # TFLM's make leaves its gen/ directory
    print(f"tflite-micro: {n} sources, {t} third-party sources")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
