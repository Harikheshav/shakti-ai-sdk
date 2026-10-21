#!/usr/bin/env python3
"""The frameworks (git submodules under frameworks/, pinned) and our changes to them (patches/<name>/*.patch).

The submodules stay pristine at their pinned commits; every change is a patch file, applied in name order.

usage: tools/frameworks.py status                  pinned commit, fetched or not, patches and whether they apply
       tools/frameworks.py fetch NAME [--recursive] fetch one framework at its pin (shallow), with the nested
                                                   submodules it needs (NESTED; --recursive: all of them)
       tools/frameworks.py apply NAME              apply NAME's patches to its working tree (after a reset)
       tools/frameworks.py check NAME              do the patches apply cleanly on the pinned commit? (no change)
       tools/frameworks.py reset NAME              back to the pinned commit, patches removed
       tools/frameworks.py record NAME TITLE       save the working tree's changes as the next patch
                                                   (patches/NAME/NNNN-TITLE.patch), then reset
"""
import os
import re
import subprocess
import sys

ROOT = os.environ.get("SHAKTI_AI_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ("tflite-micro", "tvm", "torch-mlir", "llvm-project")
# the frameworks' own submodules that a SHAKTI build needs (the rest - CUDA kernels, OpenCL headers ... - stay out)
NESTED = {"tvm": ["3rdparty/tvm-ffi"]}


def git(*args, cwd=ROOT, check=True, capture=True):
    r = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=capture)
    if check and r.returncode:
        sys.exit(f"git {' '.join(args)} failed in {cwd}:\n{r.stderr.strip()}")
    return r


def path(name):
    if name not in NAMES:
        sys.exit(f"unknown framework '{name}' (one of {', '.join(NAMES)})")
    return os.path.join(ROOT, "frameworks", name)


def pin(name):
    out = git("ls-files", "-s", f"frameworks/{name}").stdout.split()
    return out[1] if out else None


def fetched(name):
    return os.path.exists(os.path.join(path(name), ".git"))


def patches(name):
    d = os.path.join(ROOT, "patches", name)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".patch")) if os.path.isdir(d) else []


def reset(name):
    p = path(name)
    git("checkout", "-q", "--force", pin(name), cwd=p)
    git("clean", "-qfdx", "-e", "/tensorflow/lite/micro/tools/make/downloads/", cwd=p)   # keep TFLM's cache


def ensure_patched(name):
    """The framework's working tree carries its patches: applied again (reset first) only when the patches or the
    pin changed since the last time (stamp: build/<name>.patches)."""
    import hashlib
    h = hashlib.sha256(pin(name).encode())
    for f in patches(name):
        with open(f, "rb") as fh:
            h.update(fh.read())
    stamp = os.path.join(ROOT, "build", f"{name}.patches")
    if not fetched(name):
        sys.exit(f"frameworks/{name} is not fetched: tools/frameworks.py fetch {name}")
    if os.path.isfile(stamp) and open(stamp).read() == h.hexdigest():
        return
    if not apply(name):
        sys.exit(f"patches/{name} do not apply")
    os.makedirs(os.path.dirname(stamp), exist_ok=True)
    with open(stamp, "w") as fh:
        fh.write(h.hexdigest())


def apply(name, only_check=False):
    p = path(name)
    if not fetched(name):
        sys.exit(f"{name} is not fetched: tools/frameworks.py fetch {name}")
    reset(name)
    for f in patches(name):
        r = git("apply", "--check", "--whitespace=nowarn", f, cwd=p, check=False)
        if r.returncode:
            print(f"{os.path.basename(f)}: does not apply\n{r.stderr.strip()}")
            reset(name)
            return False
        git("apply", "--whitespace=nowarn", f, cwd=p)
        print(f"{os.path.basename(f)}: applied")
    if only_check:
        reset(name)
    return True


def record(name, title):
    """The working tree's changes beyond the existing patches become the next patch: the diff between the tree of
    pin + patches (built in a temporary index) and the working tree."""
    p = path(name)
    git("add", "-A", cwd=p)
    work = git("write-tree", cwd=p).stdout.strip()
    git("reset", "-q", cwd=p)
    idx = os.path.join(git("rev-parse", "--absolute-git-dir", cwd=p).stdout.strip(), "shakti-record-index")
    env = {**os.environ, "GIT_INDEX_FILE": idx}

    def g(*args):
        r = subprocess.run(["git", *args], cwd=p, env=env, text=True, capture_output=True)
        if r.returncode:
            sys.exit(f"git {' '.join(args)}: {r.stderr.strip()}")
        return r.stdout
    try:
        g("read-tree", pin(name))
        have = patches(name)
        for f in have:
            g("apply", "--cached", "--whitespace=nowarn", f)
        base = g("write-tree").strip()
    finally:
        if os.path.exists(idx):
            os.remove(idx)
    diff = git("diff", "--binary", base, work, cwd=p).stdout
    if not diff.strip():
        sys.exit("no changes beyond the existing patches")
    n = 1 + max([int(os.path.basename(f)[:4]) for f in have if re.match(r"\d{4}-", os.path.basename(f))] or [0])
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    out = os.path.join(ROOT, "patches", name, f"{n:04d}-{slug}.patch")
    with open(out, "w") as f:
        f.write(diff)
    print(f"wrote {os.path.relpath(out, ROOT)}")


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    if cmd == "status":
        for n in NAMES:
            if pin(n) is None:                          # not registered in this checkout
                continue
            ps = patches(n)
            ok = "" if not fetched(n) or not ps else ("  patches apply" if apply(n, only_check=True) else "  PATCHES FAIL")
            print(f"{n:13s} {pin(n)[:12]}  {'fetched' if fetched(n) else 'not fetched'}  {len(ps)} patch(es){ok}")
    elif cmd == "fetch":
        n = args[0]
        git("submodule", "update", "--init", "--depth", "1", *(["--recursive"] if "--recursive" in args else []),
            f"frameworks/{n}", capture=False)
        for sub in NESTED.get(n, []):                   # only the nested submodules a SHAKTI build needs
            git("submodule", "update", "--init", "--recursive", "--depth", "1", sub, cwd=path(n), capture=False)
    elif cmd == "apply":
        return 0 if apply(args[0]) else 1
    elif cmd == "check":
        return 0 if apply(args[0], only_check=True) else 1
    elif cmd == "reset":
        reset(args[0])
    elif cmd == "record":
        record(args[0], " ".join(args[1:]) or "change")
    else:
        sys.exit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
