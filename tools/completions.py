#!/usr/bin/env python3
"""Names for the AI app editor's completion, read from the frameworks installed in the conversion environment:
each public class and function of the modules a model file uses, with its signature and the first line of its
documentation.

usage: tools/completions.py OUT.json      (conversion environment)
OUT.json: {"pytorch": [{"name", "sig", "doc", "kind"}], "keras": [...], "common": [...]}
"""
import importlib
import inspect
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

MODULES = {
    "pytorch": ["torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.utils.data", "torchvision.datasets",
                "torchvision.transforms", "torchvision.models"],
    "keras": ["keras", "keras.layers", "keras.optimizers", "keras.losses", "keras.metrics", "keras.datasets",
              "keras.callbacks", "keras.utils", "keras.activations"],
    "common": ["numpy"],
}


def entries(modname):
    try:
        mod = importlib.import_module(modname)
    except Exception:                                # noqa: BLE001 - a module this environment lacks
        return []
    out = []
    for name in sorted(dir(mod)):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(mod, name)
        except Exception:                            # noqa: BLE001
            continue
        if inspect.ismodule(obj):
            out.append({"name": f"{modname}.{name}", "sig": "", "doc": "module", "kind": "module"})
            continue
        if not callable(obj):
            continue
        kind = "class" if inspect.isclass(obj) else "function"
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            sig = "(...)"
        doc = (inspect.getdoc(obj) or "").strip().split("\n")[0][:160]
        out.append({"name": f"{modname}.{name}", "sig": sig[:300], "doc": doc, "kind": kind})
    return out


def main(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    data = {k: [e for m in mods for e in entries(m)] for k, mods in MODULES.items()}
    os.makedirs(os.path.dirname(os.path.abspath(argv[0])), exist_ok=True)
    with open(argv[0], "w") as f:
        json.dump(data, f)
    print({k: len(v) for k, v in data.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
