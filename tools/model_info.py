#!/usr/bin/env python3
"""A model file's summary without training it: framework, input and output shapes, layers with their parameters,
parameters in all, multiply-accumulates per inference (dense and convolution layers), sizes as float and int8.

usage: tools/model_info.py MODEL.py      (conversion environment) -> JSON on stdout

The model file defines model() -> the untrained network (PyTorch torch.nn.Module or keras.Model) and INPUT_SHAPE
(the AI app templates do).  On a mistake in the file the JSON has "error", the line ("line") and the Python
message, for the editor to show.
"""
import importlib.util
import json
import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def load(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location("model_info_target", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def torch_info(net, shape):
    import torch
    layers, macs = [], [0]

    def hook(mod, inp, out):
        x, y = inp[0], out
        if isinstance(mod, torch.nn.Linear):
            macs[0] += int(y.numel()) * mod.in_features
        elif isinstance(mod, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            k = 1
            for d in mod.kernel_size:
                k *= d
            macs[0] += int(y.numel()) * (mod.in_channels // mod.groups) * k
        layers.append({"name": layer_name(mod), "type": type(mod).__name__, "output": list(y.shape) if hasattr(y, "shape") else None,
                       "params": sum(p.numel() for p in mod.parameters(recurse=False))})
    names = {m: n for n, m in net.named_modules()}

    def layer_name(mod):
        return names.get(mod, "")
    hooks = [m.register_forward_hook(hook) for m in net.modules() if not list(m.children())]
    net.eval()
    with torch.no_grad():
        out = net(torch.zeros(shape))
    for h in hooks:
        h.remove()
    return {"framework": "pytorch", "output": list(out.shape), "layers": layers,
            "params": int(sum(p.numel() for p in net.parameters())), "macs": macs[0]}


def keras_info(net, shape):
    import numpy as np
    out = net(np.zeros(shape, dtype=np.float32))
    layers, macs = [], 0
    for layer in net.layers:
        o = layer.output.shape if hasattr(layer, "output") else None
        o = [d if d is not None else 1 for d in o] if o is not None else None
        kind = type(layer).__name__
        cfg = layer.get_config()
        if kind == "Dense" and o:
            macs += int(np.prod(o)) * int(layer.kernel.shape[0])
        elif kind.startswith("Conv") and o and hasattr(layer, "kernel"):
            macs += int(np.prod(o)) * int(np.prod(layer.kernel.shape[:-1])) // max(1, cfg.get("groups", 1))
        elif kind == "DepthwiseConv2D" and o:
            macs += int(np.prod(o)) * int(np.prod(layer.kernel.shape[:2]))
        layers.append({"name": layer.name, "type": kind, "output": o, "params": int(layer.count_params())})
    return {"framework": "keras", "output": list(out.shape), "layers": layers, "params": int(net.count_params()),
            "macs": macs}


def main(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    path = os.path.abspath(argv[0])
    try:
        m = load(path)
        if not hasattr(m, "model"):
            raise ValueError("the model file has no model() returning the untrained network")
        shape = tuple(int(d) for d in m.INPUT_SHAPE)
        net = m.model()
        if any(c.__module__.split(".")[0] in ("keras", "tf_keras") for c in type(net).__mro__):
            info = keras_info(net, shape)
        else:
            info = torch_info(net, shape)
        info["input"] = list(shape)
        info["bytes"] = {"float": info["params"] * 4, "int8": info["params"]}
        for f in ("build", "calibration", "test_inputs"):
            if not callable(getattr(m, f, None)):
                info.setdefault("missing", []).append(f)
    except Exception as e:                           # noqa: BLE001 - the user's code: report it
        tb = traceback.extract_tb(e.__traceback__)
        line = next((f.lineno for f in reversed(tb) if os.path.abspath(f.filename) == path), None)
        if isinstance(e, SyntaxError) and os.path.abspath(e.filename or "") == path:
            line = e.lineno
        info = {"error": f"{type(e).__name__}: {e}", "line": line,
                "trace": "".join(traceback.format_exception(type(e), e, e.__traceback__))[-3000:]}
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
