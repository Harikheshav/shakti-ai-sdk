"""The common examples' test data for the ahead-of-time compilers (tools/tvm_aot.py, tools/mlir_aot.py): the inputs
and host references written by tools/export.py, in the model's own layout."""
import json
import os

import numpy as np


def tests(model_dir, shape, out_size):
    """(number of tests, inputs then reference outputs as float32 bytes).  shape: the model's input shape; 4-D
    inputs are NHWC in input_float.csv (TFLite's layout): transposed to NCHW for a PyTorch model's, kept for a
    Keras model's (NHWC already; model.json "source")."""
    def read(name):
        with open(os.path.join(model_dir, name)) as f:
            return np.array([float(v) for v in f.read().split(",")], dtype=np.float32)
    x, ref = read("input_float.csv"), read("output_ref_float.csv")
    n = ref.size // out_size
    info = os.path.join(model_dir, "model.json")
    nchw = len(shape) == 4 and (not os.path.isfile(info) or json.load(open(info)).get("source", "pytorch") != "keras")
    x = x.reshape([n] + shape[1:] if not nchw else [n, shape[2], shape[3], shape[1]])
    if nchw:
        x = x.transpose(0, 3, 1, 2)                       # NHWC -> NCHW
    return n, np.ascontiguousarray(x).tobytes() + ref.tobytes()


def asm(path, prefix, n_in_bytes):
    """Assembly linking tests.bin as <prefix>_tests_input and <prefix>_tests_ref."""
    return ['  .section .rodata.tests, "a"', "  .balign 16", f"  .global {prefix}_tests_input, {prefix}_tests_ref",
            f"{prefix}_tests_input:", f'  .incbin "{path}", 0, {n_in_bytes}',
            f"{prefix}_tests_ref:", f'  .incbin "{path}", {n_in_bytes}']
