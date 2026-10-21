#!/usr/bin/env python3
"""What TensorFlow Lite Micro runs for an exported model, step by step (the counterpart of tvm_aot / mlir_aot
--stages): the operators in execution order with their tensors, and the tensors (shape, type, quantisation, size).

usage: tools/tflm_stages.py MODEL_DIR float|int8 OUTDIR      (MODEL_DIR from tools/export.py; conversion environment)
Writes OUTDIR/01_operators.txt, 02_tensors.txt.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    return it


def describe(t):
    q = t.get("quantization") or (0.0, 0)
    quant = f"  scale {q[0]:.6g} zero {q[1]}" if q and q[0] else ""
    return f"{t['name'][:60]:60} {str([int(d) for d in t['shape']]):18} {t['dtype'].__name__:8}{quant}"


def main(argv):
    if len(argv) != 3 or argv[1] not in ("float", "int8"):
        sys.exit(__doc__)
    model_dir, check, out = argv
    path = os.path.join(model_dir, f"model_{check}.tflite")
    it = interpreter(path)
    tensors = {t["index"]: t for t in it.get_tensor_details()}
    os.makedirs(out, exist_ok=True)
    lines = [f"# {os.path.basename(path)}: the operators TensorFlow Lite Micro runs, in order "
             f"({os.path.getsize(path)} bytes in the C array)", ""]
    for n, op in enumerate(it._get_ops_details()):
        lines.append(f"{n:3}  {op['op_name']}")
        for kind in ("inputs", "outputs"):
            for i in op[kind]:
                if i >= 0:
                    lines.append(f"       {kind[:-1]:6} {describe(tensors[i])}")
    open(os.path.join(out, "01_operators.txt"), "w").write("\n".join(lines) + "\n")
    total = 0
    lines = [f"# {os.path.basename(path)}: tensors (constant ones are in the C array, the others in the arena)", ""]
    for t in tensors.values():
        size = int(t["dtype"]().itemsize) * int(max(1, __import__("numpy").prod(t["shape"])))
        total += size
        lines.append(f"{t['index']:3}  {describe(t)}  {size} bytes")
    lines += ["", f"# {len(tensors)} tensors, {total} bytes"]
    open(os.path.join(out, "02_tensors.txt"), "w").write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
