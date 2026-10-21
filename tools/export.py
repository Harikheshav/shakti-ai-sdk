#!/usr/bin/env python3
"""Export a model for every framework, with reference outputs.

usage: tools/export.py NAME|MODEL.py OUTDIR      (run with the conversion environment: env/requirements-convert.lock)

NAME: a common example (models/<name>/model.py); MODEL.py: any model file (an AI app's).  It provides INPUT_SHAPE,
build() -> a trained model in PyTorch (torch.nn.Module, eval mode) or TensorFlow / Keras (keras.Model),
calibration() and test_inputs() -> lists of numpy arrays (one input each, INPUT_SHAPE; NCHW for PyTorch images,
NHWC for Keras).  The model file's directory is on the import path (a data.py next to it).  Written to OUTDIR:
  model.onnx                      PyTorch -> ONNX, or Keras -> ONNX (tf2onnx) (for TVM; MLIR takes a PyTorch model
                                  itself, a Keras one through this ONNX)
  model_int8.onnx                 the same, QDQ int8 (tools/onnx_int8.py), for TVM and MLIR
  model_float.tflite              ONNX -> TensorFlow (onnx2tf) -> TFLite (PyTorch), or Keras -> TFLite, float32
  model_int8.tflite               the same, full integer: int8 weights, activations, input and output
                                  (calibrated with calibration())
  input_float.csv / output_ref_float.csv     test inputs and the float model's outputs on the host
  input_int8.csv / output_ref_int8.csv       the inputs quantised with the model's input scale, and its int8 outputs
  ops.h                           MODEL_OPS(X): the TFLite operators the models use (for a MicroMutableOpResolver)
  model.json                      shapes, quantisation, sizes, reference values from PyTorch and both TFLite models
The host references come from TensorFlow Lite Micro itself built for the host (tools/tflm_host.py); the TFLite
interpreter's outputs (builtin kernels) are a cross-check in model.json.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")     # CPU only (TensorFlow's graph tools probe CUDA otherwise)


def model_path(name):
    """A common example's model file, or the path given."""
    if name.endswith(".py"):
        return os.path.abspath(name)
    return os.path.join(ROOT, "models", name, "model.py")


def load(name):
    path = model_path(name)
    if not os.path.isfile(path):
        sys.exit(f"export: no model file {path}")
    sys.path.insert(0, os.path.dirname(path))          # its own helpers (data.py ...)
    stem = os.path.basename(os.path.dirname(path)) if os.path.basename(path) == "model.py" else os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(f"model_{stem}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def is_keras(net):
    return any(c.__module__.split(".")[0] in ("keras", "tf_keras") for c in type(net).__mro__)


def write_csv(path, rows, fmt):
    with open(path, "w") as f:
        f.write(",".join(fmt % v for v in np.concatenate([r.reshape(-1) for r in rows])) + "\n")


def interpreter(path, allocate=True):
    # the TFLite interpreter is a cross-check only (TFLite Micro on the host is the reference); its reference
    # kernels (BUILTIN_REF) abort on ResNet-18 int8 and its XNNPACK delegate cannot prepare it (ai-edge-litert
    # 2.3.0), so the builtin optimised kernels without delegates are used, in a child process (crosscheck())
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    it = Interpreter(model_path=path, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    if allocate:
        it.allocate_tensors()
    return it


def run(it, x):
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], x.astype(i["dtype"]))
    it.invoke()
    return it.get_tensor(o["index"]).copy()


def crosscheck(model, inputs):
    """The TFLite interpreter's outputs for these inputs, or None when it fails (run in a child process: it can
    abort the whole process)."""
    tmp = model + ".xcheck"
    np.save(tmp + ".in.npy", np.stack(inputs))
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--crosscheck", model, tmp + ".in.npy",
                        tmp + ".out.npy"], capture_output=True, text=True)
    out = list(np.load(tmp + ".out.npy")) if r.returncode == 0 else None
    for f in (tmp + ".in.npy", tmp + ".out.npy"):
        if os.path.exists(f):
            os.remove(f)
    return out


MICRO_NAMES = {"PADV2": "PadV2"}                     # TFLite op names whose TFLM Add<Name>() is not the CamelCase


def op_names(path):
    it = interpreter(path, allocate=False)
    names = sorted({d["op_name"] for d in it._get_ops_details()})

    def micro(n):                                    # CONV_2D -> Conv2D, FULLY_CONNECTED -> FullyConnected
        return MICRO_NAMES.get(n) or "".join(w if w[:1].isdigit() else w.capitalize() for w in n.split("_"))
    return [micro(n) for n in names]


def main(argv):
    if argv and argv[0] == "--crosscheck":            # child process of crosscheck()
        it = interpreter(argv[1])
        np.save(argv[3], np.stack([run(it, x) for x in np.load(argv[2])]))
        return 0
    if len(argv) != 2:
        sys.exit(__doc__)
    name, out = argv[0], os.path.abspath(argv[1])
    os.makedirs(out, exist_ok=True)
    m = load(name)
    net = m.build()
    if is_keras(net):
        return export_keras(m, net, name, out)
    import torch
    x0 = torch.from_numpy(m.test_inputs()[0])

    onnx_path = os.path.join(out, "model.onnx")
    torch.onnx.export(net, x0, onnx_path, input_names=["input"], output_names=["output"], opset_version=17,
                      dynamo=False)
    # fold torch's shape arithmetic (e.g. the pads of a ZeroPad2d) into constants, so onnx2tf and TVM see plain
    # operators; a Pad is kept separate from the pool after it (onnxsim would merge a zero pad into a MaxPool,
    # which onnx2tf then pads with the type's minimum: an int8 scale of ~1e36)
    import onnx
    import onnxsim
    simple, ok = onnxsim.simplify(onnx.load(onnx_path), skipped_optimizers=["fuse_pad_into_pool"])
    if not ok:
        sys.exit("onnxsim could not check the simplified model")
    onnx.save(simple, onnx_path)

    # ONNX -> TensorFlow SavedModel (onnx2tf), then TFLite float and full-integer int8
    tfdir = os.path.join(out, "tf")
    shutil.rmtree(tfdir, ignore_errors=True)
    r = subprocess.run([sys.executable, "-m", "onnx2tf", "-i", onnx_path, "-o", tfdir, "-tb", "tf_converter", "-osd", "-nuo", "-v", "error"],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"onnx2tf failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    import tensorflow as tf
    calib = m.calibration()

    def layout(a):                                   # onnx2tf turns NCHW inputs into NHWC
        return np.transpose(a, (0, 2, 3, 1)) if a.ndim == 4 else a

    conv = tf.lite.TFLiteConverter.from_saved_model(tfdir)
    open(os.path.join(out, "model_float.tflite"), "wb").write(conv.convert())
    conv = tf.lite.TFLiteConverter.from_saved_model(tfdir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([layout(c)] for c in calib)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    open(os.path.join(out, "model_int8.tflite"), "wb").write(conv.convert())

    tests = [layout(t) for t in m.test_inputs()]
    with torch.no_grad():
        torch_out = [net(torch.from_numpy(t)).numpy() for t in m.test_inputs()]
    shutil.rmtree(tfdir, ignore_errors=True)
    return references(m, name, out, onnx_path, tests, torch_out, "pytorch")


def export_keras(m, net, name, out):
    """Keras: TFLite straight from the model (TensorFlow's converter), ONNX through tf2onnx; inputs are NHWC as the
    model takes them."""
    import onnx
    import onnxsim
    import tensorflow as tf
    import tf2onnx
    shape = (1,) + tuple(int(d) for d in m.INPUT_SHAPE[1:])
    fn = tf.function(lambda input: net(input))
    proto, _ = tf2onnx.convert.from_function(fn, input_signature=(tf.TensorSpec(shape, tf.float32, name="input"),),
                                             opset=17)
    simple, ok = onnxsim.simplify(proto)
    onnx_path = os.path.join(out, "model.onnx")
    onnx.save(simple if ok else proto, onnx_path)
    conv = tf.lite.TFLiteConverter.from_keras_model(net)
    open(os.path.join(out, "model_float.tflite"), "wb").write(conv.convert())
    conv = tf.lite.TFLiteConverter.from_keras_model(net)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    calib = m.calibration()
    conv.representative_dataset = lambda: ([c] for c in calib)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    open(os.path.join(out, "model_int8.tflite"), "wb").write(conv.convert())
    tests = list(m.test_inputs())
    keras_out = [np.asarray(net(t)) for t in tests]
    return references(m, name, out, onnx_path, tests, keras_out, "keras")


def references(m, name, out, onnx_path, tests, src_out, source):
    """Test data and host references for the exported files: TFLite Micro on the host, the TFLite interpreter as a
    cross-check, the int8 ONNX; model.json."""
    q_info = interpreter(os.path.join(out, "model_int8.tflite"), allocate=False)
    qi, qo = q_info.get_input_details()[0], q_info.get_output_details()[0]
    (si, zi), (so, zo) = qi["quantization"], qo["quantization"]
    q_in = [np.clip(np.round(t / si) + zi, -128, 127).astype(np.int8) for t in tests]
    out_size = int(np.prod(q_info.get_output_details()[0]["shape"]))
    f_out = crosscheck(os.path.join(out, "model_float.tflite"), tests)
    q_out = crosscheck(os.path.join(out, "model_int8.tflite"), q_in)

    write_csv(os.path.join(out, "input_float.csv"), tests, "%.9g")
    write_csv(os.path.join(out, "input_int8.csv"), q_in, "%d")
    ops = sorted(set(op_names(os.path.join(out, "model_float.tflite"))) | set(op_names(os.path.join(out, "model_int8.tflite"))))
    with open(os.path.join(out, "ops.h"), "w") as f:
        f.write("/* generated by tools/export.py: the TFLite operators of model_float / model_int8 */\n")
        f.write("#define MODEL_OPS(X) " + " ".join(f"X({o})" for o in ops) + "\n")
        f.write(f"#define MODEL_NUM_OPS {len(ops)}\n")
        f.write(f"#define MODEL_NUM_TESTS {len(tests)}\n")
        f.write(f"#define MODEL_INPUT_SIZE {tests[0].size}\n#define MODEL_OUTPUT_SIZE {out_size}\n")
        f.write(f"#define MODEL_OUTPUT_SCALE {so:.9g}f\n#define MODEL_OUTPUT_ZERO {zo}\n")
    # the reference the target is compared with: TensorFlow Lite Micro itself on the host (bit-exact for int8);
    # the TFLite interpreter's outputs above stay as a cross-check
    arena = os.environ.get("TFLM_HOST_ARENA_KB", "65536")
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "tflm_host.py"), out, arena], check=True)

    def read(path, dtype):
        return np.array([float(x) for x in open(path).read().split(",")], dtype=dtype)
    import onnx_int8
    onnx_int8.quantize(m, out)
    onnx_int8_diff = onnx_int8.compare(onnx_path, os.path.join(out, "model_int8.onnx"), m.test_inputs())
    tflm_q = read(os.path.join(out, "output_ref_int8.csv"), np.int32)
    tflm_f = read(os.path.join(out, "output_ref_float.csv"), np.float32)
    tflm_deq = (tflm_q.astype(np.float32) - zo) * so
    src_flat = np.concatenate([np.asarray(o).reshape(-1) for o in src_out])
    info = {
        "model": os.path.basename(name), "source": source, "input_shape": list(tests[0].shape), "output_size": out_size,
        "input_quant": [float(si), int(zi)], "output_quant": [float(so), int(zo)], "ops": ops,
        "sizes": {f: os.path.getsize(os.path.join(out, f)) for f in ("model.onnx", "model_float.tflite", "model_int8.tflite")},
        "tflm_float_vs_source_max_abs": float(np.abs(tflm_f - src_flat).max()),
        "tflm_int8_vs_source_max_abs": float(np.abs(tflm_deq - src_flat).max()),
        "onnx_int8_vs_float_max_abs": onnx_int8_diff,
        "tflite_crosscheck": {
            "float_max_abs": None if f_out is None else float(np.abs(tflm_f - np.concatenate([o.reshape(-1) for o in f_out])).max()),
            "int8_max_steps": None if q_out is None else int(np.abs(tflm_q - np.concatenate([o.reshape(-1) for o in q_out])).max()),
            "note": "TFLite interpreter (builtin kernels); None: it failed on this model"},
    }
    if getattr(m, "NOTE", None):
        info["note"] = m.NOTE
    json.dump(info, open(os.path.join(out, "model.json"), "w"), indent=1)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
