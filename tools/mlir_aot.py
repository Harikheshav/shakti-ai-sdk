#!/usr/bin/env python3
"""Compile a model with MLIR into LLVM IR for a bare-metal SHAKTI target: ahead of time, no runtime.

usage: tools/mlir_aot.py (--torch MODEL | --onnx FILE) OUTDIR [--tests MODEL_DIR [--host-ref]]
       (conversion environment; MLIR: tools/mlir_build.py build)

--torch MODEL: models/<MODEL>/model.py, imported by torch-mlir's FX importer.  --onnx FILE: an ONNX model (the
QDQ int8 export: tools/onnx_int8.py), imported by torch-mlir's ONNX importer; its quantised convolutions and
matrix products become integer linalg operations (torch-mlir's QDQ fusion; patches/torch-mlir).  Either way the
model becomes linalg on tensors; upstream MLIR bufferizes it (one-shot), lowers linalg to loops and everything to
the LLVM dialect, and mlir-translate writes LLVM IR.  Written to OUTDIR:
  model64.ll, model32.ll  the model for RV64 and RV32 (MLIR's index type is the XLEN); opt and llc compile one
                          for the target (mk/mlir.mk).  The weights are constants in it.
  model.h, model.c        mlir_model_run(input, output): calls the model through its C interface (memref
                          descriptors); the buffers it allocates come from malloc
--stages DIR: also the model at each step (linalg, bufferized, loops, LLVM dialect, LLVM IR, after opt -O2).
With --tests, a common example's test data (MODEL_DIR from tools/export.py), linked by tests.S as
mlir_tests_input and mlir_tests_ref, with MLIR_NUM_TESTS in model.h.  --host-ref: the reference outputs are this
compiled model's own, built for and run on the host (opt and llc for the host; HOST_CC, default gcc), for a
bit-exact check (int8 models).
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlir_build  # noqa: E402
from aot_tests import asm, tests  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# linalg on tensors -> LLVM dialect, with upstream MLIR passes only; STAGES: where --stages shows the IR
PASSES = ["func.func(linalg-fuse-elementwise-ops)",
          "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
          "buffer-deallocation-pipeline",
          "convert-bufferization-to-memref",
          "func.func(convert-linalg-to-loops)",
          "func.func(arith-int-range-narrowing{int-bitwidths-supported=32})",
          "lower-affine", "expand-strided-metadata", "convert-scf-to-cf",
          "convert-math-to-llvm", "convert-math-to-libm",
          "convert-arith-to-llvm{index-bitwidth=%(w)d}",
          "finalize-memref-to-llvm{index-bitwidth=%(w)d}",
          "convert-func-to-llvm{index-bitwidth=%(w)d}",
          "convert-cf-to-llvm{index-bitwidth=%(w)d}",
          "reconcile-unrealized-casts"]
PIPELINE = "builtin.module(" + ",".join(PASSES) + ")"
STAGES = [("02_bufferized.mlir", 4), ("03_loops.mlir", 5), ("04_llvm_dialect.mlir", len(PASSES))]


def load(name):
    """A common example's model (models/<name>/model.py), or a model file given by its path."""
    path = os.path.abspath(name) if name.endswith(".py") else os.path.join(ROOT, "models", name, "model.py")
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location("model_" + os.path.basename(os.path.dirname(path)), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def c_interface(mod):
    """MLIR text of the module, its one function renamed @model, with a C interface."""
    from torch_mlir import ir
    funcs = [op for op in mod.body.operations if op.operation.name == "func.func"]
    if len(funcs) != 1:
        sys.exit("mlir_aot: the model must be one function")
    with mod.context:
        funcs[0].attributes["sym_name"] = ir.StringAttr.get("model")
        funcs[0].attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    return str(mod)


def linear_as_matmul(net):
    """Each nn.Linear as x @ W^T + b with W^T stored (a constant), in place: torch-mlir would transpose W at run
    time into a heap buffer each inference (its constants are resource blobs, which MLIR does not fold)."""
    import torch

    class Matmul(torch.nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.wt = torch.nn.Parameter(lin.weight.detach().t().contiguous(), requires_grad=False)
            self.b = None if lin.bias is None else torch.nn.Parameter(lin.bias.detach(), requires_grad=False)

        def forward(self, x):
            y = torch.matmul(x, self.wt)
            return y if self.b is None else y + self.b

    for name, child in list(net.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(net, name, Matmul(child))
        else:
            linear_as_matmul(child)
    return net


def from_torch(m):
    """The PyTorch model as linalg on tensors; its input and output shapes."""
    import torch
    from torch_mlir import fx
    net = linear_as_matmul(m.build())
    x = torch.from_numpy(m.test_inputs()[0])
    with torch.no_grad():
        out_shape = list(net(x).shape)
    mod = fx.export_and_import(net, x, output_type="linalg-on-tensors", func_name="model")
    return c_interface(mod), list(m.INPUT_SHAPE), out_shape


def gemm_weights_transposed(model):
    """Gemm(x, W, transB=1) with W a constant (or a per-tensor DequantizeLinear of one) -> W transposed, transB=0:
    no transpose at run time (see linear_as_matmul)."""
    from onnx import helper, numpy_helper
    inits = {i.name: i for i in model.graph.initializer}
    producer = {o: n for n in model.graph.node for o in n.output}
    users = {}
    for n in model.graph.node:
        for i in n.input:
            users[i] = users.get(i, 0) + 1
    for n in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        if n.op_type != "Gemm" or attrs.get("transB", 0) != 1 or attrs.get("transA", 0):
            continue
        w = n.input[1]
        dq = producer.get(w)
        if dq is not None and dq.op_type == "DequantizeLinear" and dq.input[0] in inits:
            scale = inits.get(dq.input[1])
            if scale is None or numpy_helper.to_array(scale).size != 1:
                continue
            w = dq.input[0]
        if w not in inits or users.get(w, 0) != 1 or (dq is not None and users.get(n.input[1], 0) != 1):
            continue
        t = numpy_helper.to_array(inits[w]).T.copy()
        inits[w].CopyFrom(numpy_helper.from_array(t, w))
        for a in n.attribute:
            if a.name == "transB":
                a.i = 0
    return model


# ONNX (torch dialect with onnx operators) -> torch backend -> QDQ fusion -> linalg on tensors
ONNX_PIPELINE = ("builtin.module(torch-onnx-to-torch-backend-pipeline,"
                 "func.func(torch-match-quantized-custom-ops,torch-fuse-quantized-ops,canonicalize),"
                 "torch-backend-to-linalg-on-tensors-backend-pipeline)")


def from_onnx(path, bindir):
    import onnx
    from torch_mlir import ir
    from torch_mlir.extras import onnx_importer
    from torch_mlir.dialects import torch as torch_d
    model = gemm_weights_transposed(onnx.load(path))
    model = onnx.shape_inference.infer_shapes(model, data_prop=True)   # as torch-mlir's import_onnx

    def shape(v):
        return [d.dim_value for d in v.type.tensor_type.shape.dim]
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    info = onnx_importer.ModelInfo(model)
    mod = info.create_module(context=ctx).operation
    onnx_importer.NodeImporter.define_function(info.main_graph, mod).import_all()
    r = subprocess.run([os.path.join(bindir, "torch-mlir-opt"), f"--pass-pipeline={ONNX_PIPELINE}"],
                       input=str(mod), capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"torch-mlir-opt failed:\n{r.stderr[-3000:]}")
    return (c_interface(ir.Module.parse(r.stdout, ctx)), shape(model.graph.input[0]),
            shape(model.graph.output[0]))


def lower(linalg, bindir, out, xlen):
    ll = os.path.join(out, f"model{xlen}.ll")
    opt = subprocess.run([os.path.join(bindir, "mlir-opt"), f"--pass-pipeline={PIPELINE % {'w': xlen}}"],
                         input=linalg, capture_output=True, text=True)
    if opt.returncode:
        sys.exit(f"mlir-opt failed:\n{opt.stderr[-3000:]}")
    tr = subprocess.run([os.path.join(bindir, "mlir-translate"), "--mlir-to-llvmir", "-o", ll], input=opt.stdout,
                        capture_output=True, text=True)
    if tr.returncode:
        sys.exit(f"mlir-translate failed:\n{tr.stderr[-3000:]}")


def stages(linalg, bindir, out, ll):
    """The model at each step (RV64): linalg on tensors, bufferized, loops, LLVM dialect, LLVM IR, after opt -O2."""
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "01_linalg.mlir"), "w") as f:
        f.write(linalg)
    for name, n in STAGES:
        r = subprocess.run([os.path.join(bindir, "mlir-opt"),
                            "--pass-pipeline=builtin.module(" + ",".join(PASSES[:n]) % {"w": 64} + ")"],
                           input=linalg, capture_output=True, text=True)
        with open(os.path.join(out, name), "w") as f:
            f.write(r.stdout if r.returncode == 0 else r.stderr)
    shutil.copy(ll, os.path.join(out, "05_llvm_ir.ll"))
    subprocess.run([os.path.join(bindir, "opt"), "-O2", "-S", ll, "-o", os.path.join(out, "06_llvm_ir_opt.ll")],
                   check=False)


def emit(out, name, in_shape, out_shape, n_tests):
    r_in, r_out = len(in_shape), len(out_shape)
    h = [f"/* generated by tools/mlir_aot.py: {name} */", "#ifndef MLIR_MODEL_H", "#define MLIR_MODEL_H",
         f"#define MLIR_INPUT_SIZE {int(np.prod(in_shape))}      /* float32 {in_shape} */",
         f"#define MLIR_OUTPUT_SIZE {int(np.prod(out_shape))}     /* float32 {out_shape} */"]
    if n_tests:
        h += [f"#define MLIR_NUM_TESTS {n_tests}", "extern const float mlir_tests_input[], mlir_tests_ref[];"]
    h += ["/* runs the model: 0, or 1 when its output is not where it should be */",
          "int mlir_model_run(const float *input, float *output);", "#endif"]

    def desc(r):        # MLIR's memref descriptor of rank r (index = intptr_t: the XLEN)
        return f"struct memref{r} {{ float *allocated, *aligned; intptr_t offset, sizes[{r}], strides[{r}]; }}"
    strides = [int(np.prod(in_shape[i + 1:])) for i in range(r_in)]
    c = [f"/* generated by tools/mlir_aot.py: {name}; the model is model<XLEN>.ll (opt, llc) */",
         "#include <stdint.h>", "#include <stdlib.h>", "#include <string.h>", '#include "model.h"', "",
         desc(r_in) + ";"] + ([desc(r_out) + ";"] if r_out != r_in else []) + [
         f"void _mlir_ciface_model(struct memref{r_out} *result, struct memref{r_in} *input);", "",
         "int mlir_model_run(const float *input, float *output)", "{",
         f"  struct memref{r_in} in = {{(float *)input, (float *)input, 0, {{{', '.join(map(str, in_shape))}}}, "
         f"{{{', '.join(map(str, strides))}}}}};",
         f"  struct memref{r_out} res;", "  _mlir_ciface_model(&res, &in);",
         "  if (!res.aligned)", "    return 1;",
         "  memcpy(output, res.aligned + res.offset, sizeof(float) * MLIR_OUTPUT_SIZE);",
         "  if (res.allocated != (float *)input)", "    free(res.allocated);", "  return 0;", "}"]
    for f, lines in (("model.h", h), ("model.c", c)):
        with open(os.path.join(out, f), "w") as fh:
            fh.write("\n".join(lines) + "\n")


def host_reference(out, bindir, n_tests, out_size, n_in_bytes):
    """Compile model64.ll for the host (opt and llc, as for the target), run the test inputs, and put its outputs
    into tests.bin as the reference."""
    bc, obj, exe, ref = (os.path.join(out, f) for f in ("host_model.bc", "host_model.o", "host_ref", "host_ref.bin"))
    subprocess.run([os.path.join(bindir, "opt"), "-O2", os.path.join(out, "model64.ll"), "-o", bc], check=True)
    subprocess.run([os.path.join(bindir, "llc"), "-O2", "-relocation-model=pic", "-filetype=obj", bc, "-o", obj],
                   check=True)
    subprocess.run([os.environ.get("HOST_CC", "gcc"), "-O2", "-ffp-contract=off", f"-I{out}",
                    '-DAOT_HEADER="model.h"', "-DAOT_RUN=mlir_model_run", "-DAOT_INPUT=mlir_tests_input",
                    "-DAOT_IN_SIZE=MLIR_INPUT_SIZE", "-DAOT_OUT_SIZE=MLIR_OUTPUT_SIZE", "-DAOT_TESTS=MLIR_NUM_TESTS",
                    os.path.join(ROOT, "tools", "aot_host_main.c"), os.path.join(out, "model.c"),
                    os.path.join(out, "tests.S"), os.path.join(ROOT, "runtime", "mlir", "mlir_rt.c"), obj,
                    "-lm", "-o", exe], check=True)
    subprocess.run([exe, ref], check=True)
    with open(os.path.join(out, "tests.bin"), "rb") as f:
        data = f.read(n_in_bytes)
    with open(ref, "rb") as f:
        outs = f.read()
    if len(outs) != n_tests * out_size * 4:
        sys.exit("mlir_aot: the host reference wrote the wrong amount")
    with open(os.path.join(out, "tests.bin"), "wb") as f:
        f.write(data + outs)
    for f in (obj, exe, ref):
        os.remove(f)


def main(argv):
    ap = argparse.ArgumentParser(usage=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--torch")
    src.add_argument("--onnx")
    ap.add_argument("outdir")
    ap.add_argument("--tests")
    ap.add_argument("--host-ref", action="store_true")
    ap.add_argument("--stages", help="also write the model at each step into this directory")
    a = ap.parse_args(argv)
    if a.host_ref and not a.tests:
        sys.exit(__doc__)
    out = os.path.abspath(a.outdir)
    bindir = mlir_build.use()
    os.makedirs(out, exist_ok=True)
    if a.torch:
        name = a.torch
        linalg, in_shape, out_shape = from_torch(load(name))
    else:
        name = os.path.basename(a.onnx)
        linalg, in_shape, out_shape = from_onnx(os.path.abspath(a.onnx), bindir)
    for xlen in (64, 32):
        lower(linalg, bindir, out, xlen)
    if a.stages:
        stages(linalg, bindir, os.path.abspath(a.stages), os.path.join(out, "model64.ll"))
    n_tests = 0
    if a.tests:
        n_tests, data = tests(a.tests, in_shape, int(np.prod(out_shape)))
        with open(os.path.join(out, "tests.bin"), "wb") as f:
            f.write(data)
    with open(os.path.join(out, "tests.S"), "w") as f:
        lines = ["/* generated by tools/mlir_aot.py: the test data */"]
        if n_tests:
            lines += asm(os.path.join(out, "tests.bin"), "mlir", n_tests * int(np.prod(in_shape)) * 4)
        f.write("\n".join(lines) + "\n")
    emit(out, name, in_shape, out_shape, n_tests)
    if a.host_ref:
        host_reference(out, bindir, n_tests, int(np.prod(out_shape)), n_tests * int(np.prod(in_shape)) * 4)
    print(f"{name}: MLIR -> model64.ll, model32.ll ({os.path.getsize(os.path.join(out, 'model64.ll'))} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
