#!/usr/bin/env python3
"""Compile an ONNX model with TVM into C for a bare-metal SHAKTI target: ahead of time, no TVM runtime.

usage: tools/tvm_aot.py MODEL.onnx OUTDIR [--tests MODEL_DIR [--host-ref]] [--stages DIR]
       (conversion environment; TVM: tools/tvm_build.py build)

TVM imports the model (Relax ONNX front end; a QDQ int8 model's convolutions and matrix products become integer
arithmetic: tools/tvm_int8.py), lowers every operator to TIR, folds constants and fuses operators,
and its C back end writes the kernels: model_kernels.c, one TVM-FFI packed function each.  What TVM's executors
would do at run time is written here as static C:
  model.h, model.c    tvm_model_run(input, output): the kernels in order on static DLTensors.  The intermediate
                      tensors share one arena (offsets planned from their lifetimes); the kernels' own scratch
                      memory comes from TVMBackendAllocWorkspace (runtime/tvm/tvm_rt.c) in a workspace of the
                      most they ask for at once.
  model_params.bin    the weights (the model's constants), linked by model_data.S (.incbin) as tvm_params
With --tests, a common example's test data (MODEL_DIR from tools/export.py): input_float.csv (4-D inputs there are
NHWC, TFLite's layout: transposed to the model's NCHW) and output_ref_float.csv, linked as tvm_tests_input and
tvm_tests_ref, with TVM_NUM_TESTS in model.h.  --host-ref: the reference outputs are this compiled model's own,
built for and run on the host (HOST_CC, default gcc; no floating-point contraction on either side), for a
bit-exact check (int8 models).  --stages DIR: also the model at each step (imported Relax, integer arithmetic,
lowered with its TIR kernels, the C kernels, the call sequence).
"""
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvm_build  # noqa: E402
import tvm_int8  # noqa: E402
from aot_tests import asm, tests  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIGN = 16
DLPACK = {"int": 0, "uint": 1, "float": 2, "bool": 6}


def dl_dtype(dtype):
    m = re.fullmatch(r"(bool|int|uint|float)(\d*)", dtype)
    if not m:
        sys.exit(f"tvm_aot: unsupported tensor type {dtype}")
    return DLPACK[m.group(1)], int(m.group(2) or 8)


def up(n, a=ALIGN):
    return (n + a - 1) // a * a


def pad_values_folded(model):
    """Pad's constant value given as DequantizeLinear of constants (a QDQ model) -> the float constant: Relax's
    ONNX front end takes only a constant there."""
    from onnx import numpy_helper
    inits = {i.name: i for i in model.graph.initializer}
    producer = {o: n for n in model.graph.node for o in n.output}
    for n in model.graph.node:
        if n.op_type != "Pad" or len(n.input) < 3 or not n.input[2]:
            continue
        dq = producer.get(n.input[2])
        if dq is None or dq.op_type != "DequantizeLinear" or not all(i in inits for i in dq.input if i):
            continue
        q, scale = (numpy_helper.to_array(inits[i]) for i in dq.input[:2])
        zero = numpy_helper.to_array(inits[dq.input[2]]) if len(dq.input) > 2 and dq.input[2] else 0
        value = ((q.astype(np.float32) - np.float32(zero)) * scale).astype(np.float32)
        name = n.input[2] + "_folded"
        model.graph.initializer.append(numpy_helper.from_array(value, name))
        n.input[2] = name
    return model


def lower(onnx_path, stages=None):
    """The model as Relax with every operator a call_tir of a fused TIR kernel.  stages: a directory for the model
    at each step (imported Relax, integer arithmetic, lowered with its TIR kernels)."""
    import onnx
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    model = pad_values_folded(onnx.load(onnx_path))
    mod = from_onnx(model, keep_params_in_input=False)
    if stages:
        os.makedirs(stages, exist_ok=True)
        open(os.path.join(stages, "01_relax_imported.py"), "w").write(mod.script())
    mod = tvm_int8.rewrite(mod)
    if stages:
        open(os.path.join(stages, "02_relax_integer.py"), "w").write(mod.script())
    mod = tvm.ir.transform.Sequential([
        relax.transform.DecomposeOpsForInference(), relax.transform.LegalizeOps(), relax.transform.FoldConstant(),
        relax.transform.AnnotateTIROpPattern(), relax.transform.FuseOps(), relax.transform.FuseTIR(),
        relax.transform.DeadCodeElimination(), relax.transform.AttachGlobalSymbol()])(mod)
    if stages:
        open(os.path.join(stages, "03_relax_tir.py"), "w").write(mod.script())
    return mod


class Tensor:
    def __init__(self, name, shape, dtype):
        self.name, self.shape, self.dtype = name, [int(s) for s in shape], str(dtype)
        self.nbytes = int(np.prod(self.shape)) * dl_dtype(self.dtype)[1] // 8
        self.kind, self.offset = None, 0          # kind: input, output, param, arena
        self.first = self.last = None             # the steps that write it first and read it last


def schedule(mod):
    """The kernel calls of main, in order: [(kernel name, [Tensor ...] (arguments, then the output))], the
    tensors, the parameter blob."""
    import tvm
    from tvm import relax
    main = mod["main"]
    if len(main.params) != 1:
        sys.exit("tvm_aot: models with one input only")
    tensors, steps, params = {}, [], bytearray()

    def tensor(expr, step):
        if isinstance(expr, tvm.ir.expr.GenericConst):
            if expr in tensors:                       # TVM objects hash by node: the same constant again
                return tensors[expr]
            data = expr.value.numpy()
            t = Tensor(f"c{len(tensors)}", data.shape, str(data.dtype))
            t.kind, t.offset = "param", len(params)
            params.extend(data.tobytes())
            params.extend(b"\0" * (up(len(params)) - len(params)))
            tensors[expr] = t
            return t
        t = tensors.get(expr)
        if t is None:
            sys.exit(f"tvm_aot: {expr} is used before it is computed")
        t.last = step
        return t

    x = main.params[0]
    tensors[x] = Tensor("t_input", x.ty.shape, x.ty.dtype)
    tensors[x].kind, tensors[x].first = "input", -1
    for block in main.body.blocks:
        for b in block.bindings:
            v = b.value
            if isinstance(v, relax.Var):                       # an alias (R.output of a dataflow block)
                tensors[b.var] = tensor(v, len(steps))
                continue
            op = getattr(v.op, "name", "") if isinstance(v, relax.Call) else ""
            if op not in ("relax.call_tir", "relax.call_dps_packed"):
                sys.exit(f"tvm_aot: cannot run {v.op} ahead of time (only TIR kernels and C functions)")
            gv, args = v.args[0], v.args[1].fields
            # a TIR kernel (__tvm_ffi_<name>, TVM's packed calling convention) or a C function of ours
            # (int name(DLTensor *const *tensors, int n))
            name = gv.name_hint if op == "relax.call_tir" else "@" + str(gv.global_symbol)
            out_si = v.ty_args[0]
            if not isinstance(out_si, relax.TensorType):
                sys.exit(f"tvm_aot: {name} has several outputs")
            step = len(steps)
            ins = [tensor(a, step) for a in args]
            out = Tensor(f"t{len(tensors)}", out_si.shape, out_si.dtype)
            out.kind, out.first, out.last = "arena", step, step
            tensors[b.var] = out
            steps.append((name, ins + [out]))
    result = main.body.body
    if not isinstance(result, relax.Var) or tensors[result].kind != "arena":
        sys.exit("tvm_aot: the model's output must be one computed tensor")
    tensors[result].kind = "output"
    return steps, list(dict.fromkeys(tensors.values())), bytes(params)


def plan(tensors):
    """Arena offsets: largest first, each at the lowest offset free during its lifetime.  Returns the arena size."""
    placed = []
    for t in sorted((t for t in tensors if t.kind == "arena"), key=lambda t: -t.nbytes):
        live = sorted((p.offset, p.offset + up(p.nbytes)) for p in placed if p.first <= t.last and t.first <= p.last)
        off = 0
        for lo, hi in live:
            if off + t.nbytes <= lo:
                break
            off = max(off, hi)
        t.offset = off
        placed.append(t)
    return max((t.offset + up(t.nbytes) for t in placed), default=0)


def kernels_c(mod):
    import tvm
    prims = tvm.IRModule({gv: f for gv, f in mod.functions.items() if isinstance(f, tvm.tirx.PrimFunc)})
    return tvm.tirx.build(prims, target="c").inspect_source()


def workspace(src):
    """The most scratch memory any one kernel takes at once (all of its TVMBackendAllocWorkspace sizes)."""
    most = 0
    for body in src.split("TVM_DLL int32_t __tvm_ffi_")[1:]:
        sizes = re.findall(r"TVMBackendAllocWorkspace\(1, dev_id, \(uint64_t\)(\d+)ULL", body)
        if len(sizes) != body.count("TVMBackendAllocWorkspace("):
            sys.exit("tvm_aot: a kernel asks for a workspace of a size known only at run time")
        most = max(most, sum(up(int(s), 64) for s in sizes))
    return most


def emit(out, steps, tensors, arena, ws, n_tests):
    inp = next(t for t in tensors if t.kind == "input")
    outp = next(t for t in tensors if t.kind == "output")
    name = re.sub(r"\W", "_", os.path.basename(out))
    h = [f"/* generated by tools/tvm_aot.py: {name} */", "#ifndef TVM_MODEL_H", "#define TVM_MODEL_H",
         "#include <stdint.h>",
         f"#define TVM_INPUT_SIZE {int(np.prod(inp.shape))}      /* {inp.dtype} {inp.shape} */",
         f"#define TVM_OUTPUT_SIZE {int(np.prod(outp.shape))}     /* {outp.dtype} {outp.shape} */",
         f"#define TVM_ARENA_BYTES {arena}", f"#define TVM_WORKSPACE_BYTES {ws}",
         f"#define TVM_NUM_KERNELS {len(steps)}"]
    if n_tests:
        h += [f"#define TVM_NUM_TESTS {n_tests}", "extern const float tvm_tests_input[], tvm_tests_ref[];"]
    h += ["/* runs the model: 0, or a kernel's error (reported by TVMFFIErrorSetRaisedFromCStrParts) */",
          "int tvm_model_run(const void *input, void *output);", "#endif"]
    c = [f"/* generated by tools/tvm_aot.py: {name}, {len(steps)} kernels (model_kernels.c, TVM's C back end) */",
         "#include <stddef.h>", '#include "tvm/ffi/c_api.h"', '#include "model.h"', "",
         "extern const uint8_t tvm_params[];",
         f"static uint8_t tvm_arena[{max(arena, ALIGN)}] __attribute__((aligned(64)));",
         f"uint8_t tvm_workspace[{max(ws, ALIGN)}] __attribute__((aligned(64)));",
         f"const size_t tvm_workspace_size = {ws};", ""]
    for k in sorted({k for k, _ in steps}):
        c.append(f"int {k[1:]}(DLTensor *const *tensors, int n);" if k[0] == "@" else
                 f"int32_t __tvm_ffi_{k}(void *self, void *args, int32_t num_args, void *result);")
    c.append("")
    for t in tensors:
        code, bits = dl_dtype(t.dtype)
        data = {"param": f"(void *)(tvm_params + {t.offset})", "arena": f"tvm_arena + {t.offset}"}.get(t.kind, "NULL")
        c.append(f"static int64_t {t.name}_shape[] = {{{', '.join(map(str, t.shape)) or '1'}}};")
        c.append(f"static DLTensor {t.name} = {{{data}, {{kDLCPU, 0}}, {len(t.shape)}, {{{code}, {bits}, 1}}, "
                 f"{t.name}_shape, NULL, 0}};   /* {t.kind} {t.dtype} {t.shape} */")
    most = max(len(ts) for _, ts in steps)
    c += ["", "static int call(int32_t (*kernel)(void *, void *, int32_t, void *), DLTensor *const *ts, int n)",
          "{", f"  TVMFFIAny args[{most}], result;", "  for (int i = 0; i < n; i++) {",
          "    args[i].type_index = kTVMFFIDLTensorPtr;", "    args[i].zero_padding = 0;",
          "    args[i].v_ptr = ts[i];", "  }", "  result.type_index = kTVMFFINone;", "  result.zero_padding = 0;",
          "  result.v_int64 = 0;", "  return kernel(NULL, args, n, &result);", "}", "",
          "int tvm_model_run(const void *input, void *output)", "{", "  int rc;",
          f"  {inp.name}.data = (void *)input;", f"  {outp.name}.data = output;"]
    for k, ts in steps:
        args = ", ".join(f"&{t.name}" for t in ts)
        call = f"{k[1:]}(a, {len(ts)})" if k[0] == "@" else f"call(__tvm_ffi_{k}, a, {len(ts)})"
        c.append(f"  {{ DLTensor *const a[] = {{{args}}}; if ((rc = {call})) return rc; }}")
    c += ["  return 0;", "}"]
    s = ['/* generated by tools/tvm_aot.py: the weights' + (" and the test data" if n_tests else "") + " */",
         '  .section .rodata.tvm_params, "a"', "  .balign 64", "  .global tvm_params", "tvm_params:",
         f'  .incbin "{os.path.join(out, "model_params.bin")}"']
    if n_tests:
        n_in = n_tests * int(np.prod(inp.shape)) * 4
        s += asm(os.path.join(out, "tests.bin"), "tvm", n_in)
    for f, lines in (("model.h", h), ("model.c", c), ("model_data.S", s)):
        with open(os.path.join(out, f), "w") as fh:
            fh.write("\n".join(lines) + "\n")


def host_reference(out, n_tests, out_size, n_in_bytes):
    """Run the generated C on the host and put its outputs into tests.bin as the reference."""
    import subprocess
    t = os.path.join(ROOT, "frameworks", "tvm")
    exe = os.path.join(out, "host_ref")
    subprocess.run([os.environ.get("HOST_CC", "gcc"), "-O2", "-ffp-contract=off", f"-I{out}",
                    f"-I{t}/include", f"-I{t}/3rdparty/tvm-ffi/include", f"-I{t}/3rdparty/tvm-ffi/3rdparty/dlpack/include",
                    '-DAOT_HEADER="model.h"', "-DAOT_RUN=tvm_model_run", "-DAOT_INPUT=tvm_tests_input",
                    "-DAOT_IN_SIZE=TVM_INPUT_SIZE", "-DAOT_OUT_SIZE=TVM_OUTPUT_SIZE", "-DAOT_TESTS=TVM_NUM_TESTS",
                    os.path.join(ROOT, "tools", "aot_host_main.c"), os.path.join(out, "model.c"),
                    os.path.join(out, "model_kernels.c"), os.path.join(out, "model_data.S"),
                    os.path.join(ROOT, "runtime", "tvm", "tvm_rt.c"), "-lm", "-o", exe], check=True)
    ref = os.path.join(out, "host_ref.bin")
    subprocess.run([exe, ref], check=True)
    with open(os.path.join(out, "tests.bin"), "rb") as f:
        data = f.read(n_in_bytes)
    with open(ref, "rb") as f:
        outs = f.read()
    if len(outs) != n_tests * out_size * 4:
        sys.exit("tvm_aot: the host reference wrote the wrong amount")
    with open(os.path.join(out, "tests.bin"), "wb") as f:
        f.write(data + outs)
    os.remove(exe)
    os.remove(ref)


def main(argv):
    host_ref = "--host-ref" in argv
    argv = [a for a in argv if a != "--host-ref"]
    stages = None
    if "--stages" in argv:
        i = argv.index("--stages")
        stages = os.path.abspath(argv[i + 1]) if i + 1 < len(argv) else sys.exit(__doc__)
        argv = argv[:i] + argv[i + 2:]
    if len(argv) not in (2, 4) or (len(argv) == 4 and argv[2] != "--tests") or (host_ref and len(argv) != 4):
        sys.exit(__doc__)
    onnx_path, out = os.path.abspath(argv[0]), os.path.abspath(argv[1])
    tvm_build.use()
    os.makedirs(out, exist_ok=True)
    mod = lower(onnx_path, stages)
    steps, tensors, arena, ws, params, n_tests = build(out, mod, argv[3] if len(argv) == 4 else None)
    if stages:                                       # the C TVM wrote and the call sequence around it
        shutil.copy(os.path.join(out, "model_kernels.c"), os.path.join(stages, "04_kernels.c"))
        shutil.copy(os.path.join(out, "model.c"), os.path.join(stages, "05_model.c"))
    if host_ref:
        inp = next(t for t in tensors if t.kind == "input")
        outp = next(t for t in tensors if t.kind == "output")
        host_reference(out, n_tests, int(np.prod(outp.shape)), n_tests * int(np.prod(inp.shape)) * 4)
    print(f"{os.path.basename(onnx_path)}: {len(steps)} kernels, params {len(params)} bytes, arena {arena} bytes, "
          f"workspace {ws} bytes")
    return 0


def build(out, mod, test_dir):
    """The C of a lowered model in `out` (with the test data of test_dir)."""
    steps, tensors, params = schedule(mod)
    arena = plan(tensors)
    src = kernels_c(mod)
    ws = workspace(src)
    with open(os.path.join(out, "model_kernels.c"), "w") as f:
        f.write(src)
    with open(os.path.join(out, "model_params.bin"), "wb") as f:
        f.write(params)
    n_tests = 0
    if test_dir:
        inp = next(t for t in tensors if t.kind == "input")
        outp = next(t for t in tensors if t.kind == "output")
        if inp.dtype != "float32" or outp.dtype != "float32":
            sys.exit("tvm_aot: the test data are float; this model is not")
        n_tests, data = tests(test_dir, inp.shape, int(np.prod(outp.shape)))
        with open(os.path.join(out, "tests.bin"), "wb") as f:
            f.write(data)
    emit(out, steps, tensors, arena, ws, n_tests)
    return steps, tensors, arena, ws, params, n_tests


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
