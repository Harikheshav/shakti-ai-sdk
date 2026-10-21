"""Integer arithmetic for a QDQ int8 model in TVM (tools/tvm_aot.py): Relax's ONNX front end keeps QDQ as
dequantize -> float operator -> quantize; for convolutions and matrix products this rewrites

    quantize(conv2d(dequantize(x), dequantize(W)) [+ dequantize(b)], s_y, z_y)
    quantize(matmul(dequantize(x), permute_dims(dequantize(W))) [+ dequantize(b)], s_y, z_y)

into an int32 accumulation of the int8 values and one requantisation:

    acc = conv2d(pad(x, z_x), W) + (b - z_x * sum(W))             (int8 x, W; the sum per output channel)
    y   = quantize(float(acc), s_y / (s_x * s_W), z_y)            (per output channel when W is)

which is sum((x - z_x) * W) + b, with the padding at z_x (zero in the real domain).  Weights stay int8.  Symmetric
weights (z_W = 0) and per-tensor activations only; the bias must be in the accumulator's scale
(s_b = s_x * s_W, as onnxruntime writes it).  Other operators keep TVM's dequantize / float / quantize.
"""
import numpy as np


def rewrite(mod):
    import tvm
    from tvm import relax
    from tvm.relax.dpl import is_const, is_op, rewrite_call, wildcard

    def dq(data):
        s, z = is_const(), is_const()
        return is_op("relax.dequantize")(data, s, z), s, z

    def const(e):
        return e.value.numpy()

    x = wildcard()
    xd, sx, zx = dq(x)
    w = is_const()
    wd, sw, zw = dq(w)
    b = is_const()
    bd, sb, zb = dq(b)
    sy, zy = is_const(), is_const()
    conv = is_op("relax.nn.conv2d")(xd, wd)
    conv_b = is_op("relax.add")(conv, is_op("relax.reshape")(bd, wildcard()))
    mm = is_op("relax.matmul")(xd, is_op("relax.permute_dims")(wd))
    mm_b = is_op("relax.add")(mm, bd)
    patterns = [(is_op("relax.quantize")(body, sy, zy), body, core)
                for body, core in ((conv_b, conv), (conv, conv), (mm_b, mm), (mm, mm))]

    def make(core, has_bias):
        def fn(expr, m):
            s_x, z_x, s_w, z_w = (const(m[p]) for p in (sx, zx, sw, zw))
            s_y = const(m[sy])
            if np.ndim(s_x) or np.ndim(z_x) or np.any(z_w != 0):
                return expr                              # per-tensor activations, symmetric weights only
            acc_scale = (s_x.astype(np.float64) * s_w.astype(np.float64)).reshape(-1)
            weights, z = const(m[w]), int(z_x)
            xq, call = m[x], m[core]
            if core is conv:
                a = call.attrs
                if a.data_layout != "NCHW" or a.kernel_layout != "OIHW":
                    return expr
                top, left, bottom, right = (int(p) for p in a.padding)
                if z and (top or left or bottom or right):  # pad with z_x here, the convolution pads nothing
                    xq = relax.op.nn.pad(xq, [0, 0, 0, 0, top, bottom, left, right], pad_value=float(z))
                    top = left = bottom = right = 0
                acc = relax.op.nn.conv2d(xq, relax.const(weights), strides=a.strides,
                                         padding=[top, left, bottom, right], dilation=a.dilation,
                                         groups=a.groups, data_layout="NCHW", kernel_layout="OIHW",
                                         out_dtype="int32")
                wsum = weights.astype(np.int64).reshape(weights.shape[0], -1).sum(axis=1)
                shape, axis = (1, -1, 1, 1), 1
            else:
                acc = relax.op.matmul(xq, relax.const(np.ascontiguousarray(weights.T)), out_dtype="int32")
                wsum = weights.astype(np.int64).sum(axis=1)
                shape, axis = (1, -1), 1
            bias = -z * wsum
            if has_bias:
                s_b = const(m[sb]).astype(np.float64).reshape(-1)
                if np.any(const(m[zb]) != 0) or not np.allclose(s_b, acc_scale, rtol=1e-6):
                    return expr
                bias = bias + const(m[b]).astype(np.int64).reshape(-1)
            if np.any(bias):
                acc = relax.op.add(acc, relax.const(bias.astype(np.int32).reshape(shape)))
            scale = (float(s_y) / acc_scale).astype(np.float32)
            scale = relax.const(scale if scale.size > 1 else scale.reshape(()))
            return relax.op.quantize(relax.op.astype(acc, "float32"), scale, m[zy], axis=axis,
                                     out_dtype=str(expr.attrs.out_dtype))
        return fn

    for pattern, body, core in patterns:
        main = rewrite_call(pattern, make(core, body is not core), mod["main"])
        mod = tvm.IRModule({**{gv: f for gv, f in mod.functions.items() if gv.name_hint != "main"},
                            mod.get_global_var("main"): main}, attrs=mod.attrs)
    return mod
