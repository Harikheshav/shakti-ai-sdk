#!/usr/bin/env python3
"""The int8 ONNX model of a common example, for TVM and MLIR: onnxruntime's static post-training quantisation.

usage: tools/onnx_int8.py NAME MODEL_DIR     (conversion environment; MODEL_DIR: tools/export.py's output)

MODEL_DIR/model.onnx (float) -> MODEL_DIR/model_int8.onnx: QDQ format (QuantizeLinear / DequantizeLinear around
each operator), int8 activations (asymmetric, calibrated with the model's calibration() inputs) and int8 weights
(symmetric, per output channel), int32 biases.  Input and output stay float.  Prints the largest difference from
the float model on the test inputs (onnxruntime on the host).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export import load  # noqa: E402


def quantize(m, model_dir):
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    src = os.path.join(model_dir, "model.onnx")
    pre = os.path.join(model_dir, "model_int8_pre.onnx")
    dst = os.path.join(model_dir, "model_int8.onnx")
    quant_pre_process(src, pre, skip_symbolic_shape=True)
    name = onnx.load(pre).graph.input[0].name

    class Calibration(CalibrationDataReader):
        def __init__(self):
            self.data = iter([{name: c} for c in m.calibration()])

        def get_next(self):
            return next(self.data, None)

    quantize_static(pre, dst, Calibration(), quant_format=QuantFormat.QDQ, per_channel=False,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    os.remove(pre)
    return src, dst


def compare(src, dst, inputs):
    import onnxruntime as ort
    f, q = (ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for p in (src, dst))
    name = f.get_inputs()[0].name
    return max(float(np.abs(f.run(None, {name: x})[0] - q.run(None, {name: x})[0]).max()) for x in inputs)


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)
    m = load(argv[0])
    src, dst = quantize(m, os.path.abspath(argv[1]))
    print(f"{argv[0]}: model_int8.onnx, max |int8 - float| on the test inputs "
          f"{compare(src, dst, m.test_inputs()):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
