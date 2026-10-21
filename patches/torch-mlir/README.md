Patches to frameworks/torch-mlir, applied in name order on the pinned commit (tools/frameworks.py apply torch-mlir).
One change per file: NNNN-what.patch, made with tools/frameworks.py record torch-mlir TITLE.

- 0001-match-typed-quantized-ops: the ONNX importer writes QuantizeLinear / DequantizeLinear as the typed
  `torch.quantized_decomposed.*` ops, but torch-match-quantized-custom-ops matched them only as generic
  `torch.operator` ops, so torch-fuse-quantized-ops never saw a QDQ chain and an int8 ONNX model ran in float.
  The pattern now matches both forms (same operands).
