# AI app templates

A model file (`./aisdk build|run MODEL.py ...`, `tools/export.py MODEL.py OUTDIR`, or an editor built on the SDK),
in PyTorch (`pytorch/`) or TensorFlow / Keras (`keras/`).  Its first docstring line is the template's title.  Every
model file defines:

| name | what |
|---|---|
| `INPUT_SHAPE` | the input of one inference, batch 1: NCHW for PyTorch images, NHWC for Keras |
| `model()` | the network, untrained (the summary reads it without training) |
| `build()` | the trained network (eval mode for PyTorch) |
| `calibration()` | a list of inputs (numpy, `INPUT_SHAPE`) for int8 quantisation: a few hundred typical ones |
| `test_inputs()` | the inputs the SoC runs and is checked on (a few: RTL simulation is slow) |

Files next to the model file (e.g. `data.py`) can be imported.  Datasets are downloaded into
`SHAKTI_AI_CACHE` (default `~/.cache/shakti-ai-sdk`) or Keras's own cache.
