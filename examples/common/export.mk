# The model of a common example (included by <framework>_example.mk): exported by tools/export.py (with the
# conversion environment: PYTHON_AI, default the .venv-ai next to this repository) into build/models/<model>/.
# MODEL_SRC / MODEL_DIR: any model file and where its exports go (an app's own model; the files next to it, e.g.
# its data.py, are dependencies too); MODEL_KIND=keras for a TensorFlow / Keras model (MLIR then takes its ONNX).
# TESTS=n checks only the first n test inputs (RTL simulation); RUN_CYCLES: the models run for 1e5..1e11
# instructions, so the limit is left to the caller's timeout.
EX_SDK := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
PYTHON_AI ?= $(abspath $(EX_SDK)/../.venv-ai/bin/python)
MODEL_SRC ?= $(EX_SDK)/models/$(MODEL)/model.py
MODEL_DIR ?= $(EX_SDK)/build/models/$(MODEL)
MODEL_KIND ?= pytorch
RUN_CYCLES ?= 1000000000000

$(MODEL_DIR)/model.json: $(MODEL_SRC) $(filter-out $(MODEL_SRC),$(wildcard $(dir $(MODEL_SRC))*.py)) $(addprefix $(EX_SDK)/tools/,export.py onnx_int8.py tflm_host.py tflm_host_runner.cc)
	TF_CPP_MIN_LOG_LEVEL=3 $(PYTHON_AI) $(EX_SDK)/tools/export.py $(MODEL_SRC) $(MODEL_DIR)
$(addprefix $(MODEL_DIR)/,model.onnx model_int8.onnx ops.h $(foreach v,int8 float,model_$(v).tflite input_$(v).csv output_ref_$(v).csv)): $(MODEL_DIR)/model.json
