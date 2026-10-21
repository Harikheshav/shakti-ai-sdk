# A common example on MLIR (examples/<model>/mlir/Makefile: MODEL := <model>, include this), compiled ahead of time
# through torch-mlir and MLIR (tools/mlir_aot.py).  CHECK=float (default): the PyTorch model, checked against the
# host reference within a tolerance; CHECK=int8: model_int8.onnx (QDQ), checked bit-exact against the same compiled
# code on the host.
include $(dir $(lastword $(MAKEFILE_LIST)))export.mk
CHECK ?= float
APP ?= $(MODEL)_mlir_$(CHECK)$(if $(TESTS),_t$(TESTS))
FRAMEWORK := mlir
APP_SRCS := $(EX_SDK)/examples/common/mlir_check.c
MLIR_MODEL := $(MODEL_SRC)
MLIR_ONNX := $(if $(filter int8,$(CHECK)),$(MODEL_DIR)/model_int8.onnx,$(if $(filter keras,$(MODEL_KIND)),$(MODEL_DIR)/model.onnx))
MLIR_TESTS := $(MODEL_DIR)
MLIR_HOST_REF := $(filter int8,$(CHECK))
CFLAGS += -DCHECK_NAME='"$(CHECK)"' $(if $(MLIR_HOST_REF),-DCHECK_EXACT) $(if $(TESTS),-DCHECK_TESTS=$(TESTS))

include $(EX_SDK)/mk/common.mk
