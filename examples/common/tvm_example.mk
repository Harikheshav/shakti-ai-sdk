# A common example on Apache TVM (examples/<model>/tvm/Makefile: MODEL := <model>, include this): the model's ONNX
# export compiled ahead of time (tools/tvm_aot.py).  CHECK=float (default): model.onnx, checked against the host
# reference within a tolerance; CHECK=int8: model_int8.onnx (QDQ), checked bit-exact against the same compiled
# code on the host.
include $(dir $(lastword $(MAKEFILE_LIST)))export.mk
CHECK ?= float
APP ?= $(MODEL)_tvm_$(CHECK)$(if $(TESTS),_t$(TESTS))
FRAMEWORK := tvm
APP_SRCS := $(EX_SDK)/examples/common/tvm_check.c
TVM_ONNX := $(MODEL_DIR)/$(if $(filter int8,$(CHECK)),model_int8,model).onnx
TVM_TESTS := $(MODEL_DIR)
TVM_HOST_REF := $(filter int8,$(CHECK))
CFLAGS += -DCHECK_NAME='"$(CHECK)"' $(if $(TVM_HOST_REF),-DCHECK_EXACT) $(if $(TESTS),-DCHECK_TESTS=$(TESTS))

include $(EX_SDK)/mk/common.mk
