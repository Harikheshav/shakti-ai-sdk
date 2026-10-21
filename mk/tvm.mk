# Apache TVM for an app (FRAMEWORK := tvm; included by mk/common.mk).  TVM_ONNX: the model, compiled on the host by
# tools/tvm_aot.py (TVM built by tools/tvm_build.py; Python: PYTHON_AI, the conversion environment) into C in
# build/tvm-aot/<app>/ (target independent: shared by all SoCs); TVM_TESTS: a common example's model directory
# (its test data are linked in), TVM_HOST_REF=1: the reference outputs from the same code run on the host.  The app
# calls tvm_model_run() (model.h).  No floating-point contraction, as in the host reference.
TVM_ROOT := $(SDK)/frameworks/tvm
PYTHON_AI ?= $(abspath $(SDK)/../.venv-ai/bin/python)
TVM_GEN := $(SDK)/build/tvm-aot/$(APP)
TVM_GEN_SRCS := $(TVM_GEN)/model_kernels.c $(TVM_GEN)/model.c $(TVM_GEN)/model_data.S
APP_SRCS += $(TVM_GEN_SRCS) $(SDK)/runtime/tvm/tvm_rt.c
GEN_SRCS += $(TVM_GEN_SRCS)
CFLAGS += -ffp-contract=off
APP_INC += -I$(TVM_GEN) -I$(TVM_ROOT)/include -I$(TVM_ROOT)/3rdparty/tvm-ffi/include \
           -I$(TVM_ROOT)/3rdparty/tvm-ffi/3rdparty/dlpack/include

$(TVM_GEN_SRCS) &: $(TVM_ONNX) $(addprefix $(SDK)/tools/,tvm_aot.py tvm_int8.py aot_tests.py aot_host_main.c) $(if $(TVM_TESTS),$(TVM_TESTS)/input_float.csv $(TVM_TESTS)/output_ref_float.csv)
	$(PYTHON_AI) $(SDK)/tools/tvm_aot.py $(TVM_ONNX) $(TVM_GEN) $(if $(TVM_TESTS),--tests $(TVM_TESTS) $(if $(TVM_HOST_REF),--host-ref)) $(if $(TVM_STAGES),--stages $(TVM_STAGES))
