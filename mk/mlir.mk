# MLIR for an app (FRAMEWORK := mlir; included by mk/common.mk).  MLIR_MODEL: a model of models/ (PyTorch), or
# MLIR_ONNX: an ONNX file, compiled on the host by tools/mlir_aot.py (torch-mlir and MLIR built by
# tools/mlir_build.py; Python: PYTHON_AI) into LLVM IR in build/mlir-aot/<app>/ (shared by all SoCs), then by llc
# for the target's ISA and ABI; MLIR_TESTS: a common example's model directory (its test data are linked in),
# MLIR_HOST_REF=1: the reference outputs from the same code run on the host.  The app calls mlir_model_run() (model.h).
PYTHON_AI ?= $(abspath $(SDK)/../.venv-ai/bin/python)
MLIR_BIN := $(SDK)/build/mlir/bin
MLIR_GEN := $(SDK)/build/mlir-aot/$(APP)
MLIR_XLEN := $(if $(filter rv64%,$(MARCH))$(HOST),64,32)
# rv64imafdc_zicsr_zifencei -> +m,+a,+f,+d,+c,+zicsr,+zifencei
MLIR_MATTR := $(if $(HOST),,$(shell $(PYTHON) -c "import sys; b, *z = sys.argv[1][4:].split('_'); print(','.join(['+' + c for c in b if c != 'i'] + ['+' + e for e in z]))" $(MARCH)))
MLIR_GEN_SRCS := $(MLIR_GEN)/model.c $(MLIR_GEN)/tests.S
MLIR_OBJ := $(OUT)/mlir/model.o
APP_SRCS += $(MLIR_GEN_SRCS) $(SDK)/runtime/mlir/mlir_rt.c
GEN_SRCS += $(MLIR_GEN_SRCS)
APP_INC += -I$(MLIR_GEN)
APP_LIBS += $(MLIR_OBJ)

MLIR_SOURCE := $(if $(MLIR_ONNX),$(MLIR_ONNX),$(if $(filter %.py,$(MLIR_MODEL)),$(MLIR_MODEL),$(SDK)/models/$(MLIR_MODEL)/model.py))
$(MLIR_GEN)/model64.ll $(MLIR_GEN)/model32.ll $(MLIR_GEN_SRCS) &: $(MLIR_SOURCE) $(addprefix $(SDK)/tools/,mlir_aot.py aot_tests.py aot_host_main.c) $(SDK)/runtime/mlir/mlir_rt.c $(if $(MLIR_TESTS),$(MLIR_TESTS)/input_float.csv $(MLIR_TESTS)/output_ref_float.csv)
	$(PYTHON_AI) $(SDK)/tools/mlir_aot.py $(if $(MLIR_ONNX),--onnx $(MLIR_ONNX),--torch $(MLIR_MODEL)) $(MLIR_GEN) $(if $(MLIR_TESTS),--tests $(MLIR_TESTS) $(if $(MLIR_HOST_REF),--host-ref)) $(if $(MLIR_STAGES),--stages $(MLIR_STAGES))

# LLVM's optimiser (opt: the loops MLIR writes get their invariants hoisted, accumulators kept in registers ...),
# then llc, both for the target
# (SOC=host: the host's own triple, position-independent as gcc links by default)
MLIR_TARGET := $(if $(HOST),,-mtriple=riscv$(MLIR_XLEN)-unknown-elf -mattr=$(MLIR_MATTR))
MLIR_LLC_TARGET := $(if $(HOST),-relocation-model=pic,-target-abi=$(MABI) -code-model=medium -relocation-model=static)
$(MLIR_OBJ): $(MLIR_GEN)/model$(MLIR_XLEN).ll $(BUILD)/soc/soc.mk
	@mkdir -p $(dir $@)
	$(MLIR_BIN)/opt -O2 $(MLIR_TARGET) $< -o $(@:.o=.bc)
	$(MLIR_BIN)/llc -O2 $(MLIR_TARGET) $(MLIR_LLC_TARGET) -function-sections -data-sections -filetype=obj $(@:.o=.bc) -o $@
$(OUT)/$(APP).elf: $(MLIR_OBJ)
