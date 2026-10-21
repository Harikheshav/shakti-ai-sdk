# A common example on TensorFlow Lite Micro (examples/<model>/tflm/Makefile: MODEL := <model>, include this):
# int8 and float (CHECK: both by default, or one) checked against TFLite Micro on the host (tools/tflm_host.py,
# run by tools/export.py).
include $(dir $(lastword $(MAKEFILE_LIST)))export.mk
CHECK ?= int8 float
APP ?= $(MODEL)_tflm$(if $(filter 1,$(words $(CHECK))),_$(CHECK))$(if $(TESTS),_t$(TESTS))
FRAMEWORK := tflm
APP_SRCS := $(EX_SDK)/examples/common/tflm_check.cc
APP_INC += -I$(MODEL_DIR)
CXXFLAGS += $(if $(filter int8,$(CHECK)),-DCHECK_INT8) $(if $(filter float,$(CHECK)),-DCHECK_FLOAT) $(if $(ARENA_KB),-DTFLM_ARENA_KB=$(ARENA_KB)) $(if $(TESTS),-DCHECK_TESTS=$(TESTS))
TFLM_DATA := $(foreach v,$(CHECK),$(MODEL_DIR)/model_$(v).tflite $(MODEL_DIR)/input_$(v).csv $(MODEL_DIR)/output_ref_$(v).csv)

include $(EX_SDK)/mk/common.mk
