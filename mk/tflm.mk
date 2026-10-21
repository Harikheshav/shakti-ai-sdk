# TensorFlow Lite Micro for an app (FRAMEWORK := tflm; included by mk/common.mk).  Builds build/<soc>/tflm/libtflm.a
# from TFLM's own source list and flags (tools/tflm.py: patches applied, downloads unpacked) with the SoC's ISA and
# ABI.  TFLM_DATA: .tflite / .bmp / .wav / .npy files (paths in frameworks/tflite-micro, or absolute) turned into C
# arrays by TFLM's own generate_cc_arrays.py (<file>_model_data.cc/.h ..., included as in upstream's examples).

TFLM_OUT := $(SDK)/build/tflm
TFLM_PREP := $(shell $(PYTHON) $(SDK)/tools/tflm.py prepare $(TFLM_OUT) >/dev/null 2>&1 && echo ok)
ifneq ($(TFLM_PREP),ok)
$(error tools/tflm.py prepare failed: run it directly to see why)
endif
include $(TFLM_OUT)/sources.mk

TFLM_LIB := $(BUILD)/tflm/libtflm.a
# SOC=host: TFLM's generic platform files instead of the SHAKTI target's (they read mcycle), and a failed check
# aborts (as tools/tflm_host.py builds it)
TFLM_HOST_FLAGS := $(if $(HOST),'-DTFLITE_ABORT=__builtin_abort();')
TFLM_BUILD_FLAGS = $(ARCH) $(OPT) -g $(TFLM_INCLUDES) -DSHAKTI_CLOCK_HZ=$(CLOCK_HZ) $(TFLM_HOST_FLAGS)
APP_INC += $(TFLM_INCLUDES)
APP_LIBS += $(TFLM_LIB)
CXXFLAGS += $(filter -D%,$(TFLM_CXXFLAGS)) -funsigned-char $(TFLM_HOST_FLAGS)

TFLM_ALL := $(if $(HOST),$(subst /tensorflow/lite/micro/shakti/,/tensorflow/lite/micro/,$(TFLM_SRCS)),$(TFLM_SRCS)) $(TFLM_THIRD_PARTY_SRCS)
TFLM_OBJS := $(foreach s,$(TFLM_ALL),$(BUILD)/tflm/obj/$(subst /,_,$(subst $(SDK)/,,$(s))).o)

define tflm_compile
$(BUILD)/tflm/obj/$(subst /,_,$(subst $(SDK)/,,$(1))).o: $(1) $(TFLM_OUT)/sources.mk
	@mkdir -p $$(dir $$@)
	$$(if $$(filter %.c,$(1)),$$(CC) $$(TFLM_BUILD_FLAGS) $$(TFLM_CCFLAGS),$$(CXX) $$(TFLM_BUILD_FLAGS) $$(TFLM_CXXFLAGS)) -c $$< -o $$@
endef
$(foreach s,$(TFLM_ALL),$(eval $(call tflm_compile,$(s))))

$(TFLM_LIB): $(TFLM_OBJS)
	@rm -f $@
	$(CROSS)ar rcs $@ $^

$(OUT)/$(APP).elf: $(TFLM_LIB)

TFLM_GEN := $(BUILD)/tflm/gen
TFLM_ROOT := $(SDK)/frameworks/tflite-micro
tflm_gen_name = $(if $(filter /%,$(1)),,$(TFLM_GEN)/)$(basename $(1))$(if $(filter %.tflite,$(1)),_model_data,$(if $(filter %.bmp,$(1)),_image_data,$(if $(filter %.wav,$(1)),_audio_data,_test_data))).cc
TFLM_GEN_SRCS := $(foreach d,$(TFLM_DATA),$(call tflm_gen_name,$(d)))
APP_SRCS += $(TFLM_GEN_SRCS)
GEN_SRCS += $(TFLM_GEN_SRCS)              # generated before any app source compiles (their headers)
APP_INC += -I$(TFLM_GEN) $(sort $(foreach d,$(filter /%,$(TFLM_DATA)),-I$(dir $(d))))   # absolute data: generated beside it
define tflm_gen
$(call tflm_gen_name,$(1)): $(if $(filter /%,$(1)),$(1),$(TFLM_ROOT)/$(1))
	cd $(TFLM_ROOT) && $(PYTHON) tensorflow/lite/micro/tools/generate_cc_arrays.py $(TFLM_GEN) $(1) >/dev/null
endef
$(foreach d,$(TFLM_DATA),$(eval $(call tflm_gen,$(d))))
