# shakti-ai-sdk: build rules shared by examples/ and projects/.
#
#   make SOC=<target dir>                             a SoC described by <target dir>/target.json (tools/soc.py);
#                                                     built only: the SoC's own tool runs the ELF
#   make SOC=spike:rv64imafdc_zicsr_zifencei/lp64d    Spike (the default)
#   make SOC=host                                     this machine: the same program built with gcc, run directly
#   CROSS=<prefix>   toolchain prefix (default riscv64-unknown-elf-, the multilib GCC on PATH; see env/README.md)
#
# An app Makefile sets APP and APP_SRCS (C, C++, assembly), FRAMEWORK (tflm, tvm, mlir) if it uses one, and
# includes this file.  Output:
#   build/<soc>/<app>/<app>.elf, .bin (raw image from the memory base), .dump

SDK := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
.DEFAULT_GOAL := all
SOC ?= spike:rv64imafdc_zicsr_zifencei/lp64d
CROSS ?= riscv64-unknown-elf-
PYTHON_AI ?= $(abspath $(SDK)/../.venv-ai/bin/python)
# the tools' Python: the conversion environment when there is one (TFLM's generate_cc_arrays.py needs numpy and
# Pillow), else python3
PYTHON ?= $(if $(wildcard $(PYTHON_AI)),$(PYTHON_AI),python3)
SOCID := $(if $(filter spike:%,$(SOC)),$(subst /,-,$(subst spike:,spike-,$(SOC))),$(if $(filter host,$(SOC)),host,$(notdir $(abspath $(SOC)))))
BUILD ?= $(SDK)/build/$(SOCID)

SOC_OK := $(shell $(PYTHON) $(SDK)/tools/soc.py '$(SOC)' $(BUILD)/soc >/dev/null 2>&1 && echo ok)
ifneq ($(SOC_OK),ok)
$(error tools/soc.py '$(SOC)' failed: run it directly to see why)
endif
include $(BUILD)/soc/soc.mk
ifneq ($(HOST),)
CROSS :=                                         # SOC=host: the machine's own gcc, C library and start-up
endif

CC := $(CROSS)gcc
CXX := $(CROSS)g++
OBJCOPY := $(CROSS)objcopy
OBJDUMP := $(CROSS)objdump

ARCH := $(if $(HOST),,-march=$(MARCH) -mabi=$(MABI) -mcmodel=medany)
OPT ?= -O2
COMMON = $(ARCH) $(OPT) -g -ffunction-sections -fdata-sections -Wall -MMD -MP -DCONSOLE_$(CONSOLE) \
          $(if $(SOC_BOARD_INIT),-DSOC_BOARD_INIT -DBOARD_INIT_ONLY) $(SOC_INC) -I$(SDK)/runtime/shakti $(APP_INC)
CFLAGS += $(COMMON) -std=gnu11
CXXFLAGS += $(COMMON) -std=c++17 -fno-exceptions -fno-rtti -fno-threadsafe-statics
ASFLAGS += $(COMMON)
LDFLAGS += $(ARCH) -Wl,--gc-sections $(if $(HOST),,-static -nostartfiles -T $(BUILD)/soc/link.ld -Wl,--no-warn-rwx-segments --specs=nano.specs)

OUT := $(BUILD)/$(APP)
ifneq ($(FRAMEWORK),)
include $(SDK)/mk/$(FRAMEWORK).mk                 # the framework's library and generated sources (mk/tflm.mk ...)
endif
RT_SRCS := $(if $(HOST),$(SDK)/runtime/host/rt.c,$(SDK)/runtime/shakti/crt0.S $(SDK)/runtime/shakti/rt.c) $(SOC_SRCS)
SRCS := $(RT_SRCS) $(APP_SRCS)
OBJS := $(foreach s,$(SRCS),$(OUT)/obj/$(subst /,_,$(subst $(SDK)/,,$(abspath $(s)))).o)

all: $(OUT)/$(APP).elf $(if $(HOST),,$(OUT)/$(APP).bin) $(OUT)/$(APP).dump

define compile
$(OUT)/obj/$(subst /,_,$(subst $(SDK)/,,$(abspath $(1)))).o: $(1) $(BUILD)/soc/soc.mk | $(GEN_SRCS)
	@mkdir -p $$(dir $$@)
	$$(if $$(filter %.S,$(1)),$$(CC) $$(ASFLAGS) -c,$$(if $$(filter %.c,$(1)),$$(CC) $$(CFLAGS) -c,$$(CXX) $$(CXXFLAGS) -c)) $$< -o $$@
endef
$(foreach s,$(SRCS),$(eval $(call compile,$(s))))
-include $(OBJS:.o=.d)                           # the headers (and included C) each object was compiled from

$(OUT)/$(APP).elf: $(OBJS) $(BUILD)/soc/link.ld
	$(CXX) $(LDFLAGS) $(OBJS) $(APP_LIBS) -lm -Wl,-Map,$(OUT)/$(APP).map -o $@

$(OUT)/$(APP).bin: $(OUT)/$(APP).elf
	$(OBJCOPY) -O binary $< $@

$(OUT)/$(APP).dump: $(OUT)/$(APP).elf
	$(OBJDUMP) -d -S $< > $@

run: $(OUT)/$(APP).elf $(if $(HOST),,$(OUT)/$(APP).bin)
	$(PYTHON) $(SDK)/tools/run.py '$(SOC)' $(OUT)/$(APP).elf $(if $(RUN_CYCLES),--cycles $(RUN_CYCLES))

clean:
	rm -rf $(OUT)

.PHONY: all run clean
