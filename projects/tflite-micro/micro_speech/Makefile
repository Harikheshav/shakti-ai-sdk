# TensorFlow Lite Micro's own micro_speech example (an int8 keyword model behind TFLM's audio front-end, the
# signal library's preprocessor model), its test program unchanged:
# frameworks/tflite-micro/tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc.  Prints
# "~~~ALL TESTS PASSED~~~" when every recording gives its word and the features match upstream's.
APP := micro_speech
FRAMEWORK := tflm
MS := tensorflow/lite/micro/examples/micro_speech
APP_SRCS := ../../../frameworks/tflite-micro/$(MS)/micro_speech_test.cc
TFLM_DATA := $(MS)/models/micro_speech_quantized.tflite $(MS)/models/audio_preprocessor_int8.tflite \
             $(foreach w,no_1000ms yes_1000ms silence_1000ms noise_1000ms no_30ms yes_30ms,$(MS)/testdata/$(w).wav)
RUN_CYCLES ?= 1000000000000
include ../../../mk/common.mk
