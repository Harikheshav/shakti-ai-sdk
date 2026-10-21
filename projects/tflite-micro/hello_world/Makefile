# TensorFlow Lite Micro's own hello_world example (a sine model, float and int8), its test program unchanged:
# frameworks/tflite-micro/tensorflow/lite/micro/examples/hello_world/hello_world_test.cc.  Prints its profile and
# "~~~ALL TESTS PASSED~~~" when the outputs are within its tolerance.
APP := hello_world
FRAMEWORK := tflm
HW := tensorflow/lite/micro/examples/hello_world
APP_SRCS := ../../../frameworks/tflite-micro/$(HW)/hello_world_test.cc
TFLM_DATA := $(HW)/models/hello_world_float.tflite $(HW)/models/hello_world_int8.tflite
include ../../../mk/common.mk
