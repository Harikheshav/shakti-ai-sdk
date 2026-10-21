# TensorFlow Lite Micro's own person_detection example (an int8 MobileNet on 96x96 greyscale images), its test
# program unchanged: frameworks/tflite-micro/tensorflow/lite/micro/examples/person_detection/person_detection_test.cc.
# Prints "~~~ALL TESTS PASSED~~~" when the person image scores above the no-person image and the other way round.
APP := person_detection
FRAMEWORK := tflm
PD := tensorflow/lite/micro/examples/person_detection
APP_SRCS := ../../../frameworks/tflite-micro/$(PD)/person_detection_test.cc ../../../frameworks/tflite-micro/$(PD)/model_settings.cc
TFLM_DATA := tensorflow/lite/micro/models/person_detect.tflite $(PD)/testdata/person.bmp $(PD)/testdata/no_person.bmp
RUN_CYCLES ?= 1000000000000
include ../../../mk/common.mk
