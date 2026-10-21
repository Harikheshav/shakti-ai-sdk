// A common example on TensorFlow Lite Micro: run the exported model (tools/export.py) on its test inputs and
// compare with the host reference (TensorFlow Lite Micro built for the host, tools/tflm_host.py).  int8: every
// output exactly the reference's; float: within 1e-4 + 1e-3 * |reference| (the compilers may contract differently).  Prints cycles per inference and the arena used.
// CHECK_INT8 / CHECK_FLOAT choose the models (both by default); TFLM_ARENA_KB the arena.
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "ops.h"
#include "shakti_rt.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#if !defined(CHECK_INT8) && !defined(CHECK_FLOAT)
#define CHECK_INT8 1
#define CHECK_FLOAT 1
#endif
#ifdef CHECK_INT8
#include "model_int8_model_data.h"
#include "input_int8_test_data.h"
#include "output_ref_int8_test_data.h"
#endif
#ifdef CHECK_FLOAT
#include "model_float_model_data.h"
#include "input_float_test_data.h"
#include "output_ref_float_test_data.h"
#endif
#ifndef CHECK_TESTS
#define CHECK_TESTS MODEL_NUM_TESTS                   // TESTS=n: only the first n test inputs
#endif
#ifndef TFLM_ARENA_KB
#define TFLM_ARENA_KB 64
#endif

namespace {
alignas(16) uint8_t arena[TFLM_ARENA_KB * 1024];

using Resolver = tflite::MicroMutableOpResolver<MODEL_NUM_OPS>;
#define ADD_OP(name) if (r.Add##name() != kTfLiteOk) { std::printf("cannot add op %s\n", #name); std::exit(10); }

int check(const char *which, const unsigned char *model_data, bool int8, const void *inputs, const void *refs)
{
  Resolver r;
  MODEL_OPS(ADD_OP)
  const tflite::Model *model = tflite::GetModel(model_data);
  tflite::MicroInterpreter it(model, r, arena, sizeof(arena));
  if (it.AllocateTensors() != kTfLiteOk) {
    std::printf("%s: AllocateTensors failed (arena %d KiB)\n", which, TFLM_ARENA_KB);
    return 1;
  }
  int bad = 0;
  double worst = 0;
  uint64_t cycles = 0;
  char num[21];
  for (int t = 0; t < CHECK_TESTS; t++) {
    TfLiteTensor *in = it.input(0), *out = it.output(0);
    for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
      if (int8)
        in->data.int8[i] = static_cast<const int8_t *>(inputs)[t * MODEL_INPUT_SIZE + i];
      else
        in->data.f[i] = static_cast<const float *>(inputs)[t * MODEL_INPUT_SIZE + i];
    }
    uint64_t c0 = rt_cycles();
    if (it.Invoke() != kTfLiteOk) {
      std::printf("%s: Invoke failed on test %d\n", which, t);
      return 1;
    }
    cycles += rt_cycles() - c0;
    for (int o = 0; o < MODEL_OUTPUT_SIZE; o++) {
      double got, want, tol;
      if (int8) {
        got = out->data.int8[o];
        want = static_cast<const int8_t *>(refs)[t * MODEL_OUTPUT_SIZE + o];
        tol = 0;                                   // the reference is TFLite Micro itself on the host
      } else {
        got = out->data.f[o];
        want = static_cast<const float *>(refs)[t * MODEL_OUTPUT_SIZE + o];
        tol = 1e-4 + 1e-3 * std::fabs(want);
      }
      double d = std::fabs(got - want);
      if (d > worst) worst = d;
      if (d > tol) bad++;
    }
  }
  std::printf("%s: %d tests, %d outputs off, max |diff| %d.%06d, %s cycles per inference, arena %lu bytes\n",
              which, CHECK_TESTS, bad, static_cast<int>(worst),
              static_cast<int>((worst - static_cast<int>(worst)) * 1e6),
              rt_u64(cycles / CHECK_TESTS, num),
              static_cast<unsigned long>(it.arena_used_bytes()));
  return bad ? 1 : 0;
}
}  // namespace

int main()
{
  int fail = 0;
#ifdef CHECK_INT8
  fail |= check("int8", g_model_int8_model_data, true, g_input_int8_test_data, g_output_ref_int8_test_data);
#endif
#ifdef CHECK_FLOAT
  fail |= check("float", g_model_float_model_data, false, g_input_float_test_data, g_output_ref_float_test_data);
#endif
  std::printf(fail ? "RESULT FAIL\n" : "RESULT PASS\n");
  return fail;
}
