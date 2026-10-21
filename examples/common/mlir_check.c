// A common example on MLIR: the model compiled ahead of time (model.h, tools/mlir_aot.py) on the example's test
// inputs, checked against the host reference: within 1e-4 + 1e-3 * |reference| (as the TFLM float check), or
// exactly (CHECK_EXACT: the reference is the same code run on the host).
#include <math.h>
#include <stdio.h>

#include "model.h"
#include "shakti_rt.h"

#ifndef CHECK_NAME
#define CHECK_NAME "float"
#endif
#ifdef CHECK_EXACT
#define TOLERANCE(want) 0.0
#else
#define TOLERANCE(want) (1e-4 + 1e-3 * fabs(want))
#endif
#ifndef CHECK_TESTS
#define CHECK_TESTS MLIR_NUM_TESTS                     // TESTS=n: only the first n test inputs
#endif

static float output[MLIR_OUTPUT_SIZE];

int main(void)
{
  int bad = 0;
  double worst = 0;
  uint64_t cycles = 0;
  char num[21];
  for (int t = 0; t < CHECK_TESTS; t++) {
    uint64_t c0 = rt_cycles();
    int rc = mlir_model_run(mlir_tests_input + t * MLIR_INPUT_SIZE, output);
    cycles += rt_cycles() - c0;
    if (rc) {
      printf(CHECK_NAME ": mlir_model_run failed (%d) on test %d\nRESULT FAIL\n", rc, t);
      return 1;
    }
    for (int o = 0; o < MLIR_OUTPUT_SIZE; o++) {
      double want = mlir_tests_ref[t * MLIR_OUTPUT_SIZE + o], d = fabs(output[o] - want);
      if (d > worst) worst = d;
      if (d > TOLERANCE(want)) bad++;
    }
  }
  printf(CHECK_NAME ": %d tests, %d outputs off, max |diff| %d.%06d, %s cycles per inference, heap %lu bytes\n",
         CHECK_TESTS, bad, (int)worst, (int)((worst - (int)worst) * 1e6), rt_u64(cycles / CHECK_TESTS, num),
         (unsigned long)rt_heap_used());
  printf(bad ? "RESULT FAIL\n" : "RESULT PASS\n");
  return bad ? 1 : 0;
}
