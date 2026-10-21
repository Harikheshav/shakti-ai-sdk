// The host reference of an ahead-of-time compiled model (tools/tvm_aot.py, tools/mlir_aot.py --host-ref): the same
// generated code built for the host runs the test inputs; the outputs go to argv[1] (float32).  The compiler
// defines AOT_HEADER ("model.h"), AOT_RUN, AOT_INPUT, AOT_IN_SIZE, AOT_OUT_SIZE, AOT_TESTS.
#include <stdio.h>

#include AOT_HEADER

int main(int argc, char **argv)
{
  static float out[AOT_OUT_SIZE];
  FILE *f = argc == 2 ? fopen(argv[1], "wb") : NULL;
  if (!f)
    return 2;
  for (int t = 0; t < AOT_TESTS; t++) {
    if (AOT_RUN(AOT_INPUT + t * AOT_IN_SIZE, out)) {
      fprintf(stderr, "test %d failed\n", t);
      return 1;
    }
    fwrite(out, sizeof(float), AOT_OUT_SIZE, f);
  }
  return fclose(f) ? 1 : 0;
}
