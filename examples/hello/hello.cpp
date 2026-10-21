// hello: the runtime's check on every target.  A global object's constructor runs before main (the start-up
// calls the C++ constructors), printf goes to the console, malloc uses the heap, mcycle counts the time.
#include <cstdio>
#include <cstdlib>
#include "shakti_rt.h"

namespace {
struct Ready {
  int value = 0;
  Ready() { value = 42; }
} ready;
}

int main()
{
  std::printf("hello from shakti-ai-sdk (xlen %d)\n", __riscv_xlen);
  if (ready.value != 42) {
    std::printf("global constructor did not run\n");
    return 1;
  }
  uint64_t t0 = rt_cycles();
  int *buf = static_cast<int *>(std::malloc(1024 * sizeof(int)));
  if (!buf) {
    std::printf("malloc failed\n");
    return 2;
  }
  long sum = 0;
  for (int i = 0; i < 1024; i++) {
    buf[i] = i;
    sum += buf[i];
  }
  std::free(buf);
#ifdef __riscv_flen
  volatile float x = 1.5f;
  std::printf("float: %d\n", static_cast<int>(x * 4.0f));     // nano printf has no %f
#endif
  std::printf("sum %ld in %lu cycles\n", sum, static_cast<unsigned long>(rt_cycles() - t0));
  return sum == 523776 ? 0 : 3;
}
