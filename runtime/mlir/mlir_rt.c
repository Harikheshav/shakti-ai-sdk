// What MLIR's lowered code calls at run time (tools/mlir_aot.py), on bare metal: memrefCopy, a copy between
// strided buffers of any rank (as in MLIR's CRunnerUtils.cpp), and what newlib does not have: C23's fmaximum /
// fminimum (LLVM calls them for arith.maximumf / minimumf without hardware floating point) and roundeven.
// MLIR's index type is the XLEN here: intptr_t.
#include <math.h>
#include <stdint.h>
#include <string.h>

// IEEE 754-2019 maximum / minimum: NaN if either is NaN, and -0 < +0
#define MAXMIN(name, T, max)                                                    \
  T name(T a, T b)                                                              \
  {                                                                             \
    if (isnan(a) || isnan(b))                                                   \
      return a + b;                                                             \
    if (a == b)                                                                 \
      return (max) ? (signbit(a) ? b : a) : (signbit(a) ? a : b);               \
    return (max) ? (a > b ? a : b) : (a < b ? a : b);                           \
  }
MAXMIN(fmaximumf, float, 1)
MAXMIN(fminimumf, float, 0)
MAXMIN(fmaximum, double, 1)
MAXMIN(fminimum, double, 0)

// C23 roundeven (requantisation): to nearest, ties to even - rint in the default rounding mode, which nothing here
// changes
float roundevenf(float x)
{
  return rintf(x);
}

double roundeven(double x)
{
  return rint(x);
}

struct unranked {
  intptr_t rank;
  void *desc;               // {char *allocated, *aligned; intptr_t offset, sizes[rank], strides[rank]}
};

struct ranked {
  char *allocated, *aligned;
  intptr_t offset, sizes_strides[];
};

void memrefCopy(intptr_t elem, struct unranked *src_arg, struct unranked *dst_arg)
{
  intptr_t rank = src_arg->rank;
  struct ranked *src = src_arg->desc, *dst = dst_arg->desc;
  const intptr_t *sizes = src->sizes_strides, *s_str = sizes + rank, *d_str = dst->sizes_strides + rank;
  for (intptr_t r = 0; r < rank; r++)
    if (sizes[r] == 0)
      return;
  char *s = src->aligned + src->offset * elem, *d = dst->aligned + dst->offset * elem;
  if (rank == 0) {
    memcpy(d, s, elem);
    return;
  }
  intptr_t idx[rank];
  for (intptr_t r = 0; r < rank; r++)
    idx[r] = 0;
  for (;;) {
    memcpy(d, s, elem);
    intptr_t r = rank - 1;                         // the next index, last dimension fastest
    for (; r >= 0; r--) {
      s += s_str[r] * elem;
      d += d_str[r] * elem;
      if (++idx[r] < sizes[r])
        break;
      s -= sizes[r] * s_str[r] * elem;
      d -= sizes[r] * d_str[r] * elem;
      idx[r] = 0;
    }
    if (r < 0)
      return;
  }
}
