/* shakti-ai-sdk runtime: what programs may call besides the C library. */
#ifndef SHAKTI_RT_H
#define SHAKTI_RT_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* core clock cycles since reset (mcycle: user-mode rdcycle traps on cores without U mode); built for the host
   (SOC=host): nanoseconds, the same unit on every machine */
#if !defined(__riscv)
#include <time.h>
static inline uint64_t rt_cycles(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}
#else
static inline uint64_t rt_cycles(void)
{
#if __riscv_xlen == 64
  uint64_t c;
  __asm__ volatile ("csrr %0, mcycle" : "=r"(c));
  return c;
#else
  uint32_t hi, lo, hi2;
  do {
    __asm__ volatile ("csrr %0, mcycleh" : "=r"(hi));
    __asm__ volatile ("csrr %0, mcycle" : "=r"(lo));
    __asm__ volatile ("csrr %0, mcycleh" : "=r"(hi2));
  } while (hi != hi2);
  return ((uint64_t)hi << 32) | lo;
#endif
}
#endif

/* v in decimal, in buf (21 bytes); for printf("%s"): the C library's printf has no 64-bit integers on RV32 */
static inline const char *rt_u64(uint64_t v, char *buf)
{
  char *p = buf + 20;
  *p = 0;
  do *--p = (char)('0' + v % 10); while (v /= 10);
  return p;
}

void rt_putc(char c);                  /* the console: the SoC's console UART, or Spike's host interface */
void rt_puts(const char *s);
void rt_exit(int code) __attribute__((noreturn));   /* "[exit N]" and the end of a simulation */
unsigned long rt_heap_used(void);     /* bytes of heap malloc has taken (it never gives them back) */

#ifdef __cplusplus
}
#endif
#endif
