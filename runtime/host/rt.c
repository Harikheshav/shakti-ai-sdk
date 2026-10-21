/* shakti-ai-sdk runtime for SOC=host: the program built for this machine (gcc, its C library and start-up), so the
   same app runs directly on the CPU.  Console: stdout; exit: the process's. */
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include "shakti_rt.h"

void rt_putc(char c)
{
  putchar(c);
}

void rt_puts(const char *s)
{
  fputs(s, stdout);
}

void rt_exit(int code)
{
  fflush(stdout);
  exit(code);
}

unsigned long rt_heap_used(void)
{
  return (unsigned long)mallinfo2().uordblks;
}
