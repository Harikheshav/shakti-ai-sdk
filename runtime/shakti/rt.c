/* shakti-ai-sdk runtime: start, console, exit, traps and the newlib system calls.

   Console: CONSOLE_UART writes the SoC's console UART (UART0_BASE from the generated soc.h; the uart_v3 status
   register at +0xC, bit 1 = transmit FIFO full, data at +4); CONSOLE_HTIF uses Spike's host interface.
   Exit: "[exit N]" and an EOT character, which ends an RTL simulation; on Spike the HTIF exit code. */
#include <errno.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/times.h>
#include <unistd.h>
#include "shakti_rt.h"

#ifdef CONSOLE_UART
#include "soc.h"
#define REG8(a)  (*(volatile uint8_t *)(uintptr_t)(a))
#define REG16(a) (*(volatile uint16_t *)(uintptr_t)(a))

void rt_putc(char c)
{
  while (REG16(UART0_BASE + 0xC) & 2)
    ;
  REG8(UART0_BASE + 4) = (uint8_t)c;
}

static void console_flush(void)
{
  while (!(REG16(UART0_BASE + 0xC) & 1))  /* bit 0: transmitter empty */
    ;
}
#endif

#ifdef CONSOLE_HTIF
volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

static void htif(uint64_t cmd)
{
  tohost = cmd;
  while (fromhost == 0)
    ;
  fromhost = 0;
}

void rt_putc(char c)
{
  static volatile uint64_t req[8] __attribute__((aligned(64)));
  static volatile char ch;
  ch = c;
  req[0] = 64;                          /* SYS_write(1, &ch, 1), proxied by Spike */
  req[1] = 1;
  req[2] = (uintptr_t)&ch;
  req[3] = 1;
  __asm__ volatile ("fence" ::: "memory");
  htif((uintptr_t)req);
}

#endif

void rt_puts(const char *s)
{
  while (*s)
    rt_putc(*s++);
}

static void put_dec(long v)
{
  char b[24];
  int i = 0;
  unsigned long u = v < 0 ? -(unsigned long)v : (unsigned long)v;
  do { b[i++] = (char)('0' + u % 10); u /= 10; } while (u);
  if (v < 0) rt_putc('-');
  while (i) rt_putc(b[--i]);
}

static void put_hex(uintptr_t v)
{
  rt_puts("0x");
  for (int s = (int)sizeof(v) * 8 - 4; s >= 0; s -= 4)
    rt_putc("0123456789abcdef"[(v >> s) & 15]);
}

void rt_exit(int code)
{
  rt_puts("\n[exit ");
  put_dec(code);
  rt_puts("]\n");
#ifdef CONSOLE_HTIF
  for (;;)
    tohost = ((uint64_t)code << 1) | 1;
#else
  rt_putc(4);                           /* EOT: the end of an RTL simulation */
  console_flush();
  for (;;)
    __asm__ volatile ("wfi");
#endif
}

void __attribute__((interrupt("machine"), aligned(4))) rt_trap_entry(void)
{
  uintptr_t cause, epc, tval;
  __asm__ volatile ("csrr %0, mcause" : "=r"(cause));
  __asm__ volatile ("csrr %0, mepc" : "=r"(epc));
  __asm__ volatile ("csrr %0, mtval" : "=r"(tval));
  rt_puts("\ntrap: mcause ");
  put_hex(cause);
  rt_puts(" mepc ");
  put_hex(epc);
  rt_puts(" mtval ");
  put_hex(tval);
  rt_exit(128 + (int)(cause & 0xff));
}

/* ---- start: board set-up (the SoC's generated board_init.c, e.g. clocks of gated devices), constructors, main */
extern void __libc_init_array(void);
extern int main(void);
#ifdef SOC_BOARD_INIT
extern void board_init(void);
#endif

void rt_start(void)
{
#ifdef SOC_BOARD_INIT
  board_init();
#endif
  __libc_init_array();
  rt_exit(main());
}

/* ---- newlib system calls */
extern char __heap_start[], __heap_end[];
static char *brk = __heap_start;

void *_sbrk(ptrdiff_t incr)
{
  if (brk + incr > __heap_end || brk + incr < __heap_start) {
    static int told;
    if (!told++)
      rt_puts("heap exhausted\n");               /* malloc returns NULL: say why, once */
    errno = ENOMEM;
    return (void *)-1;
  }
  char *old = brk;
  brk += incr;
  return old;
}

unsigned long rt_heap_used(void)
{
  return (unsigned long)(brk - __heap_start);
}

ssize_t _write(int fd, const void *buf, size_t n)
{
  if (fd != 1 && fd != 2) {
    errno = EBADF;
    return -1;
  }
  const char *p = buf;
  for (size_t i = 0; i < n; i++) {
    if (p[i] == '\n')
      rt_putc('\r');
    rt_putc(p[i]);
  }
  return (ssize_t)n;
}

ssize_t _read(int fd, void *buf, size_t n) { (void)fd; (void)buf; (void)n; return 0; }
int _close(int fd) { (void)fd; errno = EBADF; return -1; }
off_t _lseek(int fd, off_t o, int w) { (void)fd; (void)o; (void)w; return 0; }
int _isatty(int fd) { return fd <= 2; }
int _fstat(int fd, struct stat *st) { (void)fd; st->st_mode = S_IFCHR; return 0; }
int _getpid(void) { return 1; }
int _kill(int pid, int sig) { (void)pid; (void)sig; errno = EINVAL; return -1; }
void _exit(int code) { rt_exit(code); }
clock_t _times(struct tms *t) { clock_t c = (clock_t)rt_cycles(); if (t) { t->tms_utime = c; t->tms_stime = t->tms_cutime = t->tms_cstime = 0; } return c; }
