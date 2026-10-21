/* shakti-ai-sdk runtime: start-up.  Global pointer and stack, trap vector, floating point and vectors on (mstatus.FS, VS),
   .bss cleared, then rt_start (rt.c): board set-up, C/C++ constructors, main, exit. */
#if __riscv_xlen == 64
#define REG_S sd
#define REGBYTES 8
#else
#define REG_S sw
#define REGBYTES 4
#endif

  .section .text.init, "ax"
  .globl _start
_start:
  .option push
  .option norelax
  la gp, __global_pointer$
  .option pop
  la sp, __stack_top
  la t0, rt_trap_entry
  csrw mtvec, t0
  csrw mie, zero
#ifdef __riscv_flen
  li t0, (1 << 13)                     /* mstatus.FS = initial: FP instructions trap until it is set */
  csrs mstatus, t0
  csrwi fcsr, 0
#endif
#ifdef __riscv_vector
  li t0, (1 << 9)                      /* mstatus.VS = initial: vector instructions trap until it is set */
  csrs mstatus, t0
#endif
  la a0, __bss_start
  la a1, __bss_end
1:
  bgeu a0, a1, 2f
  REG_S zero, 0(a0)
  addi a0, a0, REGBYTES
  j 1b
2:
  call rt_start
3:
  j 3b
