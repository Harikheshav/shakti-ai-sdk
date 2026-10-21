// What TVM's C kernels call at run time (tools/tvm_aot.py), on bare metal.  Workspace: a stack in the model's
// tvm_workspace (model.c: the most the kernels ask for at once); a block freed out of order is released when
// those above it are.
#include <stdint.h>
#include <stdio.h>

#include "tvm/ffi/c_api.h"
#include "tvm/runtime/c_backend_api.h"

extern uint8_t tvm_workspace[];
extern const size_t tvm_workspace_size;

#define MAX_BLOCKS 64
static struct { size_t start; int freed; } blocks[MAX_BLOCKS];
static int nblocks;
static size_t top;

void *TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_hint,
                               int dtype_bits_hint)
{
  (void)device_type, (void)device_id, (void)dtype_code_hint, (void)dtype_bits_hint;
  size_t start = (top + 63) & ~(size_t)63;
  if (nblocks == MAX_BLOCKS || start + nbytes > tvm_workspace_size) {
    printf("TVM workspace: %lu bytes more do not fit\n", (unsigned long)nbytes);
    return NULL;
  }
  blocks[nblocks].start = top;
  blocks[nblocks++].freed = 0;
  top = start + nbytes;
  return tvm_workspace + start;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void *ptr)
{
  (void)device_type, (void)device_id;
  for (int i = nblocks - 1; i >= 0; i--) {
    if (tvm_workspace + ((blocks[i].start + 63) & ~(size_t)63) == (uint8_t *)ptr && !blocks[i].freed) {
      blocks[i].freed = 1;
      while (nblocks && blocks[nblocks - 1].freed)
        top = blocks[--nblocks].start;
      return 0;
    }
  }
  return -1;
}

void TVMFFIErrorSetRaisedFromCStrParts(const char *kind, const char **message_parts, int32_t num_parts)
{
  printf("TVM %s: ", kind);
  for (int i = 0; i < num_parts; i++)
    printf("%s", message_parts[i]);
  printf("\n");
}
