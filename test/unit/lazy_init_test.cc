/*
 * Unit test: devrState lazy initialization.
 *
 * This test verifies that devrState uses lazy initialization: after communicator
 * creation (ncclCommInitRank, ncclCommInitAll), devrState.bigSize should remain zero,
 * with ncclDevrInitOnce deferred until the first window registration. If this lazy
 * pattern is ever intentionally changed to eager initialization, this test should be
 * updated or removed accordingly. If ncclDevrInitOnce is inadvertently triggered
 * during initialization, bigSize becomes non-zero and this test fails.
 *
 * Covers two initialization paths:
 *   1. ncclCommInitRank  (single-rank communicator)
 *   2. ncclCommInitAll   (multi-GPU, all visible devices)
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include "comm.h"

static void checkDevrLazy(ncclComm_t comm, const char* label) {
  if (comm->devrState.bigSize != 0) {
    printf("Check devrState lazy init after %s [FAIL] "
           "(bigSize=%zu, expected 0)\n", label, comm->devrState.bigSize);
    exit(1);
  }
  printf("Check devrState lazy init after %s [PASS]\n", label);
}

int main() {
  int deviceCount;
  cudaError_t cuda_err = cudaGetDeviceCount(&deviceCount);
  assert(cuda_err == cudaSuccess);

  if (deviceCount < 1) {
    printf("Skip lazy init test: no GPUs [WAIVE]\n");
    return 0;
  }

  ncclResult_t res;

  // Path 1: ncclCommInitRank (single rank) ////////////////////////////////////
  {
    ncclComm_t comm;
    ncclUniqueId id;

    res = ncclGetUniqueId(&id);
    assert(res == ncclSuccess);

    res = ncclCommInitRank(&comm, 1, id, 0);
    assert(res == ncclSuccess);

    checkDevrLazy(comm, "ncclCommInitRank");

    res = ncclCommDestroy(comm);
    assert(res == ncclSuccess);
  }

  // Path 2: ncclCommInitAll (all visible GPUs) ////////////////////////////////
  {
    ncclComm_t* comms = (ncclComm_t*)calloc(deviceCount, sizeof(ncclComm_t));
    assert(comms != NULL);

    res = ncclCommInitAll(comms, deviceCount, NULL);
    assert(res == ncclSuccess);

    for (int i = 0; i < deviceCount; i++) {
      char label[64];
      snprintf(label, sizeof(label), "ncclCommInitAll rank %d/%d",
               i, deviceCount);
      checkDevrLazy(comms[i], label);
    }

    for (int i = 0; i < deviceCount; i++) {
      res = ncclCommDestroy(comms[i]);
      assert(res == ncclSuccess);
    }
    free(comms);
  }

  printf("All lazy init checks [PASS]\n");
  return 0;
}
