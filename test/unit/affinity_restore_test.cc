/*
 * Unit test: CPU affinity restore after NCCL communicator init/destroy.
 *
 * NCCL may temporarily set the calling thread's CPU affinity during
 * ncclCommInitRank (and related paths) for GPU-local allocation. It must
 * restore the original affinity when init returns (at the exit path) and
 * must leave it restored after destroy. This test checks both points to
 * catch regressions where the restore step uses the wrong value
 * (e.g. restoring to comm->cpuAffinity instead of the saved affinity).
 *
 * We widen the process affinity to all visible CPUs before init so that
 * affinitySave (what NCCL saves) is broader than comm->cpuAffinity (the
 * GPU-local set, e.g. 0-31). A wrong restore to comm->cpuAffinity then
 * leaves us with the narrow mask instead of the wide one, and the test fails.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <nccl.h>
#include <cuda_runtime.h>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#define HAVE_AFFINITY_CHECK 1
#endif

int main() {
  int deviceCount;
  cudaError_t cuda_err = cudaGetDeviceCount(&deviceCount);
  assert(cuda_err == cudaSuccess);

  if (deviceCount < 1) {
    printf("Skip affinity restore test: no GPUs [WAIVE]\n");
    return 0;
  }

#ifdef HAVE_AFFINITY_CHECK
  cpu_set_t affinityBefore;
  if (sched_getaffinity(0, sizeof(cpu_set_t), &affinityBefore) != 0) {
    printf("Skip affinity restore test: sched_getaffinity failed [WAIVE]\n");
    return 0;
  }
  /* Widen to all configured CPUs so that our mask is broader than the
   * GPU-local set NCCL will use. Then wrong restore to comm->cpuAffinity
   * (narrow) leaves affinity different from affinityBefore (wide). */
  long nconf = sysconf(_SC_NPROCESSORS_CONF);
  if (nconf <= 0 || nconf > CPU_SETSIZE) {
    printf("Skip affinity restore test: cannot get CPU count [WAIVE]\n");
    return 0;
  }
  CPU_ZERO(&affinityBefore);
  for (int i = 0; i < nconf && i < CPU_SETSIZE; i++) {
    CPU_SET(i, &affinityBefore);
  }
  if (sched_setaffinity(0, sizeof(cpu_set_t), &affinityBefore) != 0) {
    printf("Skip affinity restore test: sched_setaffinity (widen) failed [WAIVE]\n");
    return 0;
  }
  /* Use the effective affinity after set (cgroup may restrict the mask). */
  if (sched_getaffinity(0, sizeof(cpu_set_t), &affinityBefore) != 0) {
    printf("Skip affinity restore test: sched_getaffinity after set failed [WAIVE]\n");
    return 0;
  }
#endif

  ncclComm_t comm;
  ncclUniqueId id;
  ncclResult_t res;

  res = ncclGetUniqueId(&id);
  assert(res == ncclSuccess);

  res = ncclCommInitRank(&comm, 1, id, 0);
  assert(res == ncclSuccess);

#ifdef HAVE_AFFINITY_CHECK
  /* Restore happens at init exit, so affinity should already be restored when Init returns. */
  cpu_set_t affinityAfterInit;
  if (sched_getaffinity(0, sizeof(cpu_set_t), &affinityAfterInit) != 0) {
    printf("Check CPU affinity restore after ncclCommInitRank [FAIL] (getaffinity)\n");
    return 1;
  }
  if (!CPU_EQUAL(&affinityBefore, &affinityAfterInit)) {
    printf("Check CPU affinity restore after ncclCommInitRank [FAIL] (affinity not restored on return)\n");
    return 1;
  }
#endif

  res = ncclCommDestroy(comm);
  assert(res == ncclSuccess);

#ifdef HAVE_AFFINITY_CHECK
  cpu_set_t affinityAfterDestroy;
  if (sched_getaffinity(0, sizeof(cpu_set_t), &affinityAfterDestroy) != 0) {
    printf("Check CPU affinity restore after ncclCommDestroy [FAIL] (getaffinity)\n");
    return 1;
  }
  if (!CPU_EQUAL(&affinityBefore, &affinityAfterDestroy)) {
    printf("Check CPU affinity restore after ncclCommDestroy [FAIL] (affinity changed and not restored)\n");
    return 1;
  }
  printf("Check CPU affinity restore after ncclCommInitRank and ncclCommDestroy [PASS]\n");
#else
  printf("Check CPU affinity restore: skipped (non-Linux) [WAIVE]\n");
#endif

  return 0;
}
