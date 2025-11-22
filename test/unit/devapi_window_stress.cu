#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include <cassert>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e_69578e73e9 = cmd;                              \
  if (e_69578e73e9 != cudaSuccess ) {                          \
    char hostname[1024];                            \
    gethostname(hostname, 1024);                    \
    printf("%s: Cuda failure %s:%d '%s'\n",         \
         hostname,                                  \
        __FILE__,__LINE__,cudaGetErrorString(e_69578e73e9));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t e_69578e73e9 = cmd;                  \
  if (e_69578e73e9 != ncclSuccess && e_69578e73e9 != ncclInProgress) {     \
    char hostname[1024];                            \
    gethostname(hostname, 1024);                    \
    printf("%s: NCCL failure %s:%d '%s'\n",         \
         hostname,                                  \
        __FILE__,__LINE__,ncclGetErrorString(e_69578e73e9));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

__global__ void checkFindWin(ncclDevComm comm, void* ptr, ncclWindow_t win, uint32_t rando) {
  if (threadIdx.x * 0xdeadbeef <= rando) {
    ncclWindow_t got = ncclFindWindow(ncclCoopCoalesced(), comm, ptr);
    if (got != win) {
      if (threadIdx.x == 0) printf("ERROR: r=%d find(addr=%p)=%p  ; want=%p\n", comm.rank, ptr, got, win);
      if (threadIdx.x == 0) assert(0);
    }
  }
}

__global__ void dumpWins(ncclDevComm comm) {
  if (threadIdx.x == 0) {
    ncclDevCommWindowTable *p = comm.windowTable;
    while (p) {
      printf("Device table node %p:\n", p);
      for (int i=0; i < 32; i++) {
        auto e = p->entries[i];
        if (e.window) {
          printf("  i=%d win=%p\tlo=%p\thi=%p\n", i, e.window, (void*)e.base, (char*)e.base+e.size);
        }
      }
      p = p->next;
    }
  }
}

uint64_t hash(uint64_t i) {
  i *= 0xdcc162351ffdf937;
  i ^= i>>32;
  i *= 0x2aac028ff9996e1b;
  i ^= i>>32;
  return i;
}

constexpr bool Prints = false;
//constexpr bool Prints = true;

constexpr int WinCount = 512;
//constexpr int WinCount = 4;
//constexpr int WinCount = 1;

constexpr int winStart(int w) {
  return 4096*(32*(w/32)*((w/32)+1)/2 + w%32);
}
constexpr int ArenaSize = winStart(WinCount+1);

int main(int argc, char** argv) {
  ncclComm_t comm[2];
  int rank2dev[2] = {0, 1};
  NCCLCHECK(ncclCommInitAll(comm, 2, rank2dev));
  printf("Created comms\n");

  cudaStream_t stream[2];
  void* arena[2];
  for (int r=0; r < 2; r++) {
    CUDACHECK(cudaSetDevice(r));
    CUDACHECK(cudaStreamCreate(&stream[r]));
    NCCLCHECK(ncclMemAlloc(&arena[r], ArenaSize));
  }
  printf("Allocated arenas at {%p, %p}\n", arena[0], arena[1]);

  ncclDevComm dcomm[2];
  NCCLCHECK(ncclGroupStart());
  for (int r=0; r < 2; r++) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    NCCLCHECK(ncclDevCommCreate(comm[r], &reqs, &dcomm[r]));
  }
  NCCLCHECK(ncclGroupEnd());
  printf("Created devcomms\n");

  ncclWindow_t wins[2][WinCount] = {};
  uint64_t rng = 0;

  int nIters = 4*WinCount;
  for (int iter=0; iter <  nIters; iter++) {
    NCCLCHECK(ncclGroupStart());
    int w = hash(rng++) % WinCount;
    if (wins[0][w] == nullptr) {
      if (Prints) printf("Window insert at %d\n", w);
      for (int r=0; r < 2; r++) {
        NCCLCHECK(ncclCommWindowRegister(comm[r], (char*)arena[r] + winStart(w), winStart(w+1)-winStart(w), &wins[r][w], 0));
      }
    } else {
      if (Prints) printf("Window collision at %d\n", w);
      for (int r=0; r < 2; r++) {
        NCCLCHECK(ncclCommWindowDeregister(comm[r], wins[r][w]));
        wins[r][w] = nullptr;
      }
    }
    NCCLCHECK(ncclGroupEnd());
  }

  if (Prints) {
    printf("Host windows:\n");
    for (int w=0; w < WinCount; w++) {
      if (wins[0][w]) {
        int lo = winStart(w);
        int hi = winStart(w+1);
        if (Prints) printf("  i=%d win=%p\tbeg=%p\tend=%p\n", w, wins[0][w], (char*)arena[0] + lo, (char*)arena[0] + hi);
      }
    }
  }

  printf("Launching kernels\n");
  if (Prints) {
    printf("Launching dumpWins\n");
    cudaSetDevice(0);
    dumpWins<<<1, 128, 0, stream[0]>>>(dcomm[0]);
  }
  for (int w=0; w < WinCount; w++) {
    int lo = winStart(w);
    int hi = winStart(w+1);
    int off = hash(rng++) % (hi-lo);
    uint32_t rando = hash(rng++);
    if (wins[0][w] != nullptr) {
      for (int r=0; r < 2; r++) {
        CUDACHECK(cudaSetDevice(r));
        checkFindWin<<<1, 128, 0, stream[r]>>>(dcomm[r], (char*)arena[r] + lo + off, wins[r][w], rando);
      }
    }
  }

  for (int r=0; r < 2; r++) {
    CUDACHECK(cudaStreamSynchronize(stream[r]));
  }
  printf("Completed kernels\n");

  // cleanup
  NCCLCHECK(ncclGroupStart());
  for (int w=0; w < WinCount; w++) {
    if (wins[0][w] != nullptr) {
      for (int r=0; r < 2; r++) {
        NCCLCHECK(ncclCommWindowDeregister(comm[r], wins[r][w]));
      }
    }
  }
  NCCLCHECK(ncclGroupEnd());
  printf("Deregistered windows\n");

  NCCLCHECK(ncclGroupStart());
  for (int r=0; r < 2; r++) {
    NCCLCHECK(ncclDevCommDestroy(comm[r], &dcomm[r]));
  }
  NCCLCHECK(ncclGroupEnd());

  NCCLCHECK(ncclGroupStart());
  for (int r=0; r < 2; r++) {
    NCCLCHECK(ncclCommDestroy(comm[r]));
  }
  NCCLCHECK(ncclGroupEnd());
  return 0;
}
