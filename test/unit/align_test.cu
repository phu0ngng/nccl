#include <stdio.h>
#include "nccl.h"

#include "device.h"
#include "reduce_kernel.h"
#include "common_kernel.h"

#include <unistd.h>
static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
	return;
    }
  }
}

#undef CUDACHECK
#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Cuda failure %s:%d '%s'\n",         \
         hostname,                                  \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#undef NCCLCHECK
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: NCCL failure %s:%d '%s'\n",         \
         hostname,                                  \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define MAXRANKS 1
#define MAXSIZE (32*1024*1024+16)

#if __CUDA_ARCH__ >= 800
#define UNROLL 8
#else
#define UNROLL 4
#endif

template<typename T, int NSRCS, int NDSTS, int NREPS>
__global__ void ReduceCopyMultiKernel(const T** srcs, T** dsts, int nsrcs, int ndsts, const int nelem) {
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  int n = nelem/nblocks;
  for (int i=0; i<NSRCS; i++) srcs[i] += bid*n;
  for (int i=0; i<NDSTS; i++) dsts[i] += bid*n;
  for (int i=0; i<NREPS; i++) {
    reduceCopyFull<UNROLL, FuncSum<T>, T, 0, NSRCS >= 2 ? 2 : 1, NSRCS, 0, NDSTS >= 2 ? 2 : 1, NDSTS, 0>
      (threadIdx.x, blockDim.x, 0, NULL, false, nsrcs, (void**)srcs, ndsts, (void**)dsts, n);
  }
}

#include <sys/time.h>

double getTimeUsec() {
  struct timeval tv;
  gettimeofday(&tv, NULL);

  return tv.tv_usec*1.0 + tv.tv_sec*1E6;
}

#define NREPS 20
const int offsets[] = { 0, 1, 2, 4, 8, 3, 7, 15 };
#define NOFFSETS 8

template<typename T, int NSRCS, int NDSTS, int NBLOCKS, int NTHREADS>
ncclResult_t testReduceCopy(T ** srcs, T ** dsts, const T ** devSrcs, T ** devDsts, int nsrcs, int ndsts) {
  const int size = MAXSIZE - 16*sizeof(T); // Max unalign
  const int sizeT = size/sizeof(T);
  T* s[NSRCS];
  T* d[NDSTS];

  printf("Offset D -> ");
  for (int dOffset = 0; dOffset < NOFFSETS; dOffset++) {
    printf("%7d ", offsets[dOffset]);
  }
  printf("\nOffset S -v\n");

  for (int sOffset = 0; sOffset < NOFFSETS; sOffset++) {
    printf("%11d ", offsets[sOffset]);
    for (int dOffset = 0; dOffset < NOFFSETS; dOffset++) {
      for (int i=0; i<NSRCS; i++) s[i] = srcs[i] + offsets[sOffset];
      for (int i=0; i<NDSTS; i++) d[i] = dsts[i] + offsets[dOffset];
      CUDACHECK(cudaMemcpy(devSrcs, s, sizeof(void*)*MAXRANKS, cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(devDsts, d, sizeof(void*)*MAXRANKS, cudaMemcpyHostToDevice));
      CUDACHECK(cudaDeviceSynchronize());
      double time = getTimeUsec();
      for (int i=0; i<NREPS; i++) {
        ReduceCopyMultiKernel<T, NSRCS, NDSTS, 1><<<NBLOCKS, NTHREADS>>>(devSrcs, devDsts, nsrcs, ndsts, sizeT);
      }
      cudaError_t ret;
      if ((ret = cudaDeviceSynchronize()) != cudaSuccess) {
        printf("%4s%3d ", "Err", ret);
	while (cudaGetLastError());
      } else {
        time = getTimeUsec() - time;
        printf("%7g ", (((size_t)size)*NREPS)/time);
      }
      fflush(stdout);
    }
    printf("\n");
  }
  return ncclSuccess;
}

int main() {
  char* mem;
  void* srcs[MAXRANKS];
  void* dsts[MAXRANKS];
  void** devSrcs;
  void** devDsts;
  setlinebuf(stdout);
  CUDACHECK(cudaMalloc(&mem, MAXRANKS*2*MAXSIZE));
  CUDACHECK(cudaMalloc(&devSrcs, sizeof(void*)*MAXRANKS));
  CUDACHECK(cudaMalloc(&devDsts, sizeof(void*)*MAXRANKS));
  for (int i=0; i<MAXRANKS; i++) {
    srcs[i] = mem+MAXSIZE*i;
    dsts[i] = mem+MAXSIZE*(MAXRANKS+i);
  }

  printf(" ==== Typesize  8 ==== \n");
  NCCLCHECK((testReduceCopy<int8_t, 1, 1, 1, 512>((int8_t**)srcs, (int8_t**)dsts, (const int8_t**)devSrcs, (int8_t**)devDsts, 1, 1)));
  printf("\n ==== Typesize 16 ==== \n");
  NCCLCHECK((testReduceCopy<half, 1, 1, 1, 512>((half**)srcs, (half**)dsts, (const half**)devSrcs, (half**)devDsts, 1, 1)));
  printf("\n ==== Typesize 32 ==== \n");
  NCCLCHECK((testReduceCopy<int32_t, 1, 1, 1, 512>((int32_t**)srcs, (int32_t**)dsts, (const int32_t**)devSrcs, (int32_t**)devDsts, 1, 1)));
  printf("\n ==== Typesize 64 ==== \n");
  NCCLCHECK((testReduceCopy<int64_t, 1, 1, 1, 512>((int64_t**)srcs, (int64_t**)dsts, (const int64_t**)devSrcs, (int64_t**)devDsts, 1, 1)));
  return 0;
}
