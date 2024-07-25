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


#define MAXRANKS 16
#define MAXSIZE (32*1024*1024)
#define UNROLL 4

template<typename T>
__global__ void InitKernel(T** srcs, T** dsts, int nsrcs, int ndsts, int size, int totalSize) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int nthreads = blockDim.x * gridDim.x;
  // Set only that interval
  for (int offset = tid; offset < size; offset += nthreads) {
    for (int i=0; i<nsrcs; i++) {
      srcs[i][offset] = (offset << 8) + i;
    }
  }
  // Clear everything else
  for (int offset = size+tid; offset < totalSize; offset += nthreads) {
    for (int i=0; i<nsrcs; i++) {
      srcs[i][offset] = 0;
    }
  }
  for (int offset = tid; offset < totalSize; offset += nthreads) {
    for (int i=0; i<ndsts; i++) {
      dsts[i][offset] = 1;
    }
  }
}

#ifdef COPYCOPY
template<typename T>
__global__ void CopyCopyRef(const T** srcs, T** dsts, int nptrs, int startOffset, int size) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int nthreads = blockDim.x * gridDim.x;
  for (int offset = startOffset+tid; offset < startOffset+size; offset += nthreads) {
    for (int i=0; i<nptrs; i++) {
      dsts[i][offset] = srcs[i][offset];
    }
  }
}
#endif

template<typename T, int NREPS>
__global__ void ReduceCopyRef(T** srcs, T** dsts, int nsrcs, int ndsts, int size) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int nthreads = blockDim.x * gridDim.x;
  for (int i=0; i<NREPS; i++) {
    for (int offset = tid; offset < size; offset += nthreads) {
      T val = 0;
      for (int i=0; i<nsrcs; i++) {
        val += srcs[i][offset];
      }
      for (int i=0; i<ndsts; i++) {
        dsts[i][offset] = val;
      }
    }
  }
}

#ifdef COPYCOPY
template<typename T>
__global__ void CheckCopyKernel(T** srcs, T** dsts, int nptrs, int startOffset, int size, int totalSize) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int nthreads = blockDim.x * gridDim.x;
  for (int offset = tid; offset < startOffset; offset += nthreads) {
    for (int i=0; i<nptrs; i++) {
      if (srcs[i][offset] != 0) printf("[%X/%X](%p) src = %X != %X\n", i, offset, srcs[i]+offset, srcs[i][offset], 0);
      if (dsts[i][offset] != 0) printf("[%X/%X](%p) dst = %X != %X\n", i, offset, dsts[i]+offset, dsts[i][offset], 0);
    }
  }
  for (int offset = startOffset+tid; offset < startOffset+size; offset += nthreads) {
    for (int i=0; i<nptrs; i++) {
      if (srcs[i][offset] != (offset << 8) + i) printf("[%X/%X](%p) src = %X != %X\n", i, offset, srcs[i]+offset, srcs[i][offset], (offset << 8) + i);
      if (dsts[i][offset] != (offset << 8) + i) printf("[%X/%X](%p) dst = %X != %X\n", i, offset, dsts[i]+offset, dsts[i][offset], (offset << 8) + i);
    }
  }
  for (int offset = startOffset+size+tid; offset < totalSize; offset += nthreads) {
    for (int i=0; i<nptrs; i++) {
      if (srcs[i][offset] != 0) printf("[%X/%X](%p) src = %X != %X\n", i, offset, srcs[i]+offset, srcs[i][offset], 0);
      if (dsts[i][offset] != 0) printf("[%X/%X](%p) dst = %X != %X\n", i, offset, dsts[i]+offset, dsts[i][offset], 0);
    }
  }
}
#endif

template<typename T>
__global__ void CheckReduceKernel(T** srcs, T** dsts, int nsrcs, int ndsts, int size, int totalSize) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int nthreads = blockDim.x * gridDim.x;
  for (int offset = tid; offset < size; offset += nthreads) {
    for (int i=0; i<nsrcs; i++) {
      if (srcs[i][offset] != (offset << 8) + i) printf("[%X/%X](%p) src = %X != %X\n", i, offset, srcs[i]+offset, srcs[i][offset], (offset << 8) + i);
    }
    for (int i=0; i<ndsts; i++) {
      T expected =  (offset << 8)*nsrcs + nsrcs*(nsrcs-1)/2;
      if (dsts[i][offset] != expected) printf("[%X/%X](%p) dst = %X != %X\n", i, offset, dsts[i]+offset, dsts[i][offset], expected);
    }
  }
  for (int offset = size+tid; offset < totalSize; offset += nthreads) {
    for (int i=0; i<nsrcs; i++) {
      if (srcs[i][offset] != 0) printf("[%X/%X](%p) src = %X != %X\n", i, offset, srcs[i]+offset, srcs[i][offset], 0);
    }
    for (int i=0; i<ndsts; i++) {
      if (dsts[i][offset] != 1) printf("[%X/%X](%p) dst = %X != %X\n", i, offset, dsts[i]+offset, dsts[i][offset], 1);
    }
  }
}

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

template<typename T, int NSRCS, int NDSTS, int NBLOCKS, int NTHREADS>
ncclResult_t testReduceCopy(T ** srcs, T ** dsts, int nsrcs, int ndsts) {
  const int maxSizeT = MAXSIZE/sizeof(T);

  printf("============== ReduceCopy %d -> %d / %dx%d ==============\n", nsrcs, ndsts, NBLOCKS, NTHREADS);
  printf("     Nbytes      Nelem Basic(us) Basic(MB/s)   Multi(us)  Multi(MB/s)\n");
  for (int size = 2; size < MAXSIZE; size <<= 1) {
    int sizeT = size/sizeof(T);
    printf(" %10d %10d ", size, sizeT);

    double time = getTimeUsec();
    for (int i=0; i<NREPS; i++)
      ReduceCopyRef<T, NREPS><<<64, 256>>>(srcs, dsts, nsrcs, ndsts, sizeT);
    CUDACHECK(cudaDeviceSynchronize());
    time = getTimeUsec() - time;
    printf("%11g %11g |", time, (((size_t)size)*NREPS*NREPS)/time);

    InitKernel<<<32, 256>>>(srcs, dsts, nsrcs, ndsts, sizeT, maxSizeT);
    CUDACHECK(cudaDeviceSynchronize());

    time = getTimeUsec();
    for (int i=0; i<NREPS; i++) {
      ReduceCopyMultiKernel<T, NSRCS, NDSTS, NREPS><<<NBLOCKS, NTHREADS>>>((const T**)srcs, dsts, nsrcs, ndsts, sizeT);
    }
    CUDACHECK(cudaDeviceSynchronize());
    time = getTimeUsec() - time;
    printf("%11g %11g |", time, (((size_t)size)*NREPS*NREPS)/time);

    CheckReduceKernel<<<64, 256>>>(srcs, dsts, nsrcs, ndsts, sizeT, maxSizeT);
    CUDACHECK(cudaDeviceSynchronize());

    printf("\n");
  }
  return ncclSuccess;
}

#ifdef COPYCOPY
template<typename T>
ncclResult_t testCopyCopySimple(T ** srcs, T ** dsts, int nptrs) {
  const int maxSizeT = MAXSIZE/sizeof(T);

  printf("===  CopyCopy  %4d ===\n", nptrs);
  for (int size = 2; size < MAXSIZE; size <<= 1) {
    const int sizeT = size/sizeof(T);
    printf(" %10d ", size);

    for (int offset = 0; offset <= size/2; offset += size/2) {
      const int offsetT = offset/sizeof(T);
      InitKernel<<<32, 256>>>(srcs, dsts, nptrs, offsetT, sizeT, maxSizeT);
      CUDACHECK(cudaDeviceSynchronize());

      double time = getTimeUsec();
      for (int i=0; i<NREPS; i++)
        CopyCopyKernel<<<1, 256>>>(srcs, dsts, nptrs, offsetT, sizeT);
      CUDACHECK(cudaDeviceSynchronize());
      time = getTimeUsec() - time;
      printf("%9g  ", (((size_t)size)*NREPS*nptrs)/time);

      CheckCopyKernel<<<32, 256>>>(srcs, dsts, nptrs, offsetT, sizeT, maxSizeT);
      CUDACHECK(cudaDeviceSynchronize());

      InitKernel<<<32, 256>>>(srcs, dsts, nptrs, offsetT, sizeT, maxSizeT);
      CUDACHECK(cudaDeviceSynchronize());

      time = getTimeUsec();
      for (int i=0; i<NREPS; i++)
        CopyCopyKernel2<<<1, 256>>>(srcs, dsts, nptrs, offsetT, sizeT);
      CUDACHECK(cudaDeviceSynchronize());
      time = getTimeUsec() - time;
      printf("%9g  ", (((size_t)size)*NREPS*nptrs)/time);

      CheckCopyKernel<<<32, 256>>>(srcs, dsts, nptrs, offsetT, sizeT, maxSizeT);
      CUDACHECK(cudaDeviceSynchronize());

      time = getTimeUsec();
      for (int i=0; i<NREPS; i++)
        CopyCopyRef<<<1, 256>>>(srcs, dsts, nptrs, offsetT, sizeT);
      CUDACHECK(cudaDeviceSynchronize());
      time = getTimeUsec() - time;
      printf("%9g  ", (((size_t)size)*NREPS*nptrs)/time);

    }
    printf("\n");
  }
  return ncclSuccess;
}
#endif

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
    //printf("[%d] %p %p\n", i, srcs[i], dsts[i]);
  }
  CUDACHECK(cudaMemcpy(devSrcs, srcs, sizeof(void*)*MAXRANKS, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(devDsts, dsts, sizeof(void*)*MAXRANKS, cudaMemcpyHostToDevice));

#ifdef COPYCOPY
  NCCLCHECK(testCopyCopySimple((int32_t**)devSrcs, (int32_t**)devDsts, n));
#endif
  ncclResult_t ret;
  ret = testReduceCopy<int32_t, 1, 1, 1, 256>((int32_t**)devSrcs, (int32_t**)devDsts, 1, 1); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 1, 2, 1, 256>((int32_t**)devSrcs, (int32_t**)devDsts, 1, 2); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 2, 1, 1, 256>((int32_t**)devSrcs, (int32_t**)devDsts, 2, 1); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 2, 2, 1, 256>((int32_t**)devSrcs, (int32_t**)devDsts, 2, 2); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 1, 3, 1, 256>((int32_t**)devSrcs, (int32_t**)devDsts, 1, 3); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 1, 4, 1, 256>((int32_t**)devSrcs, (int32_t**)devDsts, 1, 4); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 3, 1, 1, 256>((int32_t**)devSrcs, (int32_t**)devDsts, 3, 1); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 4, 1, 1, 256>((int32_t**)devSrcs, (int32_t**)devDsts, 4, 1); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 1, 1, 1, 128>((int32_t**)devSrcs, (int32_t**)devDsts, 1, 1); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 1, 2, 1, 128>((int32_t**)devSrcs, (int32_t**)devDsts, 1, 2); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 2, 1, 1, 128>((int32_t**)devSrcs, (int32_t**)devDsts, 2, 1); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 2, 2, 1, 128>((int32_t**)devSrcs, (int32_t**)devDsts, 2, 2); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 1, 3, 1, 128>((int32_t**)devSrcs, (int32_t**)devDsts, 1, 3); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 1, 4, 1, 128>((int32_t**)devSrcs, (int32_t**)devDsts, 1, 4); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 3, 1, 1, 128>((int32_t**)devSrcs, (int32_t**)devDsts, 3, 1); NCCLCHECK(ret);
  ret = testReduceCopy<int32_t, 4, 1, 1, 128>((int32_t**)devSrcs, (int32_t**)devDsts, 4, 1); NCCLCHECK(ret);
  return 0;
}
