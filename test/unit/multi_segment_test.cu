#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <nccl.h>
#include <mpi.h>

// Helper macros for error checking
#define CUDACHECK(cmd) do {                             \
  cudaError_t err = cmd;                                \
  if(err != cudaSuccess) {                              \
    printf("CUDA Error at %s:%d - %s\n",                \
           __FILE__, __LINE__,                          \
           cudaGetErrorString(err));                    \
    exit(1);                                            \
  }                                                     \
} while(0)

#define CUCHECK(cmd) do {                               \
  CUresult err = cmd;                                   \
  if(err != CUDA_SUCCESS) {                             \
    const char* errStr;                                 \
    cuGetErrorString(err, &errStr);                     \
    printf("CUDA Driver Error at %s:%d - %s\n",         \
           __FILE__, __LINE__, errStr);                 \
    exit(1);                                            \
  }                                                     \
} while(0)

#define NCCLCHECK(cmd) do {                             \
  ncclResult_t res = cmd;                               \
  if(res != ncclSuccess) {                              \
    printf("NCCL Error at %s:%d - %s\n",                \
           __FILE__, __LINE__,                          \
           ncclGetErrorString(res));                    \
    exit(1);                                            \
  }                                                     \
} while(0)

#define MPICHECK(cmd) do {                              \
  int e = cmd;                                          \
  if(e != MPI_SUCCESS) {                                \
    printf("MPI Error at %s:%d\n",                      \
           __FILE__, __LINE__);                         \
    exit(1);                                            \
  }                                                     \
} while(0)

// Align size to granularity
#define ALIGN_SIZE(size, granularity) \
  (((size) + (granularity) - 1) / (granularity) * (granularity))

#if CUDART_VERSION >= 11030
void allocateSegmentedMemory(void** ptr, size_t totalSize, int* outNumSegments) {
  size_t segmentSize = 2 * 1024 * 1024; // 2MB per segment
  size_t memGran = 0;
  CUdevice currentDev;
  CUmemAllocationProp memprop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle* handles;
  int cudaDev;
  int dcnt;

  // Get current device
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  CUDACHECK(cudaGetDeviceCount(&dcnt));

  // Setup memory allocation properties
  int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  // Query device for FABRIC handle support (CUDA 12.3+)
  int flag = 0;
#if CUDART_VERSION >= 12030
  cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev);
  if (flag) requestedHandleTypes |= CU_MEM_HANDLE_TYPE_FABRIC;
#endif

  memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  memprop.requestedHandleTypes = (CUmemAllocationHandleType)requestedHandleTypes;
  memprop.location.id = currentDev;

  // Query device for RDMA support
  flag = 0;
  CUCHECK(cuDeviceGetAttribute(&flag,
          CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
          currentDev));
  if (flag) memprop.allocFlags.gpuDirectRDMACapable = 1;

  // Get allocation granularity
  CUCHECK(cuMemGetAllocationGranularity(&memGran, &memprop,
          CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

  // Align segment sizes to granularity
  size_t alignedSegmentSize = ALIGN_SIZE(segmentSize, memGran);
  size_t alignedTotalSize = ALIGN_SIZE(totalSize, alignedSegmentSize);
  int numSegments = alignedTotalSize/alignedSegmentSize;

  CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, alignedTotalSize, memGran, 0, 0));

  handles = (CUmemGenericAllocationHandle *) calloc(numSegments, sizeof(CUmemGenericAllocationHandle));

  // Try with FABRIC handle first, fall back if not supported
  CUresult createErr = cuMemCreate(&handles[0], alignedSegmentSize, &memprop, 0);
#if CUDART_VERSION >= 12030
  if ((requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) && (createErr == CUDA_ERROR_NOT_PERMITTED || createErr == CUDA_ERROR_NOT_SUPPORTED)) {
    requestedHandleTypes &= ~CU_MEM_HANDLE_TYPE_FABRIC;
    memprop.requestedHandleTypes = (CUmemAllocationHandleType)requestedHandleTypes;
    CUCHECK(cuMemCreate(&handles[0], alignedSegmentSize, &memprop, 0));
  } else
#endif
  {
    CUCHECK(createErr);
  }

  for (int segment = 1; segment < numSegments; segment++) {
    CUCHECK(cuMemCreate(&handles[segment], alignedSegmentSize, &memprop, 0));
  }
  CUdeviceptr basePtr = (CUdeviceptr)*ptr;
  for (int segment = 0; segment < numSegments; segment++) {
    CUCHECK(cuMemMap(basePtr, alignedSegmentSize, 0, handles[segment], 0));
    basePtr = basePtr + alignedSegmentSize;
  }

  for (int i = 0; i < dcnt; ++i) {
    int p2p = 0;
    if (i == cudaDev ||
        ((cudaDeviceCanAccessPeer(&p2p, i, cudaDev) == cudaSuccess) && p2p)) {
      accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      accessDesc.location.id = i;
      accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

      basePtr = (CUdeviceptr)*ptr;
      CUCHECK(cuMemSetAccess(basePtr, totalSize, &accessDesc, 1));
    }
    if (0 == p2p && i != cudaDev) {
      fprintf(stderr, "P2P not supported between GPU: %d and GPU: %d\n", cudaDev, i);
    }
  }
  *outNumSegments = numSegments;
  free(handles);
}

void freeSegmentedMemory(void* ptr, int numSegments) {
  CUdeviceptr basePtr = (CUdeviceptr)ptr;
  size_t totalSize = 0;
  for (int segment = 0; segment < numSegments; segment++) {
    size_t size = 0;
    CUmemGenericAllocationHandle handle;
    CUCHECK(cuMemRetainAllocationHandle(&handle, (void *) (basePtr + totalSize)));
    CUCHECK(cuMemRelease(handle));
    CUCHECK(cuMemGetAddressRange(NULL, &size, basePtr + totalSize));
    CUCHECK(cuMemUnmap(basePtr + totalSize, size));
    CUCHECK(cuMemRelease(handle));
    totalSize += size;
  }
  CUCHECK(cuMemAddressFree(basePtr, totalSize));
}

int test_graph_registration(void* sendbuf, void* recvbuf, int count, ncclComm_t comm, cudaStream_t stream) {
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;

  // Begin graph capture
  CUDACHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  NCCLCHECK(ncclAllReduce(sendbuf, recvbuf, count, ncclFloat,
                          ncclSum, comm, stream));
  // End graph capture
  CUDACHECK(cudaStreamEndCapture(stream, &graph));

  // Instantiate graph
  CUDACHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  CUDACHECK(cudaGraphLaunch(graphExec, stream));

  CUDACHECK(cudaDeviceSynchronize());

  CUDACHECK(cudaGraphExecDestroy(graphExec));
  CUDACHECK(cudaGraphDestroy(graph));
  return 0;
}

int test_local_registration(void* sendbuf, void* recvbuf, int count, ncclComm_t comm, cudaStream_t stream) {
  void* sendHandle;
  void* recvHandle;
  NCCLCHECK(ncclCommRegister(comm, sendbuf, count * sizeof(float), &sendHandle));
  NCCLCHECK(ncclCommRegister(comm, recvbuf, count * sizeof(float), &recvHandle));
  NCCLCHECK(ncclAllReduce(sendbuf, recvbuf, count, ncclFloat,
                          ncclSum, comm, stream));
  CUDACHECK(cudaDeviceSynchronize());
  NCCLCHECK(ncclCommDeregister(comm, sendHandle));
  NCCLCHECK(ncclCommDeregister(comm, recvHandle));

  return 0;
}

#endif /* CUDART_VERSION >= 11030 */

int main(int argc, char* argv[]) {
#if CUDART_VERSION >= 11030
  int rank, nranks;

  // Initialize MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

  // Determine number of ranks per node
  int local_rank = 0, local_size = 0;
  MPI_Comm lcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &lcomm);
  MPI_Comm_rank(lcomm, &local_rank);
  MPI_Comm_size(lcomm, &local_size);
  MPI_Comm_free(&lcomm);

  // Set GPU device (assumes one GPU per rank)
  CUDACHECK(cudaSetDevice(local_rank));

  const size_t totalSize = 8 * 1024 * 1024;
  const size_t count = totalSize / sizeof(float);

  float *sendbuf, *recvbuf;
  int numSegments = 0;
  allocateSegmentedMemory((void**)&sendbuf, totalSize, &numSegments);
  allocateSegmentedMemory((void**)&recvbuf, totalSize, &numSegments);

  float initValue = (float)(rank + 1);
  float* h_data = (float*)malloc(totalSize);
  for (size_t i = 0; i < count; i++) {
    h_data[i] = initValue;
  }
  CUDACHECK(cudaMemcpy(sendbuf, h_data, totalSize, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Initialize NCCL
  ncclUniqueId id;
  ncclComm_t comm;

  if (rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&id));
  }
  MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));

  int res = test_graph_registration(sendbuf, recvbuf, count, comm, stream);
  if (res != 0) return res;
  // Verify result
  CUDACHECK(cudaMemcpy(h_data, recvbuf, totalSize, cudaMemcpyDeviceToHost));

  float expected = (float)(nranks * (nranks + 1) / 2);
  bool success = true;

  for (int i = 0; i < count; i++) {
    if (h_data[i] != expected) {
      success = false;
    }
  }
  fprintf(stderr, "[rank %d] Done with graph registration\n", rank);

  res = test_local_registration(sendbuf, recvbuf, count, comm, stream);
  if (res != 0) return res;
  // Verify result
  CUDACHECK(cudaMemcpy(h_data, recvbuf, totalSize, cudaMemcpyDeviceToHost));

  success = true;

  for (int i = 0; i < count; i++) {
    if (h_data[i] != expected) {
      success = false;
    }
  }
  fprintf(stderr, "[rank %d] Done with local registration\n", rank);

  freeSegmentedMemory(sendbuf, numSegments);
  freeSegmentedMemory(recvbuf, numSegments);
  free(h_data);

  CUDACHECK(cudaStreamDestroy(stream));

  NCCLCHECK(ncclCommDestroy(comm));

  MPICHECK(MPI_Finalize());
  return success ? 0 : 1;
#else
  return 0;
#endif /* CUDART_VERSION >= 11030 */
}
