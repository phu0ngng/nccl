#include <iostream>
#include <thread>
#include <chrono>
#include <nccl.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <sys/wait.h>
#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>

#define NUM_GPUS 2
#define NUM_ELEMENTS 1000000
#define NUM_ALL_REDUCES 32

#define MPI_TRY(call)                        \
  do {                                       \
    int status = call;                       \
    if (MPI_SUCCESS != status) {             \
      fprintf(stderr,"MPI call='%s' failed. Error %d\n", #call, status); \
      exit(EXIT_FAILURE);                    \
    }                                        \
  } while (0)

#undef CUDACHECK
#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Cuda failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#undef NCCLCHECK
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("NCCL failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void runAllReduceProcess(int rank, int num_devices, ncclUniqueId id) {
    cudaStream_t stream;
    ncclComm_t comm;

    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaStreamCreate(&stream));
    NCCLCHECK(ncclCommInitRank(&comm, num_devices, id, rank));

    float* device_tensor;
    float* host_tensor;
    CUDACHECK(cudaMalloc(&device_tensor, NUM_ELEMENTS * sizeof(float)));
    CUDACHECK(cudaMallocHost(&host_tensor, NUM_ELEMENTS * sizeof(float)));
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        host_tensor[i] = 1.0f;
    }
    CUDACHECK(cudaMemcpy(device_tensor, host_tensor, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));

    // for some reason this all-reduce is necessary
    NCCLCHECK(ncclAllReduce(device_tensor, device_tensor, NUM_ELEMENTS, ncclFloat32, ncclSum, comm, stream));

    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaDeviceSynchronize());

    // Perform multiple all-reduce operations
    if (rank == 0) {
        for (int i = 0; i < NUM_ALL_REDUCES; i++) {
            NCCLCHECK(ncclAllReduce(device_tensor, device_tensor, NUM_ELEMENTS, ncclFloat32, ncclSum, comm, stream));
        }
    }

    std::cout << "ABORT " << rank << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    NCCLCHECK(ncclCommAbort(comm));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;


    CUDACHECK(cudaMemcpy(host_tensor, device_tensor, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaDeviceSynchronize());
    // Clean up
    CUDACHECK(cudaFree(device_tensor));
    CUDACHECK(cudaStreamDestroy(stream));
    //NCCLCHECK(ncclCommDestroy(comm));

    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Process " << rank << " completed all-reduce operations successfully!" << std::endl;
    return;
}

int main(int argc, char** argv) {
    ncclUniqueId id;
    int comm_rank = 0;
    int comm_size = 0;

    MPI_TRY(MPI_Init(&argc, &argv));
    MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));
    MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));

    if (comm_rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
    MPI_TRY(MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_CHAR, 0, MPI_COMM_WORLD));

    // Only need 2 ranks
    assert(comm_size == 2);
    runAllReduceProcess(comm_rank, comm_size, id);

    MPI_TRY(MPI_Finalize());
    return 0;
}
