/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

// MPI error checking macro
#define MPICHECK(cmd)                                                          \
  do {                                                                         \
    int err = cmd;                                                             \
    if (err != MPI_SUCCESS) {                                                  \
      char error_string[MPI_MAX_ERROR_STRING];                                 \
      int length;                                                              \
      MPI_Error_string(err, error_string, &length);                            \
      fprintf(stderr, "MPI error at %s:%d - %s\n", __FILE__, __LINE__,         \
              error_string);                                                   \
      fprintf(stderr, "Failed MPI operation: %s\n", #cmd);                     \
      MPI_Abort(MPI_COMM_WORLD, err);                                          \
    }                                                                          \
  } while (0)

// Error checking
#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t res = cmd;                                                    \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,   \
              ncclGetErrorString(res));                                        \
      fprintf(stderr, "Failed NCCL operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,   \
              cudaGetErrorString(err));                                        \
      fprintf(stderr, "Failed CUDA operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define TIMEOUT_CYCLES 100000000

#define PRINT_RANK0(my_rank, msg)                                              \
  do {                                                                         \
    if (my_rank == 0) {                                                        \
      printf("%s\n", msg);                                                     \
    }                                                                          \
  } while (0)

/*
 * NCCL barrier with timeout test
 *
 * This example demonstrates NCCL's GPU-Initiated Networking (GIN) capabilities
 * for barrier with timeout synchronization.
 */

// Device API kernel launch configuration
// CTA count must match railGinBarrierCount for proper barrier synchronization
#define NCCL_DEVICE_CTA_COUNT 1
#define NCCL_DEVICE_THREADS_PER_CTA 512
#define CHECK_TIMEOUT(barsync, rank, msg)                              \
 do {                                                                  \
  ncclResult_t result = barsync;                                       \
  if (result == ncclTimeout) {                                         \
    printf("rank %d caught timeout, message: %s\n", rank, msg);        \
  }                                                                    \
} while (0)
// ==========================================================================
// Device Kernel Implementations
// ==========================================================================

// Barrier with timeout kernel - uses GIN for barrier synchronization
// This kernel demonstrates network-based barrier synchronization using GPU-initiated networking
__global__ void GinBarrierWithTimeoutKernel(struct ncclDevComm devComm) {
  int ginContext = 0;
  ncclGin gin { devComm, ginContext };

  // GIN barriers enable coordination between GPU threads across different ranks over network
  ncclGinBarrierSession<ncclCoopCta> bar { ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x };
  CHECK_TIMEOUT(bar.sync(ncclCoopCta(),
   cuda::memory_order_relaxed,
   ncclGinFenceLevel::None,
   TIMEOUT_CYCLES), devComm.rank, "GIN barrier normal");
}

__global__ void GinBarrierReturnForTimeoutKernel(struct ncclDevComm devComm) {
  int ginContext = 0;
  ncclGin gin { devComm, ginContext };
  ncclGinBarrierSession<ncclCoopCta> bar { ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x };
  // some devComm will just return without the barrier, lead to timeout on others
  if (devComm.rank == 1) {
    return;
  }

  CHECK_TIMEOUT(bar.sync(ncclCoopCta(),
   cuda::memory_order_relaxed,
   ncclGinFenceLevel::None,
   TIMEOUT_CYCLES), devComm.rank, "GIN barrier timeout");
}

__global__ void LsaBarrierWithTimeoutKernel(struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(),
    blockIdx.x, devComm.lsaMultimem.mcBasePtr != nullptr };
  CHECK_TIMEOUT(bar.sync(ncclCoopCta(),
   cuda::memory_order_relaxed,
   TIMEOUT_CYCLES), devComm.rank, "LSA barrier normal");
}

__global__ void LsaBarrierWithTimeoutUnicastKernel(struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(),
    blockIdx.x, false};
  CHECK_TIMEOUT(bar.sync(ncclCoopCta(),
   cuda::memory_order_relaxed,
   TIMEOUT_CYCLES), devComm.rank, "LSA barrier unicast normal");
}


__global__ void LsaBarrierReturnForTimeoutKernel(struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(),
    blockIdx.x, devComm.lsaMultimem.mcBasePtr != nullptr };
  // some devComm will just return without the barrier, lead to timeout on others
  if (devComm.lsaRank == 1) {
    return;
  }

  CHECK_TIMEOUT(bar.sync(ncclCoopCta(),
   cuda::memory_order_relaxed,
   TIMEOUT_CYCLES), devComm.rank, "LSA barrier timeout");

  return;
}

__global__ void LsaBarrierReturnForTimeoutUnicastKernel(struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(),
    blockIdx.x, false};
  // some devComm will just return without the barrier, lead to timeout on others
  if (devComm.lsaRank == 1) {
    return;
  }

  CHECK_TIMEOUT(bar.sync(ncclCoopCta(),
   cuda::memory_order_relaxed,
   TIMEOUT_CYCLES), devComm.rank, "LSA barrier unicast timeout");

  return;
}

__global__ void BarrierWithTimeoutKernel(struct ncclDevComm devComm) {
  int ginContext = 0;
  ncclGin gin { devComm, ginContext };
  // GIN barriers for cross-node synchronization
  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  CHECK_TIMEOUT(bar.sync(ncclCoopCta(),
   cuda::memory_order_relaxed,
   ncclGinFenceLevel::None,
   TIMEOUT_CYCLES), devComm.rank, "Barrier normal");
}

__global__ void BarrierReturnForTimeoutKernel(struct ncclDevComm devComm) {
  int ginContext = 0;
  ncclGin gin { devComm, ginContext };
  // GIN barriers for cross-node synchronization
  ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
  // some devComm will just return without the barrier, lead to timeout on others
  if (devComm.rank == 1) {
    return;
  }

  CHECK_TIMEOUT(bar.sync(ncclCoopCta(),
   cuda::memory_order_relaxed,
   ncclGinFenceLevel::None,
   TIMEOUT_CYCLES), devComm.rank, "Barrier timeout");
}

struct testKernel {
  void (*kernel)(struct ncclDevComm devComm);
  const char* kernelName;
};


void barrierWithTimeout(ncclComm_t comm, cudaStream_t stream, ncclCommProperties_t &props,
                        int my_rank, int total_ranks, int local_device, const testKernel &test) {
  // ==========================================================================
  // Create Device Communicator with GIN Support
  // ==========================================================================

  // Create device communicator with GIN support
  ncclDevComm devComm;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  if (props.ginType != NCCL_GIN_TYPE_NONE) {
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;  // Enable full GIN connectivity
    reqs.worldGinBarrierCount = NCCL_DEVICE_CTA_COUNT;  // GIN barriers for network synchronization
    reqs.barrierCount = NCCL_DEVICE_CTA_COUNT;  // Hybrid LSA-GIN barriers
  }
  reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;  // LSA barriers for shared memory synchronization
  reqs.lsaMultimem = props.multimemSupport;
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));

  // Launch barrier kernel
  test.kernel<<<NCCL_DEVICE_CTA_COUNT, NCCL_DEVICE_THREADS_PER_CTA, 0, stream>>>(devComm);
  CUDACHECK(cudaStreamSynchronize(stream));

  // ==========================================================================
  // Cleanup Resources
  // ==========================================================================

  // Device API specific cleanup
  NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
}

int main(int argc, char* argv[]) {
  int my_rank, total_ranks, local_device;
  ncclComm_t comm;
  ncclUniqueId nccl_unique_id;

  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &total_ranks));

  if (my_rank == 0) {
    printf("Number of processes: %d\n", total_ranks);
  }
  // Only for printing the output in order
  MPI_Barrier(MPI_COMM_WORLD);
  printf("MPI initialized: rank %d of %d\n", my_rank, total_ranks);

  // Split the communicator based on shared memory (i.e., nodes)
  MPI_Comm node_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, my_rank,
                      MPI_INFO_NULL, &node_comm);

  // Get the rank within the node communicator
  MPI_Comm_rank(node_comm, &local_device);

  // Clean up the node communicator
  MPI_Comm_free(&node_comm);

  // Standard NCCL communicator initialization
  if (my_rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&nccl_unique_id));
  }

  // Distribute unique ID
  MPICHECK(MPI_Bcast(&nccl_unique_id, sizeof(nccl_unique_id), MPI_BYTE, 0, MPI_COMM_WORLD));
  MPI_Barrier(MPI_COMM_WORLD);

  // Set device context for this rank
  CUDACHECK(cudaSetDevice(local_device));

  // Initialize NCCL communicator
  NCCLCHECK(ncclCommInitRank(&comm, total_ranks, nccl_unique_id, my_rank));

  // Create stream for kernel execution
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Check for Device API and GIN support
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCLCHECK(ncclCommQueryProperties(comm, &props));

  if (!props.deviceApiSupport) {
    printf("ERROR: rank %d communicator does not support Device API!\n", my_rank);
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    return 1;
  }

  if (props.ginType != NCCL_GIN_TYPE_NONE) {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    PRINT_RANK0(my_rank, "Starting GIN barrier test");

    barrierWithTimeout(comm, stream, props, my_rank, total_ranks, local_device,
                       testKernel{GinBarrierWithTimeoutKernel, "GIN barrier test"});
  } else {
    PRINT_RANK0(my_rank, "# Skipping GIN barrier test due to lack of GIN world support!");
  }

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  PRINT_RANK0(my_rank, "Starting LSA barrier test");

  barrierWithTimeout(comm, stream, props, my_rank, total_ranks, local_device,
     testKernel{LsaBarrierWithTimeoutKernel, "LSA barrier test"});

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  PRINT_RANK0(my_rank, "Starting LSA barrier unicast test");

  barrierWithTimeout(comm, stream, props, my_rank, total_ranks, local_device,
    testKernel{LsaBarrierWithTimeoutUnicastKernel, "LSA barrier unicast test"});

  if (props.ginType != NCCL_GIN_TYPE_NONE) {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    PRINT_RANK0(my_rank, "Starting Barrier test");

    barrierWithTimeout(comm, stream, props, my_rank, total_ranks, local_device,
                       testKernel{BarrierWithTimeoutKernel, "Barrier test"});

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    PRINT_RANK0(my_rank, "Starting GIN barrier with timeout");

    barrierWithTimeout(comm, stream, props, my_rank, total_ranks, local_device,
                       testKernel{GinBarrierReturnForTimeoutKernel, "GIN barrier with timeout"});
  } else {
    PRINT_RANK0(my_rank, "# Skipping Barrier test due to lack of GIN world support!");
    PRINT_RANK0(my_rank, "# Skipping GIN barrier with timeout test due to lack of GIN world support!");
  }

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  PRINT_RANK0(my_rank, "Starting LSA barrier with timeout");

  barrierWithTimeout(comm, stream, props, my_rank, total_ranks, local_device,
    testKernel{LsaBarrierReturnForTimeoutKernel, "LSA barrier with timeout"});

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  PRINT_RANK0(my_rank, "Starting LSA barrier unicast with timeout");

  barrierWithTimeout(comm, stream, props, my_rank, total_ranks, local_device,
    testKernel{LsaBarrierReturnForTimeoutUnicastKernel, "LSA barrier unicast timeout"});

  if (props.ginType != NCCL_GIN_TYPE_NONE) {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    PRINT_RANK0(my_rank, "Starting Barrier with timeout");

    barrierWithTimeout(comm, stream, props, my_rank, total_ranks, local_device,
                       testKernel{BarrierReturnForTimeoutKernel, "Barrier with timeout"});
  } else {
    PRINT_RANK0(my_rank, "# Skipping Barrier with timeout test due to lack of GIN world support!");
  }

  // Standard NCCL cleanup
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  CUDACHECK(cudaStreamDestroy(stream));

  MPICHECK(MPI_Finalize());
  return 0;
}
