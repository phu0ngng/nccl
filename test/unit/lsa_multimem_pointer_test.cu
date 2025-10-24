/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// LSA Multimem Pointer Test
// Tests ncclGetLsaMultimemPointerHost and ncclGetPeerPointerHost functions
// with full multimem support across multiple ranks using MPI
//
// REQUIREMENTS:
// - MUST be run with MPI (mpirun -np <num_ranks>)
// - Requires at least 2 ranks (LSA multimem needs 2+ ranks for proper testing)
// - Requires exactly 1 GPU per rank (num_ranks <= num_gpus_available)
// - REQUIRES LSA multimem support (multicast capability on the system)
// - Each rank will use GPU ID = local_rank
//
// USAGE:
//   mpirun -np 2 ./lsa_multimem_pointer_test
//   (Requires at least 2 GPUs available and multicast support)
//
// NOTE: For basic LSA testing without multicast, use lsa_pointer_test.cu instead

#include <nccl.h>
#include <nccl_device.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define NCCL_DEVICE_CTA_COUNT 1
#define NCCL_DEVICE_THREADS_PER_CTA 32

#define MPI_TRY(call)                        \
  do {                                       \
    int status = call;                       \
    if (MPI_SUCCESS != status) {             \
      fprintf(stderr,"MPI call='%s' failed. Error %d\n", #call, status); \
      exit(EXIT_FAILURE);                    \
    }                                        \
  } while (0)

#define CUDA_TRY(call)                        \
  do {                                        \
    cudaError_t const status = (call);        \
    if (cudaSuccess != status) {              \
      fprintf(stderr,"CUDA call='%s' failed. Error %s (%d)\n", #call, cudaGetErrorString(status), status); \
      exit(EXIT_FAILURE);                     \
    }                                         \
  } while (0)

#define NCCL_TRY(call)                                                                  \
  do {                                                                                  \
    ncclResult_t const status = (call);                                                 \
    if (ncclSuccess != status) {                                                        \
      fprintf(stderr,"NCCL call='%s' failed. Reason:%s\n", #call, ncclGetErrorString(status)); \
      exit(EXIT_FAILURE);                                                               \
    }                                                                                   \
  } while (0)

// Device kernel to get device pointers
__global__ void getDevicePointersKernel(ncclWindow_t sendwin, size_t sendoffset,
                                       struct ncclDevComm devComm,
                                       void* d_device_ptr) {
    ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm),
                                             devComm.lsaBarrier, blockIdx.x };
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

    // Get pointers to local send and receive buffers for this rank
    float* sendPtr = (float*)ncclGetMultimemPointer(sendwin, sendoffset, devComm.lsaMultimem);
    if(threadIdx.x == 0 and blockIdx.x == 0) {
        *(void**)d_device_ptr = sendPtr;
    }

    bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

// Test function for LSA multimem pointer functionality
int testLsaMultimemPointers(int my_rank, int total_ranks, int local_device) {
    printf("Rank %d: Starting LSA multimem pointer test\n", my_rank);

    ncclComm_t comm;
    ncclUniqueId nccl_unique_id;

    // Standard NCCL communicator initialization
    if (my_rank == 0) {
        NCCL_TRY(ncclGetUniqueId(&nccl_unique_id));
    }

    // Distribute unique ID using MPI
    MPI_TRY(MPI_Bcast(&nccl_unique_id, sizeof(nccl_unique_id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Set device context for this rank
    CUDA_TRY(cudaSetDevice(local_device));

    // Initialize NCCL communicator
    NCCL_TRY(ncclCommInitRank(&comm, total_ranks, nccl_unique_id, my_rank));

    // Allocate memory for testing
    size_t size_bytes = 1024; // Small test size

    void* d_sendbuff=nullptr;
    ncclWindow_t send_win = nullptr;
    // Device API requires ncclMemAlloc for symmetric memory allocation
    NCCL_TRY(ncclMemAlloc(&d_sendbuff, size_bytes));
    // Register window for LSA access
    NCCL_TRY(ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC));

    // Variables to store pointers for verification
    void* host_ptr = nullptr;
    void* d_device_ptr = nullptr;
    CUDA_TRY(cudaMalloc(&d_device_ptr, sizeof(void*)));

    // Create stream for kernel execution
    cudaStream_t stream;
    CUDA_TRY(cudaStreamCreate(&stream));

    // Create device communicator with team requirements to get multimem handle
    ncclDevComm devComm;
    ncclDevCommRequirements reqs;
    ncclTeamRequirements teamReqs;
    ncclMultimemHandle multimemHandle;

    memset(&reqs, 0, sizeof(reqs));
    memset(&teamReqs, 0, sizeof(teamReqs));
    memset(&multimemHandle, 0, sizeof(multimemHandle));

    reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;
    reqs.lsaMultimem = true;

    // Set up team requirements to get multimem handle
    teamReqs.team = ncclTeamLsa(comm);
    teamReqs.multimem = true;
    teamReqs.outMultimemHandle = &multimemHandle;
    reqs.teamRequirementsList = &teamReqs;

    // Try to create device communicator with multimem support
    ncclResult_t result = ncclDevCommCreate(comm, &reqs, &devComm);
    if (result != ncclSuccess) {
        if (my_rank == 0) {
            fprintf(stderr, "ERROR: Failed to create device communicator with LSA multimem support\n");
            fprintf(stderr, "This test requires LSA multimem functionality which is not available on this system\n");
            fprintf(stderr, "Error: %s\n", ncclGetErrorString(result));
            fprintf(stderr, "Please run on a system that supports LSA multimem (requires multicast support)\n");
        }
        return 1; // Test failure
    }

    // Test 1: Original ncclGetLsaMultimemDevicePointer function
    NCCL_TRY(ncclGetLsaMultimemDevicePointer(send_win, 0, &host_ptr));

    getDevicePointersKernel<<<NCCL_DEVICE_CTA_COUNT, NCCL_DEVICE_THREADS_PER_CTA, 0, stream>>>(
        (ncclWindow_t)send_win, 0, devComm, d_device_ptr);

    // Wait for kernel completion
    CUDA_TRY(cudaStreamSynchronize(stream));

    void* device_ptr;
    CUDA_TRY(cudaMemcpy(&device_ptr, d_device_ptr, sizeof(void*), cudaMemcpyDeviceToHost));

    if (device_ptr == host_ptr) {
      printf("\n✓ SUCCESS: host and device multimem pointer match for rank %d! %p %p\n", my_rank, device_ptr, host_ptr);
    } else {
        printf("\n✗ FAILURE:  host and device multimem pointer doesn't match for rank %d! %p %p\n", my_rank, device_ptr, host_ptr);
        return 1; // Test failure
    }

    // Test 2: New ncclGetMultimemDevicePointer function
    void* handle_ptr = nullptr;
    printf("Testing ncclGetMultimemDevicePointer with multimem handle...\n");

    // Test the new function with the multimem handle we obtained
    NCCL_TRY(ncclGetMultimemDevicePointer(send_win, 0, multimemHandle, &handle_ptr));

    // Compare the results from both functions
    if (handle_ptr == host_ptr) {
        printf("✓ SUCCESS: ncclGetMultimemDevicePointer matches original function for rank %d! %p %p\n", my_rank, handle_ptr, host_ptr);
    } else {
        printf("✗ FAILURE: ncclGetMultimemDevicePointer doesn't match original function for rank %d! %p %p\n", my_rank, handle_ptr, host_ptr);
        return 1; // Test failure
    }

    // Test 3: Error handling - test with invalid multimem handle
    void* invalid_ptr = nullptr;
    ncclMultimemHandle invalidHandle;
    memset(&invalidHandle, 0, sizeof(invalidHandle)); // mcBasePtr will be nullptr

    ncclResult_t invalidResult = ncclGetMultimemDevicePointer(send_win, 0, invalidHandle, &invalid_ptr);
    if (invalidResult == ncclInvalidArgument) {
        printf("✓ SUCCESS: ncclGetMultimemDevicePointer correctly rejects invalid multimem handle for rank %d\n", my_rank);
    } else {
        printf("✗ FAILURE: ncclGetMultimemDevicePointer should reject invalid multimem handle for rank %d\n", my_rank);
        return 1; // Test failure
    }

    printf("===============================================\n\n");

    // Cleanup resources
    CUDA_TRY(cudaFree(d_device_ptr));

    // Device API specific cleanup
    NCCL_TRY(ncclDevCommDestroy(comm, &devComm));
    NCCL_TRY(ncclCommWindowDeregister(comm, send_win));
    NCCL_TRY(ncclMemFree(d_sendbuff));
    CUDA_TRY(cudaStreamDestroy(stream));

    // Standard NCCL cleanup
    NCCL_TRY(ncclCommFinalize(comm));
    NCCL_TRY(ncclCommDestroy(comm));

    printf("Rank %d: LSA multimem pointer test completed successfully\n", my_rank);
    return 0; // Test success
}

int main(int argc, char** argv) {
    // Make sure every line is flushed so that we see the progress of the test
    setlinebuf(stdout);

    // HARD REQUIREMENT: This test MUST use MPI
    MPI_TRY(MPI_Init(&argc, &argv));

    // Determine COMM_WORLD rank and size
    int comm_rank = 0;
    int comm_size = 0;
    MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));
    MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));

    // Determine number of ranks per node
    int local_rank = 0, local_size = 0;
    MPI_Comm lcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &lcomm);
    MPI_Comm_rank(lcomm, &local_rank);
    MPI_Comm_size(lcomm, &local_size);
    MPI_Comm_free(&lcomm);

    // HARD REQUIREMENT: One GPU per rank
    int phys_num_gpus = 0;
    CUDA_TRY(cudaGetDeviceCount(&phys_num_gpus));

    // Check if we have at least 2 ranks (LSA multimem requires 2+ ranks for proper testing)
    if (comm_size < 2) {
        if (comm_rank == 0) {
            fprintf(stderr, "ERROR: This test requires at least 2 ranks for LSA multimem functionality\n");
            fprintf(stderr, "Usage: mpirun -np <num_ranks> ./lsa_multimem_pointer_test\n");
            fprintf(stderr, "Where <num_ranks> >= 2 and <= number of available GPUs\n");
        }
        MPI_TRY(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    // Check if we have enough GPUs for all ranks
    if (phys_num_gpus < local_size) {
        if (comm_rank == 0) {
            fprintf(stderr, "ERROR: This test requires at least %d GPUs (one per rank), but only %d GPUs available\n",
                    comm_size, phys_num_gpus);
            fprintf(stderr, "Usage: mpirun -np <num_ranks> ./lsa_multimem_pointer_test\n");
            fprintf(stderr, "Where <num_ranks> <= number of available GPUs\n");
        }
        MPI_TRY(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    // Each rank uses exactly one GPU: local_rank * 1 (since we enforce 1 GPU per rank)
    int local_device = local_rank;

    if (comm_rank == 0) {
        printf("Starting LSA multimem pointer test on %d ranks (nodes %d local %d) with 1 GPU per rank\n",
               comm_size, comm_size/local_size, local_size);
        printf("Total GPUs available: %d, GPUs required: %d\n", phys_num_gpus, comm_size);
        printf("Note: This test requires LSA multimem support (multicast capability)\n");
    }

    // Run the test
    int test_result = testLsaMultimemPointers(comm_rank, comm_size, local_device);

    // Collect results from all ranks
    int global_test_result = 0;
    MPI_TRY(MPI_Allreduce(&test_result, &global_test_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));

    if (comm_rank == 0) {
        if (global_test_result == 0) {
            printf("\n✓ ALL TESTS PASSED: LSA multimem pointer test completed successfully across all ranks\n");
        } else {
            printf("\n✗ SOME TESTS FAILED: LSA multimem pointer test failed on one or more ranks\n");
        }
    }

    MPI_TRY(MPI_Finalize());
    return global_test_result;
}
