/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// LSA Pointer Test
// Tests both LSA and Peer pointer functionality without requiring multimem/multicast support
// across multiple ranks using MPI. Tests both host-side and device-side pointer functions.
//
// REQUIREMENTS:
// - MUST be run with MPI (mpirun -np <num_ranks>)
// - Requires at least 2 ranks (LSA needs 2+ ranks for proper testing)
// - Requires exactly 1 GPU per rank (num_ranks <= num_gpus_available)
// - Each rank will use GPU ID = local_rank
//
// TESTS:
// 1. ncclGetLsaPointerHost - Host-side LSA pointer access
// 2. ncclGetPeerPointerHost - Host-side Peer pointer access
// 3. ncclGetLsaPointer - Device-side LSA pointer access
// 4. ncclGetPeerPointer - Device-side Peer pointer access
// 5. Cross-validation between host and device pointers for both LSA and Peer functions
//
// USAGE:
//   mpirun -np 2 ./lsa_pointer_test
//   (Requires at least 2 GPUs available)

#include <nccl.h>
#include <nccl_device.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define NCCL_DEVICE_CTA_COUNT 16
#define NCCL_DEVICE_THREADS_PER_CTA 512

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

// Device kernel to test both LSA and Peer pointer functionality
__global__ void testLsaKernel(ncclWindow_t sendwin, size_t sendoffset,
                              struct ncclDevComm devComm,
                              void** d_device_lsa_ptrs, void** d_device_peer_ptrs) {
    // LSA barriers enable coordination between GPU threads across different ranks
    ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamLsa(devComm),
                                             devComm.lsaBarrier, blockIdx.x };
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

    const int nRanks = devComm.nRanks;

    // Get pointer to local send buffer for this rank
    if(threadIdx.x == 0 and blockIdx.x == 0) {
        // Get device pointers for all ranks using both LSA and Peer functions
        for (int peer = 0; peer < nRanks; peer++) {
            // Test ncclGetLsaPointer (takes LSA rank directly)
            float* lsaPtr = (float*)ncclGetLsaPointer(sendwin, sendoffset, peer);
            d_device_lsa_ptrs[peer] = lsaPtr;

            // Test ncclGetPeerPointer (takes world rank, converts internally)
            float* peerPtr = (float*)ncclGetPeerPointer(sendwin, sendoffset, peer);
            d_device_peer_ptrs[peer] = peerPtr;
        }
    }

    bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

// Test function for basic LSA functionality
int testLsaPointers(int my_rank, int total_ranks, int local_device) {
    printf("Rank %d: Starting LSA pointer test\n", my_rank);

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

    void* d_sendbuff;
    ncclWindow_t send_win;
    // Device API requires ncclMemAlloc for symmetric memory allocation
    NCCL_TRY(ncclMemAlloc(&d_sendbuff, size_bytes));

    // Register window for LSA access
    NCCL_TRY(ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC));

    // Arrays to store all pointers for this rank
    void** host_lsa_ptrs = (void**)malloc(total_ranks * sizeof(void*));
    void** host_peer_ptrs = (void**)malloc(total_ranks * sizeof(void*));
    void** device_lsa_ptrs = (void**)malloc(total_ranks * sizeof(void*));
    void** device_peer_ptrs = (void**)malloc(total_ranks * sizeof(void*));
    void** d_device_lsa_ptrs = nullptr;
    void** d_device_peer_ptrs = nullptr;
    CUDA_TRY(cudaMalloc(&d_device_lsa_ptrs, total_ranks * sizeof(void*)));
    CUDA_TRY(cudaMalloc(&d_device_peer_ptrs, total_ranks * sizeof(void*)));

    // Create stream for kernel execution
    cudaStream_t stream;
    CUDA_TRY(cudaStreamCreate(&stream));

    // Create device communicator (without multimem requirement)
    ncclDevComm devComm;
    ncclDevCommRequirements reqs;
    memset(&reqs, 0, sizeof(reqs));
    reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;
    reqs.lsaMultimem = false; // Basic LSA, no multimem required
    NCCL_TRY(ncclDevCommCreate(comm, &reqs, &devComm));

    // Test 1: Get host LSA pointers for all ranks (using LSA ranks directly)
    printf("\n=== Testing ncclGetLsaPointerHost ===\n");
    for (int lsa_peer = 0; lsa_peer < total_ranks; lsa_peer++) {
        NCCL_TRY(ncclGetLsaDevicePointer(send_win, 0, lsa_peer, &host_lsa_ptrs[lsa_peer]));
        printf("Rank %d: Host LSA peer %d pointer = %p\n", my_rank, lsa_peer, host_lsa_ptrs[lsa_peer]);
    }

    // Test 2: Get host peer pointers for all ranks (using world ranks)
    printf("\n=== Testing ncclGetPeerPointerHost ===\n");
    for (int peer = 0; peer < total_ranks; peer++) {
        NCCL_TRY(ncclGetPeerDevicePointer(send_win, 0, peer, &host_peer_ptrs[peer]));
        printf("Rank %d: Host peer %d pointer = %p\n", my_rank, peer, host_peer_ptrs[peer]);
    }

    // Test 3: Get device pointers using kernel
    printf("\n=== Testing Device ncclGetLsaPointer and ncclGetPeerPointer ===\n");
    testLsaKernel<<<NCCL_DEVICE_CTA_COUNT, NCCL_DEVICE_THREADS_PER_CTA, 0, stream>>>(
        (ncclWindow_t)send_win, 0, devComm, d_device_lsa_ptrs, d_device_peer_ptrs);

    // Wait for kernel completion
    CUDA_TRY(cudaStreamSynchronize(stream));

    // Copy device pointers back to host
    CUDA_TRY(cudaMemcpy(device_lsa_ptrs, d_device_lsa_ptrs, total_ranks * sizeof(void*), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(device_peer_ptrs, d_device_peer_ptrs, total_ranks * sizeof(void*), cudaMemcpyDeviceToHost));

    // Test 4: Print comparison results
    printf("\n=== Host vs Device Pointer Comparison Results for Rank %d ===\n", my_rank);
    printf("Peer | Host LSA Ptr   | Device LSA Ptr  | Host Peer Ptr  | Device Peer Ptr | LSA Match | Peer Match\n");
    printf("-----|----------------|-----------------|----------------|-----------------|-----------|----------\n");

    bool all_lsa_match = true;
    bool all_peer_match = true;
    for (int peer = 0; peer < total_ranks; peer++) {
        bool lsa_match = (host_lsa_ptrs[peer] == device_lsa_ptrs[peer]);
        bool peer_match = (host_peer_ptrs[peer] == device_peer_ptrs[peer]);
        if (!lsa_match) all_lsa_match = false;
        if (!peer_match) all_peer_match = false;
        const char* lsa_status = lsa_match ? "YES" : "NO";
        const char* peer_status = peer_match ? "YES" : "NO";
        printf("%4d | %14p | %15p | %14p | %15p | %9s | %10s\n",
               peer, host_lsa_ptrs[peer], device_lsa_ptrs[peer],
               host_peer_ptrs[peer], device_peer_ptrs[peer],
               lsa_status, peer_status);
    }

    printf("\n=== Test Results Summary ===\n");
    if (all_lsa_match) {
        printf("✓ SUCCESS: All LSA host and device pointers match for rank %d!\n", my_rank);
    } else {
        printf("✗ FAILURE: Some LSA host and device pointers don't match for rank %d!\n", my_rank);
    }

    if (all_peer_match) {
        printf("✓ SUCCESS: All Peer host and device pointers match for rank %d!\n", my_rank);
    } else {
        printf("✗ FAILURE: Some Peer host and device pointers don't match for rank %d!\n", my_rank);
    }

    if (all_lsa_match && all_peer_match) {
        printf("✓ OVERALL SUCCESS: All host vs device pointer tests passed for rank %d!\n", my_rank);
    } else {
        printf("✗ OVERALL FAILURE: Some host vs device pointer tests failed for rank %d!\n", my_rank);
        return 1; // Test failure
    }
    printf("===============================================\n\n");

    // Cleanup resources
    free(host_lsa_ptrs);
    free(host_peer_ptrs);
    free(device_lsa_ptrs);
    free(device_peer_ptrs);
    CUDA_TRY(cudaFree(d_device_lsa_ptrs));
    CUDA_TRY(cudaFree(d_device_peer_ptrs));

    // Device API specific cleanup
    NCCL_TRY(ncclDevCommDestroy(comm, &devComm));
    NCCL_TRY(ncclCommWindowDeregister(comm, send_win));
    NCCL_TRY(ncclMemFree(d_sendbuff));
    CUDA_TRY(cudaStreamDestroy(stream));

    // Standard NCCL cleanup
    NCCL_TRY(ncclCommFinalize(comm));
    NCCL_TRY(ncclCommDestroy(comm));

    printf("Rank %d: LSA pointer test completed successfully\n", my_rank);
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

    // Check if we have at least 2 ranks (LSA requires 2+ ranks for proper testing)
    if (comm_size < 2) {
        if (comm_rank == 0) {
            fprintf(stderr, "ERROR: This test requires at least 2 ranks for LSA functionality\n");
            fprintf(stderr, "Usage: mpirun -np <num_ranks> ./lsa_pointer_test\n");
            fprintf(stderr, "Where <num_ranks> >= 2 and <= number of available GPUs\n");
        }
        MPI_TRY(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    // Check if we have enough GPUs for all ranks
    if (phys_num_gpus < local_size) {
        if (comm_rank == 0) {
            fprintf(stderr, "ERROR: This test requires at least %d GPUs (one per rank), but only %d GPUs available\n",
                    local_size, phys_num_gpus);
            fprintf(stderr, "Usage: mpirun -np <num_ranks> ./lsa_pointer_test\n");
            fprintf(stderr, "Where <num_ranks> <= number of available GPUs\n");
        }
        MPI_TRY(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    // Each rank uses exactly one GPU: local_rank * 1 (since we enforce 1 GPU per rank)
    int local_device = local_rank;

    if (comm_rank == 0) {
        printf("Starting LSA pointer test on %d ranks (nodes %d local %d) with 1 GPU per rank\n",
               comm_size, comm_size/local_size, local_size);
        printf("Total GPUs available: %d, GPUs required: %d\n", phys_num_gpus, comm_size);
        printf("Note: This test uses basic LSA functionality (no multicast required)\n");
    }

    // Run the test
    int test_result = testLsaPointers(comm_rank, comm_size, local_device);

    // Collect results from all ranks
    int global_test_result = 0;
    MPI_TRY(MPI_Allreduce(&test_result, &global_test_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));

    if (comm_rank == 0) {
        if (global_test_result == 0) {
            printf("\n✓ ALL TESTS PASSED: LSA and Peer pointer tests completed successfully across all ranks\n");
            printf("✓ Tested: ncclGetLsaPointerHost vs ncclGetLsaPointer and ncclGetPeerPointerHost vs ncclGetPeerPointer\n");
        } else {
            printf("\n✗ SOME TESTS FAILED: LSA and Peer pointer tests failed on one or more ranks\n");
        }
    }

    MPI_TRY(MPI_Finalize());
    return global_test_result;
}
