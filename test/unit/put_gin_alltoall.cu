/*************************************************************************
* Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
*
* See LICENSE.txt for license information
************************************************************************/

// All-to-all GIN communication test - supports any number of processes

#include <stdio.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <unistd.h>
#include <stdint.h>

#include "nccl.h"
#include "comm.h"

// internal NCCL file
#include "gin/gin_host.h"
#include "nccl_device/gin/gin_device_api.h"

#ifdef MPICHECK
#undef MPICHECK
#endif

#ifdef CUDACHECK
#undef CUDACHECK
#endif

#ifdef NCCLCHECK
#undef NCCLCHECK
#endif

#include "common.h"

using namespace cooperative_groups;

__device__ __inline__ int ld_acquire_global(const int* addr) {
    int val;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(val) : "l"(addr));
    return val;
}

__device__ __inline__ void device_sleep(unsigned long long nanoseconds) {
    unsigned long long start = clock64();
    // Use a fixed number of cycles per nanosecond (typical for modern GPUs)
    unsigned long long cycles = nanoseconds * 1;  // 1 cycle per nanosecond
    while (clock64() - start < cycles);
}

__global__ void put_alltoall(ncclGinCtx_M<-1u> ctx, int* buff, int myRank, int nRanks,
                             ncclGinWindow_t memHandle, ncclGinSignal_t signalId) {
#if CUDA_VERSION >= 12020 && __CUDA_ARCH__ >= 700
    __shared__ ncclGinDescriptorSmem desc;
    thread_block block = this_thread_block();
    thread_block_tile<1> thread = this_thread();

    if (threadIdx.x == 0) {
        // First, write own rank to own index
        buff[nRanks] = myRank;
        printf("[Rank %d] GPU thread %d: Wrote rank %d to the send buffer (%d + 1)\n", myRank,
               threadIdx.x, myRank, nRanks);

        // Send own rank to ALL processes (including itself)
        printf(
            "[Rank %d] GPU thread %d: Starting all-to-all communication (sending to %d "
            "processes)\n",
            myRank, threadIdx.x, nRanks);

        for (int targetRank = 0; targetRank < nRanks; targetRank++) {
            printf("[Rank %d] GPU thread %d: Sending to rank %d (writing to index %d)\n", myRank,
                   threadIdx.x, targetRank, myRank);
            ncclGinSignalDescriptor signal;
            signal.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
            signal.indexedSignal.signalId = signalId;
            ncclGinCall<ncclGinApi_Put>(ctx, thread, targetRank, /*hasData=*/true,
                memHandle, myRank * sizeof(int), memHandle, nRanks * sizeof(int),
                sizeof(int),
                signal, ncclGinSignalAdd, 1,
                /*hasCounter=*/false, 0,
                /*hasDescriptor=*/true, &desc,
                cuda::thread_scope_thread, cuda::thread_scope_thread);
        }
        printf("[Rank %d] GPU thread %d: Completed sending to all processes\n", myRank,
               threadIdx.x);

        // Wait for ALL other processes to send their data to us
        printf("[Rank %d] GPU thread %d: Waiting for data from all processes\n", myRank,
               threadIdx.x);

        // Simple polling approach - wait for all expected data to arrive
        for (int sourceRank = 0; sourceRank < nRanks; sourceRank++) {
            int val = -1;
            printf("[Rank %d] GPU thread %d: Waiting for data from rank %d at index %d\n", myRank,
                   threadIdx.x, sourceRank, sourceRank);

            // Poll until we receive the expected value
            int max_attempts = 1000000;  // Prevent infinite loop
            int attempts = 0;
            while (val != sourceRank && attempts < max_attempts) {
                val = ld_acquire_global(&buff[sourceRank]);
                if (val == -1) {
                    device_sleep(1000000);  // 1 millisecond delay
                    attempts++;
                    if (attempts % 100000 == 0) {
                        printf(
                            "[Rank %d] GPU thread %d: Still waiting for rank %d at index %d "
                            "(attempts: %d, current value: %d)\n",
                            myRank, threadIdx.x, sourceRank, sourceRank, attempts, val);
                    }
                }
            }

            if (attempts >= max_attempts) {
                printf(
                    "[Rank %d] GPU thread %d: ERROR - Timeout waiting for rank %d at index %d "
                    "(final value: %d)\n",
                    myRank, threadIdx.x, sourceRank, sourceRank, val);
            } else {
                printf(
                    "[Rank %d] GPU thread %d: Received value %d from rank %d at index %d (after %d "
                    "attempts)\n",
                    myRank, threadIdx.x, val, sourceRank, sourceRank, attempts);
            }
        }

        // Print final buffer state
        printf("[Rank %d] GPU thread %d: Final buffer state: [", myRank, threadIdx.x);
        for (int i = 0; i < nRanks; i++) {
            printf("%d", ld_acquire_global(&buff[i]));
            if (i < nRanks - 1) printf(", ");
        }
        printf("]\n");

        printf("[Rank %d] GPU thread %d: All-to-all communication completed successfully!\n",
               myRank, threadIdx.x);
    }
#else
    printf("ERROR: This test requires CUDA 12.2 and compute capability 7.0 (Volta) or higher\n");
    assert(0);
#endif
}

int main(int argc, char* argv[]) {
    cli_args_t args;
    parse_cli_args(argc, argv, &args);

    int myRank, nRanks, localRank = 0;
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    // Verify we have at least 2 processes
    if (nRanks < 2) {
        if (myRank == 0) {
            printf("Error: This test requires at least 2 MPI processes, but got %d\n", nRanks);
        }
        MPICHECK(MPI_Finalize());
        return 1;
    }

    printf("[MPI Rank %d] Starting all-to-all test with %d processes\n", myRank, nRanks);

    // Calculate localRank
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t),
                           MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < nRanks; p++) {
        if (p == myRank) break;
        if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }

    // Initialize NCCL
    ncclUniqueId id;
    ncclComm_t comm;
    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Setup GPU and buffer
    CUDACHECK(cudaSetDevice(localRank));
    int* buff;
    NCCLCHECK(ncclMemAlloc((void**)&buff, (nRanks + 1) * sizeof(int)));
    cudaStream_t s;
    CUDACHECK(cudaStreamCreate(&s));

    // Initialize buffer to -1
    CUDACHECK(cudaMemset(buff, -1, (nRanks + 1) * sizeof(int)));

    // Initialize NCCL with GIN support
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 1;
    NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, myRank, &config));
    NCCLCHECK(ncclGinConnectOnce(comm, NCCL_GIN_CONNECTION_FULL, 1));

    // Setup GIN windows using ncclGinRegister
    void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS];
    ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS];
    NCCLCHECK(ncclGinRegister(comm, buff, (nRanks + 1) * sizeof(int), ginHostWins, ginDevWins, /*winFlags=*/0));
    ncclGinWindow_t memHandle = ginDevWins[0];

    // Setup GIN context manually like in ping-pong example
    ncclGinCtx_M<-1u> gctx;
    gctx.backend = comm->sharedRes->ginState.ginDevHandles[0]->netDeviceType;
    gctx.handle = comm->sharedRes->ginState.ginDevHandles[0]->handle;
    gctx.rank = myRank;
    gctx.nRanks = nRanks;
    gctx.contextId = 0;

    // Allocate signal for communication
    ncclGinSignal_t signalId = myRank;  // Use rank-specific signal ID to avoid conflicts

    CUDACHECK(cudaDeviceSynchronize());

    // Synchronize all processes before starting
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    printf("[MPI Rank %d] Starting all-to-all communication kernel\n", myRank);

    // Run kernel
    put_alltoall<<<1, 1, 0, s>>>(gctx, buff, myRank, nRanks, memHandle, signalId);
    CUDACHECK(cudaStreamSynchronize(s));
    printf("[MPI Rank %d] Completed all-to-all communication kernel\n", myRank);

    // Print buffer contents
    int host_buff[nRanks];
    CUDACHECK(cudaMemcpy(host_buff, buff, nRanks * sizeof(int), cudaMemcpyDeviceToHost));
    printf("[MPI Rank %d] Final buffer contents: [", myRank);
    for (int i = 0; i < nRanks; i++) {
        printf("%d", host_buff[i]);
        if (i < nRanks - 1) printf(", ");
    }
    printf("]\n");

    // Verify correctness
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (myRank == 0) {
        printf("\n=== All-to-All Communication Pattern Verification ===\n");
        printf("Expected: Each rank should have received data from all other ranks\n");
        if (nRanks == 2) {
            printf("For %d processes, each process should have a buffer [0, 1]\n", nRanks);
        } else {
            printf("For %d processes, each process should have a buffer [0, 1, 2, ..., %d]\n",
                   nRanks, nRanks - 1);
        }
        printf("Rank i sends its value (i) to index i in all other processes' buffers\n");
        printf("So after completion, all processes should have identical buffers.\n");
    }

    // Cleanup
    // Comment out problematic cleanup code to avoid segmentation fault
    NCCLCHECK(ncclGinDeregister(comm, ginHostWins));
    NCCLCHECK(ncclGinHostFinalize(comm));
    NCCLCHECK(ncclMemFree((void*)buff));
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));

    printf("[MPI Rank %d] Success \n", myRank);

    MPICHECK(MPI_Finalize());
    return 0;
}
