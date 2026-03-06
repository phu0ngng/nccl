/*
 * Unit test for ncclSignal() and ncclWaitSignal() APIs
 *
 * This test implements a barrier using ncclSignal/ncclWaitSignal
 * to verify the standalone signal API works correctly.
 *
 * Pattern: Each rank signals all other ranks and waits for signals from all other ranks.
 *
 * Usage:
 *   mpirun -np <nranks> ./signal_barrier_rma
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIGNAL PATTERN (4 Ranks Example)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Each rank signals ALL other ranks, then waits for signals FROM all others │
│                                                                             │
│      Rank 0          Rank 1          Rank 2          Rank 3                 │
│        │               │               │               │                    │
│        │──signal──────▶│               │               │                    │
│        │──signal───────────────────────▶               │                    │
│        │──signal──────────────────────────────────────▶│                    │
│        │               │               │               │                    │
│        │◀──signal──────│               │               │                    │
│        │               │──signal──────▶│               │                    │
│        │               │──signal───────────────────────▶                    │
│        │               │               │               │                    │
│        │◀──signal─────────────────────│               │                     │
│        │               │◀──signal─────│               │                     │
│        │               │               │──signal──────▶│                    │
│        │               │               │               │                    │
│        │◀──signal─────────────────────────────────────│                     │
│        │               │◀──signal────────────────────│                      │
│        │               │               │◀──signal─────│                     │
│        │               │               │               │                    │
│        ▼               ▼               ▼               ▼                    │
│    [WAIT for       [WAIT for       [WAIT for       [WAIT for                │
│     3 signals]      3 signals]      3 signals]      3 signals]              │
│        │               │               │               │                    │
│        ▼               ▼               ▼               ▼                    │
│    ═══════════════════════════════════════════════════════                  │
│                    ALL SYNCHRONIZED (Barrier Complete)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
 *   Timeline:
═════════════════════════════════════════════════════════════════════════════

    Rank 0              Rank 1              Rank 2              ...
       │                   │                   │
       │ Print header      │                   │
       │ message           │                   │
       ▼                   ▼                   ▼
  ─────┴───────────────────┴───────────────────┴─────────────────────────────
                    MPI_Barrier #1
       │                   │                   │
       │  Ensures all ranks start the NCCL signal loop TOGETHER
       │  (header is printed before any rank starts testing)
       │                   │                   │
       ▼                   ▼                   ▼
  ┌────────────────────────────────────────────────────────────┐
  │              NCCL Signal Barrier Loop (5 iterations)       │
  │                                                            │
  │    signal_barrier() × 5                                    │
  └────────────────────────────────────────────────────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
  ─────┴───────────────────┴───────────────────┴─────────────────────────────
                    MPI_Barrier #2
       │                   │                   │
       │  Ensures ALL ranks finished before collecting results
       │  (no rank does MPI_Allreduce before others are done)
       │                   │                   │
       ▼                   ▼                   ▼
       │ MPI_Allreduce()   │                   │
       │ (collect success) │                   │
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include "nccl.h"
#include "mpi.h"
#include "cuda_runtime.h"

// Warning threshold for coefficient of variation
#define CV_WARN_THRESHOLD       0.20      // Warn if CV > 20%

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if (e != MPI_SUCCESS) {                           \
    printf("MPI error %s:%d '%d'\n",                \
        __FILE__, __LINE__, e);                     \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if (e != cudaSuccess) {                           \
    printf("CUDA error %s:%d '%s'\n",               \
        __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r != ncclSuccess) {                           \
    printf("NCCL error %s:%d '%s'\n",               \
        __FILE__, __LINE__, ncclGetErrorString(r)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Helper to get hostname hash for local rank calculation
static uint64_t getHostHash(const char* string) {
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

// Signal barrier: each rank signals all other ranks and waits for signals from all.
// Caller is responsible for stream synchronization.
static ncclResult_t signal_barrier(ncclComm_t comm, int myRank, int nRanks,
                                   ncclWaitSignalDesc_t* waitDescs, int ctx,
                                   cudaStream_t stream) {
    NCCLCHECK(ncclGroupStart());
    for (int peer = 0; peer < nRanks; peer++) {
        if (peer != myRank) {
            NCCLCHECK(ncclSignal(peer, 0, ctx, 0, comm, stream));
        }
    }
    NCCLCHECK(ncclGroupEnd());

    NCCLCHECK(ncclWaitSignal(nRanks - 1, waitDescs, comm, stream));

    return ncclSuccess;
}

// Print timing stats and CV (warns if CV exceeds threshold, but does not fail the test)
static void print_timing_stats(double elapsed_us, int num_iterations, int myRank, int nRanks) {
    // Gather timing stats across ranks
    double max_elapsed_us, min_elapsed_us, sum_elapsed_us, sum_sq_elapsed_us;
    double elapsed_sq = elapsed_us * elapsed_us;
    MPICHECK(MPI_Reduce(&elapsed_us, &max_elapsed_us, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&elapsed_us, &min_elapsed_us, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&elapsed_us, &sum_elapsed_us, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(&elapsed_sq, &sum_sq_elapsed_us, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

    if (myRank == 0) {
        double avg_us = sum_elapsed_us / nRanks;
        double variance = (sum_sq_elapsed_us / nRanks) - (avg_us * avg_us);
        double stddev = (variance > 0) ? sqrt(variance) : 0.0;
        double skew_us = max_elapsed_us - min_elapsed_us;
        double cv = (avg_us > 0) ? (stddev / avg_us) : 0.0;

        printf("Timing (%d iterations):\n", num_iterations);
        printf("  Total:    avg=%.2f us, min=%.2f us, max=%.2f us\n",
               avg_us, min_elapsed_us, max_elapsed_us);
        printf("  Per-iter: %.2f us\n", avg_us / num_iterations);
        printf("  Skew:     %.2f us\n", skew_us);
        printf("  Stddev:   %.2f us\n", stddev);

        if (cv > CV_WARN_THRESHOLD) {
            printf("  CV:       %.2f%% [WARNING: > %.0f%%]\n", cv * 100.0, CV_WARN_THRESHOLD * 100.0);
        } else {
            printf("  CV:       %.2f%%\n", cv * 100.0);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    setlinebuf(stdout);
    int myRank, nRanks, localRank = 0;

    // Initialize MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    if (nRanks < 2) {
        printf("This test requires at least 2 ranks\n");
        MPI_Finalize();
        return 1;
    }

    // Calculate localRank based on hostname
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    gethostname(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < nRanks; p++) {
        if (p == myRank) break;
        if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }

    // Initialize NCCL
    ncclUniqueId id;
    ncclComm_t comm;
    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Set device and create stream
    CUDACHECK(cudaSetDevice(localRank));
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // Configure NCCL with RMA support (required for signal/wait operations)
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.numRmaCtx = 1;
    config.blocking = 1;
    NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, myRank, &config));

    int ctx = 0;  // RMA context
    int num_iterations = 5;
    int success = 1;

    // Pre-allocate wait descriptors once (nRanks - 1 peers)
    ncclWaitSignalDesc_t* waitDescs = (ncclWaitSignalDesc_t*)malloc(sizeof(ncclWaitSignalDesc_t) * (nRanks - 1));
    if (waitDescs == NULL) {
        printf("[Rank %d] Failed to allocate wait descriptors\n", myRank);
        MPI_Finalize();
        return 1;
    }
    int descIdx = 0;
    for (int peer = 0; peer < nRanks; peer++) {
        if (peer != myRank) {
            waitDescs[descIdx].opCnt = 1;
            waitDescs[descIdx].peer = peer;
            waitDescs[descIdx].sigIdx = 0;
            waitDescs[descIdx].ctx = ctx;
            descIdx++;
        }
    }

    if (myRank == 0) {
        printf("=== ncclSignal/ncclWaitSignal Unit Test ===\n");
        printf("Ranks: %d, Iterations: %d\n\n", nRanks, num_iterations);
    }

    // Warm up
    ncclResult_t res = signal_barrier(comm, myRank, nRanks, waitDescs, ctx, stream);
    CUDACHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start_ev, stop_ev;
    CUDACHECK(cudaEventCreate(&start_ev));
    CUDACHECK(cudaEventCreate(&stop_ev));

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    CUDACHECK(cudaEventRecord(start_ev, stream));

    for (int i = 0; i < num_iterations; i++) {
        ncclResult_t res = signal_barrier(comm, myRank, nRanks, waitDescs, ctx, stream);
        if (res != ncclSuccess) {
            printf("[Rank %d] Iteration %d failed: %s\n", myRank, i, ncclGetErrorString(res));
            success = 0;
            break;
        }
    }

    CUDACHECK(cudaEventRecord(stop_ev, stream));
    CUDACHECK(cudaEventSynchronize(stop_ev));

    float elapsed_ms;
    CUDACHECK(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev));
    double elapsed_us = (double)elapsed_ms * 1000.0;

    // Print timing stats (includes CV with warning if high)
    print_timing_stats(elapsed_us, num_iterations, myRank, nRanks);

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Gather results
    int global_success = 0;
    MPICHECK(MPI_Allreduce(&success, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

    if (myRank == 0) {
        printf("Result: %s\n", global_success ? "PASS" : "FAIL");
    }

    // Cleanup
    free(waitDescs);
    CUDACHECK(cudaEventDestroy(start_ev));
    CUDACHECK(cudaEventDestroy(stop_ev));
    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));

    printf("[Rank %d] %s\n", myRank, global_success ? "Success" : "FAILED");

    MPICHECK(MPI_Finalize());

    return global_success ? 0 : 1;
}

