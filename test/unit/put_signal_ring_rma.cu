#include <errno.h>
#include "nccl.h"
#include "comm.h"

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

#define CUDACHECK_DEBUG(cmd, rank) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("[%d] Failed: Cuda error %s:%d '%s'\n",       \
        rank, __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

static const bool DEBUG = false;

// Test-specific initialization values
#define SEND_VALUE_BASE 0x100
#define RECV_VALUE_BASE 0x200

// Initialize buffers for this test
static void initialize_buffers(int* sendbuff, int* recvbuff, int rank, int nelems) {
    for (int i = 0; i < nelems; i++) {
        sendbuff[i] = SEND_VALUE_BASE + rank;
        recvbuff[i] = RECV_VALUE_BASE + rank;
    }
}

// Host-side ring function
static ncclResult_t host_ring(
    ncclComm_t comm, int ctx,
    void *sendbuff, void *recvbuff, ncclWindow_t recvWindow,
    int nelems, int iter, cudaStream_t stream) {

    int nRanks = comm->nRanks;
    int myRank = comm->rank;

    // In a ring: send to downstream (rank + 1), receive from upstream (rank - 1)
    int downstream = (myRank + 1) % nRanks;
    int upstream = (myRank - 1 + nRanks) % nRanks;

    if (DEBUG) printf("[Rank %d/%d] Starting host ring with %d iterations, nelems=%d, upstream=%d, downstream=%d\n",
                      myRank, nRanks, iter, nelems, upstream, downstream);

    for (int i = 1; i <= iter; i++) {
        if (DEBUG) printf("[Rank %d] Starting iteration %d\n", myRank, i);

        // Each rank sends to downstream peer
        if (DEBUG) printf("[Rank %d] Sending data with signal to downstream rank %d\n", myRank, downstream);
        NCCLCHECK(ncclPutSignal(sendbuff, nelems, ncclInt, downstream, recvWindow, 0,
                         0, ctx, 0, comm, stream));

        if (DEBUG) printf("[Rank %d] Waiting for signal from upstream rank %d\n", myRank, upstream);

        // Each rank waits for signal from upstream peer
        ncclWaitSignalDesc_t waitDesc = {.opCnt = 1, .peer = upstream, .sigIdx = 0, .ctx = ctx};
        NCCLCHECK(ncclWaitSignal(1, &waitDesc, comm, stream));

        if (DEBUG) printf("[Rank %d] Received signal from upstream rank %d\n", myRank, upstream);

        // Synchronize stream after each iteration
        CUDACHECK(cudaStreamSynchronize(stream));
    }

    if (DEBUG) printf("[Rank %d] Completed all iterations\n", myRank);
    return ncclSuccess;
}


#ifndef READ_ONCE
#define READ_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

#ifndef WRITE_ONCE
#define WRITE_ONCE(x, v) do { \
    (*(volatile typeof(x) *)&(x)) = (v); \
} while (0)
#endif

int main(int argc, char* argv[]) {
    setlinebuf(stdout);
    int myRank, nRanks, localRank = 0;
    cli_args_t args;
    parse_cli_args(argc, argv, &args);

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
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p=0; p<nRanks; p++) {
        if (p == myRank) break;
        if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }

    if (DEBUG) printf("[Rank %d] Using GPU %d\n", myRank, localRank);

    // Initialize NCCL with GIN support
    ncclUniqueId id;
    ncclComm_t comm;
    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Set device and create stream
    CUDACHECK(cudaSetDevice(localRank));
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // Configure NCCL with RMA support
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.numRmaCtx = 1;  // Enable RMA with 1 context
    config.blocking = 1;
    NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, myRank, &config));

    if (DEBUG) printf("[Rank %d] NCCL communicator initialized successfully\n", myRank);

    // Initialize RMA context for put/signal operations
    int ctx = 0;  // Use context 0 for RMA operations

    // Allocate symmetric memory for RMA operations
    void *sendbuff, *recvbuff;
    NCCLCHECK(ncclMemAlloc((void**)&sendbuff, args.end_size));
    NCCLCHECK(ncclMemAlloc((void**)&recvbuff, args.end_size));

    // Register both send and receive buffers as symmetric windows for RMA operations
    ncclWindow_t sendWindow, recvWindow;
    NCCLCHECK(ncclCommWindowRegister(comm, sendbuff, args.end_size, &sendWindow, NCCL_WIN_COLL_SYMMETRIC));
    NCCLCHECK(ncclCommWindowRegister(comm, recvbuff, args.end_size, &recvWindow, NCCL_WIN_COLL_SYMMETRIC));

    // Ensure all ranks have completed window registration before proceeding
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (DEBUG) printf("[Rank %d] Registered windows and allocated %zu bytes\n", myRank, args.end_size);

    // Timing setup
    float milliseconds;
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    if (myRank == 0) {
        printf("Note: This test measures ring latency with %d ranks\n", nRanks);
        if (args.verify) {
            printf("Data verification enabled\n");
        }
        printf("Warmup iterations: %d\n", args.warmup_iters);
        printf("Normal iterations: %d\n", args.normal_iters);
        printf("Message size range: %zu to %zu bytes\n", args.begin_size, args.end_size);
    }

    // Print header once before running tests
    if (myRank == 0) {
        printf("size(B)     latency (us)    status\n");
    }

    // Track verification status
    int all_tests_passed = 1;

    // Run tests for different message sizes
    for (size_t size = args.begin_size; size <= args.end_size; size *= 2) {

        int nelems = size / sizeof(int);
        if (DEBUG) printf("[Rank %d] Testing size %zu bytes (%d elements, %zu actual bytes)\n", myRank, size, nelems, nelems * sizeof(int));

        if (args.verify && nelems > 0) {
            int* h_sendbuff = (int*)malloc(nelems * sizeof(int));
            int* h_recvbuff = (int*)malloc(nelems * sizeof(int));
            initialize_buffers(h_sendbuff, h_recvbuff, myRank, nelems);
            CUDACHECK(cudaMemcpy(sendbuff, h_sendbuff, nelems * sizeof(int), cudaMemcpyHostToDevice));
            CUDACHECK(cudaMemcpy(recvbuff, h_recvbuff, nelems * sizeof(int), cudaMemcpyHostToDevice));
            free(h_sendbuff);
            free(h_recvbuff);
        }

        // Warmup phase
        if (args.warmup_iters > 0) {
            if (DEBUG) printf("[Rank %d] Running warmup\n", myRank);
            NCCLCHECK(host_ring(comm, ctx, sendbuff, recvbuff, recvWindow, nelems, args.warmup_iters, stream));
            if (DEBUG) printf("[Rank %d] Ran warmup\n", myRank);
            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
            if (DEBUG) printf("[Rank %d] Ran warmup (after MPI_Barrier)\n", myRank);
        }

        // Measurement phase
        if (DEBUG) printf("[Rank %d] Running measurement\n", myRank);
        CUDACHECK(cudaEventRecord(start, stream));
        NCCLCHECK(host_ring(comm, ctx, sendbuff, recvbuff, recvWindow, nelems, args.normal_iters, stream));
        CUDACHECK(cudaEventRecord(stop, stream));
        CUDACHECK_DEBUG(cudaStreamSynchronize(stream), myRank);
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Calculate latency
        CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        double latency = (milliseconds * 1000) / args.normal_iters;  // Convert to microseconds

        int verification_result = 1;
        if (args.verify && nelems > 0) {
            // Calculate upstream peer in ring topology
            int upstream = (myRank - 1 + nRanks) % nRanks;

            // Get current values from GPU
            int* h_recvbuff = (int*)malloc(nelems * sizeof(int));
            CUDACHECK(cudaMemcpy(h_recvbuff, recvbuff, nelems * sizeof(int), cudaMemcpyDeviceToHost));

            // Verify that recvbuff contains data from upstream peer
            int expected = SEND_VALUE_BASE + upstream;
            for (int i = 0; i < nelems; i++) {
                if (h_recvbuff[i] != expected) {
                    if (verification_result) {
                        verification_result = 0;
                        printf("\n=== VERIFICATION FAILED: SIZE %zu BYTES ===\n", size);
                        printf("[Rank %d] Mismatch at index %d: expected 0x%X, got 0x%X\n",
                               myRank, i, expected, h_recvbuff[i]);
                        printf("[Rank %d] Expected to receive from upstream rank %d\n", myRank, upstream);
                        printf("=====================================\n");
                        break;
                    }
                }
            }

            // Update overall test status
            if (!verification_result) {
                all_tests_passed = 0;
            }

            free(h_recvbuff);

            // Reset GPU buffers for next message size iteration
            int* h_sendbuff_reset = (int*)malloc(nelems * sizeof(int));
            int* h_recvbuff_reset = (int*)malloc(nelems * sizeof(int));
            initialize_buffers(h_sendbuff_reset, h_recvbuff_reset, myRank, nelems);
            CUDACHECK(cudaMemcpy(sendbuff, h_sendbuff_reset, nelems * sizeof(int), cudaMemcpyHostToDevice));
            CUDACHECK(cudaMemcpy(recvbuff, h_recvbuff_reset, nelems * sizeof(int), cudaMemcpyHostToDevice));
            free(h_sendbuff_reset);
            free(h_recvbuff_reset);
        }

        if (myRank == 0) {
            printf("%-12zu %-18.2f %-10s\n", size, latency,
                   (args.verify && nelems > 0) ? (verification_result ? "PASS" : "FAIL") : "N/A");
        }
    }

    // Cleanup
    if (DEBUG) printf("[Rank %d] Cleaning up\n", myRank);

    NCCLCHECK(ncclCommWindowDeregister(comm, sendWindow));
    NCCLCHECK(ncclCommWindowDeregister(comm, recvWindow));
    NCCLCHECK(ncclMemFree(sendbuff));
    NCCLCHECK(ncclMemFree(recvbuff));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));

    if (all_tests_passed) {
        printf("[MPI Rank %d] Success \n", myRank);
    } else {
        printf("[MPI Rank %d] FAILED - Verification errors detected\n", myRank);
    }

    MPICHECK(MPI_Finalize());

    return all_tests_passed ? 0 : 1;
}
