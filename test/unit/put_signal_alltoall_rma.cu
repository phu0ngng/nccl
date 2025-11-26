#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include "nccl.h"
#include "comm.h"
#include "mpi.h"
#include "cuda_runtime.h"

#ifdef MPICHECK
#undef MPICHECK
#endif

#ifdef CUDACHECK
#undef CUDACHECK
#endif

#ifdef NCCLCHECK
#undef NCCLCHECK
#endif

#include <getopt.h>

// CLI argument structure
typedef struct {
    int verify;          // -v flag
    int warmup_iters;    // -w flag
    int normal_iters;    // -i flag
    size_t begin_size;   // -b flag
    size_t end_size;     // -e flag
    int debug;           // -d flag
} cli_args_t;

// Function to parse CLI arguments
static void parse_cli_args(int argc, char* argv[], cli_args_t* args) {
    // Default values
    args->verify = 0;
    args->warmup_iters = 50;
    args->normal_iters = 500;
    args->begin_size = sizeof(int);
    args->end_size = 4 * 1024 * 1024;  // 4MB
    args->debug = 0;  // Default to no debug

    int opt;
    while ((opt = getopt(argc, argv, "vdw:i:b:e:")) != -1) {
        switch (opt) {
            case 'v':
                args->verify = 1;
                break;
            case 'd':
                args->debug = 1;
                break;
            case 'w':
                args->warmup_iters = atoi(optarg);
                break;
            case 'i':
                args->normal_iters = atoi(optarg);
                break;
            case 'b':
                args->begin_size = atoi(optarg);
                break;
            case 'e':
                args->end_size = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s [-v] [-d] [-w warmup_iters] [-i normal_iters] [-b begin_size] [-e end_size]\n", argv[0]);
                fprintf(stderr, "  -v: enable verification\n");
                fprintf(stderr, "  -d: enable debug output\n");
                exit(EXIT_FAILURE);
        }
    }
}

// Error checking macros
#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Helper functions
static uint64_t getHostHash(const char* string) {
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  memset(hostname, 0, maxlen);
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen && hostname[i] != '\0'; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

#define CUDACHECK_DEBUG(cmd, rank) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("[%d] Failed: Cuda error %s:%d '%s'\n",       \
        rank, __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Global debug flag (set from command line)
static int DEBUG = 0;

// Test-specific initialization values
#define SEND_VALUE_BASE 0x100
#define RECV_VALUE_BASE 0x200

// Verification helper function
static int verify_alltoall(void *recvbuff, int nelems_per_rank, int nRanks, int myRank,
                           size_t size, const char* mode_name) {
    int total_nelems = nelems_per_rank * nRanks;
    int* h_recvbuff = (int*)malloc(total_nelems * sizeof(int));
    CUDACHECK(cudaMemcpy(h_recvbuff, recvbuff, total_nelems * sizeof(int), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaDeviceSynchronize());

    if (DEBUG) printf("[Rank %d] Verifying data received from all %d ranks\n", myRank, nRanks);

    int verification_result = 1;
    for (int peer = 0; peer < nRanks; peer++) {
        int expected = SEND_VALUE_BASE + peer;
        int offset = peer * nelems_per_rank;

        for (int i = 0; i < nelems_per_rank; i++) {
            if (h_recvbuff[offset + i] != expected) {
                // Print detailed error only on first failure
                if (verification_result) {
                    verification_result = 0;
                    printf("\n=== VERIFICATION FAILED: SIZE %zu BYTES (%s) ===\n", size, mode_name);
                    printf("[Rank %d] Mismatch from rank %d at index %d (global offset %d): expected 0x%X, got 0x%X\n",
                           myRank, peer, i, offset + i, expected, h_recvbuff[offset + i]);
                    printf("[Rank %d] Data from rank %d: first=0x%X, last=0x%X (expected 0x%X)\n",
                           myRank, peer, h_recvbuff[offset], h_recvbuff[offset + nelems_per_rank - 1], expected);
                    printf("=====================================\n");
                    goto done;  // Exit early on first failure
                }
            }
        }
    }

done:
    free(h_recvbuff);
    return verification_result;
}

// Buffer reset helper function
static void reset_buffers(void *sendbuff, void *recvbuff, int nelems_per_rank, int nRanks, int myRank) {
    int total_nelems = nelems_per_rank * nRanks;
    int* h_sendbuff_reset = (int*)malloc(nelems_per_rank * sizeof(int));
    int* h_recvbuff_reset = (int*)malloc(total_nelems * sizeof(int));

    for (int i = 0; i < nelems_per_rank; i++) {
        h_sendbuff_reset[i] = SEND_VALUE_BASE + myRank;
    }
    for (int i = 0; i < total_nelems; i++) {
        h_recvbuff_reset[i] = RECV_VALUE_BASE + myRank;
    }

    CUDACHECK(cudaMemcpy(sendbuff, h_sendbuff_reset, nelems_per_rank * sizeof(int), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(recvbuff, h_recvbuff_reset, total_nelems * sizeof(int), cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    free(h_sendbuff_reset);
    free(h_recvbuff_reset);
}

// Host-side all-to-all function
static ncclResult_t host_alltoall(
    ncclComm_t comm, int ctx,
    void *sendbuff, void *recvbuff, ncclWindow_t recvWindow,
    int nelems_per_rank, int iter, ncclSignalMode_t signal_type, cudaStream_t stream) {

    int nRanks = comm->nRanks;
    int myRank = comm->rank;

    const char* signal_name = "NCCL_SIGNAL";
    if (DEBUG) printf("[Rank %d/%d] Starting host alltoall with %d iterations, nelems_per_rank=%d, signal_type=%s\n",
                      myRank, nRanks, iter, nelems_per_rank, signal_name);

    for (int i = 1; i <= iter; i++) {
        if (DEBUG) printf("[Rank %d] Starting iteration %d\n", myRank, i);

        // Each rank sends to all ranks (including itself)
        for (int peer = 0; peer < nRanks; peer++) {
            // Calculate offset in remote recvbuff where this rank's data should go (in bytes)
            size_t remote_offset = myRank * nelems_per_rank * sizeof(int);

            if (DEBUG) printf("[Rank %d] Iteration %d: Sending data with signal to rank %d at offset %zu\n",
                            myRank, i, peer, remote_offset);

            NCCLCHECK(ncclPutSignal(sendbuff, nelems_per_rank, ncclInt, peer,
                            recvWindow, remote_offset, signal_type, ctx, comm, stream));
        }

        if (DEBUG) printf("[Rank %d] Iteration %d: Waiting for signals from all ranks\n", myRank, i);

        // Wait for signals from all ranks (including itself)
        int peer_list[nRanks];
        int nsignals_list[nRanks];
        for (int peer = 0; peer < nRanks; peer++) {
            peer_list[peer] = peer;
            // Each put generates 1 signal
            nsignals_list[peer] = 1;
        }

        NCCLCHECK(ncclWaitSignal(nRanks, peer_list, nsignals_list, signal_type, ctx, comm, stream));

        if (DEBUG) printf("[Rank %d] Iteration %d: Received signals from all %d ranks\n", myRank, i, nRanks);

        // Synchronize stream after each iteration
        CUDACHECK(cudaStreamSynchronize(stream));
    }

    if (DEBUG) printf("[Rank %d] Completed all %d iterations\n", myRank, iter);
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

    // Set global debug flag from command line
    DEBUG = args.debug;

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
    // For all-to-all, recvbuff needs to hold data from all ranks
    void *sendbuff, *recvbuff;
    NCCLCHECK(ncclMemAlloc((void**)&sendbuff, args.end_size));
    NCCLCHECK(ncclMemAlloc((void**)&recvbuff, args.end_size * nRanks));

    // Register both send and receive buffers as symmetric windows for RMA operations
    ncclWindow_t sendWindow, recvWindow;
    NCCLCHECK(ncclCommWindowRegister(comm, sendbuff, args.end_size, &sendWindow, NCCL_WIN_COLL_SYMMETRIC));
    NCCLCHECK(ncclCommWindowRegister(comm, recvbuff, args.end_size * nRanks, &recvWindow, NCCL_WIN_COLL_SYMMETRIC));

    // Ensure all ranks have completed window registration before proceeding
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (DEBUG) printf("[Rank %d] Registered windows and allocated %zu bytes\n", myRank, args.end_size);

    // Timing setup
    float milliseconds;
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    // Use NCCL_SIGNAL mode
    ncclSignalMode_t signal_mode = NCCL_SIGNAL;
    const char* signal_name = "NCCL_SIGNAL";

    if (myRank == 0) {
        printf("Note: This test measures all-to-all latency with %d ranks\n", nRanks);
        printf("Each rank sends to all %d ranks (including itself) and receives from all %d ranks\n", nRanks, nRanks);
        printf("Signal mode: %s\n", signal_name);
        if (args.verify) {
            printf("Data verification enabled\n");
        }
        printf("Warmup iterations: %d\n", args.warmup_iters);
        printf("Normal iterations: %d\n", args.normal_iters);
        printf("Message size range: %zu to %zu bytes per rank\n", args.begin_size, args.end_size);
    }

    // Print header once before running tests
    if (myRank == 0) {
        printf("size(B)     latency (us)    status\n");
    }

    // Track verification status
    int all_tests_passed = 1;

    // Run tests for different message sizes
    for (size_t size = args.begin_size; size <= args.end_size; size *= 2) {

        int nelems_per_rank = size / sizeof(int);
        int total_nelems = nelems_per_rank * nRanks;
        if (DEBUG) printf("[Rank %d] Testing size %zu bytes per rank (%d elements per rank, %d total elements)\n",
                         myRank, size, nelems_per_rank, total_nelems);

        // Initialize buffers
        if (args.verify && nelems_per_rank > 0) {
            reset_buffers(sendbuff, recvbuff, nelems_per_rank, nRanks, myRank);
        }

        if (DEBUG && myRank == 0) printf("Testing %s\n", signal_name);

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Warmup phase
        if (args.warmup_iters > 0) {
            if (DEBUG) printf("[Rank %d] Running warmup\n", myRank);
            NCCLCHECK(host_alltoall(comm, ctx, sendbuff, recvbuff, recvWindow, nelems_per_rank,
                                   args.warmup_iters, signal_mode, stream));
            CUDACHECK(cudaStreamSynchronize(stream));
            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        }

        // Measurement phase
        if (DEBUG) printf("[Rank %d] Running measurement\n", myRank);
        CUDACHECK(cudaEventRecord(start, stream));
        NCCLCHECK(host_alltoall(comm, ctx, sendbuff, recvbuff, recvWindow, nelems_per_rank,
                               args.normal_iters, signal_mode, stream));
        CUDACHECK(cudaEventRecord(stop, stream));
        CUDACHECK_DEBUG(cudaStreamSynchronize(stream), myRank);
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Calculate latency
        CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        double latency = (milliseconds * 1000) / args.normal_iters;  // Convert to microseconds

        // Verify results
        int verification_result = 1;
        if (args.verify && nelems_per_rank > 0) {
            verification_result = verify_alltoall(recvbuff, nelems_per_rank, nRanks, myRank, size, signal_name);
            if (!verification_result) {
                all_tests_passed = 0;
            }
        }

        // Print results
        if (myRank == 0) {
            printf("%-12zu %-18.2f %-10s\n", size, latency,
                   (args.verify && nelems_per_rank > 0) ? (verification_result ? "PASS" : "FAIL") : "N/A");
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
