/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Dynamic Memory Offload Unit Test
 *
 * Tests NCCL dynamic memory suspend/resume with configurable options:
 * - Symmetric window registration (ncclCommWindowRegister)
 * - User buffer data integrity verification
 * - Multiple release/restore cycles
 * - Symmetric vs non-symmetric allocations
 */

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
#include "alloc.h"
#include "cudawrap.h"

#include <vector>
#include <cstring>
#include <algorithm>

// Debug output control
static bool DEBUG = false;

#define DEBUG_PRINT(...) do { if (DEBUG) printf(__VA_ARGS__); } while(0)

// CLI argument structure
typedef struct {
    int verify;             // -v flag: enable collective result verification
    int repeat_cycles;      // -r flag: number of release/restore cycles
    int use_symmetric_win;  // -w flag: use symmetric window registration
    int verify_integrity;   // -i flag: verify user buffer data integrity
    size_t count;           // -c flag: element count for collectives
    int debug;              // -d flag: enable debug output
} local_cli_args_t;

// Parse CLI arguments
static void parse_local_cli_args(int argc, char* argv[], local_cli_args_t* args) {
    // Default values
    args->verify = 1;
    args->repeat_cycles = 3;
    args->use_symmetric_win = 0;
    args->verify_integrity = 0;
    args->count = 1024000000;
    args->debug = 0;

    int opt;
    while ((opt = getopt(argc, argv, "vr:wic:dh")) != -1) {
        switch (opt) {
            case 'v':
                args->verify = 1;
                break;
            case 'r':
                args->repeat_cycles = atoi(optarg);
                if (args->repeat_cycles < 1) {
                    fprintf(stderr, "Error: repeat_cycles must be >= 1\n");
                    exit(EXIT_FAILURE);
                }
                break;
            case 'w':
                args->use_symmetric_win = 1;
                break;
            case 'i':
                args->verify_integrity = 1;
                break;
            case 'c':
                args->count = (size_t)atoll(optarg);
                break;
            case 'd':
                args->debug = 1;
                break;
            case 'h':
            default:
                fprintf(stderr, "Usage: %s [-v] [-r cycles] [-w] [-i] [-c count] [-d]\n", argv[0]);
                fprintf(stderr, "  -v          Enable collective result verification (default: on)\n");
                fprintf(stderr, "  -r cycles   Number of suspend/resume cycles (default: 1)\n");
                fprintf(stderr, "  -w          Use symmetric window registration (ncclCommWindowRegister)\n");
                fprintf(stderr, "  -i          Verify user buffer data integrity across suspend/resume\n");
                fprintf(stderr, "  -c count    Element count for collectives (default: 1024)\n");
                fprintf(stderr, "  -d          Enable debug output\n");
                exit(opt == 'h' ? 0 : EXIT_FAILURE);
        }
    }
}

// Test-specific buffer values
#define PATTERN_VALUE 0xDEAD

// Device buffer structure
struct DeviceBuffers {
    int* send;
    int* recv;
    cudaStream_t stream;
    size_t allocBytes;
    ncclWindow_t sendWindow;
    ncclWindow_t recvWindow;
    bool windowsRegistered;
};

// Helper: fill buffer with value
static void fillBuffer(int* ptr, size_t count, int value) {
    std::vector<int> host(count, value);
    CUDACHECK(cudaMemcpy(ptr, host.data(), count * sizeof(int), cudaMemcpyHostToDevice));
}

// Helper: zero buffer
static void zeroBuffer(int* ptr, size_t count) {
    CUDACHECK(cudaMemset(ptr, 0, count * sizeof(int)));
}

// Helper: copy to host
static std::vector<int> copyToHost(int* ptr, size_t count) {
    std::vector<int> host(count, 0);
    CUDACHECK(cudaMemcpy(host.data(), ptr, count * sizeof(int), cudaMemcpyDeviceToHost));
    return host;
}

// Helper: verify all elements equal expected value
static bool verifyAllEqual(const std::vector<int>& data, int expected, const char* label, int rank) {
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] != expected) {
            printf("[Rank %d] %s FAILED at index %zu: expected 0x%X, got 0x%X\n",
                   rank, label, i, expected, data[i]);
            return false;
        }
    }
    DEBUG_PRINT("[Rank %d] %s PASSED\n", rank, label);
    return true;
}

// Helper: verify allgather result
static bool verifyAllGather(const std::vector<int>& data, size_t sendcount, int nranks, int rank) {
    for (size_t i = 0; i < data.size(); ++i) {
        int expected = static_cast<int>(i / sendcount) + 1;
        if (data[i] != expected) {
            printf("[Rank %d] allgather FAILED at index %zu: expected %d, got %d\n",
                   rank, i, expected, data[i]);
            return false;
        }
    }
    DEBUG_PRINT("[Rank %d] allgather PASSED\n", rank);
    return true;
}

// Helper: print memory statistics using new ncclCommMemStats API
static void printMemoryStats(ncclComm_t comm) {
    uint64_t total, persist, suspendable, suspended;

    NCCLCHECK(ncclCommMemStats(comm, ncclStatGpuMemTotal, &total));
    NCCLCHECK(ncclCommMemStats(comm, ncclStatGpuMemPersist, &persist));
    NCCLCHECK(ncclCommMemStats(comm, ncclStatGpuMemSuspend, &suspendable));
    NCCLCHECK(ncclCommMemStats(comm, ncclStatGpuMemSuspended, &suspended));
    printf("=== NCCL Memory Statistics ===\n");
    printf("State:           %s\n", suspended ? "SUSPENDED" : "ACTIVE");
    printf("Total tracked:   %llu bytes (%.2f MB)\n",
           (unsigned long long)total, total / (1024.0 * 1024.0));
    printf("Persistent:      %llu bytes (%.2f MB) [cannot be suspended]\n",
           (unsigned long long)persist, persist / (1024.0 * 1024.0));
    printf("Suspendable:     %llu bytes (%.2f MB) [can be suspended]\n",
           (unsigned long long)suspendable, suspendable / (1024.0 * 1024.0));
    printf("==============================\n");
}

// Helper: sync stream
static void syncStream(const DeviceBuffers& buf) {
    CUDACHECK(cudaStreamSynchronize(buf.stream));
}

// ============================================================================
// Collective Operations
// ============================================================================

static bool runAllReduce(const DeviceBuffers& buf, ncclComm_t comm,
                         size_t count, int nranks, int rank, bool verify) {
    fillBuffer(buf.send, count, rank + 1);
    zeroBuffer(buf.recv, count);
    NCCLCHECK(ncclAllReduce(buf.send, buf.recv, count, ncclInt, ncclSum, comm, buf.stream));
    syncStream(buf);

    if (verify) {
        int expected = (nranks * (nranks + 1)) / 2;
        auto host = copyToHost(buf.recv, count);
        return verifyAllEqual(host, expected, "AllReduce", rank);
    }
    return true;
}

static bool runAllGather(const DeviceBuffers& buf, ncclComm_t comm,
                         size_t sendcount, int nranks, int rank, bool verify) {
    size_t recvcount = sendcount * nranks;
    fillBuffer(buf.send, sendcount, rank + 1);
    zeroBuffer(buf.recv, recvcount);
    NCCLCHECK(ncclAllGather(buf.send, buf.recv, sendcount, ncclInt, comm, buf.stream));
    syncStream(buf);

    if (verify) {
        auto host = copyToHost(buf.recv, recvcount);
        return verifyAllGather(host, sendcount, nranks, rank);
    }
    return true;
}

static bool runAlltoAll(const DeviceBuffers& buf, ncclComm_t comm,
                        size_t countPerRank, int nranks, int rank, bool verify) {
    size_t totalCount = countPerRank * nranks;
    // Fill send buffer: each chunk for peer p gets value (rank * 100 + p)
    std::vector<int> sendData(totalCount);
    for (int p = 0; p < nranks; ++p) {
        for (size_t i = 0; i < countPerRank; ++i) {
            sendData[p * countPerRank + i] = rank * 100 + p;
        }
    }
    CUDACHECK(cudaMemcpy(buf.send, sendData.data(), totalCount * sizeof(int), cudaMemcpyHostToDevice));
    zeroBuffer(buf.recv, totalCount);
    NCCLCHECK(ncclAlltoAll(buf.send, buf.recv, countPerRank, ncclInt, comm, buf.stream));
    syncStream(buf);

    if (verify) {
        auto host = copyToHost(buf.recv, totalCount);
        // Each chunk from peer p should have value (p * 100 + rank)
        for (int p = 0; p < nranks; ++p) {
            int expected = p * 100 + rank;
            for (size_t i = 0; i < countPerRank; ++i) {
                if (host[p * countPerRank + i] != expected) {
                    printf("[Rank %d] AlltoAll FAILED at chunk %d index %zu: expected %d, got %d\n",
                           rank, p, i, expected, host[p * countPerRank + i]);
                    return false;
                }
            }
        }
        DEBUG_PRINT("[Rank %d] AlltoAll PASSED\n", rank);
    }
    return true;
}

// Run all collectives and return success status
static bool runAllCollectives(const DeviceBuffers& buf, ncclComm_t comm,
                              size_t baseCount, int nranks, int rank, bool verify) {
    bool success = true;
    success = runAllReduce(buf, comm, baseCount, nranks, rank, verify) && success;
    success = runAllGather(buf, comm, baseCount, nranks, rank, verify) && success;
    success = runAlltoAll(buf, comm, baseCount, nranks, rank, verify) && success;
    return success;
}

// ============================================================================
// Unified Suspend/Resume Memory Test
// ============================================================================
static bool runDynMemTest(
    ncclComm_t comm,
    int rank,
    int nranks,
    size_t baseCount,
    bool useSymmetricWindow,    // Register buffers with ncclCommWindowRegister
    bool verifyDataIntegrity,   // Verify user buffer data survives suspend/resume
    int numCycles,              // Number of suspend/resume cycles
    bool verifyCollectives      // Verify collective results
) {
    DEBUG_PRINT("[Rank %d] === Running Suspend/Resume Test ===\n", rank);
    DEBUG_PRINT("[Rank %d]   useSymmetricWindow=%d, verifyDataIntegrity=%d, cycles=%d\n",
                rank, useSymmetricWindow, verifyDataIntegrity, numCycles);

    // Calculate allocation size (same on all ranks for collectives)
    size_t allocCount = baseCount * nranks;
    size_t allocBytes = allocCount * sizeof(int);

    DEBUG_PRINT("[Rank %d] Allocating %zu elements (%zu bytes)\n", rank, allocCount, allocBytes);

    // Allocate buffers
    DeviceBuffers buf{};
    CUDACHECK(cudaStreamCreate(&buf.stream));
    NCCLCHECK(ncclMemAlloc((void**)&buf.send, allocBytes));
    NCCLCHECK(ncclMemAlloc((void**)&buf.recv, allocBytes));
    buf.allocBytes = allocBytes;
    buf.windowsRegistered = false;

    // Register symmetric windows if requested
    if (useSymmetricWindow) {
        DEBUG_PRINT("[Rank %d] Registering symmetric windows...\n", rank);
        NCCLCHECK(ncclCommWindowRegister(comm, buf.send, allocBytes, &buf.sendWindow, NCCL_WIN_COLL_SYMMETRIC));
        NCCLCHECK(ncclCommWindowRegister(comm, buf.recv, allocBytes, &buf.recvWindow, NCCL_WIN_COLL_SYMMETRIC));
        buf.windowsRegistered = true;
    }

    // Barrier after window registration
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    bool success = true;
    double totalSuspendTime = 0.0;
    double totalResumeTime = 0.0;

    for (int cycle = 0; cycle < numCycles; ++cycle) {
        DEBUG_PRINT("[Rank %d] Cycle %d/%d\n", rank, cycle + 1, numCycles);

        bool isLastCycle = (cycle == numCycles - 1);

        // Fill buffers with known patterns (for integrity check in last cycle)
        int sendPattern = PATTERN_VALUE + rank;
        int recvPattern = PATTERN_VALUE + rank + 1000;
        if (isLastCycle && verifyDataIntegrity) {
            fillBuffer(buf.send, allocCount, sendPattern);
            fillBuffer(buf.recv, allocCount, recvPattern);
            syncStream(buf);
        }

        // Run collectives before suspend (verify only in last cycle)
        DEBUG_PRINT("[Rank %d] Running collectives before suspend...\n", rank);
        success = runAllCollectives(buf, comm, baseCount, nranks, rank, isLastCycle && verifyCollectives) && success;

        // Refill buffers with patterns before suspend (for integrity check)
        if (isLastCycle && verifyDataIntegrity) {
            fillBuffer(buf.send, allocCount, sendPattern);
            fillBuffer(buf.recv, allocCount, recvPattern);
            syncStream(buf);
        }

        // Sync before timing suspend
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Suspend NCCL internal memory (with timing)
        DEBUG_PRINT("[Rank %d] Suspending memory...\n", rank);
        double suspendStart = MPI_Wtime();
        NCCLCHECK(ncclCommSuspend(comm, NCCL_SUSPEND_MEM));
        double suspendEnd = MPI_Wtime();
        double suspendTime = (suspendEnd - suspendStart) * 1000.0; // ms
        totalSuspendTime += suspendTime;

        if (rank == 0) {
            printf("[Cycle %d] Suspend time: %.3f ms\n", cycle + 1, suspendTime);
        }

        // Verify user data integrity after suspend (only in last cycle)
        if (isLastCycle && verifyDataIntegrity) {
            auto hostSend = copyToHost(buf.send, allocCount);
            auto hostRecv = copyToHost(buf.recv, allocCount);
            success = verifyAllEqual(hostSend, sendPattern, "send after suspend", rank) && success;
            success = verifyAllEqual(hostRecv, recvPattern, "recv after suspend", rank) && success;
        }

        // Sync before timing resume
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Resume NCCL internal memory (with timing)
        DEBUG_PRINT("[Rank %d] Resuming memory...\n", rank);
        double resumeStart = MPI_Wtime();
        NCCLCHECK(ncclCommResume(comm));
        double resumeEnd = MPI_Wtime();
        double resumeTime = (resumeEnd - resumeStart) * 1000.0; // ms
        totalResumeTime += resumeTime;

        if (rank == 0) {
            printf("[Cycle %d] Resume time: %.3f ms\n", cycle + 1, resumeTime);
            // Print memory status only in the last cycle
            if (isLastCycle) {
                printf("\n--- Memory Status After Resume ---\n");
                // Print detailed stats using new public API
                printMemoryStats(comm);
            }
        }

        // Verify user data integrity after resume (only in last cycle)
        if (isLastCycle && verifyDataIntegrity) {
            auto hostSend = copyToHost(buf.send, allocCount);
            auto hostRecv = copyToHost(buf.recv, allocCount);
            success = verifyAllEqual(hostSend, sendPattern, "send after resume", rank) && success;
            success = verifyAllEqual(hostRecv, recvPattern, "recv after resume", rank) && success;
        }

        // Run collectives after resume (verify only in last cycle)
        DEBUG_PRINT("[Rank %d] Running collectives after resume...\n", rank);
        success = runAllCollectives(buf, comm, baseCount, nranks, rank, isLastCycle && verifyCollectives) && success;
    }

    // Print average timing summary
    if (rank == 0 && numCycles > 0) {
        printf("\n--- Timing Summary ---\n");
        printf("Average suspend time: %.3f ms\n", totalSuspendTime / numCycles);
        printf("Average resume time: %.3f ms\n", totalResumeTime / numCycles);
        printf("Total suspend time:   %.3f ms\n", totalSuspendTime);
        printf("Total resume time:   %.3f ms\n", totalResumeTime);
    }

    // Cleanup
    if (buf.windowsRegistered) {
        DEBUG_PRINT("[Rank %d] Deregistering windows...\n", rank);
        NCCLCHECK(ncclCommWindowDeregister(comm, buf.sendWindow));
        NCCLCHECK(ncclCommWindowDeregister(comm, buf.recvWindow));
    }
    NCCLCHECK(ncclMemFree(buf.send));
    NCCLCHECK(ncclMemFree(buf.recv));
    CUDACHECK(cudaStreamDestroy(buf.stream));

    DEBUG_PRINT("[Rank %d] Test: %s\n", rank, success ? "PASSED" : "FAILED");
    return success;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    setlinebuf(stdout);

    local_cli_args_t args;
    parse_local_cli_args(argc, argv, &args);
    DEBUG = args.debug;

    // Initialize MPI
    int rank = 0, nranks = 0, localRank = 0;
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    if (nranks < 2) {
        if (rank == 0) {
            printf("suspend_test: SKIP (need at least 2 ranks, found %d)\n", nranks);
        }
        MPICHECK(MPI_Finalize());
        return 0;
    }

    // Calculate localRank based on hostname hash
    int nvis = 0;
    CUDACHECK(cudaGetDeviceCount(&nvis));

    uint64_t hostHashs[nranks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[rank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
                           sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

    for (int p = 0; p < nranks; ++p) {
        if (p == rank) break;
        if (hostHashs[p] == hostHashs[rank]) localRank++;
    }

    DEBUG_PRINT("[Rank %d] Using GPU %d (nvis=%d)\n", rank, localRank, nvis);

    if (localRank >= nvis) {
        if (rank == 0) {
            printf("suspend_test: SKIP (localRank %d >= device count %d)\n", localRank, nvis);
        }
        MPICHECK(MPI_Finalize());
        return 0;
    }

    // Set device
    CUDACHECK(cudaSetDevice(localRank));

    // Query GPU memory and adjust count if necessary
    {
        cudaDeviceProp prop;
        CUDACHECK(cudaGetDeviceProperties(&prop, localRank));
        size_t maxMem = prop.totalGlobalMem;

        // Reserve memory: 1GB per 16GB of GPU memory, capped at 4GB (same formula as perf tests)
        size_t reserveMem = std::min((maxMem + (16ULL << 30) - 1) / (16ULL << 30) * (1ULL << 30), 4ULL << 30);

        // We need 2 buffers (send + recv), each of size: count * nranks * sizeof(int)
        // Plus 1GB additional reserve for NCCL internal buffers
        size_t availableMem = maxMem - reserveMem - (1ULL << 30);
        size_t maxAllocBytes = availableMem / 2;  // Divide by 2 for send + recv buffers
        size_t maxCount = maxAllocBytes / (nranks * sizeof(int));

        if (args.count > maxCount) {
            if (rank == 0) {
                printf("# Reducing element count from %zu to %zu due to memory limitation (GPU has %zu MB)\n",
                       args.count, maxCount, maxMem / (1024 * 1024));
            }
            args.count = maxCount;
        }
    }

    // Initialize NCCL
    ncclUniqueId id;
    ncclComm_t comm;
    if (rank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 1;
    NCCLCHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));

    DEBUG_PRINT("[Rank %d] NCCL communicator initialized\n", rank);

    // Print test configuration
    if (rank == 0) {
        printf("=== Dynamic Memory Offload Unit Test ===\n");
        printf("Ranks: %d\n", nranks);
        printf("Element count: %zu\n", args.count);
        printf("Options:\n");
        printf("  Repeat cycles: %d\n", args.repeat_cycles);
        printf("  Symmetric window: %s\n", args.use_symmetric_win ? "yes" : "no");
        printf("  Verify integrity: %s\n", args.verify_integrity ? "yes" : "no");
        printf("  Verify collectives: %s\n", args.verify ? "yes" : "no");
        printf("=========================================\n");
    }
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Run the unified test
    bool success = runDynMemTest(
        comm,
        rank,
        nranks,
        args.count,
        args.use_symmetric_win,
        args.verify_integrity,
        args.repeat_cycles,
        args.verify
    );

    // Aggregate results
    int localSuccess = success ? 1 : 0;
    int globalSuccess = 0;
    MPICHECK(MPI_Allreduce(&localSuccess, &globalSuccess, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

    // Cleanup
    DEBUG_PRINT("[Rank %d] Cleaning up...\n", rank);
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));

    // Final result
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) {
        printf("\n=========================================\n");
        printf("suspend_test: %s\n", globalSuccess ? "PASS" : "FAIL");
        printf("=========================================\n");
    }

    MPICHECK(MPI_Finalize());
    return globalSuccess ? 0 : 1;
}
