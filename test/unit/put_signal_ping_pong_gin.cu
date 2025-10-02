#include <errno.h>
#include <cooperative_groups.h>
#include "nccl.h"
#include "comm.h"
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

#define CUDACHECK_DEBUG(cmd, rank) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("[%d] Failed: Cuda error %s:%d '%s'\n",       \
        rank, __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

using namespace cooperative_groups;

static const bool DEBUG = false;

// Kernel for ping-pong test
__global__ void ping_pong_kernel(
    ncclGinCtx_M<-1u> ctx,
    void *sendbuff, ncclGinWindow_t ginHandle_src,
    void *recvbuff, ncclGinWindow_t ginHandle_dst,
    ncclGinSignal_t signal_id,
    size_t nelems, int pe, int iter) {

#if CUDA_VERSION >= 12020 && __CUDA_ARCH__ >= 700
    __shared__ ncclGinDescriptorSmem desc;
    int i, peer;
    peer = !pe;

    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    thread_block_tile<1> thread = this_thread();

    if (DEBUG) printf("[Rank %d] Starting kernel with %d iterations, nelems=%llu, signal_id=%u\n", pe, iter, nelems, signal_id);

    uint64_t current_signal_value = 0;
    for (i = 1; i <= iter; i++) {
        if (DEBUG) printf("[Rank %d] Starting iteration %d\n", pe, i);

        uint64_t expected_signal_value = i;

        if (pe) {  // Rank 1
            if (DEBUG) printf("[Rank %d] Waiting for signal from peer (signal_id=%u, expected_value=%lu) current_signal_value=%lu\n", pe, signal_id, expected_signal_value, current_signal_value);

            // Wait for signal from peer with expected iteration count
            { uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(ctx, signal_id);
              cuda::atomic_ref<uint64_t> ref{*ptr};
              do current_signal_value = ref.load(cuda::memory_order_acquire);
              while (current_signal_value < expected_signal_value);
            }
            if (DEBUG) printf("[Rank %d] Received signal (signal_id=%u, expected_value=%lu) current_signal_value=%lu, sending data (peer=%d, nelems=%llu, signal_id=%u)\n", pe, signal_id, expected_signal_value, current_signal_value, peer, nelems, signal_id);

            // Put data with counted signal
            if (DEBUG) printf("[Rank %d] About to call ncclGinPut with signal (hasSignal=true, signalOp=ncclGinSignalAdd, signalVal=%lu)\n", pe, expected_signal_value);

            ncclGinCall<ncclGinApi_Put>(ctx, thread, peer, /*hasWins=*/true,
                ginHandle_dst, 0, ginHandle_src, 0, nelems * sizeof(int),
                /*hasSignal=*/true, signal_id, ncclGinSignalAdd, 1,
                /*hasCounter=*/false, 0,
                /*hasDescriptor=*/true, &desc,
                cuda::thread_scope_thread, cuda::thread_scope_thread);
            if (DEBUG) printf("[Rank %d] Sent data with signal\n", pe);

        } else {   // Rank 0
            if (DEBUG) printf("[Rank %d] Sending data with signal (peer=%d, nelems=%llu, signal_id=%u, signal_value=%lu)\n", pe, peer, nelems, signal_id, expected_signal_value);

            // Put data with counted signal
            if (DEBUG) printf("[Rank %d] About to call ncclGinPut with signal (hasSignal=true, signalOp=ncclGinSignalAdd, signalVal=%lu)\n", pe, expected_signal_value);

            ncclGinCall<ncclGinApi_Put>(ctx, thread, peer, /*hasWins=*/true,
                ginHandle_dst, 0, ginHandle_src, 0, nelems * sizeof(int),
                /*hasSignal=*/true, signal_id, ncclGinSignalAdd, 1,
                /*hasCounter=*/false, 0,
                /*hasDescriptor=*/true, &desc,
                cuda::thread_scope_thread, cuda::thread_scope_thread);
            if (DEBUG) printf("[Rank %d] Sent data with signal\n", pe);

            if (DEBUG) printf("[Rank %d] Sent data, waiting for signal (signal_id=%u, expected_value=%lu) current_signal_value=%lu\n", pe, signal_id, expected_signal_value, current_signal_value);

            // Wait for signal from peer with expected iteration count
            { uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(ctx, signal_id);
              cuda::atomic_ref<uint64_t> ref{*ptr};
              do current_signal_value = ref.load(cuda::memory_order_acquire);
              while (current_signal_value < expected_signal_value);
            }
            if (DEBUG) printf("[Rank %d] Received signal (signal_id=%u, expected_value=%lu) current_signal_value=%lu\n", pe, signal_id, expected_signal_value, current_signal_value);
        }
    }
    if (DEBUG) printf("[Rank %d] Completed all iterations\n", pe);
    if (threadIdx.x==0) ncclGinCall<ncclGinApi_ResetSignal>(ctx, signal_id);
#else
    printf("ERROR: This test requires CUDA 12.2 and compute capability 7.0 (Volta) or higher\n");
    assert(0);
#endif
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

    if (nRanks != 2) {
        printf("This test requires exactly 2 ranks\n");
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

    // Configure NCCL with GIN support
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 1;
    NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, myRank, &config));
    NCCLCHECK(ncclGinConnectOnce(comm));

    // Allocate and register symmetric memory
    void *sendbuff, *recvbuff;
    NCCLCHECK(ncclMemAlloc((void**)&sendbuff, args.end_size));
    NCCLCHECK(ncclMemAlloc((void**)&recvbuff, args.end_size));

    // Get window handles
    void* srcGinHostWins[NCCL_GIN_MAX_CONTEXTS];
    ncclGinWindow_t srcGinDevWins[NCCL_GIN_MAX_CONTEXTS];
    NCCLCHECK(ncclGinRegister(comm, sendbuff, args.end_size, srcGinHostWins, srcGinDevWins));
    ncclGinWindow_t srcGinWindow = srcGinDevWins[0];

    void* dstGinHostWins[NCCL_GIN_MAX_CONTEXTS];
    ncclGinWindow_t dstGinDevWins[NCCL_GIN_MAX_CONTEXTS];
    NCCLCHECK(ncclGinRegister(comm, recvbuff, args.end_size, dstGinHostWins, dstGinDevWins));
    ncclGinWindow_t dstGinWindow = dstGinDevWins[0];

    // Get GIN resources
    ncclGinCtx_M<-1u> gctx;
    gctx.backend = comm->sharedRes->ginState.ginDevHandles[0]->netDeviceType;
    gctx.handle = comm->sharedRes->ginState.ginDevHandles[0]->handle;
    gctx.rank = myRank;
    gctx.nRanks = nRanks;

    ncclGinSignal_t signalIDs = 0;

    if (DEBUG) printf("[Rank %d] Allocating %zu bytes\n", myRank, args.end_size);

    // Timing setup
    float milliseconds;
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    if (myRank == 0) {
        printf("Note: This test measures full round-trip latency\n");
        if (args.verify) {
            printf("Data verification enabled\n");
        }
        printf("Warmup iterations: %d\n", args.warmup_iters);
        printf("Normal iterations: %d\n", args.normal_iters);
        printf("Message size range: %zu to %zu bytes\n", args.begin_size, args.end_size);
    }

    // Print header once before running tests
    if (myRank == 0) {
        printf("size(B)     latency (us)\n");
    }

    // Run tests for different message sizes
    for (size_t size = args.begin_size; size <= args.end_size; size *= 2) {

        size_t nelems = size / sizeof(int);
        if (DEBUG) printf("[Rank %d] Testing size %zu bytes (%zu elements, %zu actual bytes)\n", myRank, size, nelems, nelems * sizeof(int));

        if (args.verify && nelems > 0) {
            int* h_sendbuff = (int*)malloc(nelems * sizeof(int));
            int* h_recvbuff = (int*)malloc(nelems * sizeof(int));
            initialize(h_sendbuff, h_recvbuff, myRank, nelems);
            CUDACHECK(cudaMemcpy(sendbuff, h_sendbuff, nelems * sizeof(int), cudaMemcpyHostToDevice));
            CUDACHECK(cudaMemcpy(recvbuff, h_recvbuff, nelems * sizeof(int), cudaMemcpyHostToDevice));
            free(h_sendbuff);
            free(h_recvbuff);
        }

        // Warmup phase
        if (args.warmup_iters > 0) {
            if (DEBUG) printf("[Rank %d] Running warmup\n", myRank);
            ping_pong_kernel<<<1, 1, 0, stream>>>(gctx, sendbuff, srcGinWindow, recvbuff, dstGinWindow, signalIDs, nelems, myRank, args.warmup_iters);
            CUDACHECK(cudaStreamSynchronize(stream));
            if (DEBUG) printf("[Rank %d] Ran warmup (after cudaStreamSynchronize)\n", myRank);
            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
            if (DEBUG) printf("[Rank %d] Ran warmup (after MPI_Barrier)\n", myRank);
        }

        // Measurement phase
        if (DEBUG) printf("[Rank %d] Running measurement\n", myRank);
        CUDACHECK(cudaEventRecord(start, stream));
        ping_pong_kernel<<<1, 1, 0, stream>>>(gctx, sendbuff, srcGinWindow, recvbuff, dstGinWindow, signalIDs, nelems, myRank, args.normal_iters);
        CUDACHECK(cudaEventRecord(stop, stream));
        CUDACHECK_DEBUG(cudaStreamSynchronize(stream), myRank);
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Calculate latency
        CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        double latency = (milliseconds * 1000) / args.normal_iters;  // Convert to microseconds

        if (args.verify && nelems > 0) {
            // Get original values for comparison
            int* h_sendbuff_orig = (int*)malloc(nelems * sizeof(int));
            int* h_recvbuff_orig = (int*)malloc(nelems * sizeof(int));
            initialize(h_sendbuff_orig, h_recvbuff_orig, myRank, nelems);

            // Get current values from GPU
            int* h_sendbuff = (int*)malloc(nelems * sizeof(int));
            int* h_recvbuff = (int*)malloc(nelems * sizeof(int));
            CUDACHECK(cudaMemcpy(h_sendbuff, sendbuff, nelems * sizeof(int), cudaMemcpyDeviceToHost));
            CUDACHECK(cudaMemcpy(h_recvbuff, recvbuff, nelems * sizeof(int), cudaMemcpyDeviceToHost));

            // Print before/after values for last index
            printf("\n=== VERIFICATION FOR SIZE %zu BYTES ===\n", size);
            printf("[Rank %d] Buffer values at last index [%zu]:\n", myRank, nelems-1);
            printf("[Rank %d]   sendbuff: before=0x%X (%d) -> after=0x%X (%d)\n",
                   myRank, h_sendbuff_orig[nelems-1], h_sendbuff_orig[nelems-1],
                   h_sendbuff[nelems-1], h_sendbuff[nelems-1]);
            printf("[Rank %d]   recvbuff: before=0x%X (%d) -> after=0x%X (%d)\n",
                   myRank, h_recvbuff_orig[nelems-1], h_recvbuff_orig[nelems-1],
                   h_recvbuff[nelems-1], h_recvbuff[nelems-1]);
            printf("[Rank %d] Expected to receive: 0x%X (%d) from peer\n",
                   myRank, 0x100 + (!myRank), 0x100 + (!myRank));

            bool verification_result = verify(h_sendbuff, h_recvbuff, nelems, myRank);
            printf("[Rank %d] %s\n", myRank, verification_result ? "✓ PASS" : "✗ FAIL");
            printf("=====================================\n");

            free(h_sendbuff_orig);
            free(h_recvbuff_orig);
            free(h_sendbuff);
            free(h_recvbuff);

            // Reset GPU buffers for next message size iteration
            int* h_sendbuff_reset = (int*)malloc(nelems * sizeof(int));
            int* h_recvbuff_reset = (int*)malloc(nelems * sizeof(int));
            initialize(h_sendbuff_reset, h_recvbuff_reset, myRank, nelems);
            CUDACHECK(cudaMemcpy(sendbuff, h_sendbuff_reset, nelems * sizeof(int), cudaMemcpyHostToDevice));
            CUDACHECK(cudaMemcpy(recvbuff, h_recvbuff_reset, nelems * sizeof(int), cudaMemcpyHostToDevice));
            free(h_sendbuff_reset);
            free(h_recvbuff_reset);
        }

        if (myRank == 0) {
            printf("%-12zu %-12.2f\n", size, latency);
        }
    }

    // Cleanup
    if (DEBUG) printf("[Rank %d] Cleaning up\n", myRank);

    NCCLCHECK(ncclGinDeregister(comm, srcGinHostWins));
    NCCLCHECK(ncclGinDeregister(comm, dstGinHostWins));
    NCCLCHECK(ncclGinFinalize(comm));
    NCCLCHECK(ncclMemFree(sendbuff));
    NCCLCHECK(ncclMemFree(recvbuff));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));

    printf("[MPI Rank %d] Success \n", myRank);

    MPICHECK(MPI_Finalize());

    return 0;
}
