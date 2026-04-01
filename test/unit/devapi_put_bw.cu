#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "bitops.h"
#include "common.h"

#include <vector>
#include <cmath>

__device__ __forceinline__ int getLaneId() {
  int laneId;
  asm("mov.s32 %0, %laneid;" : "=r"(laneId));
  return laneId;
}

template <bool skipCreditCheck, bool aggregateRequests>
__global__ void runDevice(ncclDevComm comm, ncclDevResourceHandle hbuf, int numIters, size_t numElems) {
#if __CUDA_ARCH__ >= 700
  const int b = blockIdx.x;
  const int laneId = getLaneId();
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin net = ncclGin(comm, b);
  ncclSymPtr<int> sbuf = (ncclSymPtr<int>)ncclGetResourceBuffer(comm, hbuf);
  ncclSymPtr<int> rbuf = sbuf;

  constexpr uint32_t optFlags = skipCreditCheck ? ncclGinOptFlagsMaySkipCreditCheck : ncclGinOptFlagsDefault;

  for (int iter = 0; iter < numIters; iter++) {
    if (aggregateRequests) {
      if (laneId < warpSize - 1) {
        net.put(world, 1, rbuf, sbuf, numElems, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread,
                optFlags | ncclGinOptFlagsAggregateRequests);
      }
      __syncwarp();
      if (laneId == warpSize - 1) {
        net.put(world, 1, rbuf, sbuf, numElems, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
                ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread,
                optFlags);
      }
    }
    else {
      net.put(world, 1, rbuf, sbuf, numElems, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{},
              ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_thread,
              optFlags);
    }
    if (skipCreditCheck) {
      net.flush(ncclCoopCta{});
    }
    else {
      __syncthreads();
    }
  }
  net.flush(ncclCoopCta{});
#endif
}

typedef void (*runDevice_t)(ncclDevComm comm, ncclDevResourceHandle hbuf, int numIters, size_t numElems);

static runDevice_t getRunDeviceFunc(cli_args_t args) {
  runDevice_t func = nullptr;
  if (args.gin_skip_credit_check && args.gin_aggregate_requests) {
    func = runDevice<true, true>;
  }
  else if (args.gin_skip_credit_check) {
    func = runDevice<true, false>;
  }
  else if (args.gin_aggregate_requests) {
    func = runDevice<false, true>;
  }
  else {
    func = runDevice<false, false>;
  }
  return func;
}

// warpSize = 32 is constant for all NVIDIA GPU architectures.
static const int kWarpSize = 32;

static void validate_args(int argc, char** argv, const cli_args_t& args) {
  if (args.gin_aggregate_requests && args.num_threads % kWarpSize != 0) {
    int nearest = ((args.num_threads + kWarpSize / 2) / kWarpSize) * kWarpSize;
    if (nearest < kWarpSize) nearest = kWarpSize;
    if (nearest > 1024) nearest = 1024;
    fprintf(stderr,
      "Error: --gin_aggregate_requests requires the thread count per CTA (-t) to be a multiple of warpSize.\n"
      "  warpSize = %d (constant for all NVIDIA GPU architectures)\n"
      "  Provided: -t %d\n"
      "  Valid values: any multiple of %d in [%d, 1024], e.g. 32, 64, 96, 128, 256, 512, 1024\n"
      "  Suggestion: use -t %d\n",
      kWarpSize, args.num_threads, kWarpSize, kWarpSize, nearest);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  cli_args_t args;
  parse_cli_args(argc, argv, &args);
  validate_args(argc, argv, args);

  int rank, nRanks;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  if (nRanks != 2) {
    fprintf(stderr, "This test requires exactly 2 ranks\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  uint64_t* hosts = new uint64_t[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hosts[rank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hosts, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  int dev = 0;
  for (int r=0; r < rank; r++) {
    if (hosts[r] == hosts[rank]) dev++;
  }
  delete[] hosts;

  ncclUniqueId id;
  ncclComm_t comm;
  if (rank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  CUDACHECK(cudaSetDevice(dev));

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 1;
  NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, rank, &config));

  ncclDevComm dcomm;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;

  reqs.ginForceEnable = true;
  reqs.ginContextCount = args.num_ctas;
  if (args.gin_skip_credit_check || args.gin_aggregate_requests) {
    reqs.ginQueueDepth = args.num_threads;
  }
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginSignalCount = 0;
  reqs.ginForceEnable = true;

  ncclDevResourceHandle hBuf;
  ncclDevResourceRequirements bufReq = {};
  bufReq.bufferSize = args.end_size * sizeof(int);
  bufReq.outBufferHandle = &hBuf;
  bufReq.next = reqs.resourceRequirementsList;
  reqs.resourceRequirementsList = &bufReq;

  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &dcomm));

  if (rank == 0) {
    runDevice_t runDeviceFunc = getRunDeviceFunc(args);
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    int device_iters;
    size_t nelems = args.end_size / sizeof(int);
    if (args.warmup_iters > 0) {
      device_iters = (int)std::ceil((double)args.warmup_iters / (double)args.num_threads);
      printf("[MPI Rank %d] Starting warmup for size %zu bytes (%zu elements)\n", rank, args.end_size, nelems);
      runDeviceFunc<<<args.num_ctas, args.num_threads, 0, stream>>>(dcomm, hBuf, device_iters, nelems);
      CUDACHECK(cudaStreamSynchronize(stream));
      printf("[MPI Rank %d] Completed warmup\n", rank);
    }

    printf("Size(B)\t\tNum_Messages\t\tBandwidth(MiB/s)\t\tMessage_Rate(MPPS)\n");

    // run kernel
    for (size_t size = args.begin_size; size <= args.end_size; size *= 2) {
      size_t nelems = size / sizeof(int);
      device_iters = (int)std::ceil((double)args.normal_iters / (double)args.num_threads);

      CUDACHECK(cudaEventRecord(start, stream));
      runDeviceFunc<<<args.num_ctas, args.num_threads, 0, stream>>>(dcomm, hBuf, device_iters, nelems);
      CUDACHECK(cudaEventRecord(stop, stream));
      CUDACHECK(cudaStreamSynchronize(stream));
      float milliseconds = 0.0f;
      CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));
      double seconds = (double)milliseconds / 1e3;
      double num_messages = (double)device_iters * (double)args.num_ctas * (double)args.num_threads;
      double msgrate = num_messages / 1e6 / seconds;
      double bytes = num_messages * (double)nelems * sizeof(int);
      double bw = bytes / (1ULL << 20) / seconds;
      printf("%zu\t\t%d\t\t%9.2f\t\t%9.2f\n", size, (int)num_messages, bw, msgrate);
    }

    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
  }

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // cleanup
  CUDACHECK(cudaStreamDestroy(stream));

  NCCLCHECK(ncclDevCommDestroy(comm, &dcomm));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  MPICHECK(MPI_Finalize());
  return 0;
}
