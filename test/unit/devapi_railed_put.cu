#include <errno.h>
#include "nccl.h"
#include "nccl_device.h"
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

constexpr uint64_t NCCL_PUT_VALUE_BASE = 28; // atomic number of nickel
constexpr int SRC_RANK = 0;
constexpr int DST_RANK = 1;

__device__ uint64_t getPutValue(ncclDevComm comm) {
  int myRailTeam = ncclTeamLsa(comm).rank;
  return NCCL_PUT_VALUE_BASE + myRailTeam; // ensure different rails use different values
}

__global__ void putKernel(ncclDevComm comm, ncclWindow_t window, size_t offset) {
#if __CUDA_ARCH__ >= 700
  ncclTeam railTeam = ncclTeamRail(comm);
  // Exercise both resource sharing modes on the same contextIndex.
  ncclGin ginGpu(comm, 0, NCCL_GIN_RESOURCE_SHARING_GPU);
  ncclGin ginCta(comm, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
  ncclGinSignal_t signalIdxGpu = 0;
  ncclGinSignal_t signalIdxCta = 1;

  assert(railTeam.nRanks >= 2 && "Railed Gin put test requires at least 2 ranks per rail");

  uint64_t putValueGpu = getPutValue(comm) + 0;
  uint64_t putValueCta = getPutValue(comm) + 1000;
  if (railTeam.rank == SRC_RANK) {
    uint64_t* putSrcPtrGpu = (uint64_t*)ncclGetLocalPointer(window, offset + 0*sizeof(uint64_t));
    uint64_t* putSrcPtrCta = (uint64_t*)ncclGetLocalPointer(window, offset + 1*sizeof(uint64_t));
    assert(*putSrcPtrGpu == 0 && "putSrcPtrGpu should be 0 before put");
    assert(*putSrcPtrCta == 0 && "putSrcPtrCta should be 0 before put");
    *putSrcPtrGpu = putValueGpu;
    *putSrcPtrCta = putValueCta;

    ginGpu.put(railTeam, DST_RANK,
      window, offset + 0*sizeof(uint64_t),
      window, offset + 0*sizeof(uint64_t),
      sizeof(uint64_t), ncclGin_SignalInc{signalIdxGpu});

    ginCta.put(railTeam, DST_RANK,
      window, offset + 1*sizeof(uint64_t),
      window, offset + 1*sizeof(uint64_t),
      sizeof(uint64_t), ncclGin_SignalInc{signalIdxCta});
  }

  if (railTeam.rank == DST_RANK) {
    volatile uint64_t* putPtrGpu = (uint64_t*)ncclGetLocalPointer(window, offset + 0*sizeof(uint64_t));
    volatile uint64_t* putPtrCta = (uint64_t*)ncclGetLocalPointer(window, offset + 1*sizeof(uint64_t));

    ginGpu.waitSignal(ncclCoopCta(), signalIdxGpu, 1);
    if (*putPtrGpu != putValueGpu) {
      printf("ERROR: Rank %d expected %llu but got %llu after waitSignal (GPU mode)\n",
              comm.rank, putValueGpu, *putPtrGpu);
      assert(0 && "Put value mismatch after waitSignal (GPU mode)");
    }

    ginCta.waitSignal(ncclCoopCta(), signalIdxCta, 1);
    if (*putPtrCta != putValueCta) {
      printf("ERROR: Rank %d expected %llu but got %llu after waitSignal (CTA mode)\n",
              comm.rank, putValueCta, *putPtrCta);
      assert(0 && "Put value mismatch after waitSignal (CTA mode)");
    }

  }
#endif
}

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
    if (myRank == 0) {
      printf("This test requires at least 2 ranks (for railed communication)\n");
    }
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

  if (DEBUG) printf("[Rank %d] NCCL communicator initialized successfully\n", myRank);

  // Skip cleanly if GIN / device API isn't supported in this environment.
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCLCHECK(ncclCommQueryProperties(comm, &props));
  int supportsGin = (props.deviceApiSupport && props.ginType != NCCL_GIN_TYPE_NONE) ? 1 : 0;
  int allSupportGin = 0;
  MPICHECK(MPI_Allreduce(&supportsGin, &allSupportGin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
  if (!allSupportGin) {
    if (myRank == 0) {
      printf("SKIP: GIN/device API not supported in this environment\n");
    }
    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclCommDestroy(comm));
    MPICHECK(MPI_Finalize());
    return 0;
  }

  // Create device communicator for GIN operations
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
  reqs.ginSignalCount = 2;

  ncclDevComm_t devComm;
  CUDACHECK(cudaSetDevice(localRank));
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));

  if (DEBUG) printf("[Rank %d] Device communicator created\n", myRank);

  // Allocate window for put operations
  // We write two values to exercise both resource sharing modes (GPU/CTA).
  size_t bufferSize = 2 * sizeof(uint64_t);
  void *putBuffer;
  NCCLCHECK(ncclMemAlloc(&putBuffer, bufferSize));

  // Register window for GIN operations
  ncclWindow_t putWindow;
  NCCLCHECK(ncclCommWindowRegister(comm, putBuffer, bufferSize, &putWindow, NCCL_WIN_COLL_SYMMETRIC));

  // Initialize buffer to zero
  CUDACHECK(cudaMemset(putBuffer, 0, bufferSize));

  // Ensure all ranks have completed setup before proceeding
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (myRank == 0) {
    printf("Running Railed GIN Put test with %d ranks\n", nRanks);
    printf("Testing both signal and polling modes\n\n");
  }

  if (myRank == 0) {
    printf("Test: Railed Gin Put...");
    fflush(stdout);
  }

  CUDACHECK(cudaMemset(putBuffer, 0, bufferSize));
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  putKernel<<<1, 1, 0, stream>>>(devComm, putWindow, 0);
  CUDACHECK(cudaStreamSynchronize(stream));
  cudaError_t err = cudaGetLastError();

  int test_passed = (err == cudaSuccess) ? 1 : 0;
  int all_test_passed;
  MPICHECK(MPI_Allreduce(&test_passed, &all_test_passed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

  if (myRank == 0) {
    if (all_test_passed) {
      printf(" PASSED\n");
    } else {
      printf(" FAILED (error: %s)\n", cudaGetErrorString(err));
    }
  }


  // Cleanup
  NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
  NCCLCHECK(ncclCommWindowDeregister(comm, putWindow));
  NCCLCHECK(ncclMemFree(putBuffer));
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommDestroy(comm));


  MPICHECK(MPI_Finalize());
  return all_test_passed ? 0 : 1;
}
