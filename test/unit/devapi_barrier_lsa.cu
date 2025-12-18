#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "bitops.h"
#include "common.h"

#include <vector>

#ifdef USE_IR
#include <cuda.h>
#include "cumod_common.h"
#endif

#ifndef USE_IR
extern "C" __global__ void runDevice(ncclDevComm comm) {
#if __CUDA_ARCH__ >= 700
    // Compute LSA team on device
    ncclTeam team = ncclTeamLsa(comm);

    ncclLsaBarrierHandle handle = comm.lsaBarrier;
    ncclMultimemHandle mmHandle{}; // unused when multimem=false

    ncclLsaBarrierSession<ncclCoopCta> bar {ncclCoopCta(), comm, team, handle, blockIdx.x, /*multimem=*/false, mmHandle};

    // Barrier rounds
    for (int round = 0; round < 10; ++round) {
        bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
        if (threadIdx.x == 0 && blockIdx.x == 0 && team.rank == (round % team.nRanks)) {
            printf("Round %d\n", round);
        }
    }
#endif
}
#else
extern "C" __global__ void runDevice(ncclDevComm comm);
#endif

int main(int argc, char** argv) {
  int rank, nRanks;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

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

#ifdef USE_IR
  CUstream stream;
  CU_CHECK(cuStreamCreate(&stream, 0));
#else
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
#endif

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 1;
  NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, rank, &config));

  ncclDevComm dcomm;
  { ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = 16;
    reqs.lsaMultimem = false;
    NCCLCHECK(ncclDevCommCreate(comm, &reqs, &dcomm));
  }

  // run kernel
  printf("[MPI Rank %d] Starting kernel\n", rank);

#ifdef USE_IR
  // IR path: load kernel from cubin and launch
  CUmodule mymodule = NULL;
  initCumodule(&mymodule, "devapi_barrier_lsa_ir.cubin");
  CUfunction kernel;
  initTestCaseKernel(mymodule, &kernel, "runDevice");

  void* args[] = {&dcomm};
  CU_CHECK(cuLaunchKernel(kernel, 16, 1, 1, 512, 1, 1, 0, stream, args, NULL));
  CU_CHECK(cuStreamSynchronize(stream));
#else
  runDevice<<<16, 512, 0, stream>>>(dcomm);
  CUDACHECK(cudaStreamSynchronize(stream));
#endif

  printf("[MPI Rank %d] Completed kernel\n", rank);

  // cleanup
#ifdef USE_IR
  finiCumodule(&mymodule);
  CU_CHECK(cuStreamDestroy(stream));
#else
  CUDACHECK(cudaStreamDestroy(stream));
#endif

  NCCLCHECK(ncclDevCommDestroy(comm, &dcomm));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  MPICHECK(MPI_Finalize());
  return 0;
}
