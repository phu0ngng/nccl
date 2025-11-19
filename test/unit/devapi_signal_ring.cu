#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "bitops.h"
#include "common.h"

#include <vector>

#ifdef USE_IR
#include <cuda.h>
#endif

#ifndef USE_IR
__global__ void runDevice(ncclDevComm comm) {
#if __CUDA_ARCH__ >= 700
  int t = threadIdx.x;
  ncclTeam world = ncclTeamWorld(comm);

  ncclGin net(comm, 0);
  for (int round=0; round < 10; round++) {
    if (t==0) net.signal(world, (world.rank + 1)%world.nRanks, ncclGin_SignalAdd{0, 1});
    net.waitSignal(ncclCoopCta(), 0, round+1);
    if (world.rank == round%world.nRanks && t==0) printf("Signal received round %d\n", round);
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
  { 
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginSignalCount = 1;
    reqs.ginForceEnable = true;
    NCCLCHECK(ncclDevCommCreate(comm, &reqs, &dcomm));
  }
  // run kernel
  printf("[MPI Rank %d] Starting kernel\n", rank);

#ifdef USE_IR
  // IR path: load kernel from cubin and launch
  CUmodule mymodule = NULL;
  init_cumodule(&mymodule, "devapi_signal_ring_ir.cubin");
  CUfunction kernel;
  init_test_case_kernel(mymodule, &kernel, "runDevice");
  
  void* args[] = {&dcomm};
  CU_CHECK(cuLaunchKernel(kernel, 1, 1, 1, 512, 1, 1, 0, stream, args, NULL));
  CU_CHECK(cuStreamSynchronize(stream));
#else
  runDevice<<<1, 512, 0, stream>>>(dcomm);
  CUDACHECK(cudaStreamSynchronize(stream));
#endif

  printf("[MPI Rank %d] Completed kernel\n", rank);

  // cleanup
#ifdef USE_IR
  fini_cumodule(&mymodule);
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
