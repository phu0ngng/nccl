#include <cuda_runtime.h>

#include <vector>

#include "bitops.h"
#include "common.h"
#include "nccl.h"
#include "nccl_device.h"

__global__ void runDevice(ncclDevComm comm, ncclDevResourceHandle hGinDone) {
#if __CUDA_ARCH__ >= 700
  int t = threadIdx.x;
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin net(comm, 0);

  int* ginDone = (int*)ncclGetResourceBufferLocalPointer(comm, hGinDone);

  {
    ncclGinBarrierSession<ncclCoopCta> ginBar(ncclCoopCta(), net, ncclTeamTagRail(), blockIdx.x);
    for (int round = 0; round < 10; round++) {
      ginBar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::None);
      if (t == 0 && blockIdx.x == 0 && world.rank == round % world.nRanks) {
        printf("GIN Round %d\n", round);
      }
    }
  }

  if (t == 0) {
    atomicAdd(ginDone, 1);
  }
  __syncthreads();

  {
    ncclBarrierSession<ncclCoopCta> bar(ncclCoopCta(), ncclTeamTagWorld(), net, blockIdx.x);
    for (int round = 0; round < 10; round++) {
      bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::None);
      if (round == 0 && t == 0) {
        int done = atomicAdd(ginDone, 0);
        if (done != gridDim.x) {
          printf("ERROR [rank %d, block %d]: only %d/%d blocks exited GIN barrier "
                 "before world barrier sync\n", world.rank, blockIdx.x, done, gridDim.x);
          assert(0);
        }
      }
      if (t == 0 && blockIdx.x == 0 && world.rank == round % world.nRanks) {
        printf("World Round %d\n", round);
      }
    }
  }
#endif
}

int main(int argc, char** argv) {
  int rank, nRanks;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  uint64_t* hosts = new uint64_t[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hosts[rank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hosts, sizeof(uint64_t), MPI_BYTE,
                         MPI_COMM_WORLD));
  int dev = 0;
  for (int r = 0; r < rank; r++) {
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
  ncclDevResourceHandle hGinDone;
  {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.barrierCount = 16;
    reqs.railGinBarrierCount = 16;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;

    ncclDevResourceRequirements counterReq = {};
    counterReq.bufferSize = sizeof(int);
    counterReq.outBufferHandle = &hGinDone;
    counterReq.next = reqs.resourceRequirementsList;
    reqs.resourceRequirementsList = &counterReq;

    NCCLCHECK(ncclDevCommCreate(comm, &reqs, &dcomm));
  }

  printf("[MPI Rank %d] Starting kernel\n", rank);
  runDevice<<<16, 512, 0, stream>>>(dcomm, hGinDone);
  CUDACHECK(cudaStreamSynchronize(stream));
  printf("[MPI Rank %d] Completed kernel\n", rank);

  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclDevCommDestroy(comm, &dcomm));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  MPICHECK(MPI_Finalize());
  return 0;
}
