#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "bitops.h"
#include "common.h"

#include <cassert>

#ifdef USE_IR
#include <cuda.h>
#include "cumod_common.h"
#endif

//constexpr int BlockPerRank = 1;
constexpr int BlockPerRank = 16;

//constexpr int BufElts = 10;
//constexpr int BufElts = 129;
constexpr int BufElts = 1<<20;

constexpr bool Prints = false;
//constexpr bool Prints = true;

#ifndef USE_IR
__global__ void runDevice(ncclDevComm comm, ncclDevResourceHandle hbuf, int kernelNum) {
#if __CUDA_ARCH__ >= 700
  int t = threadIdx.x;
  int tn = blockDim.x;
  ncclTeam world = ncclTeamWorld(comm);
  int b = world.rank*gridDim.x + blockIdx.x;
  int bn = world.nRanks*gridDim.x;
  int up_rank = (b+bn-1)%bn / gridDim.x;
  int up_block = (b+bn-1)%bn % gridDim.x;
  int down_rank = (b+1)%bn / gridDim.x;
  int down_block = (b+1)%bn % gridDim.x;

  ncclSymPtr<int> sbuf = (ncclSymPtr<int>)ncclGetResourceBuffer(comm, hbuf);
  ncclSymPtr<int> rbuf = sbuf + gridDim.x*BufElts;
  unsigned sigData0 = 0;
  unsigned sigFree0 = gridDim.x;
  unsigned counter = blockIdx.x;
  uint64_t counterShadow = 0;

  ncclGin net(comm, 0);
  uint32_t roundLo = 10*kernelNum;
  uint32_t roundHi = 10*kernelNum + 10;
  if (Prints) printf("[%d:%d]: Kernel %d\n", world.rank, blockIdx.x, kernelNum);
  for (uint32_t round=roundLo; round < roundHi; round++) {
    if (Prints) printf("[%d:%d]: Round %d\n", world.rank, blockIdx.x, round);
    // Consume recv buf (rbuf) and populate send buf (sbuf)
    for (int i=t; i < BufElts; i += tn) {
      int got = (rbuf + blockIdx.x*BufElts).localPtr()[i];
      if (Prints) {
        if (got != (int)round) {
          printf("[%d:%d]: ERROR round=%d rbuf[%d]=%d but expected %d\n", world.rank, blockIdx.x, round, i, got, round);
          assert(0);
        }
      } else {
        assert(got == (int)round);
      }
      (sbuf + blockIdx.x*BufElts).localPtr()[i] = 1 + got; // Thus data received should equal its round index
    }
    __syncthreads();

    if (t==0) {
      // Send signal upstream to indicate free space
      net.signal(world, up_rank, ncclGin_SignalInc{sigFree0 + up_block});
    }
    // Wait for downstream to give us free space
    if (Prints) if (t==0) printf("[%d:%d] round=%d Send free signal\n", world.rank, blockIdx.x, round);
    net.waitSignal(ncclCoopCta(), sigFree0 + blockIdx.x, round+1);
    if (Prints) if (t==0) printf("[%d:%d] round=%d Got free signal\n", world.rank, blockIdx.x, round);

    // Send data downstream in chunks.
    int nChunks = min(BufElts, 1 + (round*0xdeadbeefu >> (32-10)));
    int chunkElts = BufElts/nChunks;
    nChunks = (BufElts + chunkElts-1)/chunkElts;
    if (Prints) if (t==0) printf("[%d:%d] round=%d Sending %d chunks\n", world.rank, blockIdx.x, round, nChunks);
    #pragma unroll 1
    for (int i=t; i < nChunks; i += tn) {
      net.increaseSignalShadow(sigData0 + blockIdx.x, 1);
      net.put(world, down_rank,
        rbuf + down_block*BufElts + i*chunkElts,
        sbuf + blockIdx.x*BufElts + i*chunkElts,
        min(chunkElts, BufElts - i*chunkElts),
        ncclGin_SignalInc{sigData0 + down_block},
        ncclGin_CounterInc{counter}
      );
    }
    if (Prints) if (t==0) printf("[%d:%d] round=%d wait chunks\n", world.rank, blockIdx.x, round);
    // Wait for all chunks from upstream.
    net.waitSignalMeetShadow(ncclCoopCta(), sigData0 + blockIdx.x);
    // Wait for all outgoing to settle
    net.waitCounter(ncclCoopCta(), counter, counterShadow + nChunks);
    counterShadow += nChunks;
    if (Prints) if (t==0) printf("[%d:%d] round=%d Got chunks\n", world.rank, blockIdx.x, round);
  }
  if (t==0) net.resetCounter(counter);
#endif
}
#else
extern "C" __global__ void runDevice(ncclDevComm comm, ncclDevResourceHandle hbuf, int kernelNum);
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
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = 2*BlockPerRank;
  reqs.ginCounterCount = BlockPerRank;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;

  ncclDevResourceHandle hBuf;
  ncclDevResourceRequirements bufReq = {};
  bufReq.bufferSize = 2*BlockPerRank*BufElts*sizeof(int);
  bufReq.outBufferHandle = &hBuf;
  bufReq.next = reqs.resourceRequirementsList;
  reqs.resourceRequirementsList = &bufReq;

  NCCLCHECK(ncclDevCommCreate(comm, &reqs, &dcomm));

  // run kernel
  printf("[MPI Rank %d] Starting kernels\n", rank);

#ifdef USE_IR
  // IR path: load kernel from cubin and launch
  CUmodule mymodule = NULL;
  initCumodule(&mymodule, "devapi_data_ring2_ir.cubin");
  CUfunction kernel;
  initTestCaseKernel(mymodule, &kernel, "runDevice");

  for (int kernelNum=0; kernelNum < 100; kernelNum++) {
    void* args[] = {&dcomm, &hBuf, &kernelNum};
    CU_CHECK(cuLaunchKernel(kernel, BlockPerRank, 1, 1, 512, 1, 1, 0, stream, args, NULL));
  }
  CU_CHECK(cuStreamSynchronize(stream));
#else
  for (int kernelNum=0; kernelNum < 100; kernelNum++) {
    runDevice<<<BlockPerRank, 512, 0, stream>>>(dcomm, hBuf, kernelNum);
  }
  CUDACHECK(cudaStreamSynchronize(stream));
#endif

  printf("[MPI Rank %d] Completed kernels\n", rank);

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
