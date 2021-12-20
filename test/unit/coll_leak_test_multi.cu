// Multi process/node reproducer for https://nvbugs/3363300

/*

mpicc -g -o coll_leak_test_multi coll_leak_test_multi.cc -I$CUDA_HOME/include -L$CUDA_HOME/lib64 -I$NCCL_HOME/include -L$NCCL_HOME/lib -lnccl -lcudart

*/

#include <nccl.h>
#include <mpi.h>

#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

#define MAX_GPUS (32)

#define MPI_TRY(call)                        \
  do {                                       \
    int status = call;                       \
    if (MPI_SUCCESS != status) {             \
      fprintf(stderr,"MPI call='%s' failed. Error %d\n", #call, status); \
      exit(EXIT_FAILURE);                    \
    }                                        \
  } while (0)

#define CUDA_TRY(call)                        \
  do {                                        \
    cudaError_t const status = (call);        \
    if (cudaSuccess != status) {              \
      fprintf(stderr,"CUDA call='%s' failed. Error %s (%d)\n", #call, cudaGetErrorString(status), status); \
      exit(EXIT_FAILURE);                     \
    }                                         \
  } while (0)

#define NCCL_TRY(call)                                                                  \
  do {                                                                                  \
    ncclResult_t const status = (call);                                                 \
    if (ncclSuccess != status) {                                                        \
      fprintf(stderr,"NCCL calll='%s' failed. Reason:%s\n", #call, ncclGetErrorString(status)); \
      exit(EXIT_FAILURE);                                                               \
    }                                                                                   \
  } while (0)

float* sendbuff[MAX_GPUS];
float* recvbuff[MAX_GPUS];
cudaStream_t s[MAX_GPUS];

ncclResult_t do_test(int comm_rank, int local_rank, int comm_size, int num_gpus, size_t size, size_t coll_reps, int abort, int alltoall)
{
  // Allocate a NCCL communicator
  ncclComm_t nccl_comm[MAX_GPUS];
  ncclUniqueId nccl_unique_id;
  if (comm_rank == 0) {
    NCCL_TRY(ncclGetUniqueId(&nccl_unique_id));
  }
  MPI_TRY(MPI_Bcast(&nccl_unique_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
  MPI_Barrier(MPI_COMM_WORLD);

  NCCL_TRY(ncclGroupStart());
  for (int g = 0; g < num_gpus; g++) {
    CUDA_TRY(cudaSetDevice((local_rank*num_gpus)+g));
    NCCL_TRY(ncclCommInitRank(&nccl_comm[g], comm_size*num_gpus, nccl_unique_id, (comm_rank*num_gpus)+g));
  }
  NCCL_TRY(ncclGroupEnd());

  MPI_Barrier(MPI_COMM_WORLD);

  // Perform collective call
  for (size_t i = 0; i < coll_reps; i++) {
    NCCL_TRY(ncclGroupStart());
    for (int g = 0; g < num_gpus; g++) {
      CUDA_TRY(cudaSetDevice((local_rank*num_gpus)+g));
      if (alltoall) {
        size_t rankOffset = (size/comm_size)*sizeof(float);
        for (int r=0; r<comm_size; r++) {
          NCCL_TRY(ncclSend(((char*)sendbuff[g])+r*rankOffset, size/comm_size, ncclFloat, r, nccl_comm[g], s[g]));
          NCCL_TRY(ncclRecv(((char*)recvbuff[g])+r*rankOffset, size/comm_size, ncclFloat, r, nccl_comm[g], s[g]));
        }
      } else {
        NCCL_TRY(ncclAllReduce((const void*)sendbuff[g], (void*)recvbuff[g], size, ncclFloat, ncclSum,
                               nccl_comm[g], s[g]));
      }
    }
    NCCL_TRY(ncclGroupEnd());
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (!abort) {
    //synchronize on CUDA streams to wait for completion of NCCL operations
    for (int g = 0; g < num_gpus; g++) {
      CUDA_TRY(cudaStreamSynchronize(s[g]));
    }

    for (int g = 0; g < num_gpus; g++) {
      NCCL_TRY(ncclCommDestroy(nccl_comm[g]));
    }
  }
  else {
    for (int g = 0; g < num_gpus; g++) {
      NCCL_TRY(ncclCommAbort(nccl_comm[g]));
    }

    //synchronize on CUDA streams to wait for completion of NCCL operations
    for (int g = 0; g < num_gpus; g++) {
      CUDA_TRY(cudaStreamSynchronize(s[g]));
    }
  }

  return ncclSuccess;
}

int main(int argc, char** argv)
{
    MPI_TRY(MPI_Init(&argc, &argv));

    // Determine COMM_WORLD rank and size
    int comm_rank = 0;
    int comm_size = 0;
    MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));
    MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));

    // Determine number of ranks per node
    int local_rank = 0, local_size = 0;
    MPI_Comm lcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &lcomm);
    MPI_Comm_rank(lcomm, &local_rank);
    MPI_Comm_size(lcomm, &local_size);
    MPI_Comm_free(&lcomm);

    int num_gpus = 1, phys_num_gpus = 0;
    size_t size = (1024*1024);
    size_t reps = 20;
    size_t coll_reps = 100;
    size_t warmup = 2;
    int abort = 0;
    int alltoall = 1;

    if (argc > 1) reps = atoi(argv[1]);
    if (argc > 2) num_gpus = atoi(argv[2]);
    if (argc > 3) warmup = atoi(argv[3]);
    if (argc > 4) abort = atoi(argv[4]);
    if (argc > 5) coll_reps = atoi(argv[5]);
    if (argc > 6) alltoall = atoi(argv[6]);

    CUDA_TRY(cudaGetDeviceCount(&phys_num_gpus));

    if (num_gpus == 0) num_gpus = phys_num_gpus;
    assert(num_gpus <= phys_num_gpus);
    assert(local_size*num_gpus <= phys_num_gpus);

    if (comm_rank == 0) printf("Starting test on %d ranks (local %d) gpus %d reps %zi warmup %zi coll_reps %zi abort %d alltoall %d\n",
                               comm_size, local_size, num_gpus, reps, warmup, coll_reps, abort, alltoall);

    for (int g = 0; g < num_gpus; g++) {
      CUDA_TRY(cudaSetDevice((local_rank*num_gpus)+g));
      CUDA_TRY(cudaMalloc(sendbuff+g, size * sizeof(float)));
      CUDA_TRY(cudaMalloc(recvbuff+g, size * sizeof(float)));
      CUDA_TRY(cudaMemset(sendbuff[g], 1, size * sizeof(float)));
      CUDA_TRY(cudaMemset(recvbuff[g], 0, size * sizeof(float)));
      CUDA_TRY(cudaStreamCreateWithFlags(&s[g], cudaStreamNonBlocking));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t i = 0; i < warmup; ++i) {
      do_test(comm_rank, local_rank, comm_size, num_gpus, size, 1, 0, alltoall);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Sample the amount of free CUDA memory on all devices
    size_t free1[MAX_GPUS];
    size_t total;
    for (int g = 0; g < num_gpus; g++) {
      CUDA_TRY(cudaSetDevice((local_rank*num_gpus)+g));
      CUDA_TRY(cudaMemGetInfo(&free1[g], &total));
    }

    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);

    for (size_t i = 0; i < reps; ++i) {
      if (comm_rank == 0) printf("Starting rep %zi/%zi\n", i, reps);
      do_test(comm_rank, local_rank, comm_size, num_gpus, size, coll_reps, abort, alltoall);

      MPI_Barrier(MPI_COMM_WORLD);
    }

    gettimeofday(&end, NULL);

    double time = (end.tv_sec-start.tv_sec)*1.0 + (end.tv_usec-start.tv_usec)*1.0E-6;
    if (comm_rank == 0) printf("%d gpus %zi reps took %gs time per rep %gs\n", num_gpus, reps, time, time/reps);

    MPI_Barrier(MPI_COMM_WORLD);

    // Sample the amount of free CUDA memory on all devices
    size_t leaked = 0;
    size_t free2[MAX_GPUS];
    for (int g = 0; g < num_gpus; g++) {
      CUDA_TRY(cudaSetDevice((local_rank*num_gpus)+g));
      CUDA_TRY(cudaMemGetInfo(&free2[g], &total));
      //printf("rank %d GPU %d free1 %zi free2 %zi leaked %zi\n", comm_rank, g, free1[g], free2[g], free1[g]-free2[g]);
      // Only record leaks of > 1 CUDA page
      if ((free1[g] - free2[g]) > (2*1024*1024)) leaked += free1[g]-free2[g];
    }

    MPI_TRY(MPI_Allreduce(MPI_IN_PLACE, &leaked, sizeof(leaked), MPI_LONG, MPI_SUM, MPI_COMM_WORLD));

    if (leaked) {
      if (comm_rank == 0) printf("ERROR: total leaked %zi bytes (%zi MiB) CUDA memory over %zi iterations on %d gpus\n", leaked, leaked/(1024*1024), reps, num_gpus);
      exit(EXIT_FAILURE);
    }

    for (int g = 0; g < num_gpus; g++) {
      CUDA_TRY(cudaFree(sendbuff[g]));
      CUDA_TRY(cudaFree(recvbuff[g]));
      CUDA_TRY(cudaStreamDestroy(s[g]));
    }

    MPI_TRY(MPI_Finalize());
    cudaDeviceReset();

    if (comm_rank == 0) printf("SUCCESS: Completed test on %d ranks of %zi iterations on %d gpus per node - no CUDA memory leaks detected\n", comm_size, reps, local_size*num_gpus);
    exit (EXIT_SUCCESS);
}
