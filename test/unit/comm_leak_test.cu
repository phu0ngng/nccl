// Single node reproducer for https://nvbugs/3363300

/*

nvcc -g -o comm_leak_test comm_leak_test.cu --compiler-options="-fsanitize=leak" -I $NCCL_HOME/include -L$NCCL_HOME/lib -lnccl

*/

#include <nccl.h>

#include <iostream>
#include <cassert>
#include <stdexcept>

#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>

#include <dirent.h>
int count_open_fds(void) {
  DIR *dp = opendir("/proc/self/fd");
  int count = -3; // Exclude '.', '..', dp

  if (dp == NULL)
    return -1;

  while (readdir(dp) != NULL)
    count++;

  (void)closedir(dp);

  return count;
}

#define MAX_GPUS (32)

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

int main(int argc, char** argv)
{
    int num_gpus = 0;
    size_t reps = 3;
    size_t warmup = 1;
    int abort = 0;

    // Make sure everyline is flushed so that we see the progress of the test
    setlinebuf(stdout);

    if (argc > 1) reps = atoi(argv[1]);
    if (argc > 2) num_gpus = atoi(argv[2]);
    if (argc > 3) warmup = atoi(argv[3]);
    if (argc > 4) abort = atoi(argv[4]);

    if (num_gpus == 0)
      CUDA_TRY(cudaGetDeviceCount(&num_gpus));

    printf("Starting test on %d gpus reps %zi warmup %zi abort %d\n", num_gpus, reps, warmup, abort);

    int dev_list[MAX_GPUS];
    for (int i = 0; i < num_gpus; i++) {
      dev_list[i] = i;
    }

    // Allocate and destroy an initial NCCL communicator
    // before sampling the CUDA free memory
    for (size_t i = 0; i < warmup; ++i) {
      ncclComm_t nccl_comm[MAX_GPUS];
      NCCL_TRY(ncclCommInitAll(nccl_comm, num_gpus, dev_list));
      for (int g = 0; g < num_gpus; g++)
        NCCL_TRY(abort ? ncclCommAbort(nccl_comm[g]) :ncclCommDestroy(nccl_comm[g]));
    }

    // Sample the amount of free CUDA memory on all devices
    size_t free1[MAX_GPUS];
    size_t total;
    for (int i = 0; i < num_gpus; i++) {
      CUDA_TRY(cudaSetDevice(i));
      CUDA_TRY(cudaMemGetInfo(&free1[i], &total));
    }
    int startOpenFds = count_open_fds();

    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);

    for (size_t i = 0; i < reps; ++i) {
      ncclComm_t nccl_comm[MAX_GPUS];
      if ((i % 10) == 0) {
        struct timeval now;
        double elapsed;
        gettimeofday(&now, NULL);
        elapsed = (now.tv_sec-start.tv_sec)*1.0 + (now.tv_usec-start.tv_usec)*1.0E-6;
        printf("Doing iteration %zi elapsed time %gs\n", i, elapsed);
      }
      NCCL_TRY(ncclCommInitAll(nccl_comm, num_gpus, dev_list));
      for (int g = 0; g < num_gpus; g++)
        NCCL_TRY(abort ? ncclCommAbort(nccl_comm[g]) : ncclCommDestroy(nccl_comm[g]));
    }

    gettimeofday(&end, NULL);

    double time = (end.tv_sec-start.tv_sec)*1.0 + (end.tv_usec-start.tv_usec)*1.0E-6;
    printf("%d gpus %zi reps took %gs time per rep %gs\n", num_gpus, reps, time, time/reps);

    // Sample the amount of free CUDA memory on all devices
    size_t leaked{0};
    size_t free2[MAX_GPUS];
    for (int i = 0; i < num_gpus; i++) {
      CUDA_TRY(cudaSetDevice(i));
      CUDA_TRY(cudaMemGetInfo(&free2[i], &total));
      //printf("GPU %d free1 %zi free2 %zi leaked %zi\n", i, free1[i], free2[i], free1[i]-free2[i]);
      if (free2[i] < free1[i]) leaked += free1[i]-free2[i];
    }

    cudaDeviceReset();

    // Only report leaks of > 1 CUDA page
    if (leaked > (2*1024*1024)) {
      printf("ERROR: GPU Memory leaked %zi bytes (%zi MiB) CUDA memory over %zi iterations on %d gpus\n", leaked, leaked/(1024*1024), reps, num_gpus);
      exit(EXIT_FAILURE);
    }

    int endOpenFds = count_open_fds();
    if ((endOpenFds-startOpenFds) > 0) {
      printf("ERROR: File Descriptor leaked %d open fds over %zi iterations on %d gpus\n", endOpenFds-startOpenFds, reps, num_gpus);
      exit(EXIT_FAILURE);
    }

    printf("SUCCESS: Completed test of %zi iterations on %d gpus - no CUDA memory leaks detected\n", reps, num_gpus);
    exit (EXIT_SUCCESS);
}
