#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <stdbool.h>

// CLI argument structure
typedef struct {
    int verify;          // -v flag
    int warmup_iters;    // -w flag
    int normal_iters;    // -i flag
    size_t begin_size;   // -b flag
    size_t end_size;     // -e flag
} cli_args_t;

// Function to parse CLI arguments
static void parse_cli_args(int argc, char* argv[], cli_args_t* args) {
    // Default values
    args->verify = 0;
    args->warmup_iters = 50;
    args->normal_iters = 500;
    args->begin_size = sizeof(int);
    args->end_size = 4 * 1024 * 1024;  // 4MB

    int opt;
    while ((opt = getopt(argc, argv, "vw:i:b:e:")) != -1) {
        switch (opt) {
            case 'v':
                args->verify = 1;
                break;
            case 'w':
                args->warmup_iters = atoi(optarg);
                break;
            case 'i':
                args->normal_iters = atoi(optarg);
                break;
            case 'b':
                args->begin_size = atoll(optarg);
                break;
            case 'e':
                args->end_size = atoll(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s [-v] [-w warmup_iters] [-i normal_iters] [-b begin_size] [-e end_size]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }
}

// Error checking macros
#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Helper functions
static uint64_t getHostHash(const char* string) {
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  memset(hostname, 0, maxlen);
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen && hostname[i] != '\0'; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

// Data verification functions
static void initialize_and_verify_data(int *data_d, int mype, int iter, size_t size, int nelems) {
    // Initialize data with expected pattern
    int *host_data = (int *)malloc(size);
    for (int j = 0; j < nelems; j++) {
        // For PE 0: start with (iter * 100 + j)
        // For PE 1: start with (iter * 100 + j + 1)
        host_data[j] = (mype == 0) ? (iter * 100 + j) : (iter * 100 + j + 1);
    }
    CUDACHECK(cudaMemcpy(data_d, host_data, size, cudaMemcpyHostToDevice));
    free(host_data);
}

static bool verify_data_pattern(int *data_d, int mype, int iter, size_t size, int nelems) {
    int *host_data = (int *)malloc(size);
    CUDACHECK(cudaMemcpy(host_data, data_d, size, cudaMemcpyDeviceToHost));

    // Verify data pattern
    bool verification_passed = true;
    for (int j = 0; j < nelems; j++) {
        // After ping-pong, each PE should have the other PE's data
        int expected_value = (mype == 0) ? (iter * 100 + j + 1) : (iter * 100 + j);
        if (host_data[j] != expected_value) {
            printf("PE %d: Data verification failed at element %d. Expected %d, got %d\n",
                   mype, j, expected_value, host_data[j]);
            verification_passed = false;
            break;
        }
    }

    if (verification_passed) {
        printf("PE %d: Data verification passed for size %zu\n", mype, size);
    }

    free(host_data);
    return verification_passed;
}

static inline void initialize(int* sendbuff, int* recvbuff, int rank, int nelems) {
    int send_val = 0x100 + rank;        // PE0: 0x100, PE1: 0x101
    int recv_val = 0x200 + rank;        // PE0: 0x200, PE1: 0x201
    for (int i = 0; i < nelems; ++i) {
        sendbuff[i] = send_val;
        recvbuff[i] = recv_val;
    }
}

static inline bool verify(int* sendbuff, int* recvbuff, int nelems, int rank) {
    // After ping-pong, recvbuff should contain the peer's original sendbuff data
    int expected_received_val = 0x100 + (!rank);  // Peer's original send value

    for (int i = 0; i < nelems; ++i) {
        if (recvbuff[i] != expected_received_val) {
            printf("[Rank %d] ✗ VERIFICATION FAILED at index %d: expected=0x%X (%d), got=0x%X (%d)\n",
                   rank, i, expected_received_val, expected_received_val, recvbuff[i], recvbuff[i]);
            return false;
        }
    }
    printf("[Rank %d] ✓ VERIFICATION PASSED: All %d elements correct\n", rank, nelems);
    return true;
}

#endif // COMMON_H
