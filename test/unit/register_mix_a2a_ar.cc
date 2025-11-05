#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// Manual test framework - simple assertion macros
#define NCCLCHECKGOTO(call, RES, label) do { \
  RES = call; \
  if (RES != ncclSuccess && RES != ncclInProgress) {  \
    /* Print the back trace*/                         \
    printf("%s:%d -> %d", __FILE__, __LINE__, RES);   \
    goto label; \
  } \
} while (0)

#define CUDACHECKGOTO(cmd, RES, label)                                         \
  do {                                                                         \
    RES = cmd;                                                                 \
    if (RES != cudaSuccess) {                                                  \
      printf("Cuda failure '%s'", cudaGetErrorString(RES));                    \
      goto label;                                                              \
    }                                                                          \
  } while (false)

// Main function to run the test
int main(int argc, char** argv) {
  int nVis = 0;
  int ret = 0;
  ncclComm_t* comms = nullptr;
  void** send_buffs = nullptr;
  void** recv_buffs = nullptr;
  void** handles = nullptr;
  cudaError_t cuda_result = cudaSuccess;
  ncclResult_t nccl_result = ncclSuccess;
  const int size = 1 << 28; // 256MB
  cudaStream_t* streams = nullptr;

  // Need to run AR with Ring algorithm to test the registration
  CUDACHECKGOTO(cudaGetDeviceCount(&nVis), cuda_result, fail);

  comms = (ncclComm_t*)calloc(sizeof(ncclComm_t), nVis);
  send_buffs = (void**)calloc(sizeof(void*), nVis);
  recv_buffs = (void**)calloc(sizeof(void*), nVis);
  handles = (void**)calloc(sizeof(void*), nVis);
  streams = (cudaStream_t*)calloc(sizeof(cudaStream_t), nVis);
  if (comms == nullptr || send_buffs == nullptr || recv_buffs == nullptr || handles == nullptr || streams == nullptr) {
    printf("calloc failed: returned comms %p or send_buffs %p or recv_buffs %p or handles %p stream %p nullptr", comms, send_buffs, recv_buffs, handles, streams);
    goto fail;
  }

  NCCLCHECKGOTO(ncclCommInitAll(comms, nVis, NULL), nccl_result, fail);

  // Test that registration is skipped when P2P_USE_CUDA_MEMCPY is enabled
  for (int i = 0; i < nVis; i++) {
    CUDACHECKGOTO(cudaSetDevice(i), cuda_result, fail);
    NCCLCHECKGOTO(ncclMemAlloc(&send_buffs[i], size), nccl_result, fail);
    NCCLCHECKGOTO(ncclMemAlloc(&recv_buffs[i], size), nccl_result, fail);
    NCCLCHECKGOTO(ncclCommRegister(comms[i], send_buffs[i], size, &handles[i]), nccl_result, fail);
    NCCLCHECKGOTO(ncclCommRegister(comms[i], recv_buffs[i], size, &handles[i]), nccl_result, fail);
    CUDACHECKGOTO(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking), cuda_result, fail);
  }

  NCCLCHECKGOTO(ncclGroupStart(), nccl_result, fail);
  for (int i = 0; i < nVis; i++) {
    for (int j = 0; j < nVis; j++) {
      NCCLCHECKGOTO(ncclSend(send_buffs[i], size, ncclInt8, j, comms[i], streams[i]), nccl_result, fail);
      NCCLCHECKGOTO(ncclRecv(recv_buffs[i], size, ncclInt8, j, comms[i], streams[i]), nccl_result, fail);
    }
  }
  NCCLCHECKGOTO(ncclGroupEnd(), nccl_result, fail);

  NCCLCHECKGOTO(ncclGroupStart(), nccl_result, fail);
  for (int i = 0; i < nVis; i++) {
    NCCLCHECKGOTO(ncclAllReduce(send_buffs[i], recv_buffs[i], size, ncclInt8, ncclSum, comms[i], streams[i]), nccl_result, fail);
  }
  NCCLCHECKGOTO(ncclGroupEnd(), nccl_result, fail);

  for (int i = 0; i < nVis; i++) {
    NCCLCHECKGOTO(ncclCommDeregister(comms[i], handles[i]), nccl_result, fail);
    NCCLCHECKGOTO(ncclMemFree(send_buffs[i]), nccl_result, fail);
    NCCLCHECKGOTO(ncclMemFree(recv_buffs[i]), nccl_result, fail);
    CUDACHECKGOTO(cudaStreamDestroy(streams[i]), cuda_result, fail);
  }

exit:
  free(comms);
  free(send_buffs);
  free(recv_buffs);
  free(handles);
  free(streams);
  return ret;
fail:
  ret = 1;
  goto exit;
}
