#include "ncclCommon_test.cuh"
#include "nccl_device.h"

class ncclCommWindowRegister_test : public ::testing::Test {
  protected:
    ncclComm_t *comms;
    int nVis;
    void **sendbuffs;
    void **recvbuffs;
    cudaStream_t *streams;
    const size_t size = 4 << 20; // 4MB
  virtual void SetUp() {
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));
    sendbuffs = (void**)calloc(nVis, sizeof(void*));
    recvbuffs = (void**)calloc(nVis, sizeof(void*));
    streams = (cudaStream_t*)calloc(nVis, sizeof(cudaStream_t));
    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      ASSERT_EQ(ncclSuccess, ncclMemAlloc(&sendbuffs[i], size));
      ASSERT_EQ(ncclSuccess, ncclMemAlloc(&recvbuffs[i], size));
      ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }
  }
  virtual void TearDown() {
    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streams[i]));
      ASSERT_EQ(ncclSuccess, ncclMemFree(sendbuffs[i]));
      ASSERT_EQ(ncclSuccess, ncclMemFree(recvbuffs[i]));
      ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    }
    free(sendbuffs);
    free(recvbuffs);
    free(streams);
    free(comms);
  }
};

TEST_F(ncclCommWindowRegister_test, basic) {
  ncclWindow_t *sendwins = (ncclWindow_t*)calloc(nVis, sizeof(ncclWindow_t));
  ncclWindow_t *recvwins = (ncclWindow_t*)calloc(nVis, sizeof(ncclWindow_t));
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], sendbuffs[i], size, &sendwins[i], NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], recvbuffs[i], size, &recvwins[i], NCCL_WIN_COLL_SYMMETRIC));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclAllReduce(sendbuffs[i], recvbuffs[i], size, ncclInt8, ncclSum, comms[i], streams[i]));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));
  }

  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclCommWindowDeregister(comms[i], sendwins[i]));
    ASSERT_EQ(ncclSuccess, ncclCommWindowDeregister(comms[i], recvwins[i]));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  free(sendwins);
  free(recvwins);
}

TEST_F(ncclCommWindowRegister_test, debug_mode) {
  ncclWindow_t *sendwins = (ncclWindow_t*)calloc(nVis, sizeof(ncclWindow_t));
  ncclWindow_t *recvwins = (ncclWindow_t*)calloc(nVis, sizeof(ncclWindow_t));

  setenv("NCCL_CHECK_MODE", "DEBUG_GLOBAL", 1);
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], sendbuffs[i], size, &sendwins[i], NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], recvbuffs[i], size, &recvwins[i], NCCL_WIN_COLL_SYMMETRIC));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclAllGather((uint8_t*)recvbuffs[i] + i * 1024, recvbuffs[i], 1024, ncclInt8, comms[i], streams[i]));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));
  }

  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclCommWindowDeregister(comms[i], sendwins[i]));
    ASSERT_EQ(ncclSuccess, ncclCommWindowDeregister(comms[i], recvwins[i]));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  free(sendwins);
  free(recvwins);
  unsetenv("NCCL_CHECK_MODE");
}

TEST_F(ncclCommWindowRegister_test, debug_mode_invalid) {
  ncclWindow_t *sendwins = (ncclWindow_t*)calloc(nVis, sizeof(ncclWindow_t));
  ncclWindow_t *recvwins = (ncclWindow_t*)calloc(nVis, sizeof(ncclWindow_t));
  ncclComm_t *localcomms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));

  setenv("NCCL_CHECK_MODE", "DEBUG_GLOBAL", 1);
  ASSERT_EQ(ncclSuccess, ncclCommInitAll(localcomms, nVis, NULL));
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(localcomms[i], sendbuffs[i], size, &sendwins[i], NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(localcomms[i], recvbuffs[i], size, &recvwins[i], NCCL_WIN_COLL_SYMMETRIC));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclAllReduce((uint8_t*)sendbuffs[i] + i * 1024, recvbuffs[i], 1024, ncclFloat32, ncclSum, localcomms[i], streams[i]));
  }
  if (sendwins[0] == nullptr || recvwins[0] == nullptr || nVis == 1) {
    // the platform does not support symmetric registration, so the group end should be successful
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  } else {
    ASSERT_EQ(ncclInvalidArgument, ncclGroupEnd());
  }

  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclCommWindowDeregister(localcomms[i], sendwins[i]));
    ASSERT_EQ(ncclSuccess, ncclCommWindowDeregister(localcomms[i], recvwins[i]));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclCommAbort(localcomms[i]));
  }
  free(localcomms);
  free(sendwins);
  free(recvwins);
  unsetenv("NCCL_CHECK_MODE");
}

TEST_F(ncclCommWindowRegister_test, win_get_user_ptr) {
  ncclWindow_t *wins = (ncclWindow_t*)calloc(nVis, sizeof(ncclWindow_t));
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], sendbuffs[i], size, &wins[i], NCCL_WIN_COLL_SYMMETRIC));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  // Verify ncclWinGetUserPtr returns the original buffer pointer for each rank
  for (int i = 0; i < nVis; i++) {
    void *userPtr = NULL;
    ASSERT_EQ(ncclSuccess, ncclWinGetUserPtr(comms[i], wins[i], &userPtr));
    ASSERT_EQ(userPtr, sendbuffs[i]) << "ncclWinGetUserPtr returned wrong pointer for rank " << i;
  }

  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclCommWindowDeregister(comms[i], wins[i]));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  free(wins);
}

TEST_F(ncclCommWindowRegister_test, win_get_user_ptr_multiple_windows) {
  ncclWindow_t *sendwins = (ncclWindow_t*)calloc(nVis, sizeof(ncclWindow_t));
  ncclWindow_t *recvwins = (ncclWindow_t*)calloc(nVis, sizeof(ncclWindow_t));
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], sendbuffs[i], size, &sendwins[i], NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], recvbuffs[i], size, &recvwins[i], NCCL_WIN_COLL_SYMMETRIC));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  // Verify ncclWinGetUserPtr returns correct pointers for both windows
  for (int i = 0; i < nVis; i++) {
    void *sendPtr = NULL, *recvPtr = NULL;
    ASSERT_EQ(ncclSuccess, ncclWinGetUserPtr(comms[i], sendwins[i], &sendPtr));
    ASSERT_EQ(ncclSuccess, ncclWinGetUserPtr(comms[i], recvwins[i], &recvPtr));
    ASSERT_EQ(sendPtr, sendbuffs[i]) << "Send window user ptr mismatch for rank " << i;
    ASSERT_EQ(recvPtr, recvbuffs[i]) << "Recv window user ptr mismatch for rank " << i;
    ASSERT_NE(sendPtr, recvPtr) << "Send and recv windows should have different user ptrs";
  }

  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclCommWindowDeregister(comms[i], sendwins[i]));
    ASSERT_EQ(ncclSuccess, ncclCommWindowDeregister(comms[i], recvwins[i]));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  free(sendwins);
  free(recvwins);
}

// ============================================================================
// Negative tests for ncclCommWindowRegister
// ============================================================================

TEST_F(ncclCommWindowRegister_test, register_null_buff) {
  ncclWindow_t win;
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  ASSERT_EQ(ncclInvalidArgument, ncclCommWindowRegister(comms[0], NULL, size, &win, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());
}

TEST_F(ncclCommWindowRegister_test, register_zero_size) {
  ncclWindow_t win;
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  ASSERT_EQ(ncclInvalidArgument, ncclCommWindowRegister(comms[0], sendbuffs[0], 0, &win, NCCL_WIN_COLL_SYMMETRIC));
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());
}

TEST_F(ncclCommWindowRegister_test, register_null_win_ptr) {
  ASSERT_EQ(ncclInvalidArgument, ncclCommWindowRegister(comms[0], sendbuffs[0], size, NULL, NCCL_WIN_COLL_SYMMETRIC));
}

// ============================================================================
// Negative tests for ncclCommWindowDeregister
// ============================================================================

TEST_F(ncclCommWindowRegister_test, deregister_null_win) {
  // Deregistering NULL window should be a no-op (succeeds silently)
  ASSERT_EQ(ncclSuccess, ncclCommWindowDeregister(comms[0], NULL));
}
