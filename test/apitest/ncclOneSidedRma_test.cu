#include "ncclCommon_test.cuh"
#include "nccl_device.h"

// ============================================================================
// One-Sided RMA test fixture
// Communicators are created with numRmaCtx = 1 to enable RMA operations.
// Requires at least 2 GPUs; tests are skipped otherwise.
// ============================================================================

class ncclOneSidedRma_test : public ::testing::Test {
  protected:
    ncclComm_t *comms = NULL;
    int nVis = 0;
    void **sendbuffs = NULL;
    void **recvbuffs = NULL;
    ncclWindow_t *sendwins = NULL;
    ncclWindow_t *recvwins = NULL;
    cudaStream_t *streams = NULL;
    const size_t size = 4 << 20; // 4MB
    const int ctx = 0;

  virtual void SetUp() {
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
    if (nVis < 2) return;

    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    sendbuffs = (void**)calloc(nVis, sizeof(void*));
    recvbuffs = (void**)calloc(nVis, sizeof(void*));
    sendwins = (ncclWindow_t*)calloc(nVis, sizeof(ncclWindow_t));
    recvwins = (ncclWindow_t*)calloc(nVis, sizeof(ncclWindow_t));
    streams = (cudaStream_t*)calloc(nVis, sizeof(cudaStream_t));

    // Create communicators with RMA support
    ncclUniqueId id;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.numRmaCtx = 1;
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    // Allocate symmetric buffers and create streams
    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
      ASSERT_EQ(ncclSuccess, ncclMemAlloc(&sendbuffs[i], size));
      ASSERT_EQ(ncclSuccess, ncclMemAlloc(&recvbuffs[i], size));
      ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    // Register windows collectively
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], sendbuffs[i], size, &sendwins[i], NCCL_WIN_COLL_SYMMETRIC));
      ASSERT_EQ(ncclSuccess, ncclCommWindowRegister(comms[i], recvbuffs[i], size, &recvwins[i], NCCL_WIN_COLL_SYMMETRIC));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  }

  virtual void TearDown() {
    if (comms == NULL) return;

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(ncclSuccess, ncclCommWindowDeregister(comms[i], sendwins[i]));
      ASSERT_EQ(ncclSuccess, ncclCommWindowDeregister(comms[i], recvwins[i]));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < nVis; i++) {
      ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streams[i]));
      ASSERT_EQ(ncclSuccess, ncclMemFree(sendbuffs[i]));
      ASSERT_EQ(ncclSuccess, ncclMemFree(recvbuffs[i]));
      ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    }
    free(sendbuffs);
    free(recvbuffs);
    free(sendwins);
    free(recvwins);
    free(streams);
    free(comms);
  }
};

// Test ncclPutSignal + ncclWaitSignal: ring put with data verification
TEST_F(ncclOneSidedRma_test, put_signal_basic) {
  if (nVis < 2) return;

  const size_t count = 1024; // 1024 ints = 4KB

  // Initialize send buffers with rank-specific values, zero recv buffers
  for (int i = 0; i < nVis; i++) {
    int *hostbuf = (int*)malloc(count * sizeof(int));
    for (size_t j = 0; j < count; j++) hostbuf[j] = 0x1000 + i;
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(sendbuffs[i], hostbuf, count * sizeof(int), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(recvbuffs[i], 0, count * sizeof(int)));
    free(hostbuf);
  }

  // Each rank puts to downstream peer's recvWindow
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    int downstream = (i + 1) % nVis;
    ASSERT_EQ(ncclSuccess, ncclPutSignal(sendbuffs[i], count, ncclInt, downstream, recvwins[i], 0,
                                          0, ctx, 0, comms[i], streams[i]));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  // Each rank waits for signal from upstream peer
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    int upstream = (i - 1 + nVis) % nVis;
    ncclWaitSignalDesc_t waitDesc = {.opCnt = 1, .peer = upstream, .sigIdx = 0, .ctx = ctx};
    ASSERT_EQ(ncclSuccess, ncclWaitSignal(1, &waitDesc, comms[i], streams[i]));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));
  }

  // Verify: each rank's recvbuff should contain upstream's sendbuff values
  for (int i = 0; i < nVis; i++) {
    int upstream = (i - 1 + nVis) % nVis;
    int expected = 0x1000 + upstream;
    int *hostbuf = (int*)malloc(count * sizeof(int));
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(hostbuf, recvbuffs[i], count * sizeof(int), cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < count; j++) {
      ASSERT_EQ(hostbuf[j], expected)
        << "Rank " << i << " recvbuff mismatch at index " << j;
    }
    free(hostbuf);
  }
}

// Test ncclSignal + ncclWaitSignal without data transfer
TEST_F(ncclOneSidedRma_test, signal_only) {
  if (nVis < 2) return;

  // Each rank signals its downstream peer (no data)
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    int downstream = (i + 1) % nVis;
    ASSERT_EQ(ncclSuccess, ncclSignal(downstream, 0, ctx, 0, comms[i], streams[i]));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  // Each rank waits for signal from upstream peer
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    int upstream = (i - 1 + nVis) % nVis;
    ncclWaitSignalDesc_t waitDesc = {.opCnt = 1, .peer = upstream, .sigIdx = 0, .ctx = ctx};
    ASSERT_EQ(ncclSuccess, ncclWaitSignal(1, &waitDesc, comms[i], streams[i]));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  // All streams should complete (signals were delivered and received)
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));
  }
}

// Test ncclWaitSignal with multiple descriptors (all-to-all signaling pattern)
TEST_F(ncclOneSidedRma_test, wait_signal_multi_peer) {
  if (nVis < 2) return;

  // All-to-all signal: each rank signals every other rank
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    for (int peer = 0; peer < nVis; peer++) {
      if (peer == i) continue;
      ASSERT_EQ(ncclSuccess, ncclSignal(peer, 0, ctx, 0, comms[i], streams[i]));
    }
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  // Each rank waits for signals from all other ranks using multi-descriptor wait
  ncclWaitSignalDesc_t *descs = (ncclWaitSignalDesc_t*)calloc(nVis - 1, sizeof(ncclWaitSignalDesc_t));
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    int nDesc = 0;
    for (int peer = 0; peer < nVis; peer++) {
      if (peer == i) continue;
      descs[nDesc].opCnt = 1;
      descs[nDesc].peer = peer;
      descs[nDesc].sigIdx = 0;
      descs[nDesc].ctx = ctx;
      nDesc++;
    }
    ASSERT_EQ(ncclSuccess, ncclWaitSignal(nDesc, descs, comms[i], streams[i]));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  free(descs);

  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));
  }
}

// ============================================================================
// Negative tests for ncclPutSignal
// ============================================================================

TEST_F(ncclOneSidedRma_test, put_signal_null_localbuff) {
  if (nVis < 2) return;

  ASSERT_EQ(ncclInvalidArgument, ncclPutSignal(NULL, 1024, ncclInt, 1, recvwins[0], 0,
                                                0, ctx, 0, comms[0], streams[0]));
}

TEST_F(ncclOneSidedRma_test, put_signal_null_peer_win) {
  if (nVis < 2) return;

  ASSERT_EQ(ncclInvalidArgument, ncclPutSignal(sendbuffs[0], 1024, ncclInt, 1, NULL, 0,
                                                0, ctx, 0, comms[0], streams[0]));
}

TEST_F(ncclOneSidedRma_test, put_signal_non_window_buffer) {
  if (nVis < 2) return;

  // Allocate a buffer that is NOT registered in any window
  void *nonWinBuf = NULL;
  ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&nonWinBuf, 4096));

  ASSERT_EQ(ncclInvalidArgument, ncclPutSignal(nonWinBuf, 1024, ncclInt, 1, recvwins[0], 0,
                                                0, ctx, 0, comms[0], streams[0]));

  ASSERT_EQ(cudaSuccess, cudaFree(nonWinBuf));
}

TEST_F(ncclOneSidedRma_test, put_signal_invalid_peer) {
  if (nVis < 2) return;

  // Peer == -1 (negative)
  ASSERT_EQ(ncclInvalidArgument, ncclPutSignal(sendbuffs[0], 1024, ncclInt, -1, recvwins[0], 0,
                                                0, ctx, 0, comms[0], streams[0]));

  // Peer == nVis (out of range)
  ASSERT_EQ(ncclInvalidArgument, ncclPutSignal(sendbuffs[0], 1024, ncclInt, nVis, recvwins[0], 0,
                                                0, ctx, 0, comms[0], streams[0]));
}

// ============================================================================
// Negative tests for ncclSignal
// ============================================================================

TEST_F(ncclOneSidedRma_test, signal_invalid_peer) {
  if (nVis < 2) return;

  // Peer == -1 (negative)
  ASSERT_EQ(ncclInvalidArgument, ncclSignal(-1, 0, ctx, 0, comms[0], streams[0]));

  // Peer == nVis (out of range)
  ASSERT_EQ(ncclInvalidArgument, ncclSignal(nVis, 0, ctx, 0, comms[0], streams[0]));
}

// ============================================================================
// Negative tests for ncclWaitSignal
// ============================================================================

TEST_F(ncclOneSidedRma_test, wait_signal_null_descs) {
  if (nVis < 2) return;

  ASSERT_EQ(ncclInvalidArgument, ncclWaitSignal(1, NULL, comms[0], streams[0]));
}

TEST_F(ncclOneSidedRma_test, wait_signal_zero_ndesc) {
  if (nVis < 2) return;

  ncclWaitSignalDesc_t desc = {.opCnt = 1, .peer = 1, .sigIdx = 0, .ctx = 0};
  ASSERT_EQ(ncclInvalidArgument, ncclWaitSignal(0, &desc, comms[0], streams[0]));
}

TEST_F(ncclOneSidedRma_test, wait_signal_invalid_opcnt) {
  if (nVis < 2) return;

  // opCnt == 0
  ncclWaitSignalDesc_t desc = {.opCnt = 0, .peer = 1, .sigIdx = 0, .ctx = 0};
  ASSERT_EQ(ncclInvalidArgument, ncclWaitSignal(1, &desc, comms[0], streams[0]));

  // opCnt == -1
  desc.opCnt = -1;
  ASSERT_EQ(ncclInvalidArgument, ncclWaitSignal(1, &desc, comms[0], streams[0]));
}

TEST_F(ncclOneSidedRma_test, wait_signal_invalid_sigidx) {
  if (nVis < 2) return;

  ncclWaitSignalDesc_t desc = {.opCnt = 1, .peer = 1, .sigIdx = 1, .ctx = 0};
  ASSERT_EQ(ncclInvalidArgument, ncclWaitSignal(1, &desc, comms[0], streams[0]));
}

TEST_F(ncclOneSidedRma_test, wait_signal_invalid_ctx) {
  if (nVis < 2) return;

  ncclWaitSignalDesc_t desc = {.opCnt = 1, .peer = 1, .sigIdx = 0, .ctx = 99};
  ASSERT_EQ(ncclInvalidArgument, ncclWaitSignal(1, &desc, comms[0], streams[0]));
}
