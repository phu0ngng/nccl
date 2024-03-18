#include "ncclCommon_test.cuh"
#include <memory>

class ncclGroup_test : public ::testing::Test {
  protected:
    ncclComm_t* comms = NULL;
    cudaStream_t *streams = nullptr;
    int ndev = 0;
    virtual void SetUp() {
        register_segv_handler();
        comms = ncclCommon_getComms(&ndev);
        streams = new cudaStream_t[ndev];
        for(int i=0; i < ndev; i++) {
          ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
          ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        }
    }
    virtual void TearDown() {}
};
TEST_F(ncclGroup_test, basic) {
  for (int i=0; i < ndev; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  }
}
TEST_F(ncclGroup_test, no_start) {
  for (int i=0; i < ndev; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(ncclInvalidUsage, ncclGroupEnd());
  }
}
#if 0
// NCCL doesn't test for that
TEST_F(ncclGroup_test, DISABLED_different_stream) {
  for (int i=0; i < ndev; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_EQ(ncclSuccess, ncclAllReduce(NULL, NULL, 0, ncclFloat, ncclSum, comms[i], NULL));
    ASSERT_EQ(ncclInvalidUsage, ncclAllReduce(NULL, NULL, 0, ncclFloat, ncclSum, comms[i], streams[i]));
    ASSERT_EQ(ncclInvalidUsage, ncclGroupEnd());
  }
}
#endif
TEST_F(ncclGroup_test, aggregation_mixed_bag) {
  std::unique_ptr<int32_t*[]> buf32(new int32_t*[ndev]);
  std::unique_ptr<int64_t*[]> buf64(new int64_t*[ndev]);

  constexpr int ops = 128;
  int32_t init32[ops];
  int64_t init64[ops];

  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i=0; i < ndev; i++) {
    for(int op=0; op < ops; op++) {
      init32[op] = op;
      init64[op] = (i*1ll)<<32;
    }
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buf32[i], ops*sizeof(int32_t)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buf64[i], ops*sizeof(int64_t)));

    ASSERT_EQ(cudaSuccess, cudaMemcpy(buf32[i], init32, ops*sizeof(int32_t), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(buf64[i], init64, ops*sizeof(int64_t), cudaMemcpyHostToDevice));

    int j=0, k=0;
    while (j < 2*ops) {
      int op = k/2;
      if (k%2) {
        ASSERT_EQ(ncclSuccess, ncclAllReduce(buf32[i]+op, buf32[i]+op, 1, ncclInt32, ncclSum, comms[i], streams[i]));
      } else {
        ASSERT_EQ(ncclSuccess, ncclAllReduce(buf64[i]+op, buf64[i]+op, 1, ncclInt64, ncclMax, comms[i], streams[i]));
      }
      j += 1;
      // k randomly iterates the values [0...2*ops) to prevent any internal
      // round-robin logic from matching our two kinds of ops when partitioning
      // over an even number of channels.
      k = (k + j)%(2*ops);
    }
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  for (int i=0; i < ndev; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));
  }

  for (int i=0; i < ndev; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    int32_t val32[ops];
    int64_t val64[ops];
    cudaMemcpy(&val32, buf32[i], ops*sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&val64, buf64[i], ops*sizeof(int64_t), cudaMemcpyDeviceToHost);
    for (int op=0; op < ops; op++) {
      EXPECT_EQ(val32[op], ndev*op);
      EXPECT_EQ(val64[op], (ndev-1)*1ll<<32);
    }
    ASSERT_EQ(ncclSuccess, cudaFree(buf32[i]));
    ASSERT_EQ(ncclSuccess, cudaFree(buf64[i]));
  }
};
// EOF
