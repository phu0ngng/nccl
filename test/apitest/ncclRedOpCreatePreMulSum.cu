#include "ncclCommon_test.cuh"
#include <memory>

class ncclRedOpCreatePreMulSum_test : public ::testing::Test {
  protected:
    ncclComm_t* comms = NULL;
    cudaStream_t *streams = nullptr;
    int ndev = 0;
    virtual void SetUp() {
        comms = ncclCommon_getComms(&ndev);
        streams = new cudaStream_t[ndev];
        for(int i=0; i < ndev; i++) {
          ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
          ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[i]));
        }
    }
    virtual void TearDown() {}
};
TEST_F(ncclRedOpCreatePreMulSum_test, everything) {
  std::unique_ptr<ncclRedOp_t[]> hostOp(new ncclRedOp_t[ndev]);
  std::unique_ptr<ncclRedOp_t[]> devOp(new ncclRedOp_t[ndev]);
  std::unique_ptr<float*[]> buf(new float*[ndev]);
  float hostScalar;

  for (int i=0; i < ndev; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buf[i], 4*sizeof(float)));

    float tmp = 1.0f; // input value for allreduce
    ASSERT_EQ(cudaSuccess, cudaMemcpy(buf[i]+0, &tmp, sizeof(float), cudaMemcpyHostToDevice));

    float *devScalar = buf[i]+1;
    tmp = 3.0f*i; // all scalars are custom to this rank (the "i")
    ASSERT_EQ(cudaSuccess, cudaMemcpy(devScalar, &tmp, sizeof(float), cudaMemcpyHostToDevice));

    hostScalar = 4.0f*i;
    ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum(&hostOp[i], &hostScalar, ncclFloat, ncclScalarHostImmediate, comms[i]));
    hostScalar = 123.456; // modifying scalar doesnt affect anything

    ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum(&devOp[i], devScalar, ncclFloat, ncclScalarDevice, comms[i]));

    ASSERT_EQ(ncclSuccess, ncclAllReduce(buf[i]+0, buf[i]+2, 1, ncclFloat, devOp[i], comms[i], streams[i]));
    ASSERT_EQ(ncclSuccess, ncclAllReduce(buf[i]+0, buf[i]+3, 1, ncclFloat, hostOp[i], comms[i], streams[i]));
  }

  for (int i=0; i < ndev; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));
  }

  for (int i=0; i < ndev; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    float vals[4];
    cudaMemcpy(vals, buf[i], 4*sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(vals[2], float(3.0f*(ndev*ndev - ndev)/2));
    EXPECT_EQ(vals[3], float(4.0f*(ndev*ndev - ndev)/2));

    ASSERT_EQ(ncclSuccess, cudaFree(buf[i]));

    ASSERT_EQ(ncclSuccess, ncclRedOpDestroy(hostOp[i], comms[i]));
    ASSERT_EQ(ncclSuccess, ncclRedOpDestroy(devOp[i], comms[i]));
  }
};
