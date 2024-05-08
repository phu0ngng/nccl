#include "ncclCommon_test.cuh"
template <typename DT>
class ncclCudaGraph_test : public ncclCommon_test<DT> {
  protected:
    static cudaGraph_t *graphs[2];
    static cudaGraphExec_t *graphExec[2];
    static cudaStream_t *graphStreams[2];
    static int driverVersion;
    static int expectMask;
    void SetUp();
    void TearDown();
    void BeginCapture(int graph=0);
    void EndCapture(int graph=0);
    void LaunchGraph(int graph=0);
};

template <typename DT>
cudaGraph_t* ncclCudaGraph_test<DT>::graphs[2] = {nullptr, nullptr};

template <typename DT>
cudaGraphExec_t* ncclCudaGraph_test<DT>::graphExec[2] = {nullptr, nullptr};

template <typename DT>
cudaStream_t* ncclCudaGraph_test<DT>::graphStreams[2] = {nullptr, nullptr};

template <typename DT>
int ncclCudaGraph_test<DT>::driverVersion = 0;
template <typename DT>
int ncclCudaGraph_test<DT>::expectMask = 1<<ncclSuccess;

template <typename DT>
void ncclCudaGraph_test<DT>::SetUp() {
    ncclCommon_test<DT>::SetUp();
    graphs[0] = (cudaGraph_t*)calloc(this->nVis, sizeof(cudaGraph_t));
    graphs[1] = (cudaGraph_t*)calloc(this->nVis, sizeof(cudaGraph_t));
    graphExec[0] = (cudaGraphExec_t*)calloc(this->nVis, sizeof(cudaGraphExec_t));
    graphExec[1] = (cudaGraphExec_t*)calloc(this->nVis, sizeof(cudaGraphExec_t));
    for (int g=0; g < 2; g++) {
#pragma GCC diagnostic push
#ifdef __has_warning // clang
#  if __has_warning("-Walloc-size-larger-than=")
#    pragma GCC diagnostic ignored "-Walloc-size-larger-than="
#  endif
#else // gcc
#  pragma GCC diagnostic ignored "-Walloc-size-larger-than="
#endif
        graphStreams[g] = (cudaStream_t*)calloc(this->nVis, sizeof(cudaStream_t));
#pragma GCC diagnostic pop
        for (int s=0; s < this->nVis; s++) {
            if (g == 0) graphStreams[g][s] = this->streams[s];
            else {
                ASSERT_EQ(cudaSuccess, cudaSetDevice(s));
                ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&graphStreams[g][s], cudaStreamNonBlocking));
            }
        }
    }
    ASSERT_EQ(cudaSuccess, cudaDriverGetVersion(&this->driverVersion));
    expectMask = 1<<ncclSuccess;
    expectMask |= (this->driverVersion < 11030) ? 1<<ncclInvalidUsage : 0;
};

template <typename DT>
void ncclCudaGraph_test<DT>::TearDown() {
    auto freeStream = [](cudaStream_t s) {
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(s));
        cudaStreamDestroy(s);
    };
    EXPECT_NO_FATAL_FAILURE(freePP<>(freeStream, graphStreams[1], this->nVis));
    auto freeGraphExec = [](cudaGraphExec_t exe) { cudaGraphExecDestroy(exe); };
    EXPECT_NO_FATAL_FAILURE(freePP<>(freeGraphExec, graphExec[0], this->nVis));
    EXPECT_NO_FATAL_FAILURE(freePP<>(freeGraphExec, graphExec[1], this->nVis));
    auto freeGraph = [](cudaGraph_t g) { cudaGraphDestroy(g); };
    EXPECT_NO_FATAL_FAILURE(freePP<>(freeGraph, graphs[0], this->nVis));
    EXPECT_NO_FATAL_FAILURE(freePP<>(freeGraph, graphs[1], this->nVis));
    ncclCommon_test<DT>::TearDown();
};

template <typename DT>
void ncclCudaGraph_test<DT>::BeginCapture(int graph) {
    // Begin cuda graph capture
    for (int i=0; i<this->nVis; i++) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(this->graphStreams[graph][i], cudaStreamCaptureModeRelaxed));
    }
};

template <typename DT>
void ncclCudaGraph_test<DT>::EndCapture(int graph) {
    // End cuda graph capture
    for (int i=0; i<this->nVis; i++) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(this->graphStreams[graph][i], this->graphs[graph]+i));
    }
    // Instantiate cuda graph
    for (int i=0; i<this->nVis; i++) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(cudaSuccess, cudaGraphInstantiate(this->graphExec[graph]+i, this->graphs[graph][i], NULL, NULL, 0));
    }
};

template <typename DT>
void ncclCudaGraph_test<DT>::LaunchGraph(int graph) {
    // Launch cuda graph
    for (int i=0; i<this->nVis; i++) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(cudaSuccess, cudaGraphLaunch(this->graphExec[graph][i], this->graphStreams[graph][i]));
    }
};

TYPED_TEST_CASE(ncclCudaGraph_test, testNoType);

#if (NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >=9)) && CUDART_VERSION >= 11030
// typical usage.
TYPED_TEST(ncclCudaGraph_test, collective) {
    this->BeginCapture();
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_NE(0, this->expectMask &
                     1<<ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                                      std::min(this->N, 1024 * 1024),
                                      this->DataType(), ncclSum,
                                      this->comms[i], this->streams[i]))
            << "i" << i << ", " << std::endl;
    }
    ASSERT_NE(0, this->expectMask & 1<<ncclGroupEnd());
    this->EndCapture();
    this->LaunchGraph();
    this->LaunchGraph();  // Launch graph twice to check task list persistency
};

TYPED_TEST(ncclCudaGraph_test, alltoall) {
    size_t size = std::min(this->N, 1024 * 1024) / this->nVis;
    this->BeginCapture();
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int p = 0; p < this->nVis; ++p) {
            ASSERT_NE(0, this->expectMask &
                         1<<ncclSend(this->sendbuffs[i] + p * size, size,
                                     this->DataType(), p,
                                     this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
            ASSERT_NE(0, this->expectMask &
                         1<<ncclRecv(this->recvbuffs[i] + p * size, size,
                                     this->DataType(), p,
                                     this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
         }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
    ASSERT_NE(0, this->expectMask & 1<<ncclGroupEnd());
    this->EndCapture();
    this->LaunchGraph();
};

TYPED_TEST(ncclCudaGraph_test, aggregation) {
    this->BeginCapture();
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_NE(0, this->expectMask &
                         1<<ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                                          std::min(this->N, 1024 * 1024),
                                          this->DataType(), ncclSum,
                                          this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
        }
    }
    ASSERT_NE(0, this->expectMask & 1<<ncclGroupEnd());
    this->EndCapture();
    this->LaunchGraph();
};

TYPED_TEST(ncclCudaGraph_test, many_graph_many_stream) {
    for (int g=0; g < 2; g++) {
        this->BeginCapture(g);
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < this->nVis; ++i) {
                ASSERT_NE(0, this->expectMask &
                             1<<ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                                              std::min(this->N, 1024 * 1024),
                                              this->DataType(), ncclSum,
                                              this->comms[i], this->graphStreams[g][i]))
                    << "i" << i << ", " << std::endl;
            }
        }
        ASSERT_NE(0, this->expectMask & 1<<ncclGroupEnd());
        this->EndCapture(g);
    }
    for (int g=0; g < 2; g++) {
        this->LaunchGraph(g);
    }
};

TYPED_TEST(ncclCudaGraph_test, many_graph_many_stream_interleaved) {
    for (int g=0; g < 2; g++) {
        this->BeginCapture(g);
    }
    for (int j = 0; j < 2; ++j) {
        for (int g=0; g < 2; g++) {
            ASSERT_EQ(ncclSuccess, ncclGroupStart());
            for (int i = 0; i < this->nVis; ++i) {
                ASSERT_NE(0, this->expectMask &
                             1<<ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                                              std::min(this->N, 1024 * 1024),
                                              this->DataType(), ncclSum,
                                              this->comms[i], this->graphStreams[g][i]))
                    << "i" << i << ", " << std::endl;
            }
            ASSERT_NE(0, this->expectMask & 1<<ncclGroupEnd());
        }
    }
    for (int g=0; g < 2; g++) this->EndCapture(g);
    for (int g=0; g < 2; g++) this->LaunchGraph(g);
};

TYPED_TEST(ncclCudaGraph_test, graph_nongraph) {
    this->BeginCapture();
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_NE(0, this->expectMask &
                          1<<ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                                           std::min(this->N, 1024 * 1024),
                                           this->DataType(), ncclSum,
                                           this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
        }
    }
    ASSERT_NE(0, this->expectMask & 1<<ncclGroupEnd());
    this->EndCapture();
    this->LaunchGraph();

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                                std::min(this->N, 1024 * 1024),
                                this->DataType(), ncclSum,
                                this->comms[i], this->streams[i]))
            << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
#endif
// EOF
