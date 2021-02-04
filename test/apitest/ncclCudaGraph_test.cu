#include "ncclCommon_test.cuh"
template <typename DT>
class ncclCudaGraph_test : public ncclCommon_test<DT> {
  protected:
    static cudaGraph_t* graphs;
    static cudaGraphExec_t* graphExec;
    void SetUp();
    void TearDown();
};

template <typename DT>
cudaGraph_t* ncclCudaGraph_test<DT>::graphs = NULL;

template <typename DT>
cudaGraphExec_t* ncclCudaGraph_test<DT>::graphExec = NULL;

template <typename DT>
void ncclCudaGraph_test<DT>::SetUp() {
    ncclCommon_test<DT>::SetUp();
    graphs = (cudaGraph_t*)calloc(this->nVis, sizeof(cudaGraph_t));
    graphExec = (cudaGraphExec_t*)calloc(this->nVis, sizeof(cudaGraphExec_t));
};

template <typename DT>
void ncclCudaGraph_test<DT>::TearDown() {
    auto freeGraphExec = [](cudaGraphExec_t exe) { cudaGraphExecDestroy(exe); };
    EXPECT_NO_FATAL_FAILURE(freePP<>(freeGraphExec, graphExec, this->nVis));
    auto freeGraph = [](cudaGraph_t g) { cudaGraphDestroy(g); };
    EXPECT_NO_FATAL_FAILURE(freePP<>(freeGraph, graphs, this->nVis));
    ncclCommon_test<DT>::TearDown();
};

TYPED_TEST_CASE(ncclCudaGraph_test, testNoType);
// typical usage.
TYPED_TEST(ncclCudaGraph_test, basic) {
    // Begin cuda graph capture
    for (int i=0; i<this->nVis; i++) {
        ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(this->streams[i], cudaStreamCaptureModeThreadLocal));
    }
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
    // End cuda graph capture
    for (int i=0; i<this->nVis; i++) {
        ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(this->streams[i], this->graphs+i));
    }
    // Instantiate cuda graph
    for (int i=0; i<this->nVis; i++) {
        ASSERT_EQ(cudaSuccess, cudaGraphInstantiate(this->graphExec+i, this->graphs[i], NULL, NULL, 0));
    }
    // Launch cuda graph
    for (int i=0; i<this->nVis; i++) {
        ASSERT_EQ(cudaSuccess, cudaGraphLaunch(this->graphExec[i], this->streams[i]));
    }
};
// EOF
