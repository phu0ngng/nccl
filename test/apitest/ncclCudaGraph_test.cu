#include "ncclCommon_test.cuh"

constexpr int MAX_GRAPHS = 64;
template <typename DT>
class ncclCudaGraph_test : public ncclCommon_test<DT> {
  protected:
    static cudaGraph_t *graphs[MAX_GRAPHS];
    static cudaGraphExec_t *graphExec[MAX_GRAPHS];
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
cudaGraph_t* ncclCudaGraph_test<DT>::graphs[MAX_GRAPHS] = {};

template <typename DT>
cudaGraphExec_t* ncclCudaGraph_test<DT>::graphExec[MAX_GRAPHS] = {};

template <typename DT>
cudaStream_t* ncclCudaGraph_test<DT>::graphStreams[2] = {};

template <typename DT>
int ncclCudaGraph_test<DT>::driverVersion = 0;
template <typename DT>
int ncclCudaGraph_test<DT>::expectMask = 1<<ncclSuccess;

template <typename DT>
void ncclCudaGraph_test<DT>::SetUp() {
    ncclCommon_test<DT>::SetUp();
    for (int g=0; g < MAX_GRAPHS; g++) {
      graphs[g] = (cudaGraph_t*)calloc(this->nVis, sizeof(cudaGraph_t));
      graphExec[g] = (cudaGraphExec_t*)calloc(this->nVis, sizeof(cudaGraphExec_t));
    }
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
    for (int g=0; g < MAX_GRAPHS; g++) {
      auto freeGraphExec = [](cudaGraphExec_t exe) { if (exe) cudaGraphExecDestroy(exe); };
      EXPECT_NO_FATAL_FAILURE(freePP<>(freeGraphExec, graphExec[g], this->nVis));
      auto freeGraph = [](cudaGraph_t g) { if (g) cudaGraphDestroy(g); };
      EXPECT_NO_FATAL_FAILURE(freePP<>(freeGraph, graphs[g], this->nVis));
    }
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

TYPED_TEST(ncclCudaGraph_test, graph_storm) {
  using EltType = typename std::remove_reference<decltype(this->sendbuffs[0][0])>::type;
  constexpr int MAX_SLOTS = 512;
  int graphSlots[MAX_GRAPHS] = {};
  int graphDist[MAX_GRAPHS] = {};
  int slotAccum[MAX_SLOTS] = {};
  int nRanks = this->nVis;

  { EltType elts[MAX_SLOTS];
    for (int r=0; r < nRanks; r++) {
      for (int e=0; e < MAX_SLOTS; e++) elts[e] = EltType(r ^ e%128);
      ASSERT_EQ(cudaSuccess, cudaSetDevice(r));
      ASSERT_EQ(cudaSuccess, cudaMemcpy(this->sendbuffs[r], elts, MAX_SLOTS*sizeof(EltType), cudaMemcpyDefault));
    }
  }

  for (int iter=0; iter < 150; iter++) {
    uint64_t rng = iter;
    rng *= 0xc8adca85516adeb3;
    rng ^= rng>>32 ^ 1;
    int g;
    if (iter < MAX_GRAPHS) g = iter;
    else g = rng%std::min<int>(iter+1, MAX_GRAPHS);

    rng *= 0xae6553ef0a2d0753;
    rng ^= rng>>32 ^ 1;
    int slots = rng%MAX_SLOTS;

    rng *= 0x54692f03043c769b;
    rng ^= rng>>32 ^ 1;
    int dist = rng%nRanks;

    // Create a graph that for the first `slots` elements rotates the values by `dist` ranks.
    graphSlots[g] = slots;
    graphDist[g] = dist;

    for (int r=0; r < nRanks; r++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(r));
      if (this->graphs[g][r] != nullptr) {
        ASSERT_EQ(cudaSuccess, cudaGraphExecDestroy(this->graphExec[g][r]));
        ASSERT_EQ(cudaSuccess, cudaGraphDestroy(this->graphs[g][r]));
        this->graphExec[g][r] = nullptr;
        this->graphs[g][r] = nullptr;
      }
      ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(this->streams[r], cudaStreamCaptureModeRelaxed));
    }

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int e=0; e < slots; e++) {
      for (int r=0; r < nRanks; r++) {
        ASSERT_EQ(ncclSuccess, ncclSend(
          this->sendbuffs[r] + e, 1, this->DataType(), (r-dist+nRanks)%nRanks,
          this->comms[r], this->streams[r]));
        ASSERT_EQ(ncclSuccess, ncclRecv(
          this->recvbuffs[r] + e, 1, this->DataType(), (r+dist+nRanks)%nRanks,
          this->comms[r], this->streams[r]));
      }
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int r=0; r < nRanks; r++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(r));
      ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(this->sendbuffs[r], this->recvbuffs[r], slots*sizeof(*this->sendbuffs[0]), cudaMemcpyDefault, this->streams[r]));
      ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(this->streams[r], &this->graphs[g][r]));
      ASSERT_EQ(cudaSuccess, cudaGraphInstantiate(&this->graphExec[g][r], this->graphs[g][r], NULL, NULL, 0));
    }

    // Pick a graph to launch
    rng *= 0x9afd829080867683;
    rng ^= rng>>32 ^ 1;
    g = rng%std::min<int>(iter+1, MAX_GRAPHS);

    if (this->graphs[g][0] != nullptr) {
      slots = graphSlots[g];
      dist = graphDist[g];
      // Keep track of the total rotation distance applied to each element.
      for (int e=0; e < slots; e++) {
        slotAccum[e] += dist;
        slotAccum[e] %= nRanks;
      }
      for (int r=0; r < nRanks; r++) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(r));
        ASSERT_EQ(cudaSuccess, cudaGraphLaunch(this->graphExec[g][r], this->streams[r]));
      }
    }
  }

  // Validate results
  { EltType elts[MAX_SLOTS];
    for (int r=0; r < nRanks; r++) {
      ASSERT_EQ(cudaSuccess, cudaSetDevice(r));
      ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(elts, this->sendbuffs[r], MAX_SLOTS*sizeof(EltType), cudaMemcpyDefault, this->streams[r]));
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(this->streams[r]));

      for (int e=0; e < MAX_SLOTS; e++) {
        EXPECT_EQ(elts[e], EltType((r+slotAccum[e])%nRanks ^ e%128));
      }
    }
  }
}

#endif
// EOF
