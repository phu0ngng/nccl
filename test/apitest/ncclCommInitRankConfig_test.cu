#include "ncclCommon_test.cuh"
#include <cstdint>

class ncclCommInitRankConfig_test : public ::testing::Test {
  protected:
    ncclComm_t *gcomms;
    int ndev;
    ncclUniqueId commId;
    const int rank0 = 0;
    ncclConfig_t gconfig = NCCL_CONFIG_INITIALIZER;
    int expectMask;

    virtual void SetUp() {
        // Some CTA tests use lots of memory. Clean up everything we can before we start
        ncclCommon_destroyComms();
        ncclCommon_destroyIBComms();
        ncclCommon_destroySocketComms();
        ncclCommon_destroySplitComms();

        register_segv_handler();
        (void) setenv("NCCL_CHECK_POINTERS", "1", 0);
        expectMask = (1 << ncclSuccess) | (1 << ncclInProgress);
        EXPECT_EQ(cudaSuccess, cudaGetDeviceCount(&ndev));
        EXPECT_NE(nullptr, gcomms = (ncclComm_t*) calloc(ndev, sizeof(ncclComm_t)));
        gconfig.blocking = 0;
        gconfig.minCTAs = 2;
        gconfig.maxCTAs = 4;
        gconfig.cgaClusterSize = 0;
        gconfig.nChannelsPerNetPeer = 4;
        gconfig.netName = "Socket";
    }

    virtual void TearDown() {
        for (int i = 0; i < ndev; ++i) {
            if (gcomms[i]) {
                ASSERT_EQ(ncclSuccess, ncclCommDestroy(gcomms[i]));
            }
        }
        free(gcomms);
    }

    void waitCommsReady(ncclComm_t *comms, int nranks) {
        int complete;
        ncclResult_t state;
        do {
            complete = 1;
            for (int i = 0; i < nranks; ++i) {
                ASSERT_EQ(ncclSuccess, ncclCommGetAsyncError(comms[i], &state));
                if (state == ncclInProgress) {
                    complete = 0;
                    break;
                }
            }
            usleep(10);
        } while(!complete);
    }
};

TEST_F(ncclCommInitRankConfig_test, basic) {
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        (void) ncclCommInitRankConfig(&gcomms[i], ndev, commId, i, &gconfig);
    }
    ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));
    waitCommsReady(gcomms, ndev);
}

TEST_F(ncclCommInitRankConfig_test, basic_null) {
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&gcomms[i], ndev, commId, i, NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
}

TEST_F(ncclCommInitRankConfig_test, attr_null) {
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&gcomms[0], 1, commId, rank0, NULL));
}

TEST_F(ncclCommInitRankConfig_test, with_config) {
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commId));
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_NE(0, expectMask & (1 << ncclCommInitRankConfig(&gcomms[0], 1, commId, rank0, &gconfig)));
    waitCommsReady(gcomms, 1);
}

TEST_F(ncclCommInitRankConfig_test, comm_null) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(NULL, ndev, commId, rank0, &gconfig));
}

TEST_F(ncclCommInitRankConfig_test, ndev_zero) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&gcomms[0], 0, commId, rank0, &gconfig));
}

TEST_F(ncclCommInitRankConfig_test, dev_negative) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&gcomms[0], -1, commId, rank0, &gconfig));
}

TEST_F(ncclCommInitRankConfig_test, rank_outofboundary) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&gcomms[0], 1, commId, ndev, &gconfig));
}

TEST_F(ncclCommInitRankConfig_test, rank_negative) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&gcomms[0], ndev, commId, -1, &gconfig));
}

TEST_F(ncclCommInitRankConfig_test, blocking) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    config.blocking = 1;
    comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclCommInitRankConfig_test, config_null) {
    ncclUniqueId id;
    ncclComm_t* comms;

    comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclCommInitRankConfig_test, invalid_config_size_0) {
    ncclUniqueId id;
    ncclComm_t comm;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.size = 0;

    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&comm, 1, id, 0, &config));
    ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
}

TEST_F(ncclCommInitRankConfig_test, invalid_config_size_large) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.size = SIZE_MAX;

    comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
}

TEST_F(ncclCommInitRankConfig_test, invalid_config_magic) {
    ncclUniqueId id;
    ncclComm_t comm;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.magic = 0;

    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&comm, 1, id, 0, &config));
    ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
}

TEST_F(ncclCommInitRankConfig_test, cta_basic) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    config.minCTAs = 8;
    config.maxCTAs = 16;
    comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));

    /* equal minCTAs and maxCTAs */
    config.minCTAs = 16;
    config.maxCTAs = 16;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclCommInitRankConfig_test, cta_large) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    config.minCTAs = 64;
    config.maxCTAs = 128;
    comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclCommInitRankConfig_test, cta_invalid) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    config.minCTAs = 0;
    comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclInvalidArgument, ncclGroupEnd());
    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));

    config.minCTAs = -256;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclInvalidArgument, ncclGroupEnd());
    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));

    config.minCTAs = 16;
    config.maxCTAs = 8;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclInvalidArgument, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclCommInitRankConfig_test, cga_basic) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    config.cgaClusterSize = 8;
    comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclCommInitRankConfig_test, cga_warn) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    config.cgaClusterSize = 16; /* we should only use maximal 8 CGA group size */
    comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclCommInitRankConfig_test, cta_less_than_cga) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    config.cgaClusterSize = 8;
    config.minCTAs = 1;
    config.maxCTAs = 1;
    comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclCommInitRankConfig_test, nChannel_cga_allreduce) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    void** sendbuffs;
    void** recvbuffs;
    float* tmpbuffs;
    size_t cnt = 1 << 20;
    size_t size = cnt * sizeof(float);
    cudaStream_t* streams;

    sendbuffs = (void**)malloc(ndev * sizeof(void*));
    recvbuffs = (void**)malloc(ndev * sizeof(void*));
    tmpbuffs = (float*)malloc(size);
    streams = (cudaStream_t*)malloc(ndev * sizeof(cudaStream_t));
    for (int i = 0; i < cnt; ++i) tmpbuffs[i] = i;
    config.cgaClusterSize = 8;
    config.minCTAs = 1;
    config.maxCTAs = 1;
    comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&sendbuffs[i], size));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&recvbuffs[i], size));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(sendbuffs[i], tmpbuffs, size, cudaMemcpyDefault));
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[i]));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess,
                    ncclAllReduce(sendbuffs[i], recvbuffs[i], cnt, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaFree(sendbuffs[i]));
        ASSERT_EQ(cudaSuccess, cudaFree(recvbuffs[i]));
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streams[i]));
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    }
    free(sendbuffs);
    free(recvbuffs);
    free(tmpbuffs);
    free(comms);
};

TEST_F(ncclCommInitRankConfig_test, nChannelsPerNetPeer_basic) {
  ncclUniqueId id;
  ncclComm_t* comms;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  config.nChannelsPerNetPeer = 13; // Will be set to the next power of 2. (16 in this case)
  comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
  ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < ndev; ++i) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  for (int i = 0; i < ndev; ++i)
    ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
  free(comms);
}

TEST_F(ncclCommInitRankConfig_test, nChannelsPerNetPeer_invalid) {
  ncclUniqueId id;
  ncclComm_t* comms;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  config.nChannelsPerNetPeer = 0; // Invalid value
  comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
  ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < ndev; ++i) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
  }
  ASSERT_EQ(ncclInvalidArgument, ncclGroupEnd());

  for (int i = 0; i < ndev; ++i)
    ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
  free(comms);
}

TEST_F(ncclCommInitRankConfig_test, nChannelsPerNetPeer_alltoall) {
  ncclUniqueId id;
  ncclComm_t* comms;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  void** sendbuffs;
  void** recvbuffs;
  float* tmpbuffs;
  size_t cnt = 1 << 20;
  size_t size = cnt * sizeof(float);
  cudaStream_t* streams;
  int nRanks;

  sendbuffs = (void**)malloc(ndev * sizeof(void*));
  recvbuffs = (void**)malloc(ndev * sizeof(void*));
  tmpbuffs = (float*)malloc(size);
  streams = (cudaStream_t*)malloc(ndev * sizeof(cudaStream_t));
  for (int i = 0; i < cnt; ++i) tmpbuffs[i] = i;

  config.nChannelsPerNetPeer = 13;
  comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));

  ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
  ASSERT_EQ(ncclSuccess, ncclGroupStart());

  for (int i = 0; i < ndev; ++i) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&sendbuffs[i], size));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&recvbuffs[i], size));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(sendbuffs[i], tmpbuffs, size, cudaMemcpyDefault));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[i]));
    ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
  }

  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  ASSERT_EQ(ncclSuccess, ncclCommCount(comms[0], &nRanks));

  ASSERT_EQ(ncclSuccess, ncclGroupStart());

  for (int i = 0; i < ndev; ++i) {
    // Loop over all ranks and use send and receive operations to build an alltoall pattern
    for (int j = 0; j < nRanks; ++j) {
        // AlltoAll
        ASSERT_EQ(ncclSuccess, ncclSend(sendbuffs[i], cnt, ncclFloat, j, comms[i], streams[i]));
        ASSERT_EQ(ncclSuccess, ncclRecv(recvbuffs[i], cnt, ncclFloat, j, comms[i], streams[i]));
    }
  }

  ASSERT_EQ(ncclSuccess, ncclGroupEnd());

  for (int i = 0; i < ndev; ++i) {
    ASSERT_EQ(cudaSuccess, cudaFree(sendbuffs[i]));
    ASSERT_EQ(cudaSuccess, cudaFree(recvbuffs[i]));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streams[i]));
    ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
  }
  free(sendbuffs);
  free(recvbuffs);
  free(tmpbuffs);
  free(comms);
}

TEST_F(ncclCommInitRankConfig_test, net_name_default) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclCommInitRankConfig_test, net_name_internal) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    config.netName = "Socket";
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclCommInitRankConfig_test, net_name_nonexist) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    config.netName = "NONEXIST";
    comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclInvalidUsage, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i)

        ASSERT_EQ(ncclSuccess, ncclCommAbort(comms[i]));
    free(comms);
}

TEST_F(ncclCommInitRankConfig_test, split_config) {
    ncclComm_t* localComms = NULL;
    ncclComm_t* childComms = NULL;
    ncclUniqueId id;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    config.splitShare = 1;
    config.cgaClusterSize = 4;
    config.minCTAs = 4;
    config.maxCTAs = 16;
    config.nChannelsPerNetPeer = 15;
    ASSERT_NE(nullptr, localComms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t)));
    ASSERT_NE(nullptr, childComms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t)));

    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        (void)ncclCommInitRankConfig(&localComms[i], ndev, id, i, &config);
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    config.splitShare = 1;
    config.cgaClusterSize = 8;
    config.minCTAs = 8;
    config.maxCTAs = 8;
    config.nChannelsPerNetPeer = 7;
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommSplit(localComms[i], 0, ndev - i, &childComms[i], &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(localComms[i]));
    }
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(childComms[i]));
    }

    free(localComms);
    free(childComms);
}

TEST_F(ncclCommInitRankConfig_test, split_share_invalid_net_name) {
    ncclComm_t* localComms = NULL;
    ncclComm_t* childComms = NULL;
    ncclUniqueId id;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    /* when split shares resource. user cannot specify a different netName for child comm from parent comm. */
    config.splitShare = 1;
    config.netName = "Socket";
    ASSERT_NE(nullptr, localComms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t)));
    ASSERT_NE(nullptr, childComms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t)));

    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        (void)ncclCommInitRankConfig(&localComms[i], ndev, id, i, &config);
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    config.netName = "IB";
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        (void) ncclCommSplit(localComms[i], 0, ndev - i, &childComms[i], &config);
    }
    ASSERT_EQ(ncclInvalidUsage, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommAbort(localComms[i]));
    }

    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommAbort(childComms[i]));
    }

    free(localComms);
    free(childComms);
}

TEST_F(ncclCommInitRankConfig_test, multi_net_plugin_ext_v7) {
    if (getenv("NCCL_NET_PLUGIN")==nullptr) {
      // GTEST_SKIP requires a more recent googletest version. For now, we'll just pass the test.
      // GTEST_SKIP() << "Skipping test since NCCL_NET_PLUGIN is not defined.");
      return;
    }

    ncclComm_t *comms = NULL;
    ncclUniqueId id;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.netName = "ncclNetPlugin_v7";

    ASSERT_NE(nullptr, comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t)));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        (void)ncclCommInitRankConfig(&comms[i], ndev, id, i, &config);
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    }

    free(comms);
}

TEST_F(ncclCommInitRankConfig_test, multi_net_plugin_int_sock) {
    if (getenv("NCCL_NET_PLUGIN")==nullptr) {
      // GTEST_SKIP requires a more recent googletest version. For now, we'll just pass the test.
      // GTEST_SKIP() << "Skipping test since NCCL_NET_PLUGIN is not defined.");
      return;
    }

    ncclComm_t *comms = NULL;
    ncclUniqueId id;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.netName = "Socket";

    ASSERT_NE(nullptr, comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t)));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        (void)ncclCommInitRankConfig(&comms[i], ndev, id, i, &config);
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    }

    free(comms);
}

TEST_F(ncclCommInitRankConfig_test, multi_net_plugin_ext_fail) {
    if (getenv("NCCL_NET_PLUGIN")==nullptr) {
      // GTEST_SKIP requires a more recent googletest version. For now, we'll just pass the test.
      // GTEST_SKIP() << "Skipping test since NCCL_NET_PLUGIN is not defined.");
      return;
    }

    ncclComm_t *comms = NULL;
    ncclUniqueId id;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.netName = "ncclNetPlugin_v9";

    ASSERT_NE(nullptr, comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t)));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        (void)ncclCommInitRankConfig(&comms[i], ndev, id, i, &config);
    }
    ASSERT_EQ(ncclInvalidUsage, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    }

    free(comms);
}

// Test Description:
// This test achieves two goals:
//   1. it makes sure that the network plugin is initialized for every new comm
//   2. it makes sure that all the plugins can be loaded from the same library
TEST_F(ncclCommInitRankConfig_test, shared_plugin_lib) {
    if (getenv("NCCL_NET_PLUGIN")==nullptr) {
      // GTEST_SKIP requires a more recent googletest version. For now, we'll just pass the test.
      // GTEST_SKIP() << "Skipping test since NCCL_NET_PLUGIN is not defined.");
      return;
    }

    ncclComm_t *comms = nullptr;
    ncclUniqueId id;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.netName = "ncclNetPlugin_v11";

    ASSERT_NE(nullptr, comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t)));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));

    // First comm initialization:
    // All plugins are loaded from the same shared library
    // All plugins initialize correctly the first time but the tuner and the profiler set the number of devices for network to 0
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        (void)ncclCommInitRankConfig(&comms[i], ndev, id, i, &config);
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    }

    // Second comm initialization:
    // Net plugin returns 0 devices, test fails with ncclInvalidUsage
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        (void)ncclCommInitRankConfig(&comms[i], ndev, id, i, &config);
    }
    ASSERT_EQ(ncclInvalidUsage, ncclGroupEnd());

    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    }

    free(comms);
}

TEST_F(ncclCommInitRankConfig_test, env_plugin) {
    // Set environment variable to load the environment plugin
    setenv("NCCL_ENV_PLUGIN", "libnccl-env-example.so", 1);

    ncclComm_t *comms = NULL;
    ncclUniqueId id;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    ASSERT_NE(nullptr, comms = (ncclComm_t*)calloc(ndev, sizeof(ncclComm_t)));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));

    // Initialize communicators - this will trigger NCCL initialization
    // which loads the environment plugin
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], ndev, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    // Clean up communicators
    for (int i = 0; i < ndev; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    }

    free(comms);

    // Clean up environment variables
    unsetenv("NCCL_ENV_PLUGIN");
}

TEST_F(ncclCommInitRankConfig_test, init_net_dev_once) {
    ncclUniqueId id, newid;
    ncclComm_t comm, newcomm;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    // this test loads the libnccl-net-plugin-init-once.so plugin lib, which fails to initialize more than once"
    config.netName = "init_once";
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comm, 1, id, 0, &config));

    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&newid));
    ASSERT_EQ(ncclInvalidUsage, ncclCommInitRankConfig(&newcomm, 1, newid, 0, &config));

    ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
}
