#include "ncclCommon_test.cuh"

class ncclConfig_test : public ::testing::Test {
  public:
    int nVis;
    int expectMask;
    void SetUp() {
        ncclCommon_destroysrComms();
        expectMask = (1 << ncclSuccess) | (1 << ncclInProgress);
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
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

TEST_F(ncclConfig_test, basic) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

    config.blocking = 0;
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        (void) ncclCommInitRankConfig(&comms[i], nVis, id, i, &config);
    }
    ASSERT_NE(0, expectMask & (1 << ncclGroupEnd()));

    waitCommsReady(comms, nVis);

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclConfig_test, blocking) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclConfig_test, config_null) {
    ncclUniqueId id;
    ncclComm_t* comms;

    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclConfig_test, cta_basic) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    
    config.minCTAs = 8;
    config.maxCTAs = 16;
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    
    /* equal minCTAs and maxCTAs */
    config.minCTAs = 16;
    config.maxCTAs = 16;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclConfig_test, cta_large) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    
    config.minCTAs = 64;
    config.maxCTAs = 128;
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclConfig_test, cta_invalid) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    
    config.minCTAs = 0;
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclInvalidArgument, ncclGroupEnd());
    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));

    config.minCTAs = -256;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclInvalidArgument, ncclGroupEnd());
    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));

    config.minCTAs = 16;
    config.maxCTAs = 8;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclInvalidArgument, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclInvalidArgument, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclConfig_test, cga_basic) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    
    config.cgaClusterSize = 8;
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclConfig_test, cga_warn) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    
    config.cgaClusterSize = 16; /* we should only use maximal 8 CGA group size */
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclConfig_test, cta_less_than_cga) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    
    config.cgaClusterSize = 8;
    config.minCTAs = 1;
    config.maxCTAs = 1;
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclConfig_test, nChannel_cga_allreduce) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    void** sendbuffs;
    void** recvbuffs;
    float* tmpbuffs;
    size_t cnt = 1 << 20;
    size_t size = cnt * sizeof(float);
    cudaStream_t* streams;
    
    sendbuffs = (void**)malloc(nVis * sizeof(void*));
    recvbuffs = (void**)malloc(nVis * sizeof(void*));
    tmpbuffs = (float*)malloc(size);
    streams = (cudaStream_t*)malloc(nVis * sizeof(cudaStream_t));
    for (int i = 0; i < cnt; ++i) tmpbuffs[i] = i;
    config.cgaClusterSize = 8;
    config.minCTAs = 1;
    config.maxCTAs = 1;
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&sendbuffs[i], size));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&recvbuffs[i], size));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(sendbuffs[i], tmpbuffs, size, cudaMemcpyDefault));
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[i]));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                    ncclAllReduce(sendbuffs[i], recvbuffs[i], cnt, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    
    for (int i = 0; i < nVis; ++i) {
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

TEST_F(ncclConfig_test, net_name_default) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclConfig_test, net_name_internal) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    
    config.netName = "IB";
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    
    config.netName = "Socket";
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
    free(comms);
}

TEST_F(ncclConfig_test, net_name_nonexist) {
    ncclUniqueId id;
    ncclComm_t* comms;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    
    config.netName = "NONEXIST";
    comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < nVis; ++i) {
        ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[i], nVis, id, i, &config));
    }
    ASSERT_EQ(ncclInvalidUsage, ncclGroupEnd());

    for (int i = 0; i < nVis; ++i)
        ASSERT_EQ(ncclSuccess, ncclCommAbort(comms[i]));
    free(comms);
}
