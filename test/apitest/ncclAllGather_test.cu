#include "ncclCommon_test.cuh"
template <typename DT>
class ncclAllGather_test : public ncclCommon_test<DT> {};
TYPED_TEST_CASE(ncclAllGather_test, testDataTypes);
// typical usage.
TYPED_TEST(ncclAllGather_test, basic) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                                std::min(this->N/this->nVis, 1024 * 1024),
                                this->DataType(), this->comms[i], this->streams[i]))
            << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclAllGather_test, pinned_mem) {
    if (this->sendbuffs_pinned_device && this->recvbuffs_pinned_device) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            EXPECT_EQ(ncclSuccess,
                    ncclAllGather(this->sendbuffs_pinned_device[i], this->recvbuffs_pinned_device[i],
                                    std::min(this->N/this->nVis, 1024 * 1024),
                                    this->DataType(), this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
};
TYPED_TEST(ncclAllGather_test, stream_null) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(
                      this->sendbuffs[i], this->recvbuffs[i],
                      std::min(this->N/this->nVis, 1024 * 1024),
                      this->DataType(), this->comms[i], NULL))
            << ", " << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclAllGather_test, stream_default) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(
                      this->sendbuffs[i], this->recvbuffs[i],
                      std::min(this->N/this->nVis, 1024 * 1024),
                      this->DataType(), this->comms[i], cudaStreamDefault))
            << ", " << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclAllGather_test, stream_legacy) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(
                      this->sendbuffs[i], this->recvbuffs[i],
                      std::min(this->N/this->nVis, 1024 * 1024),
                      this->DataType(), this->comms[i], cudaStreamLegacy))
            << ", " << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclAllGather_test, stream_per_thread) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(
                      this->sendbuffs[i], this->recvbuffs[i],
                      std::min(this->N/this->nVis, 1024 * 1024),
                      this->DataType(), this->comms[i], cudaStreamPerThread))
            << ", " << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
// sendbuff
TYPED_TEST(ncclAllGather_test, sendbuf_null) {
    int i = 0;
    EXPECT_EQ(ncclInvalidArgument,
              ncclAllGather(NULL, this->recvbuffs[i],
                            std::min(this->N/this->nVis, 1024 * 1024),
                            this->DataType(),
                            this->comms[i], this->streams[i]));
};
TYPED_TEST(ncclAllGather_test, sendbuf_wrong) {
    int i = 0, j = 1;
    EXPECT_EQ(ncclInvalidArgument,
              ncclAllGather(this->sendbuffs[j], this->recvbuffs[i],
                            std::min(this->N/this->nVis, 1024 * 1024),
                            this->DataType(),
                            this->comms[i], this->streams[i]));
};
// recvbuff
TYPED_TEST(ncclAllGather_test, recvbuf_null) {
    int i = 0;
    EXPECT_EQ(ncclInvalidArgument,
              ncclAllGather(this->sendbuffs[i], NULL,
                            std::min(this->N/this->nVis, 1024 * 1024),
                            this->DataType(),
                            this->comms[i], this->streams[i]));
}
// sendbuff and recvbuff not on the same device
TYPED_TEST(ncclAllGather_test, sendbuff_recvbuff_diff_device) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidArgument,
              ncclAllGather(this->sendbuffs[i], this->recvbuffs[j],
                            std::min(this->N/this->nVis, 1024 * 1024),
                            this->DataType(),
                            this->comms[i], this->streams[i]));
};
// N
TYPED_TEST(ncclAllGather_test, DISABLED_N_zero) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(this->sendbuffs[i], this->recvbuffs[i], 0,
                                this->DataType(),
                                this->comms[i], this->streams[i]))
            << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
// data type
TYPED_TEST(ncclAllGather_test, DataType_wrong) {
    int i = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N/this->nVis, 1024 * 1024),
                            ncclNumTypes,
                            this->comms[i], this->streams[i]));
};
// comm
TYPED_TEST(ncclAllGather_test, comm_null) {
    int i = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N/this->nVis, 1024 * 1024),
                            this->DataType(),
                            NULL, this->streams[i]));
};
TYPED_TEST(ncclAllGather_test, comm_wrong) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidArgument,
              ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N/this->nVis, 1024 * 1024),
                            this->DataType(),
                            this->comms[j], this->streams[i]));
};
// STREAM can be NULL.
// stream on a diff device
TYPED_TEST(ncclAllGather_test, DISABLED_stream_wrong) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidArgument,
              ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N/this->nVis, 1024 * 1024),
                            this->DataType(),
                            this->comms[i], this->streams[j]));
};

TYPED_TEST(ncclAllGather_test, multi_net) {
    if (this->commsIB != NULL) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                        ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                                    std::min(this->N/this->nVis, 1024 * 1024),
                                    this->DataType(),
                                    this->commsIB[i], this->streams[i]))
                << ", IB i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());

        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                        ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                                    std::min(this->N/this->nVis, 1024 * 1024),
                                    this->DataType(),
                                    this->commsSockets[i], this->streams[i]))
                << ",  Sockets i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    } else {
        std::cout << "multi_net disabled" << std::endl;
    }
};

// Aggregation
// Only for 2.2 or higher
#if NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >=2)
TYPED_TEST(ncclAllGather_test, aggregate_two_level_group_call) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int j = 0; j < 10; ++j) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                      ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                                    std::min(this->N/this->nVis, 1024 * 1024),
                                    this->DataType(),
                                    this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclAllGather_test, aggregate_one_level_group_call) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                      ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                                    std::min(this->N/this->nVis, 1024 * 1024),
                                    this->DataType(),
                                    this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
        }
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};

TYPED_TEST(ncclAllGather_test, multi_split_share) {
    ncclComm_t* localComms = NULL;
    ncclComm_t* comms2;
    
    localComms = ncclCommon_getSplitShareComms();
    comms2 = (ncclComm_t*)calloc(this->nVis, sizeof(ncclComm_t));

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i=0; i<this->nVis; i++) {
      ASSERT_EQ(ncclSuccess, ncclCommSplit(localComms[i], i/2, i%2, &comms2[i], NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(this->sendbuffs[i], this->recvbuffs[i],
                                std::min(this->N/this->nVis, 1024 * 1024),
                                this->DataType(), comms2[i], this->streams[i]))
            << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    for (int i=0; i<this->nVis; i++) {
      ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms2[i]));
    }
    free(comms2);
};
#endif
//EOF
