#include "ncclCommon_test.cuh"

struct MultiNetComms {
    ncclComm_t* commsIB; 
    ncclComm_t* commsSocket;
    bool testMultiNet;
};

template <typename DT>
class ncclAllReduce_test : public ncclCommon_test<DT> {
    public:
        static MultiNetComms* multiNetComms;

    public:
        static MultiNetComms* ncclAllReduce_GetMultiNetComms() {
            if (multiNetComms == NULL) {
                multiNetComms = new MultiNetComms;
                ncclUniqueId commIdIB;
                // ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commIdIB));
                multiNetComms->commsIB = (ncclComm_t*)calloc(sizeof(ncclComm_t), this->nVis);
                (void) setenv("NCCL_NET", "IB", 1);
                if (ncclCommInitAll(multiNetComms->commsIB, this->nVis, NULL) != ncclSuccess) {
                    std::cout << "ncclGroupEnd() failed when trying to init IB communicators. Skipping test." << std::endl;
                    // This platform doesn't have IB network, so mark the test as skipped here
                    multiNetComms->testMultiNet = false;
                } else {
                    multiNetComms->testMultiNet = true;

                    ncclUniqueId commIdSocket;
                    // ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&commIdSocket));
                    multiNetComms->commsSocket = (ncclComm_t*)calloc(sizeof(ncclComm_t), this->nVis);
                    (void) setenv("NCCL_NET", "Socket", 1);
                    // ASSERT_EQ(ncclSuccess, ncclCommInitAll(multiNetComms->commsSocket, this->nVis, NULL));
                }

                (void) unsetenv("NCCL_NET");
            }

            return multiNetComms;
        };
};

TYPED_TEST_CASE(ncclAllReduce_test, testDataTypes);
// typical usage.
TYPED_TEST(ncclAllReduce_test, basic) {
    for (ncclRedOp_t op : this->RedOps) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                                    std::min(this->N, 1024 * 1024),
                                    this->DataType(), op,
                                    this->comms[i], this->streams[i]))
                << "op: " << op << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
};
TYPED_TEST(ncclAllReduce_test, host_mem) {
    for (ncclRedOp_t op : this->RedOps) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(
                ncclInvalidArgument,
                ncclAllReduce(this->sendbuffs_host[i], this->recvbuffs_host[i],
                              std::min(this->N, 1024 * 1024),
                              this->DataType(), op,
                              this->comms[i], this->streams[i]))
                << "op: " << op << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclInvalidArgument, ncclGroupEnd());
    }
};
TYPED_TEST(ncclAllReduce_test, pinned_mem) {
    for (ncclRedOp_t op : this->RedOps) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(
                          this->sendbuffs_pinned_device[i],
                          this->recvbuffs_pinned_device[i],
                          std::min(this->N, 1024 * 1024),
                          this->DataType(), op,
                          this->comms[i], this->streams[i]))
                << "op: " << op << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
};
TYPED_TEST(ncclAllReduce_test, stream_null) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(
                      this->sendbuffs[i], this->recvbuffs[i],
                      std::min(this->N, 1024 * 1024),
                      this->DataType(), ncclSum,
                      this->comms[i], NULL))
            << ", " << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclAllReduce_test, stream_default) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(
                      this->sendbuffs[i], this->recvbuffs[i],
                      std::min(this->N, 1024 * 1024),
                      this->DataType(), ncclSum,
                      this->comms[i], cudaStreamDefault))
            << ", " << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclAllReduce_test, stream_legacy) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(
                      this->sendbuffs[i], this->recvbuffs[i],
                      std::min(this->N, 1024 * 1024),
                      this->DataType(), ncclSum,
                      this->comms[i], cudaStreamLegacy))
            << ", " << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclAllReduce_test, stream_per_thread) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(
                      this->sendbuffs[i], this->recvbuffs[i],
                      std::min(this->N, 1024 * 1024),
                      this->DataType(), ncclSum,
                      this->comms[i], cudaStreamPerThread))
            << ", " << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
// sendbuff
TYPED_TEST(ncclAllReduce_test, sendbuf_null) {
    int i = 0;
    EXPECT_EQ(ncclInvalidArgument,
              ncclAllReduce(NULL, this->recvbuffs[i],
                            std::min(this->N, 1024 * 1024),
                            this->DataType(), this->RedOps[0],
                            this->comms[i], this->streams[i]));
};
// recvbuff
TYPED_TEST(ncclAllReduce_test, recvbuf_null) {
    int i = 0;
    EXPECT_EQ(ncclInvalidArgument,
              ncclAllReduce(this->sendbuffs[i], NULL,
                            std::min(this->N, 1024 * 1024),
                            this->DataType(), this->RedOps[0],
                            this->comms[i], this->streams[i]));
};
// sendbuff and recvbuff not on the same device
TYPED_TEST(ncclAllReduce_test, sendbuff_recvbuff_diff_device) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidArgument,
              ncclAllReduce(this->sendbuffs[i], this->recvbuffs[j],
                            std::min(this->N, 1024 * 1024),
                            this->DataType(), this->RedOps[0],
                            this->comms[i], this->streams[i]));
};
// N
TYPED_TEST(ncclAllReduce_test, N_zero) {
    for (ncclRedOp_t op : this->RedOps) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i], 0,
                                    this->DataType(), this->RedOps[0],
                                    this->comms[i], this->streams[i]))
                << "op: " << op << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
};
// data type
TYPED_TEST(ncclAllReduce_test, DataType_wrong) {
    int i = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N, 1024 * 1024),
                            ncclNumTypes, this->RedOps[0],
                            this->comms[i], this->streams[i]));
};
// op
TYPED_TEST(ncclAllReduce_test, op_wrong) {
    int i = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N, 1024 * 1024),
                            this->DataType(), ncclNumOps,
                            this->comms[i], this->streams[i]));
};
// comm
TYPED_TEST(ncclAllReduce_test, comm_null) {
    int i = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N, 1024 * 1024),
                            this->DataType(), this->RedOps[0],
                            NULL, this->streams[i]));
};
TYPED_TEST(ncclAllReduce_test, comm_wrong) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidArgument,
              ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N, 1024 * 1024),
                            this->DataType(), this->RedOps[0],
                            this->comms[j], this->streams[i]));
};
// STREAM can be NULL.
// stream on a diff device
TYPED_TEST(ncclAllReduce_test, DISABLED_stream_wrong) {
    int i = 0, j = 1;
    ASSERT_EQ(ncclInvalidArgument,
              ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N, 1024 * 1024),
                            this->DataType(), this->RedOps[0],
                            this->comms[i], this->streams[j]));
};

// Aggregation
// Only for 2.2 or higher
#if NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >=2)
TYPED_TEST(ncclAllReduce_test, aggregate_two_level_group_call) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (ncclRedOp_t op : this->RedOps) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                                    std::min(this->N, 1024 * 1024),
                                    this->DataType(), op,
                                    this->comms[i], this->streams[i]))
                << "op: " << op << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclAllReduce_test, aggregate_one_level_group_call) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (ncclRedOp_t op : this->RedOps) {
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                                    std::min(this->N, 1024 * 1024),
                                    this->DataType(), op,
                                    this->comms[i], this->streams[i]))
                << "op: " << op << ", "
                << "i" << i << ", " << std::endl;
        }
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclAllReduce_test, aggregate_ll_singleRing_multiRing) {
    int sizes[5] = { 1024, 32768, 512*1024, 32768, 1024 };  //ll, single-ring, multi-ring
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int k = 0; k < 5; k++) {
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                      ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                                    sizes[k],
                                    this->DataType(), ncclSum,
                                    this->comms[i], this->streams[i]))
                << "size: " << sizes[k] << ", "
                << "i" << i << ", " << std::endl;
        }
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
#endif

TYPED_TEST(ncclAllReduce_test, multi_net) {
    MultiNetComms* mnet = ncclAllReduce_test::ncclAllReduce_GetMultiNetComms();
    if (!mnet->testMultiNet) {
        std::cout << "Skipping test." << std::endl;
        return;
    }

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                    ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                                std::min(this->N, 1024 * 1024),
                                this->DataType(), ncclSum,
                                mnet->commsIB[i], this->streams[i]))
            << "IB op: " << ncclSum << ", "
            << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());

    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                    ncclAllReduce(this->sendbuffs[i], this->recvbuffs[i],
                                std::min(this->N, 1024 * 1024),
                                this->DataType(), ncclSum,
                                mnet->commsSocket[i], this->streams[i]))
            << "Socket op: " << ncclSum << ", "
            << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};

// EOF
