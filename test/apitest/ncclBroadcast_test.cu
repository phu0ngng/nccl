#include "ncclCommon_test.cuh"
template <typename DT>
class ncclBroadcast_test : public ncclCommon_test<DT> {};
TYPED_TEST_CASE(ncclBroadcast_test, testDataTypes);
// typical usage.
TYPED_TEST(ncclBroadcast_test, basic) {
    for (int root = 0; root < this->nVis; ++root) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                      ncclBroadcast(this->sendbuffs[i], this->recvbuffs[i],
                                std::min(this->N, 1024 * 1024),
                                this->DataType(), root,
                                this->comms[i], this->streams[i]))
                << "root: " << root << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
};
TYPED_TEST(ncclBroadcast_test, host_mem) {
    for (int root = 0; root < this->nVis; ++root) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(
                ncclInvalidArgument,
                ncclBroadcast(this->sendbuffs_host[i], this->recvbuffs_host[i],
                              std::min(this->N, 1024 * 1024),
                              this->DataType(), root,
                              this->comms[i], this->streams[i]))
                << "root: " << root << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclInvalidArgument, ncclGroupEnd());
    }
};
TYPED_TEST(ncclBroadcast_test, pinned_mem) {
    for (int root = 0; root < this->nVis; ++root) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                      ncclBroadcast(this->sendbuffs_pinned_device[i],
                                    this->recvbuffs_pinned_device[i],
                                    std::min(this->N, 1024 * 1024),
                                    this->DataType(), root,
                                    this->comms[i], this->streams[i]))
                << "root: " << root << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
};
TYPED_TEST(ncclBroadcast_test, stream_null) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                  ncclBroadcast(
                      this->sendbuffs[i], this->recvbuffs[i],
                      std::min(this->N, 1024 * 1024),
                      this->DataType(), 0,
                      this->comms[i], NULL))
            << ", " << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
// sendbuff
TYPED_TEST(ncclBroadcast_test, sendbuf_null) {
    int i = 0, root = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclBroadcast(NULL, this->recvbuffs[i],
                            std::min(this->N, 1024 * 1024),
                            this->DataType(), root,
                            this->comms[i], this->streams[i]));
};
TYPED_TEST(ncclBroadcast_test, sendbuf_wrong) {
    int i = 0, j = 1, root = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclBroadcast(this->sendbuffs[j], this->recvbuffs[i],
                            std::min(this->N, 1024 * 1024),
                            this->DataType(), root,
                            this->comms[i], this->streams[i]));
};
// recvbuff
// root can't be null
// non root can be null
TYPED_TEST(ncclBroadcast_test, sendbuf_root_null) {
   for (int root = 0; root < this->nVis; ++root) {
       ASSERT_EQ(ncclSuccess, ncclGroupStart());
       for (int i = 0; i < this->nVis; ++i) {
           ASSERT_EQ(root != i ? ncclSuccess : ncclInvalidArgument,
                     ncclBroadcast(NULL, this->recvbuffs[i],
                                std::min(this->N, 1024 * 1024),
                                this->DataType(), root,
                                this->comms[i], this->streams[i]))
               << "root: " << root << ", "
               << "i" << i << ", " << std::endl;
       }
       ASSERT_EQ(ncclInvalidArgument, ncclGroupEnd());
   }
};
TYPED_TEST(ncclBroadcast_test, sendbuff_nonroot_null) {
    for (int root = 0; root < this->nVis; ++root) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                      ncclBroadcast(root == i ? this->sendbuffs[i] : NULL,
                                 this->recvbuffs[i],
                                 std::min(this->N, 1024 * 1024),
                                 this->DataType(), root,
                                 this->comms[i], this->streams[i]))
                << "root: " << root << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
};
// sendbuff and recvbuff not on the same device
TYPED_TEST(ncclBroadcast_test, sendbuff_recvbuff_diff_device) {
    int i = 0, j = 1, root = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclBroadcast(this->sendbuffs[i], this->recvbuffs[j],
                            std::min(this->N, 1024 * 1024),
                            this->DataType(), root,
                            this->comms[i], this->streams[i]));
};
// N
TYPED_TEST(ncclBroadcast_test, N_zero) {
   for (int root = 0; root < this->nVis; ++root) {
       ASSERT_EQ(ncclSuccess, ncclGroupStart());
       for (int i = 0; i < this->nVis; ++i) {
           ASSERT_EQ(ncclSuccess,
                     ncclBroadcast(this->sendbuffs[i], this->recvbuffs[i], 0,
                                   this->DataType(), root,
                                   this->comms[i], this->streams[i]))
               << "root: " << root << ", "
               << "i" << i << ", " << std::endl;
       }
       ASSERT_EQ(ncclSuccess, ncclGroupEnd());
   }
};
// data type
TYPED_TEST(ncclBroadcast_test, DataType_wrong) {
    int i = 0, root = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclBroadcast(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N, 1024 * 1024),
                            ncclNumTypes, root,
                            this->comms[i], this->streams[i]));
};
// root
TYPED_TEST(ncclBroadcast_test, root_minus1) {
    int i = 0, root = -1;
    ASSERT_EQ(ncclInvalidArgument,
              ncclBroadcast(this->sendbuffs[i], this->recvbuffs[i],
                        std::min(this->N, 1024 * 1024),
                        this->DataType(), root,
                        this->comms[i], this->streams[i]));
};
TYPED_TEST(ncclBroadcast_test, root_toobig) {
    int i = 0, root = 1000;
    ASSERT_EQ(ncclInvalidArgument,
              ncclBroadcast(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N, 1024 * 1024),
                            this->DataType(), root,
                            this->comms[i], this->streams[i]));
};
// comm
TYPED_TEST(ncclBroadcast_test, comm_null) {
    int i = 0, root = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclBroadcast(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N, 1024 * 1024),
                            this->DataType(), root,
                            NULL, this->streams[i]));
};
TYPED_TEST(ncclBroadcast_test, comm_wrong) {
    int i = 0, j = 1, root = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclBroadcast(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N, 1024 * 1024),
                            this->DataType(), root,
                            this->comms[j], this->streams[i]));
};
// STREAM can be NULL.
// stream on a diff device
TYPED_TEST(ncclBroadcast_test, DISABLED_stream_wrong) {
    int i = 0, j = 1, root = 0;
    ASSERT_EQ(ncclInvalidArgument,
              ncclBroadcast(this->sendbuffs[i], this->recvbuffs[i],
                            std::min(this->N, 1024 * 1024),
                            this->DataType(), root,
                            this->comms[i], this->streams[j]));
};
// Aggregation
// Only for 2.2 or higher
#if NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >=2)
TYPED_TEST(ncclBroadcast_test, aggregate_two_level_group_call) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int root = 0; root < this->nVis; ++root) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                      ncclBroadcast(this->sendbuffs[i], this->recvbuffs[i],
                                    std::min(this->N, 1024 * 1024),
                                    this->DataType(), root,
                                    this->comms[i], this->streams[i]))
                << "root: " << root << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclBroadcast_test, aggregate_one_level_group_call) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int root = 0; root < this->nVis; ++root) {
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                      ncclBroadcast(this->sendbuffs[i], this->recvbuffs[i],
                                    std::min(this->N, 1024 * 1024),
                                    this->DataType(), root,
                                    this->comms[i], this->streams[i]))
                << "root: " << root << ", "
                << "i" << i << ", " << std::endl;
        }
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclBroadcast_test, aggregate_exchange_loops) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int root = 0; root < this->nVis; ++root) {
            ASSERT_EQ(ncclSuccess,
                      ncclBroadcast(this->sendbuffs[i], this->recvbuffs[i],
                                    std::min(this->N, 1024 * 1024),
                                    this->DataType(), root,
                                    this->comms[i], this->streams[i]))
                << "root: " << root << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
#endif
// EOF
