#include "ncclCommon_test.cuh"
template <typename DT>
class ncclSendRecv_test : public ncclCommon_test<DT> {};
TYPED_TEST_CASE(ncclSendRecv_test, testDataTypes);
#if NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >=7)
// typical usage.
// DISABLED tests should work with NCCL_LAUNCH_MODE=PARALLEL
// coop launch doesn't support incomplete sets of ranks,
// nor different numbers of blocks.
TYPED_TEST(ncclSendRecv_test, DISABLED_simple) {
    size_t size = std::min(this->N, 1024 * 1024);
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    if (this->nVis >= 2) {
      ASSERT_EQ(ncclSuccess,
          ncclSend(this->sendbuffs[0], size,
            this->DataType(), 1,
            this->comms[0], this->streams[0]))
        << "i" << 0 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess,
          ncclRecv(this->recvbuffs[1], size,
            this->DataType(), 0,
            this->comms[1], this->streams[1]))
        << "i" << 1 << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclSendRecv_test, ring) {
    size_t size = std::min(this->N, 1024 * 1024);
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        ASSERT_EQ(ncclSuccess,
                  ncclSend(this->sendbuffs[i], size,
                                this->DataType(),
                                (i+1) % this->nVis,
                                this->comms[i], this->streams[i]))
            << "i" << i << ", " << std::endl;
        ASSERT_EQ(ncclSuccess,
                  ncclRecv(this->recvbuffs[i], size,
                                this->DataType(),
                                (i-1+this->nVis) % this->nVis,
                                this->comms[i], this->streams[i]))
            << "i" << i << ", " << std::endl;
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclSendRecv_test, alltoall) {
    size_t size = std::min(this->N, 1024 * 1024) / this->nVis;
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        for (int p = 0; p < this->nVis; ++p) {
            ASSERT_EQ(ncclSuccess,
                      ncclSend(this->sendbuffs[i] + p * size, size,
                                    this->DataType(), p,
                                    this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
            ASSERT_EQ(ncclSuccess,
                      ncclRecv(this->recvbuffs[i] + p * size, size,
                                    this->DataType(), p,
                                    this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
         }
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclSendRecv_test, alltoallv) {
    size_t maxSize = std::min(this->N, 1024 * 1024) / this->nVis;
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        for (int p = 0; p < this->nVis; ++p) {
            size_t size = std::max((size_t)0, maxSize-p-i);
            ASSERT_EQ(ncclSuccess,
                      ncclSend(this->sendbuffs[i] + p * size, size,
                                    this->DataType(), p,
                                    this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
            ASSERT_EQ(ncclSuccess,
                      ncclRecv(this->recvbuffs[i] + p * size, size,
                                    this->DataType(), p,
                                    this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
         }
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclSendRecv_test, DISABLED_scatter) {
    size_t size = std::min(this->N, 1024 * 1024) / this->nVis;
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        for (int p = 0; p < this->nVis; ++p) {
            if (i == 0) {
                ASSERT_EQ(ncclSuccess,
                          ncclSend(this->sendbuffs[i] + p * size, size,
                                        this->DataType(), p,
                                        this->comms[i], this->streams[i]))
                    << "i" << i << ", " << std::endl;
            }
            if (p == 0) {
                ASSERT_EQ(ncclSuccess,
                          ncclRecv(this->recvbuffs[i] + p * size, size,
                                        this->DataType(), p,
                                        this->comms[i], this->streams[i]))
                    << "i" << i << ", " << std::endl;
            }
         }
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclSendRecv_test, DISABLED_gather) {
    size_t size = std::min(this->N, 1024 * 1024) / this->nVis;
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        for (int p = 0; p < this->nVis; ++p) {
            if (p == 0) {
                ASSERT_EQ(ncclSuccess,
                          ncclSend(this->sendbuffs[i] + p * size, size,
                                        this->DataType(), p,
                                        this->comms[i], this->streams[i]))
                    << "i" << i << ", " << std::endl;
            }
            if (i == 0) {
                ASSERT_EQ(ncclSuccess,
                          ncclRecv(this->recvbuffs[i] + p * size, size,
                                        this->DataType(), p,
                                        this->comms[i], this->streams[i]))
                    << "i" << i << ", " << std::endl;
            }
         }
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclSendRecv_test, DISABLED_hypercube) {
    size_t size = std::min(this->N, 1024 * 1024) / this->nVis;
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        for (int mask=1; mask<this->nVis; mask <<= 1) {
            int p = i^mask;
            if (p > this->nVis) continue;
            ASSERT_EQ(ncclSuccess,
                      ncclSend(this->sendbuffs[i] + p * size, size,
                                    this->DataType(), p,
                                    this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
            ASSERT_EQ(ncclSuccess,
                      ncclRecv(this->recvbuffs[i] + p * size, size,
                                    this->DataType(), p,
                                    this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
         }
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
// send to self (without a recv)
TYPED_TEST(ncclSendRecv_test, send_self) {
    EXPECT_EQ(ncclInvalidUsage,
            ncclSend(this->sendbuffs[0], 1,
                this->DataType(), 0, this->comms[0], this->streams[0]));
};
// send to self (with a recv)
TYPED_TEST(ncclSendRecv_test, DISABLED_sendrecv_self) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    EXPECT_EQ(ncclSuccess,
            ncclSend(this->sendbuffs[0], std::min(this->N, 1024 * 1024),
                this->DataType(), 0, this->comms[0], this->streams[0]));
    EXPECT_EQ(ncclSuccess,
            ncclRecv(this->recvbuffs[0], std::min(this->N, 1024 * 1024),
                this->DataType(), 0, this->comms[0], this->streams[0]));
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
// multi ops per pair
TYPED_TEST(ncclSendRecv_test, multi_ops_per_pair) {
    int nSegs = 2;
    size_t size = std::min(this->N, 1024 * 1024) / nSegs;
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        for (int s = 0; s < nSegs; ++s) {
            ASSERT_EQ(ncclSuccess,
                      ncclSend(this->sendbuffs[i] + s * size, size,
                                    this->DataType(),
                                    (i+1) % this->nVis,
                                    this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
            ASSERT_EQ(ncclSuccess,
                      ncclRecv(this->recvbuffs[i] + s * size, size,
                                    this->DataType(),
                                    (i-1+this->nVis) % this->nVis,
                                    this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
         }
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
// sendbuff
TYPED_TEST(ncclSendRecv_test, sendbuf_null) {
    int i = 0;
    EXPECT_EQ(ncclInvalidArgument,
            ncclSend(NULL, std::min(this->N, 1024 * 1024),
                this->DataType(), i, this->comms[i], this->streams[i]));
};
// recvbuff
TYPED_TEST(ncclSendRecv_test, recvbuf_null) {
    int i = 0;
    EXPECT_EQ(ncclInvalidArgument,
            ncclRecv(NULL, std::min(this->N, 1024 * 1024),
                this->DataType(), i, this->comms[i], this->streams[i]));
};
// data type
TYPED_TEST(ncclSendRecv_test, send_type_wrong) {
    int i = 0;
    ASSERT_EQ(ncclInvalidArgument,
            ncclSend(this->sendbuffs[i], std::min(this->N, 1024 * 1024),
                ncclNumTypes, i, this->comms[i], this->streams[i]));
};
TYPED_TEST(ncclSendRecv_test, recv_type_wrong) {
    int i = 0;
    ASSERT_EQ(ncclInvalidArgument,
            ncclRecv(this->recvbuffs[i], std::min(this->N, 1024 * 1024),
                ncclNumTypes, i, this->comms[i], this->streams[i]));
};
// comm
TYPED_TEST(ncclSendRecv_test, send_comm_null) {
    int i = 0;
    ASSERT_EQ(ncclInvalidArgument,
            ncclSend(this->sendbuffs[i], std::min(this->N, 1024 * 1024),
                  this->DataType(), i, NULL, this->streams[i]));
};
TYPED_TEST(ncclSendRecv_test, recv_comm_null) {
    int i = 0;
    ASSERT_EQ(ncclInvalidArgument,
            ncclRecv(this->recvbuffs[i], std::min(this->N, 1024 * 1024),
                  this->DataType(), i, NULL, this->streams[i]));
};
#endif
// EOF
