#include "ncclCommon_test.cuh"
template <typename DT>
class ncclSendRecv_test : public ncclCommon_test<DT> {};
TYPED_TEST_CASE(ncclSendRecv_test, testDataTypes);
#if NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >=7)
// typical usage.
TYPED_TEST(ncclSendRecv_test, simple) {
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
TYPED_TEST(ncclSendRecv_test, alltoallv_JoC) {
   /* In BUG 3197885 this AlltoAllv pattern was found to causes hangs
    * on DGX A100 and DGX2
    */
    size_t sendCount[8][8] = {
      1048576, 1048576, 1048576, 1048576, 1048576, 1048576, 1048576, 1048576,
      4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304,
      4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304,
      4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304,
      4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304,
      4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304,
      4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304,
      2097152, 2097152, 2097152, 2097152, 2097152, 2097152, 2097152, 2097152,
    };
    size_t recvCount[8][8] = {
      1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
      1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
      1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
      1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
      1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
      1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
      1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
      1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
    };
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    size_t maxSize = this->N;
    for (int i = 0; i < this->nVis; ++i) {
        for (int p = 0; p < this->nVis; ++p) {
            size_t sendSize = std::min(maxSize, sendCount[i%8][p%8]);
            ASSERT_EQ(ncclSuccess,
                      ncclSend(this->sendbuffs[i], sendSize,
                                    this->DataType(), p,
                                    this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
            size_t recvSize = std::min(maxSize, recvCount[i%8][p%8]);
            ASSERT_EQ(ncclSuccess,
                      ncclRecv(this->recvbuffs[i], recvSize,
                                    this->DataType(), p,
                                    this->comms[i], this->streams[i]))
                << "i" << i << ", " << std::endl;
         }
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
};
TYPED_TEST(ncclSendRecv_test, alltoallv_vasp) {
   /* In BUG 3571899 this AlltoAllv pattern was found to causes hangs
    */
    size_t size = this->N;
    size_t sendCount[4] = { size, 0, 0, 0 };
    size_t recvCount[4] = { size, size, size, size };
    size_t maxSize = this->N;
    for (int i = 0; i < 16; i++) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int r = 0; r < std::min(4, this->nVis); ++r) {
            size_t sendSize = std::min(maxSize, sendCount[r]);
            ASSERT_EQ(ncclSuccess,
                    ncclSend(this->sendbuffs[0], sendSize,
                        this->DataType(), r,
                        this->comms[0], this->streams[0]))
                << "Send 0->" << r << ", " << std::endl;
            size_t recvSize = std::min(maxSize, recvCount[r]);
            ASSERT_EQ(ncclSuccess,
                    ncclRecv(this->recvbuffs[0], recvSize,
                        this->DataType(), r,
                        this->comms[0], this->streams[0]))
                << "Recv 0<-" << r << ", " << std::endl;
        }
        for (int r = 1; r < std::min(4, this->nVis); ++r) {
            size_t sendSize = std::min(maxSize, sendCount[r]);
            ASSERT_EQ(ncclSuccess,
                    ncclRecv(this->recvbuffs[r], sendSize,
                        this->DataType(), 0,
                        this->comms[r], this->streams[r]))
                << "Send " << r << "->0, " << std::endl;
            size_t recvSize = std::min(maxSize, recvCount[r]);
            ASSERT_EQ(ncclSuccess,
                    ncclSend(this->sendbuffs[r], recvSize,
                        this->DataType(), 0,
                        this->comms[r], this->streams[r]))
                << "Recv" << r << "<-0, " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int r = 0; r < std::min(4, this->nVis); ++r) {
            ASSERT_EQ(ncclSuccess,
                    ncclSend(this->sendbuffs[0], size,
                        this->DataType(), r,
                        this->comms[0], this->streams[0]))
                << "Send 0->" << r << ", " << std::endl;
            ASSERT_EQ(ncclSuccess,
                    ncclRecv(this->recvbuffs[0], size,
                        this->DataType(), r,
                        this->comms[0], this->streams[0]))
                << "Recv 0<-" << r << ", " << std::endl;
        }
        for (int r = 1; r < std::min(4, this->nVis); ++r) {
            ASSERT_EQ(ncclSuccess,
                    ncclRecv(this->recvbuffs[r], size,
                        this->DataType(), 0,
                        this->comms[r], this->streams[r]))
                << "Send " << r << "->0, " << std::endl;
            ASSERT_EQ(ncclSuccess,
                    ncclSend(this->sendbuffs[r], size,
                        this->DataType(), 0,
                        this->comms[r], this->streams[r]))
                << "Recv" << r << "<-0, " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
};
TYPED_TEST(ncclSendRecv_test, scatter) {
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
TYPED_TEST(ncclSendRecv_test, gather) {
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
TYPED_TEST(ncclSendRecv_test, hypercube) {
    size_t size = std::min(this->N, 1024 * 1024) / this->nVis;
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i = 0; i < this->nVis; ++i) {
        for (int mask=1; mask<this->nVis; mask <<= 1) {
            int p = i^mask;
            if (p >= this->nVis) continue;
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
TYPED_TEST(ncclSendRecv_test, sendrecv_self) {
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
// Test for NVB preconnect
TYPED_TEST(ncclSendRecv_test, preconnect) {
    size_t size = std::min(this->N, 1024 * 1024);
    if (this->nVis >= 8) {
      // Make sure 1<->0 and 1<->5 are already connected
      ASSERT_EQ(ncclSuccess, ncclGroupStart());
      ASSERT_EQ(ncclSuccess, ncclSend(this->sendbuffs[0], size, this->DataType(), 1, this->comms[0], this->streams[0])) << "i" << 0 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess, ncclRecv(this->recvbuffs[0], size, this->DataType(), 1, this->comms[0], this->streams[0])) << "i" << 0 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess, ncclSend(this->sendbuffs[1], size, this->DataType(), 0, this->comms[1], this->streams[1])) << "i" << 1 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess, ncclRecv(this->recvbuffs[1], size, this->DataType(), 0, this->comms[1], this->streams[1])) << "i" << 1 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess, ncclGroupEnd());
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(this->streams[0])) << std::endl;
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(this->streams[1])) << std::endl;
      ASSERT_EQ(ncclSuccess, ncclGroupStart());
      ASSERT_EQ(ncclSuccess, ncclSend(this->sendbuffs[1], size, this->DataType(), 5, this->comms[1], this->streams[1])) << "i" << 1 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess, ncclRecv(this->recvbuffs[1], size, this->DataType(), 5, this->comms[1], this->streams[1])) << "i" << 1 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess, ncclSend(this->sendbuffs[5], size, this->DataType(), 1, this->comms[5], this->streams[5])) << "i" << 5 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess, ncclRecv(this->recvbuffs[5], size, this->DataType(), 1, this->comms[5], this->streams[5])) << "i" << 5 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess, ncclGroupEnd());
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(this->streams[1])) << std::endl;
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(this->streams[5])) << std::endl;
      // Connect 1->4. On a cubemesh, this should allocate data through 0 or 5, while 0 and 5 should have
      // a blocking receive started which should block the remote allocation
      ASSERT_EQ(ncclSuccess, ncclGroupStart());
      ASSERT_EQ(ncclSuccess, ncclRecv(this->recvbuffs[0], size, this->DataType(), 1, this->comms[0], this->streams[0])) << "i" << 0 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess, ncclSend(this->sendbuffs[1], size, this->DataType(), 0, this->comms[1], this->streams[1])) << "i" << 1 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess, ncclRecv(this->recvbuffs[5], size, this->DataType(), 1, this->comms[5], this->streams[5])) << "i" << 5 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess, ncclSend(this->sendbuffs[1], size, this->DataType(), 5, this->comms[1], this->streams[1])) << "i" << 1 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess, ncclRecv(this->recvbuffs[1], size, this->DataType(), 4, this->comms[1], this->streams[1])) << "i" << 1 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess, ncclSend(this->sendbuffs[4], size, this->DataType(), 1, this->comms[4], this->streams[4])) << "i" << 4 << ", " << std::endl;
      ASSERT_EQ(ncclSuccess, ncclGroupEnd());
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(this->streams[0])) << std::endl;
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(this->streams[1])) << std::endl;
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(this->streams[4])) << std::endl;
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(this->streams[5])) << std::endl;
    }
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
