class ncclCommSplit_test : public ::testing::Test {
  protected:
    ncclComm_t* comms = NULL;
    ncclComm_t* comms2 = NULL;
    int nVis = 0;
    virtual void SetUp() {
        (void) setenv("NCCL_CHECK_POINTERS", "1", 0); // API tests expect this behaviour (ncclCommSplit)
        ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&nVis));
        comms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
        comms2 = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
        ASSERT_EQ(ncclSuccess, ncclCommInitAll(comms, nVis, NULL));
    };
    virtual void TearDown() {
        if (NULL != comms) {
            for (int i = 0; i < nVis; ++i) {
                ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[i]));
                comms[i] = NULL;
            }
            free(comms);
            comms = NULL;
        }
        if (NULL != comms2) {
            for (int i = 0; i < nVis; ++i) {
                if (comms2[i]) {
                    ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms2[i]));
                    comms2[i] = NULL;
                }
            }
            free(comms);
            free(comms2);
            comms = NULL;
            comms2 = NULL;
        }
    };
};
#if NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >=16)
TEST_F(ncclCommSplit_test, comm_dup) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i=0; i<nVis; i++) {
        ASSERT_EQ(ncclSuccess, ncclCommSplit(comms[i], 0, i, &comms2[i], NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    for (int i=0; i<nVis; i++) ASSERT_NE((long)comms2[i], NULL);
}
TEST_F(ncclCommSplit_test, same_key) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i=0; i<nVis; i++) {
        ASSERT_EQ(ncclSuccess, ncclCommSplit(comms[i], 0, 0, &comms2[i], NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    for (int i=0; i<nVis; i++) ASSERT_NE((long)comms2[i], NULL);
}
TEST_F(ncclCommSplit_test, half) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    int split = nVis/2;
    for (int i=0; i<nVis; i++) {
        ASSERT_EQ(ncclSuccess, ncclCommSplit(comms[i], i/(split), i%split, &comms2[i], NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    for (int i=0; i<nVis; i++) ASSERT_NE((long)comms2[i], NULL);
}
TEST_F(ncclCommSplit_test, reverse) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i=0; i<nVis; i++) {
        ASSERT_EQ(ncclSuccess, ncclCommSplit(comms[i], 0, nVis-1-i, &comms2[i], NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    for (int i=0; i<nVis; i++) ASSERT_NE((long)comms2[i], NULL);
}
TEST_F(ncclCommSplit_test, comm_partial) {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int i=0; i<nVis; i++) {
        ASSERT_EQ(ncclSuccess, ncclCommSplit(comms[i], i<2 ? 0 : -1, i, &comms2[i], NULL));
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    for (int i=0; i<nVis; i++) {
        if (i<2) {
            ASSERT_NE((long)comms2[i], NULL);
        } else {
            ASSERT_EQ((long)comms2[i], NULL);
        }
    }
}
#endif
// EOF
