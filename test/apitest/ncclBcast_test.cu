#include "ncclCommon_test.cuh"
template <typename DT>
class ncclBcast_test : public ncclCommon_test<DT> {};
TYPED_TEST_CASE(ncclBcast_test, testNoType);
// typical usage.
TYPED_TEST(ncclBcast_test, basic) {
    for (int root = 0; root < this->nVis; ++root) {
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        for (int i = 0; i < this->nVis; ++i) {
            ASSERT_EQ(ncclSuccess,
                      ncclBcast(this->sendbuffs[i],
                                std::min(this->N, 1024 * 1024),
                                this->DataType(), root,
                                this->comms[i], this->streams[i]))
                << "root: " << root << ", "
                << "i" << i << ", " << std::endl;
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());
    }
};
// EOF
