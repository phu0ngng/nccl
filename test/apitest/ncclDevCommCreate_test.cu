#include "ncclCommon_test.cuh"
#include "nccl_device.h"

class ncclDevCommCreate_test : public ::testing::Test {
  protected:
    ncclComm_t comm;
  virtual void SetUp() {
    ncclUniqueId id;
    ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
    ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comm, 1, id, 0));
  }
  virtual void TearDown() {
    ncclCommDestroy(comm);
  }
};

TEST_F(ncclDevCommCreate_test, basic) {
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  ncclDevComm_t dcomm;
  ASSERT_EQ(ncclSuccess, ncclDevCommCreate(comm, &reqs, &dcomm));
  ASSERT_EQ(ncclSuccess, ncclDevCommDestroy(comm, &dcomm));
}

TEST_F(ncclDevCommCreate_test, null_reqs) {
  ncclDevComm_t dcomm;
  ASSERT_EQ(ncclInvalidArgument, ncclDevCommCreate(comm, nullptr, &dcomm));
}

TEST_F(ncclDevCommCreate_test, null_comm) {
  ncclDevComm_t dcomm;
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  ASSERT_EQ(ncclInvalidArgument, ncclDevCommCreate(nullptr, &reqs, &dcomm));
}

TEST_F(ncclDevCommCreate_test, uninitialized_reqs) {
  ncclDevComm_t dcomm;
  ncclDevCommRequirements reqs;
  ASSERT_EQ(ncclInvalidUsage, ncclDevCommCreate(comm, &reqs, &dcomm));
}
