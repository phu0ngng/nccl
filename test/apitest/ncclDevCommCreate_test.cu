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

TEST_F(ncclDevCommCreate_test, invalid_gin_connection_type) {
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  ASSERT_EQ(ncclSuccess, ncclCommQueryProperties(comm, &props));

  ncclDevComm_t dcomm;

  auto validateResult = [&](ncclDevCommRequirements* reqs) {
    if (props.deviceApiSupport) {
      ASSERT_EQ(ncclInvalidArgument, ncclDevCommCreate(comm, reqs, &dcomm));
    } else {
      ASSERT_EQ(ncclInvalidUsage, ncclDevCommCreate(comm, reqs, &dcomm));
    }
  };

  {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginSignalCount = 1;
    validateResult(&reqs);
  }

  {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginCounterCount = 1;
    validateResult(&reqs);
  }

  {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.railGinBarrierCount = 1;
    validateResult(&reqs);
  }

  {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.barrierCount = 1;
    validateResult(&reqs);
  }
}

void validateBasedOnDeviceApiSupport(ncclComm_t& comm, const ncclCommProperties_t& props,
                                     ncclDevCommRequirements* reqs) {
  ncclDevComm_t dcomm;
  if (props.ginType != NCCL_GIN_TYPE_NONE) {
    ASSERT_EQ(ncclSuccess, ncclDevCommCreate(comm, reqs, &dcomm));
    ASSERT_EQ(ncclSuccess, ncclDevCommDestroy(comm, &dcomm));
  } else {
    if (props.deviceApiSupport) {
      ASSERT_EQ(ncclInvalidArgument, ncclDevCommCreate(comm, reqs, &dcomm));
    } else {
      ASSERT_EQ(ncclInvalidUsage, ncclDevCommCreate(comm, reqs, &dcomm));
    }
  }
}

TEST_F(ncclDevCommCreate_test, valid_gin_connection_type) {
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  ASSERT_EQ(ncclSuccess, ncclCommQueryProperties(comm, &props));

  {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginSignalCount = 1;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    validateBasedOnDeviceApiSupport(comm, props, &reqs);
  }

  {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginCounterCount = 1;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    validateBasedOnDeviceApiSupport(comm, props, &reqs);
  }

  {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.railGinBarrierCount = 1;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    validateBasedOnDeviceApiSupport(comm, props, &reqs);
  }

  {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.barrierCount = 1;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    validateBasedOnDeviceApiSupport(comm, props, &reqs);
  }
}

TEST_F(ncclDevCommCreate_test, gin_force_enable) {
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  ASSERT_EQ(ncclSuccess, ncclCommQueryProperties(comm, &props));

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginSignalCount = 1;
  reqs.ginForceEnable = true;
  validateBasedOnDeviceApiSupport(comm, props, &reqs);
}

TEST_F(ncclDevCommCreate_test, world_gin_barrier_with_railed_gin) {
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  ASSERT_EQ(ncclSuccess, ncclCommQueryProperties(comm, &props));
  if (props.ginType == NCCL_GIN_TYPE_NONE || !props.deviceApiSupport) {
    return;
  }

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.worldGinBarrierCount = 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
  ncclDevComm_t dcomm;
  ASSERT_EQ(ncclInvalidArgument, ncclDevCommCreate(comm, &reqs, &dcomm));
}