#include "ncclCommon_test.cuh"
#include "nccl.h"
#include "nccl_device.h"

class ncclCommQueryProperties_test : public ncclCommon_test<char> {
  protected:
    ncclDevComm_t* devComms = nullptr;

    void SetUp() override {
      ncclCommon_test<char>::SetUp();
      comms = ncclCommon_getComms(&nVis);
      devComms = (ncclDevComm_t*)calloc(nVis, sizeof(ncclDevComm_t));
      ASSERT_NE(nullptr, devComms);
    }
    void TearDown() override {
      if (devComms != nullptr) {
        for (int i = 0; i < nVis; i++) {
          ASSERT_EQ(ncclSuccess, ncclDevCommDestroy(comms[i], &devComms[i]));
        }
        free(devComms);
        devComms = nullptr;
      }
      ncclCommon_destroyComms();
    }
};

TEST_F(ncclCommQueryProperties_test, basic) {
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  ASSERT_EQ(ncclSuccess, ncclCommQueryProperties(comms[0], &props));
  ASSERT_EQ(props.rank, 0);
  ASSERT_EQ(props.nRanks, nVis);
  ASSERT_EQ(props.cudaDev, 0);
}

TEST_F(ncclCommQueryProperties_test, test_unitialized_props) {
  ncclCommProperties_t props;
  ASSERT_EQ(ncclInvalidUsage, ncclCommQueryProperties(comms[0], &props));
}

TEST_F(ncclCommQueryProperties_test, null_comm) {
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  ASSERT_EQ(ncclInvalidArgument, ncclCommQueryProperties(nullptr, &props));
}

TEST_F(ncclCommQueryProperties_test, null_props) {
  ASSERT_EQ(ncclInvalidArgument, ncclCommQueryProperties(comms[0], nullptr));
}

TEST_F(ncclCommQueryProperties_test, test_gin_support) {
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  ASSERT_EQ(ncclSuccess, ncclCommQueryProperties(comms[0], &props));
  ncclDevCommRequirements ginReqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  ginReqs.ginForceEnable = true;

  // We expect devComm creation to fail if we request gin resource but gin is not supported
  bool expectSuccess = props.ginType != NCCL_GIN_TYPE_NONE;
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess, ncclDevCommCreate(comms[i], &ginReqs, &devComms[i]));
  }
  if (expectSuccess) {
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  } else {
    ASSERT_NE(ncclSuccess, ncclGroupEnd());
  }
}


TEST_F(ncclCommQueryProperties_test, test_multimem_support) {
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  ASSERT_EQ(ncclSuccess, ncclCommQueryProperties(comms[0], &props));

  ncclDevCommRequirements nvlsReqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  nvlsReqs.lsaMultimem = true;

  // We expect devComm creation to fail if we request multimem resource but multimem is not supported
  bool expectSuccess = props.multimemSupport;
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int i = 0; i < nVis; i++) {
    ASSERT_EQ(ncclSuccess,ncclDevCommCreate(comms[i], &nvlsReqs, &devComms[i]));
  }
  if (expectSuccess) {
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  } else {
    ASSERT_NE(ncclSuccess, ncclGroupEnd());
  }
}

