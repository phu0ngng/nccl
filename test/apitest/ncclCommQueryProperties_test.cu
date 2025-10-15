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
  ncclCommProperties_t props;
  ASSERT_EQ(ncclSuccess, ncclCommQueryProperties(comms[0], &props));
}

TEST_F(ncclCommQueryProperties_test, null_comm) {
  ncclCommProperties_t props;
  ASSERT_EQ(ncclInvalidArgument, ncclCommQueryProperties(nullptr, &props));
}

TEST_F(ncclCommQueryProperties_test, null_props) {
  ASSERT_EQ(ncclInvalidArgument, ncclCommQueryProperties(comms[0], nullptr));
}

TEST_F(ncclCommQueryProperties_test, test_gin_support) {
  ncclCommProperties_t props;
  ASSERT_EQ(ncclSuccess, ncclCommQueryProperties(comms[0], &props));
  ncclDevCommRequirements ginReqs = {};
  ginReqs.ginForceEnable = true;

  // We expect devComm creation to fail if we request gin resource but gin is not supported
  bool expectSuccess = props.ginSupport;
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
  ncclCommProperties_t props;
  ASSERT_EQ(ncclSuccess, ncclCommQueryProperties(comms[0], &props));

  ncclDevCommRequirements nvlsReqs = {};
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

