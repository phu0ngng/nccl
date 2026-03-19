#pragma once

#include "ncclDevApiCommon_test.cuh"
#include "../ncclCommon_test.cuh"
#include <cstring>
#include <stdexcept>
#include <string>

// Common base for multi-node/multi-team Device API tests: uses a dedicated
// multiTeamComms array created with NCCL_LSA_TEAM_SIZE=2 (via ParameterChanger).
// All tests must have "multi" in the test name. Returns testSkipped if
// ncclCommQueryProperties reports < 2 teams.
class ncclMultiTeamCommon_test : public ncclDevApiCommon_test {
public:
  static ncclComm_t* multiTeamComms;

  static void SetUpTestCase() {
    ncclDevApiCommon_test::SetUpTestCase();
    {
      ParameterChanger param("NCCL_LSA_TEAM_SIZE", "2");
      multiTeamComms = (ncclComm_t*)calloc(nVis, sizeof(ncclComm_t));
      EXPECT_NE(nullptr, multiTeamComms);
      EXPECT_EQ(ncclSuccess, ncclCommInitAll(multiTeamComms, nVis, NULL));
      
      // Initialize devrState so that lsaTeamSize is initialized while NCCL_LSA_TEAM_SIZE is set to 2.
      for (int i = 0; i < nVis; i++) {
        ncclTeamLsa(multiTeamComms[i]);
      }
    }
  }

  static void TearDownTestCase() {
    if (multiTeamComms != nullptr) {
      for (int i = 0; i < nVis; i++) {
        ncclCommDestroy(multiTeamComms[i]);
      }
      free(multiTeamComms);
      multiTeamComms = nullptr;
    }
    ncclDevApiCommon_test::TearDownTestCase();
  }

  void destroyDevComms() override {
    destroyDevCommsShared(multiTeamComms);
  }

  // Return testSkipped if comm has < 2 LSA teams; otherwise delegate to base.
  TestResult_t createDevComms(const ncclDevCommRequirements& reqs) override {
    if (!testNameContains(kMultiTeamTestNameSubstring)) {
      printf("Multi-team test does not contain '%s'\n", kMultiTeamTestNameSubstring);
      return TestResult_t::testError;  // Multi-team tests must have "multi" in the test name
    }
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    ncclResult_t res = ncclCommQueryProperties(multiTeamComms[0], &props);
    if (res != ncclSuccess) {
      fprintf(stderr, "ncclCommQueryProperties failed\n");
      return TestResult_t::testError;
    }
    
    // The environment variable logic is a bit finicky. Add a check here to ensure
    // NCCL_LSA_TEAM_SIZE=2 is actually respected.
    if (nVis > 2 && props.nLsaTeams == 1) {
      printf("Expected multiple LSA teams but got only 1\n");
      return TestResult_t::testError;
    }

    if (props.nLsaTeams < 2) {
      return TestResult_t::testSkipped;
    }
    return ncclDevApiCommon_test::createDevCommsShared(reqs, multiTeamComms);
  }
};
