#include <cstddef>
#include "nccl_device.h"

// These tests are designed to catch backwards incompatible changes to Device API structs.
// The tests do not run; compilation catches any issues.

// Instructions:
// - For a new field added to the bottom of an existing struct,
//   add the field to the test and increment the expected size.
//   If the expanded structure was embedded in another structure or passed by value to the kernel,
//   then changes to the backwards compatibility layer in src/devcomm/ may be needed.
// - For a backwards-incompatible change, the test for the original
//   struct will probably need to move to the backwards compatibility layer in src/devcomm/.
// - Other edits should be made with extreme caution to ensure
//   backwards-compatibility.

void ncclDevComm_backwards_compat_test() {
  // To ensure backwards compatibility, edit this test only according to the instructions above.
  static_assert(offsetof(ncclDevComm_t, rank) == 8);
  static_assert(offsetof(ncclDevComm_t, nRanks) == 12);
  static_assert(offsetof(ncclDevComm_t, nRanks_rcp32) == 16);
  static_assert(offsetof(ncclDevComm_t, lsaRank) == 20);
  static_assert(offsetof(ncclDevComm_t, lsaSize) == 24);
  static_assert(offsetof(ncclDevComm_t, lsaSize_rcp32) == 28);
  static_assert(offsetof(ncclDevComm_t, windowTable) == 32);
  static_assert(offsetof(ncclDevComm_t, resourceWindow) == 40);
  static_assert(offsetof(ncclDevComm_t, resourceWindow_inlined) == 48);
  static_assert(offsetof(ncclDevComm_t, lsaMultimem) == 120);
  static_assert(offsetof(ncclDevComm_t, lsaBarrier) == 128);
  static_assert(offsetof(ncclDevComm_t, railGinBarrier) == 136);
  static_assert(offsetof(ncclDevComm_t, ginConnectionCount) == 144);
  static_assert(offsetof(ncclDevComm_t, ginNetDeviceTypes) == 145);
  static_assert(offsetof(ncclDevComm_t, ginHandles) == 152);
  static_assert(offsetof(ncclDevComm_t, ginSignalCount) == 184);
  static_assert(offsetof(ncclDevComm_t, ginCounterCount) == 188);
  static_assert(offsetof(ncclDevComm_t, ginSignalShadows) == 192);
  static_assert(offsetof(ncclDevComm_t, ginContextCount) == 200);
  static_assert(offsetof(ncclDevComm_t, ginIsRailed) == 204);
  static_assert(offsetof(ncclDevComm_t, abortFlag) == 208);
  static_assert(offsetof(ncclDevComm_t, hybridLsaBarrier) == 216);
  static_assert(offsetof(ncclDevComm_t, hybridRailGinBarrier) == 224);
  static_assert(offsetof(ncclDevComm_t, worldGinBarrier) == 232);

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(ncclDevComm_t) == 240);

  static_assert(offsetof(struct ncclDevCommWindowTable, entries) == 0);
  static_assert(offsetof(struct ncclDevCommWindowTable, entries[0].base) == 0);
  static_assert(offsetof(struct ncclDevCommWindowTable, entries[0].size) == 8);
  static_assert(offsetof(struct ncclDevCommWindowTable, entries[0].window) == 16);
  static_assert(offsetof(struct ncclDevCommWindowTable, next) == 768);

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(struct ncclDevCommWindowTable) == 776);

  static_assert(offsetof(ncclWindow_vidmem, winHost) == 0);
  static_assert(offsetof(ncclWindow_vidmem, lsaFlatBase) == 8);
  static_assert(offsetof(ncclWindow_vidmem, lsaRank) == 16);
  static_assert(offsetof(ncclWindow_vidmem, worldRank) == 20);
  static_assert(offsetof(ncclWindow_vidmem, stride4G) == 24);
  static_assert(offsetof(ncclWindow_vidmem, mcOffset4K) == 28);
  static_assert(offsetof(ncclWindow_vidmem, ginOffset4K) == 32);
  static_assert(offsetof(ncclWindow_vidmem, ginWins) == 40);

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(ncclWindow_vidmem) == 72);

  static_assert(offsetof(ncclMultimemHandle_t, mcBasePtr) == 0);

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(ncclMultimemHandle_t) == 8);

  static_assert(offsetof(ncclLsaBarrierHandle_t, bufHandle) == 0);
  static_assert(offsetof(ncclLsaBarrierHandle_t, nBarriers) == 4);

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(ncclLsaBarrierHandle_t) == 8);

  static_assert(offsetof(ncclGinBarrierHandle_t, signal0) == 0);

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(ncclGinBarrierHandle_t) == 8);
}

void ncclDevCommRequirements_backwards_compat_test() {
  // To ensure backwards compatibility, edit this test only according to the instructions above.
  static_assert(offsetof(ncclDevCommRequirements_t, size) == 0);
  static_assert(offsetof(ncclDevCommRequirements_t, magic) == 8);
  static_assert(offsetof(ncclDevCommRequirements_t, version) == 12);
  static_assert(offsetof(ncclDevCommRequirements_t, resourceRequirementsList) == 16);
  static_assert(offsetof(ncclDevCommRequirements_t, teamRequirementsList) == 24);
  static_assert(offsetof(ncclDevCommRequirements_t, lsaMultimem) == 32);
  static_assert(offsetof(ncclDevCommRequirements_t, barrierCount) == 36);
  static_assert(offsetof(ncclDevCommRequirements_t, lsaBarrierCount) == 40);
  static_assert(offsetof(ncclDevCommRequirements_t, railGinBarrierCount) == 44);
  static_assert(offsetof(ncclDevCommRequirements_t, lsaLLA2ABlockCount) == 48);
  static_assert(offsetof(ncclDevCommRequirements_t, lsaLLA2ASlotCount) == 52);
  static_assert(offsetof(ncclDevCommRequirements_t, ginForceEnable) == 56);
  static_assert(offsetof(ncclDevCommRequirements_t, ginContextCount) == 60);
  static_assert(offsetof(ncclDevCommRequirements_t, ginSignalCount) == 64);
  static_assert(offsetof(ncclDevCommRequirements_t, ginCounterCount) == 68);
  static_assert(offsetof(ncclDevCommRequirements_t, ginConnectionType) == 72);
  static_assert(offsetof(ncclDevCommRequirements_t, ginExclusiveContexts) == 76);
  static_assert(offsetof(ncclDevCommRequirements_t, ginQueueDepth) == 80);

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(ncclDevCommRequirements_t) == 88);

  static_assert(offsetof(ncclDevResourceRequirements_t, next) == 0);
  static_assert(offsetof(ncclDevResourceRequirements_t, bufferSize) == 8);
  static_assert(offsetof(ncclDevResourceRequirements_t, bufferAlign) == 16);
  static_assert(offsetof(ncclDevResourceRequirements_t, outBufferHandle) == 24);
  static_assert(offsetof(ncclDevResourceRequirements_t, ginSignalCount) == 32);
  static_assert(offsetof(ncclDevResourceRequirements_t, ginCounterCount) == 36);
  static_assert(offsetof(ncclDevResourceRequirements_t, outGinSignalStart) == 40);
  static_assert(offsetof(ncclDevResourceRequirements_t, outGinCounterStart) == 48);

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(ncclDevResourceRequirements_t) == 56);

  static_assert(offsetof(ncclTeamRequirements_t, next) == 0);
  static_assert(offsetof(ncclTeamRequirements_t, team) == 8);
  static_assert(offsetof(ncclTeamRequirements_t, multimem) == 20);
  static_assert(offsetof(ncclTeamRequirements_t, outMultimemHandle) == 24);

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(ncclTeamRequirements_t) == 32);

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(ncclGinConnectionType_t) == 4);
}

void ncclCommProperties_backwards_compat_test() {
  // To ensure backwards compatibility, edit this test only according to the instructions above.
  static_assert(offsetof(ncclCommProperties_t, size) == 0);
  static_assert(offsetof(ncclCommProperties_t, magic) == 8);
  static_assert(offsetof(ncclCommProperties_t, version) == 12);
  static_assert(offsetof(ncclCommProperties_t, rank) == 16);
  static_assert(offsetof(ncclCommProperties_t, nRanks) == 20);
  static_assert(offsetof(ncclCommProperties_t, cudaDev) == 24);
  static_assert(offsetof(ncclCommProperties_t, nvmlDev) == 28);
  static_assert(offsetof(ncclCommProperties_t, deviceApiSupport) == 32);
  static_assert(offsetof(ncclCommProperties_t, multimemSupport) == 33);
  static_assert(offsetof(ncclCommProperties_t, ginType) == 36);
  static_assert(offsetof(ncclCommProperties_t, nLsaTeams) == 40);
  static_assert(offsetof(ncclCommProperties_t, hostRmaSupport) == 44);
  static_assert(offsetof(ncclCommProperties_t, railedGinType) == 48);

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(ncclCommProperties_t) == 56);

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(ncclGinConnectionType_t) == 4);
}
