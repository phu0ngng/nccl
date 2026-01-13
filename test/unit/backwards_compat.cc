#include <cstddef>
#include "nccl_device.h"

// These tests are designed to catch backwards incompatible changes to Device API structs.
// The tests do not run; compilation catches any issues.

// Instructions:
// - For a new field added to the bottom of an existing struct,
//   add the field to the test and increment the expected size.
// - For a backwards-incompatible change, change the version of the
//   struct by defining a new struct object. Add a test for the new
//   struct and keep the test for the original struct.
// - Other edits should be made with extreme caution to ensure
//   backwards-compatibility.

void ncclDevComm_backwards_compat_test() {
  // To ensure backwards compatibility, edit this test only according to the instructions above.
  static_assert(offsetof(ncclDevComm_t, rank) == 0);
  static_assert(offsetof(ncclDevComm_t, nRanks) == 4);
  static_assert(offsetof(ncclDevComm_t, nRanks_rcp32) == 8);
  static_assert(offsetof(ncclDevComm_t, lsaRank) == 12);
  static_assert(offsetof(ncclDevComm_t, lsaSize) == 16);
  static_assert(offsetof(ncclDevComm_t, lsaSize_rcp32) == 20);
  static_assert(offsetof(ncclDevComm_t, windowTable) == 24);
  static_assert(offsetof(ncclDevComm_t, resourceWindow) == 32);
  static_assert(offsetof(ncclDevComm_t, resourceWindow_inlined) == 40);
  static_assert(offsetof(ncclDevComm_t, lsaMultimem) == 112);
  static_assert(offsetof(ncclDevComm_t, lsaBarrier) == 120);
  static_assert(offsetof(ncclDevComm_t, railGinBarrier) == 128);
  static_assert(offsetof(ncclDevComm_t, ginContextCount) == 136);
  static_assert(offsetof(ncclDevComm_t, ginNetDeviceTypes) == 137);
  static_assert(offsetof(ncclDevComm_t, ginHandles) == 144);
  static_assert(offsetof(ncclDevComm_t, ginSignalBase) == 176);
  static_assert(offsetof(ncclDevComm_t, ginSignalCount) == 180);
  static_assert(offsetof(ncclDevComm_t, ginCounterBase) == 184);
  static_assert(offsetof(ncclDevComm_t, ginCounterCount) == 188);
  static_assert(offsetof(ncclDevComm_t, ginSignalShadows) == 192);

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(ncclDevComm_t) == 200);
}


void ncclWindow_vidmem_backwards_compat_test() {
  // To ensure backwards compatibility, edit this test only according to the instructions above.
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

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(ncclDevCommRequirements_t) == 72);
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
  static_assert(offsetof(ncclCommProperties_t, ginType) == 34);

  // This check prompts users to update the test. Edit according to the instructions above.
  static_assert(sizeof(ncclCommProperties_t) == 40);
}