/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nvmlwrap.h"
#include "checks.h"
#include "debug.h"

#include <mutex>
#include <initializer_list>

#if NCCL_NVML_DIRECT
  #define NCCL_NVML_FN(name, rettype, arglist) constexpr rettype(*pfn_##name)arglist = name;
#else
  #include <dlfcn.h>
  #define NCCL_NVML_FN(name, rettype, arglist) static rettype(*pfn_##name)arglist = nullptr;
#endif

namespace {
  NCCL_NVML_FN(nvmlInit, nvmlReturn_t, ())
  NCCL_NVML_FN(nvmlInit_v2, nvmlReturn_t, ())
  NCCL_NVML_FN(nvmlShutdown, nvmlReturn_t, ())
  NCCL_NVML_FN(nvmlDeviceGetHandleByPciBusId, nvmlReturn_t, (const char* pciBusId, nvmlDevice_t* device))
  NCCL_NVML_FN(nvmlDeviceGetHandleByIndex, nvmlReturn_t, (unsigned int index, nvmlDevice_t *device))
  NCCL_NVML_FN(nvmlDeviceGetIndex, nvmlReturn_t, (nvmlDevice_t device, unsigned* index))
  NCCL_NVML_FN(nvmlErrorString, char const*, (nvmlReturn_t r))
  NCCL_NVML_FN(nvmlDeviceGetNvLinkState, nvmlReturn_t, (nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive))
  NCCL_NVML_FN(nvmlDeviceGetNvLinkRemotePciInfo, nvmlReturn_t, (nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci))
  NCCL_NVML_FN(nvmlDeviceGetNvLinkCapability, nvmlReturn_t, (nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int *capResult))
  NCCL_NVML_FN(nvmlDeviceGetCudaComputeCapability, nvmlReturn_t, (nvmlDevice_t device, int* major, int* minor))
  NCCL_NVML_FN(nvmlDeviceGetP2PStatus, nvmlReturn_t, (nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t* p2pStatus))

  static std::mutex lock; // NVML has had some thread safety bugs
}

namespace {
static ncclResult_t ensureInitialized() {
  ncclResult_t res = ncclSuccess;

  static bool initialized = false;
  if(initialized) return res;

  #if !NCCL_NVML_DIRECT
  if (pfn_nvmlInit == nullptr) {
    void *libhandle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (libhandle == nullptr) {
      WARN("Failed to open libnvidia-ml.so.1");
      res = ncclSystemError;
    }
    else {
      struct Symbol { void **ppfn; char const *name; };
      std::initializer_list<Symbol> symbols = {
        {(void**)&pfn_nvmlInit, "nvmlInit"},
        {(void**)&pfn_nvmlInit_v2, "nvmlInit_v2"},
        {(void**)&pfn_nvmlShutdown, "nvmlShutdown"},
        {(void**)&pfn_nvmlDeviceGetHandleByPciBusId, "nvmlDeviceGetHandleByPciBusId"},
        {(void**)&pfn_nvmlDeviceGetHandleByIndex, "nvmlDeviceGetHandleByIndex"},
        {(void**)&pfn_nvmlDeviceGetIndex, "nvmlDeviceGetIndex"},
        {(void**)&pfn_nvmlErrorString, "nvmlErrorString"},
        {(void**)&pfn_nvmlDeviceGetNvLinkState, "nvmlDeviceGetNvLinkState"},
        {(void**)&pfn_nvmlDeviceGetNvLinkRemotePciInfo, "nvmlDeviceGetNvLinkRemotePciInfo"},
        {(void**)&pfn_nvmlDeviceGetNvLinkCapability, "nvmlDeviceGetNvLinkCapability"},
        {(void**)&pfn_nvmlDeviceGetCudaComputeCapability, "nvmlDeviceGetCudaComputeCapability"},
        {(void**)&pfn_nvmlDeviceGetP2PStatus, "nvmlDeviceGetP2PStatus"}
      };
      for(Symbol sym: symbols) {
        *sym.ppfn = dlsym(libhandle, sym.name);
      }
    }
  }
  #endif

  if (res == ncclSuccess) {
    #if NCCL_NVML_DIRECT
      bool have_v2 = true;
    #else
      bool have_v2 = pfn_nvmlInit_v2 != nullptr; // if this compare is done in the NCCL_NVML_DIRECT=1 case then GCC warns about it never being null
    #endif
    nvmlReturn_t res1 = (have_v2 ? pfn_nvmlInit_v2 : pfn_nvmlInit)();
    if (res1 != NVML_SUCCESS) {
      WARN("%s() failed: %s", have_v2 ? "nvmlInit_v2" : "nvmlInit", pfn_nvmlErrorString(res1));
      res = ncclSystemError;
    }
  }

  initialized = true;
  return res;
}}

#define NVMLCHECK(name, ...) do { \
  nvmlReturn_t e44241808 = pfn_##name(__VA_ARGS__); \
  if (e44241808 != NVML_SUCCESS) { \
    WARN(#name "() failed: %s", pfn_nvmlErrorString(e44241808)); \
    return ncclSystemError; \
  } \
} while(0)

#define NVMLTRY(name, ...) do { \
  if (!NCCL_NVML_DIRECT && pfn_##name == nullptr) \
    return ncclInternalError; /* missing symbol is not a warned error */ \
  nvmlReturn_t e44241808 = pfn_##name(__VA_ARGS__); \
  if (e44241808 != NVML_SUCCESS) { \
    if (e44241808 != NVML_ERROR_NOT_SUPPORTED) \
      INFO(NCCL_INIT, #name "() failed: %s", pfn_nvmlErrorString(e44241808)); \
    return ncclSystemError; \
  } \
} while(0)

ncclResult_t wrapNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device) {
  std::lock_guard<std::mutex> locked(lock);
  NCCLCHECK(ensureInitialized());
  NVMLCHECK(nvmlDeviceGetHandleByPciBusId, pciBusId, device);
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
  std::lock_guard<std::mutex> locked(lock);
  NCCLCHECK(ensureInitialized());
  NVMLCHECK(nvmlDeviceGetHandleByIndex, index, device);
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index) {
  std::lock_guard<std::mutex> locked(lock);
  NCCLCHECK(ensureInitialized());
  NVMLCHECK(nvmlDeviceGetIndex, device, index);
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive) {
  std::lock_guard<std::mutex> locked(lock);
  NCCLCHECK(ensureInitialized());
  NVMLTRY(nvmlDeviceGetNvLinkState, device, link, isActive);
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) {
  std::lock_guard<std::mutex> locked(lock);
  NCCLCHECK(ensureInitialized());
  NVMLTRY(nvmlDeviceGetNvLinkRemotePciInfo, device, link, pci);
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetNvLinkCapability(
    nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability,
    unsigned int *capResult
  ) {
  std::lock_guard<std::mutex> locked(lock);
  NCCLCHECK(ensureInitialized());
  NVMLTRY(nvmlDeviceGetNvLinkCapability, device, link, capability, capResult);
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) {
  std::lock_guard<std::mutex> locked(lock);
  NCCLCHECK(ensureInitialized());
  NVMLCHECK(nvmlDeviceGetCudaComputeCapability, device, major, minor);
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetP2PStatus(
    nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex,
    nvmlGpuP2PStatus_t* p2pStatus
  ) {
  std::lock_guard<std::mutex> locked(lock);
  NCCLCHECK(ensureInitialized());
  NVMLCHECK(nvmlDeviceGetP2PStatus, device1, device2, p2pIndex, p2pStatus);
  return ncclSuccess;
}
