/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NVLINK_H_
#define NCCL_NVLINK_H_

#include <sys/stat.h>
#include <fcntl.h>

#define CONNECT_NVLINK 0x10
#define CONNECT_NVSWITCH 0x100

enum ncclNvLinkDeviceType {
  ncclNvLinkDeviceGpu,
  ncclNvLinkDeviceSwitch,
  ncclNvLinkDeviceCpu
};

static ncclResult_t ncclDeviceType(const char* busId, enum ncclNvLinkDeviceType* type) {
  char busPath[] =  "/sys/bus/pci/devices/0000:00:00.0";
  memcpy(busPath+sizeof("/sys/bus/pci/devices/")-1, busId, sizeof("0000:00")-1);

  char pathname[MAXPATHSIZE];
  strcpy(pathname, "/sys/bus/pci/devices/");
  int strLen = strlen(pathname);
  int linkLen = readlink(busPath, pathname+strLen, MAXPATHSIZE-strLen);
  if (linkLen == 0) {
    WARN("Could not find link %s", busPath);
    return ncclSystemError;
  }
  // readlink does not append '\0'. We have to do it.
  pathname[strLen+linkLen] = '\0';
  char* rPath = realpath(pathname, NULL);
  strncpy(pathname, rPath, MAXPATHSIZE);
  free(rPath);
  strncpy(pathname+strlen(pathname), "/class", MAXPATHSIZE-strlen(pathname));
  int fd;
  SYSCHECKVAL(open(pathname, O_RDONLY), "open", fd);
  char pciClass[9];
  strncpy(pciClass, "0x000000", 9);
  int len;
  SYSCHECKVAL(read(fd, pciClass, 8), "read", len);
  SYSCHECK(close(fd), "close");
  if (strcmp(pciClass, "0x068000") == 0) {
    // PCI device is of type "Bridge / Other Bridge Device" (NVswitch)
    *type = ncclNvLinkDeviceSwitch;
  } else if (strcmp(pciClass, "0x030200") == 0 // "3D Controller" (Tesla)
      || strcmp(pciClass, "0x030000") == 0) {  // "VGA Controller" (GeForce)
    *type = ncclNvLinkDeviceGpu;
  } else {
    *type = ncclNvLinkDeviceCpu;
  }
  return ncclSuccess;
}

static int getNvlinkGpu(const char* busId1, const char* busId2) {
  // Determine if that connection is through NVLink
  int links = 0;
  int nvswitch_links = 0;
  nvmlDevice_t nvmlDev;
  ncclResult_t res = wrapNvmlDeviceGetHandleByPciBusId(busId1, &nvmlDev);
  if (res != ncclSuccess) return 0;

  for(int l=0; l<NVML_NVLINK_MAX_LINKS; ++l) {
    // nvmlDeviceGetNvLinkState() reports whether a link is enabled or not.
    // Works only on Pascal and later
    nvmlEnableState_t linkState;
    if (wrapNvmlDeviceGetNvLinkState(nvmlDev, l, &linkState) != ncclSuccess) return 0;
    if (linkState == NVML_FEATURE_DISABLED) continue;

    // nvmlDeviceGetNvLinkCapability(NVML_NVLINK_CAP_P2P_SUPPORTED) would seem to
    // report whether the NVLink connects to a peer GPU (versus a POWER CPU?). I
    // don't know whether nvmlDeviceGetNvLinkRemotePciInfo() would succeed in
    // the POWER CPU case, so it seems best to check this as well.
    unsigned canP2P;
    if ((wrapNvmlDeviceGetNvLinkCapability(nvmlDev, l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) || !canP2P) continue;

    // nvmlDeviceGetNvLinkRemotePciInfo() will return NVML_ERROR_NOT_SUPPORTED
    // if the links don't exist, or are disabled. So checking for that return
    // here would probably make the nvmlDeviceGetNvLinkState check above
    // redundant. Presumably, we still need to check the P2P capability above,
    // since even non-GPUs would posses PCI info.
    nvmlPciInfo_t remoteProc;
    if (wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, l, &remoteProc) != ncclSuccess) continue;
    
    // Determine if the remote side is NVswitch, another GPU, or a CPU
    enum ncclNvLinkDeviceType type;
    if (ncclDeviceType(remoteProc.busId, &type) != ncclSuccess) continue;

    if (type == ncclNvLinkDeviceGpu && strncmp(busId2, remoteProc.busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE) == 0) {
      links++;
    } else if (type == ncclNvLinkDeviceSwitch) {
      nvswitch_links++;
    }
  }
  return nvswitch_links ? -nvswitch_links : links;
}

static int getNvlinkCpu() {
  int links = 0;
  int nvswitch_links = 0;
  int cudaDev;
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  if (cudaGetDevice(&cudaDev) != cudaSuccess) return 0;
  if (cudaDeviceGetPCIBusId(busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, cudaDev) != cudaSuccess) return 0;
  if (wrapNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev) != ncclSuccess) return 0;

  for(int l=0; l<NVML_NVLINK_MAX_LINKS; ++l) {
    // Determine if the remote side is NVswitch, another GPU, or a CPU
    enum ncclNvLinkDeviceType type;

    // nvmlDeviceGetNvLinkState() reports whether a link is enabled or not.
    // Works only on Pascal and later
    nvmlEnableState_t linkState;
    if (wrapNvmlDeviceGetNvLinkState(nvmlDev, l, &linkState) != ncclSuccess) return 0;
    if (linkState == NVML_FEATURE_DISABLED) continue;

    // nvmlDeviceGetNvLinkCapability(NVML_NVLINK_CAP_P2P_SUPPORTED) would seem to
    // report whether the NVLink connects to a peer GPU (versus a POWER CPU?). I
    // don't know whether nvmlDeviceGetNvLinkRemotePciInfo() would succeed in
    // the POWER CPU case, so it seems best to check this as well.
    unsigned canP2P;
    if ((wrapNvmlDeviceGetNvLinkCapability(nvmlDev, l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) || !canP2P) continue;

    // nvmlDeviceGetNvLinkRemotePciInfo() will return NVML_ERROR_NOT_SUPPORTED
    // if the links don't exist, or are disabled. So checking for that return
    // here would probably make the nvmlDeviceGetNvLinkState check above
    // redundant. Presumably, we still need to check the P2P capability above,
    // since even non-GPUs would posses PCI info.
    //
    // update:
    // nvmlDeviceGetNvLinkRemotePciInfo() will return NVML_ERROR_NOT_SUPPORTED
    // if the other side of the NVLink is a CPU (e.g. a POWER CPU)
    nvmlPciInfo_t remoteProc;
    if (wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, l, &remoteProc) != ncclSuccess && remoteProc != NULL) {
      type == ncclNvLinkDeviceCpu;
    } else {
      if (ncclDeviceType(remoteProc.busId, &type) != ncclSuccess) continue;
    }

    if (type == ncclNvLinkDeviceCpu) {
      links++;
    } else if (type == ncclNvLinkDeviceSwitch) {
      nvswitch_links++;
    }
  }
  return nvswitch_links ? -nvswitch_links : links;
}

#endif
