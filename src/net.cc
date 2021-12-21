#include "net.h"
#include "bootstrap.h"
#include "checks.h"

#include <string.h>
#include <errno.h>
#include <dlfcn.h>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

ncclNet_t ncclNet;
ncclCollNet_t ncclCollNet;

static ncclNet_v5_t ncclNet_v4_as_v5;
static ncclNet_v4_t ncclNet_v4;
static ncclCollNet_v5_t ncclCollNet_v4_as_v5;
static ncclCollNet_v4_t ncclCollNet_v4;

static ncclResult_t ncclNet_v4_as_v5_getProperties(int dev, ncclNetProperties_v5_t* props) {
  ncclNetProperties_v4_t p4;
  ncclResult_t ans = ncclNet_v4.getProperties(dev, &p4);
  if(ans != ncclSuccess) return ans;
  props->name = p4.name;
  props->pciPath = p4.pciPath;
  props->guid = p4.guid;
  props->ptrSupport = p4.ptrSupport;
  props->speed = p4.speed;
  props->port = p4.port;
  props->maxComms = p4.maxComms;
  props->maxRecvs = 1;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v4_as_v5_isend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) {
  return ncclNet_v4.isend(sendComm, data, size, mhandle, request);
}

static ncclResult_t ncclNet_v4_as_v5_irecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) {
  if(n == 0) return ncclSuccess;
  if(n > 1) return ncclInvalidArgument;
  return ncclNet_v4.irecv(recvComm, data[0], sizes[0], mhandles[0], request);
}

static ncclResult_t ncclNet_v4_as_v5_iflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  if(n == 0) return ncclSuccess;
  if(n > 1) return ncclInvalidArgument;
  return ncclNet_v4.iflush(recvComm, data[0], sizes[0], mhandles[0], request);
}

static ncclResult_t ncclCollNet_v4_as_v5_getProperties(int dev, ncclNetProperties_v5_t* props) {
  ncclNetProperties_v4_t p4;
  ncclResult_t ans = ncclCollNet_v4.getProperties(dev, &p4);
  if(ans != ncclSuccess) return ans;
  props->name = p4.name;
  props->pciPath = p4.pciPath;
  props->guid = p4.guid;
  props->ptrSupport = p4.ptrSupport;
  props->speed = p4.speed;
  props->port = p4.port;
  props->maxComms = p4.maxComms;
  props->maxRecvs = 1;
  return ncclSuccess;
}

static void initPlugin(ncclNet_v5_t** net, ncclCollNet_v5_t** collnet) {
  char ncclNetPluginName[128];
  const char* envPluginName = getenv("NCCL_NET_PLUGIN");
  if (envPluginName && strlen(envPluginName)) {
    snprintf(ncclNetPluginName, 128, "libnccl-net-%s.so", envPluginName);
    INFO(NCCL_INIT, "Plugin name set by env to %s\n", ncclNetPluginName);
  } else {
    sprintf(ncclNetPluginName, "libnccl-net.so");
  }
  void* netPluginLib = dlopen(ncclNetPluginName, RTLD_NOW | RTLD_LOCAL);
  if (netPluginLib == nullptr) {
    // dlopen does not guarantee to set errno, but dlerror only gives us a
    // string, so checking errno doesn't hurt to try to provide a better
    // error message
    if (errno == ENOENT) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin : No plugin found (%s), using internal implementation", ncclNetPluginName);
    } else {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin : Plugin load returned %d : %s.", errno, dlerror());
    }
    return;
  }

  *net = (ncclNet_v5_t*)dlsym(netPluginLib, "ncclNetPlugin_v5");
  if (*net == nullptr) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin_v5 symbol.");
    ncclNet_v4_t *v4 = (ncclNet_v4_t*)dlsym(netPluginLib, "ncclNetPlugin_v4");
    if (v4 == nullptr) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin_v4 symbol.");
      if (netPluginLib != nullptr) dlclose(netPluginLib);
      return;
    }
    ncclNet_v4 = *v4;
    *net = &ncclNet_v4_as_v5;
    ncclNet_v4_as_v5.name = v4->name;
    ncclNet_v4_as_v5.init = v4->init;
    ncclNet_v4_as_v5.devices = v4->devices;
    ncclNet_v4_as_v5.getProperties = ncclNet_v4_as_v5_getProperties;
    ncclNet_v4_as_v5.listen = v4->listen;
    ncclNet_v4_as_v5.connect = v4->connect;
    ncclNet_v4_as_v5.accept = v4->accept;
    ncclNet_v4_as_v5.regMr = v4->regMr;
    ncclNet_v4_as_v5.deregMr = v4->deregMr;
    ncclNet_v4_as_v5.isend = ncclNet_v4_as_v5_isend;
    ncclNet_v4_as_v5.irecv = ncclNet_v4_as_v5_irecv;
    ncclNet_v4_as_v5.iflush = ncclNet_v4_as_v5_iflush;
    ncclNet_v4_as_v5.test = v4->test;
    ncclNet_v4_as_v5.closeSend = v4->closeSend;
    ncclNet_v4_as_v5.closeRecv = v4->closeRecv;
    ncclNet_v4_as_v5.closeListen = v4->closeListen;
  }

  // Check for CollNet
  *collnet = (ncclCollNet_v5_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v5");
  if (*collnet == nullptr) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin_v5 symbol.");
    ncclCollNet_v4_t *v4 = (ncclCollNet_v4_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v4");
    if (v4 == nullptr) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin_v4 symbol.");
    } else {
      ncclCollNet_v4 = *v4;
      *collnet = &ncclCollNet_v4_as_v5;
      ncclCollNet_v4_as_v5.name = v4->name;
      ncclCollNet_v4_as_v5.init = v4->init;
      ncclCollNet_v4_as_v5.devices = v4->devices;
      ncclCollNet_v4_as_v5.getProperties = ncclCollNet_v4_as_v5_getProperties;
      ncclCollNet_v4_as_v5.listen = v4->listen;
      ncclCollNet_v4_as_v5.connect = v4->connect;
      ncclCollNet_v4_as_v5.reduceSupport = v4->reduceSupport;
      ncclCollNet_v4_as_v5.regMr = v4->regMr;
      ncclCollNet_v4_as_v5.deregMr = v4->deregMr;
      ncclCollNet_v4_as_v5.iallreduce = v4->iallreduce;
      ncclCollNet_v4_as_v5.iflush = v4->iflush;
      ncclCollNet_v4_as_v5.test = v4->test;
      ncclCollNet_v4_as_v5.closeColl = v4->closeColl;
      ncclCollNet_v4_as_v5.closeListen = v4->closeListen;
    }
  }
  return;
}

ncclResult_t ncclNetInit() {
  // Always initialize bootstrap network
  NCCLCHECK(bootstrapNetInit());

  // Initialize main communication network
  ncclNet_t* nets[3] = { nullptr, &ncclNetIb, &ncclNetSocket };
  ncclCollNet_t* collNets[3] = { nullptr, nullptr, nullptr };
  initPlugin(&nets[0], &collNets[0]);
  char* netName = getenv("NCCL_NET");
  bool ok = false;

  for (int i=0; i<3; i++) {
    if (nets[i] == nullptr) continue;
    if (netName && strcmp(netName, nets[i]->name) != 0) continue;

    // net plugin is already initialized
    int ndev;
    if (nets[i]->init(ncclDebugLog) != ncclSuccess) continue;
    if (nets[i]->devices(&ndev) != ncclSuccess) continue;
    if (ndev <= 0) continue;
    ncclNet = *nets[i];
    ok = true;

    if (collNets[i]) {
      do {
        if (collNets[i]->init(ncclDebugLog) != ncclSuccess) break;
        if (collNets[i]->devices(&ndev) != ncclSuccess) break;
        if (ndev <= 0) break;
        ncclCollNet = *collNets[i];
      } while(0);
    }
    break;
  }

  if (!ok) {
    WARN("Error: network %s not found.", netName ? netName : "");
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}
