/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_COMMON_H_
#define NET_IB_COMMON_H_

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "graph.h"
#include "utils.h"
#include "param.h"
#include "profiler/net_ib.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#include <mutex>
#define ENABLE_TIMER 0
#include "timer.h"

#include "ibvwrap.h"
#include "mlx5/mlx5dvwrap.h"

#define MAXSUFFIXSIZE 16
#define MAXNAMESIZE (64 + MAXSUFFIXSIZE)
extern char ncclIbIfName[MAX_IF_NAME_SIZE+1];
extern union ncclSocketAddress ncclIbIfAddr;

struct ncclIbMr {
  uintptr_t addr;
  size_t pages;
  int refs;
  ibv_mr *mr;
};

struct ncclIbMrCache {
  struct ncclIbMr *slots;
  int capacity, population;
};

extern int ncclNMergedIbDevs;
#define NCCL_IB_MAX_DEVS_PER_NIC 4
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE*NCCL_IB_MAX_DEVS_PER_NIC)+NCCL_IB_MAX_DEVS_PER_NIC
struct alignas(64) ncclIbMergedDev {
  ncclNetVDeviceProps_t vProps;
  int speed;
  char devName[MAX_MERGED_DEV_NAME]; // Up to NCCL_IB_MAX_DEVS_PER_NIC * name size, and a character for each '+'
};

struct ncclIbStats {
  int fatalErrorCount;
};

enum ncclIbProvider {
  IB_PROVIDER_NONE = 0,
  IB_PROVIDER_MLX5 = 1,
  IB_PROVIDER_MAX = 2,
};

extern int ncclNIbDevs;
struct alignas(64) ncclIbDev {
  std::mutex mutex;
  int device;
  uint64_t guid;
  uint8_t portNum;
  uint8_t link;
  int speed;
  ibv_context* context;
  int pdRefs;
  ibv_pd* pd;
  char devName[MAXNAMESIZE];
  char* pciPath;
  char* virtualPciPath;
  int realPort;
  int maxQp;
  float latency;
  struct ncclIbMrCache mrCache;
  int ar; // ADAPTIVE_ROUTING
  struct ibv_port_attr portAttr;
  struct ncclIbStats stats;
  int dmaBufSupported;
  enum ncclIbProvider ibProvider;
  union {
    struct {
      int dataDirect;
    } mlx5;
  } capsProvider;
};

#define MAX_IB_DEVS  32
#define MAX_IB_VDEVS MAX_IB_DEVS*8
extern struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS];
extern struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
extern int ncclIbRelaxedOrderingEnabled;

#define NCCL_IB_LLSTR(ll) (((ll) == IBV_LINK_LAYER_INFINIBAND) ? "IB" : (((ll) == IBV_LINK_LAYER_ETHERNET) ? "RoCE" : "UNSPECIFIED"))

// Per-Dev connection metadata
struct ncclIbDevInfo {
  uint32_t lid;
  uint8_t ib_port;
  enum ibv_mtu mtu;
  uint8_t link_layer;

  // For RoCE and IB Rounter
  union ibv_gid gid;

  // FIFO RDMA info
  uint32_t fifoRkey;

  //remote dev info
  union ibv_gid remoteGid;
};

// Retain local RoCE address for error logging
struct ncclIbGidInfo {
  uint8_t link_layer;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

#define MAX_QPS_PER_REQ 8
struct ncclProfilerInfo {
  void* qpEventHandles[MAX_QPS_PER_REQ];
  int qpIndex[MAX_QPS_PER_REQ];
  int nEventHandles;
  ncclProfilerNetIbDescr_v1_t data;
  void* pHandle;
};

#define NCCL_NET_IB_MAX_RECVS 8

#define NCCL_NET_IB_REQ_UNUSED 0
#define NCCL_NET_IB_REQ_SEND 1
#define NCCL_NET_IB_REQ_RECV 2
#define NCCL_NET_IB_REQ_FLUSH 3
#define NCCL_NET_IB_REQ_GIN_IPUT 4
extern const char* ncclIbReqTypeStr[];

struct ncclIbRequest {
  struct ncclIbNetCommBase* base;
  int type;
  struct ncclSocket* sock;
  int events[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbNetCommDevBase* devBases[NCCL_IB_MAX_DEVS_PER_NIC];
#ifdef NCCL_ENABLE_NET_PROFILING
  struct ncclProfilerInfo pInfo[NCCL_NET_IB_MAX_RECVS];
#endif
  int nreqs;
  union {
    struct {
      int size;
      void* data;
      uint32_t lkeys[NCCL_IB_MAX_DEVS_PER_NIC];
      int offset;
    } send;
    struct {
      int* sizes;
    } recv;
    struct {
      int rank;
    } iput;
  };
};

struct ncclIbNetCommDevBase {
  int ibDevN;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  uint64_t pad[2];
  struct ncclIbGidInfo gidInfo;
};

struct ncclIbSendFifo {
  uint64_t addr;
  uint64_t size;
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t nreqs;
  uint32_t tag;
  uint64_t idx;
  char padding[16];
};

struct ncclIbQp {
  struct ibv_qp* qp;
  int devIndex;
  int remDevIdx;
};

// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define NET_IB_MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
static_assert(NET_IB_MAX_REQUESTS <= 256, "request id are encoded in wr_id and we need up to 8 requests ids per completion");

struct ncclIbRemSizesFifo {
  int elems[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t addr;
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
};

// A per-dev struct for netIbSendComm
struct alignas(8) ncclIbSendCommDev {
  struct ncclIbNetCommDevBase base;
  struct ibv_mr* fifoMr;
  struct ibv_mr* putSignalScratchpadMr;
  struct ibv_mr* sizesFifoMr;
  struct ibv_sge sge;
};


// Wrapper to track an MR per-device, if needed
struct ncclIbMrHandle {
  ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];
};

#define NCCL_IB_MAX_QPS 128

struct alignas(32) ncclIbNetCommBase {
  ncclNetVDeviceProps_t vProps;
  bool isSend;
  struct ncclIbRequest reqs[NET_IB_MAX_REQUESTS];
  struct ncclIbQp qps[NCCL_IB_MAX_QPS];
  uint64_t fifoHead;
  int nqps;
  int qpIndex;
  int devIndex;
  struct ncclSocket sock;
  int ready;
  // Track necessary remDevInfo here
  int nRemDevs;
  int nDataQps;
  struct ncclIbDevInfo remDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  // statistics about the comm
  struct ncclIbStats stats;
};

struct ncclIbSendComm {
  struct ncclIbNetCommBase base;
  // Start with fifo and ibv structs as they have alignment restrictions
  struct ncclIbSendFifo fifo[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS + 1];
  // Each dev correlates to a mergedIbDev
  struct ncclIbSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbRequest* fifoReqs[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ncclIbRemSizesFifo remSizesFifo;
  int ar; // Use adaptive routing when all merged devices have it enabled
  uint64_t putSignalScratchpad;
};
// The SendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
static_assert((sizeof(struct ncclIbNetCommBase) % 32) == 0, "ncclIbNetCommBase size must be 32-byte multiple to ensure fifo is at proper offset");
static_assert((offsetof(struct ncclIbSendComm, fifo) % 32) == 0, "ncclIbSendComm fifo must be 32-byte aligned");
static_assert((sizeof(struct ncclIbSendFifo) % 32) == 0, "ncclIbSendFifo element size must be 32-byte multiples");
static_assert((offsetof(struct ncclIbSendComm, sges) % 32) == 0, "sges must be 32-byte aligned");
static_assert((offsetof(struct ncclIbSendComm, wrs) % 32) == 0, "wrs must be 32-byte aligned");

struct ncclIbGpuFlush {
  struct ibv_mr* hostMr;
  struct ibv_sge sge;
  struct ncclIbQp qp;
};

struct ncclIbRemFifo {
  struct ncclIbSendFifo elems[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t addr;
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t flags;
};

struct alignas(16) ncclIbRecvCommDev {
  struct ncclIbNetCommDevBase base;
  struct ncclIbGpuFlush gpuFlush;
  struct ibv_mr* fifoMr;
  struct ibv_mr* sizesFifoMr;
  struct ibv_sge sge;
};

struct ncclIbRecvComm {
  struct ncclIbNetCommBase base;
  struct ncclIbRecvCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbRemFifo remFifo;
  int sizesFifo[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  int gpuFlushHostMem;
  int flushEnabled;
};
static_assert((offsetof(struct ncclIbRecvComm, remFifo) % 32) == 0, "ncclIbRecvComm fifo must be 32-byte aligned");

struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;
  struct ncclIbCommStage* stage;
};

static ncclResult_t ncclIbStatsInit(struct ncclIbStats* stat) {
  __atomic_store_n(&stat->fatalErrorCount, 0, __ATOMIC_RELAXED);
  return ncclSuccess;
}
static void ncclIbStatsFatalError(struct ncclIbStats* stat){
  __atomic_fetch_add(&stat->fatalErrorCount, 1, __ATOMIC_RELAXED);
}
static void ncclIbQpFatalError(struct ibv_qp* qp) {
  ncclIbStatsFatalError((struct ncclIbStats*)qp->qp_context);
}
static void ncclIbCqFatalError(struct ibv_cq* cq) {
  ncclIbStatsFatalError((struct ncclIbStats*)cq->cq_context);
}
static void ncclIbDevFatalError(struct ncclIbDev* dev) {
  ncclIbStatsFatalError(&dev->stats);
}
ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName);

extern ncclProfilerCallback_t ncclProfilerFunction;

extern std::thread ncclIbAsyncThread;
void* ncclIbAsyncThreadMain(void* args);

ncclResult_t ncclIbGdrSupport();
ncclResult_t ncclIbDmaBufSupport(int dev);

void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex, struct ncclIbNetCommDevBase* base);
ncclResult_t ncclIbGetGidIndex(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr* portAttr, int *gidIndex);
ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req);
ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r);

ncclResult_t ncclIbRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mrFlags, void** mhandle);

// Net IB plugin entry functions.
ncclResult_t ncclIbInitDevices(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
ncclResult_t ncclIbInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
ncclResult_t ncclIbDevices(int* ndev);
ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props);
ncclResult_t ncclIbListen(void* ctx, int dev, void* opaqueHandle, void** listenComm);
ncclResult_t ncclIbConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/);
ncclResult_t ncclIbAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/);
ncclResult_t ncclIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle);
ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
ncclResult_t ncclIbDeregMr(void* comm, void* mhandle);
ncclResult_t ncclIbIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request);
ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request);
ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
ncclResult_t ncclIbTest(void* request, int* done, int* sizes);
ncclResult_t ncclIbCloseSend(void* sendComm);
ncclResult_t ncclIbCloseRecv(void* recvComm);
ncclResult_t ncclIbCloseListen(void* listenComm);
ncclResult_t ncclIbMakeVDevice(int* d, ncclNetVDeviceProps_t* props);
ncclResult_t ncclIbFinalizeDevices(void);
ncclResult_t ncclIbFinalize(void* ctx);
ncclResult_t ncclIbSetNetAttr(void *ctx, ncclNetAttr_t *netAttr);

#endif

