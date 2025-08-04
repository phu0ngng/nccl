/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef PROFILER_H_
#define PROFILER_H_

#include <stdint.h>

typedef enum {
  NCCL_LOG_NONE,
  NCCL_LOG_VERSION,
  NCCL_LOG_WARN,
  NCCL_LOG_INFO,
  NCCL_LOG_ABORT,
  NCCL_LOG_TRACE
} ncclDebugLogLevel;

typedef enum {
  NCCL_INIT      =    1,
  NCCL_COLL      =    2,
  NCCL_P2P       =    4,
  NCCL_SHM       =    8,
  NCCL_NET       =   16,
  NCCL_GRAPH     =   32,
  NCCL_TUNING    =   64,
  NCCL_ENV       =  128,
  NCCL_ALLOC     =  256,
  NCCL_CALL      =  512,
  NCCL_PROXY     = 1024,
  NCCL_NVLS      = 2048,
  NCCL_BOOTSTRAP = 4096,
  NCCL_REG       = 8192,
  NCCL_ALL       = ~0
} ncclDebugLogSubsys;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char* file, int line, const char* fmt, ...);

enum {
  ncclProfileGroup     = (1 << 0),  // group event type
  ncclProfileColl      = (1 << 1),  // host collective call event type
  ncclProfileP2p       = (1 << 2),  // host point-to-point call event type
  ncclProfileProxyOp   = (1 << 3),  // proxy operation event type
  ncclProfileProxyStep = (1 << 4),  // proxy step event type
  ncclProfileProxyCtrl = (1 << 5),  // proxy control event type
  ncclProfileKernelCh  = (1 << 6),  // kernel channel event type
  ncclProfileNetPlugin = (1 << 7),  // network plugin-defined, events
};

typedef enum {
  ncclProfilerProxyOpSendPosted        = 0,  // deprecated in v4
  ncclProfilerProxyOpSendRemFifoWait   = 1,  // deprecated in v4
  ncclProfilerProxyOpSendTransmitted   = 2,  // deprecated in v4
  ncclProfilerProxyOpSendDone          = 3,  // deprecated in v4
  ncclProfilerProxyOpRecvPosted        = 4,  // deprecated in v4
  ncclProfilerProxyOpRecvReceived      = 5,  // deprecated in v4
  ncclProfilerProxyOpRecvTransmitted   = 6,  // deprecated in v4
  ncclProfilerProxyOpRecvDone          = 7,  // deprecated in v4
  ncclProfilerProxyOpInProgress_v4     = 19,

  /* Legacy proxy profiler states */
  ncclProfilerProxyStepSendGPUWait     = 8,
  ncclProfilerProxyStepSendPeerWait_v4 = 20,
  ncclProfilerProxyStepSendWait        = 9,
  ncclProfilerProxyStepRecvWait        = 10,
  ncclProfilerProxyStepRecvFlushWait   = 11,
  ncclProfilerProxyStepRecvGPUWait     = 12,

  /* Legacy proxy control states */
  ncclProfilerProxyCtrlIdle            = 13,
  ncclProfilerProxyCtrlActive          = 14,
  ncclProfilerProxyCtrlSleep           = 15,
  ncclProfilerProxyCtrlWakeup          = 16,
  ncclProfilerProxyCtrlAppend          = 17,
  ncclProfilerProxyCtrlAppendEnd       = 18,

  /* Network defined event states */
  ncclProfilerNetPluginUpdate          = 21,

  /* Kernel event states */
  ncclProfilerKernelChStop             = 22,
} ncclProfilerEventState_v5_t;

typedef struct {
  uint8_t type;                 // event type descriptor: ncclProfileColl, ...
  void* parentObj;              // pointer to the profiler parent object (for coll is the group)
  int rank;                     // originating rank
  union {
    struct {
      uint64_t seqNumber;
      const char* func;
      void const* sendBuff;
      void* recvBuff;
      size_t count;
      int root;
      const char* datatype;
      uint8_t nChannels;
      uint8_t nWarps;
      const char* algo;
      const char* proto;
    } coll;

    struct {
      const char* func;
      void* buff;
      const char* datatype;
      size_t count;
      int peer;
      uint8_t nChannels;
    } p2p;

    struct {
      pid_t pid;                // pid of the originating process
      uint8_t channelId;        // channel id for this proxy operation
      int peer;                 // remote rank for send/recv
      int nSteps;               // number of steps for this proxy operation
      int chunkSize;            // amount of data transferred by this proxy operation
    } proxyOp;

    struct {
      int step;
    } proxyStep;

    struct {
      uint8_t channelId;
      uint64_t pTimer;          // start timestamp from GPU globaltimer
    } kernelCh;

    struct {
      int64_t id;
      void* data;
    } netPlugin;
  };
} ncclProfilerEventDescr_v5_t;

typedef struct {
  struct {
    int isSend;
  } proxyOp;

  struct {
    size_t transSize;
  } proxyStep;

  struct {
    int appendedProxyOps;
  } proxyCtrl;

  struct {
    void* data;
  } netPlugin;

  struct {
    uint64_t pTimer;
  } kernelCh;
} ncclProfilerEventStateArgs_v5_t;

typedef struct {
  const char* name;
  ncclResult_t (*init)(void** ctx, uint64_t commId, int* eMask, const char* commName, int nNodes, int nranks, int rank, ncclDebugLogger_t logfn);
  ncclResult_t (*startEvent)(void* ctx, void** eHandle, ncclProfilerEventDescr_v5_t* eDescr);
  ncclResult_t (*stopEvent)(void* eHandle);
  ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v5_t eState, ncclProfilerEventStateArgs_v5_t* eStateArgs);
  ncclResult_t (*finalize)(void* ctx);
} ncclProfiler_v5_t;

typedef ncclProfiler_v5_t ncclProfiler_t;
typedef ncclProfilerEventDescr_v5_t ncclProfilerEventDescr_t;
typedef ncclProfilerEventStateArgs_v5_t ncclProfilerEventStateArgs_t;
typedef ncclProfilerEventState_v5_t ncclProfilerEventState_t;

#define PROFILER_PLUGIN_SYM "ncclProfiler_v5"
#endif
