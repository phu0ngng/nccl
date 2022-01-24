/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_STRONGSTREAM_H_
#define NCCL_STRONGSTREAM_H_

#include "nccl.h"
#include <stdint.h>
#include <assert.h>

struct ncclCudaGraph {
#if CUDART_VERSION >= 11030
  cudaGraph_t graph;
  uint64_t graphId;
#endif
};

inline struct ncclCudaGraph ncclCudaGraphNull() {
  struct ncclCudaGraph tmp;
  #if CUDART_VERSION >= 11030
    tmp.graph = nullptr;
    tmp.graphId = ULLONG_MAX;
  #endif
  return tmp;
}

inline bool ncclCudaGraphValid(struct ncclCudaGraph graph) {
  #if CUDART_VERSION >= 11030
    return graph.graph != nullptr;
  #else
    return false;
  #endif
}

inline bool ncclCudaGraphSame(struct ncclCudaGraph a, struct ncclCudaGraph b) {
  #if CUDART_VERSION >= 11030
    return a.graphId == b.graphId;
  #else
    return true;
  #endif
}

inline ncclResult_t ncclCudaGetCapturingGraph(
    struct ncclCudaGraph* graph, cudaStream_t stream
  ) {
  #if CUDART_VERSION >= 11030
    thread_local int driver = -1;
    if (driver == -1) {
      CUDACHECK(cudaDriverGetVersion(&driver));
    }
    if (driver < 11030) {
      cudaStreamCaptureStatus status;
      unsigned long long gid;
      graph->graph = nullptr;
      CUDACHECK(cudaStreamGetCaptureInfo(stream, &status, &gid));
      if (status != cudaStreamCaptureStatusNone) {
        WARN("The installed CUDA driver is older than the minimum version (R465) required for NCCL's CUDA Graphs support");
        return ncclInvalidUsage;
      }
    } else {
      cudaStreamCaptureStatus status;
      unsigned long long gid;
      CUDACHECK(cudaStreamGetCaptureInfo_v2(stream, &status, &gid, &graph->graph, nullptr, nullptr));
      if (status != cudaStreamCaptureStatusActive) {
        graph->graph = nullptr;
        gid = ULLONG_MAX;
      }
      graph->graphId = gid;
    }
  #endif
  return ncclSuccess;
}

inline ncclResult_t ncclCudaGraphAddDestructor(struct ncclCudaGraph graph, cudaHostFn_t fn, void* arg) {
  #if CUDART_VERSION >= 11030
    cudaUserObject_t object;
    CUDACHECK(cudaUserObjectCreate(
      &object, arg, fn, /*initialRefcount=*/1, cudaUserObjectNoDestructorSync
    ));
    // Hand over ownership to CUDA Graph
    CUDACHECK(cudaGraphRetainUserObject(graph.graph, object, 1, cudaGraphUserObjectMove));
    return ncclSuccess;
  #else
    return ncclInvalidUsage;
  #endif
}

/* Strong streams: An abstraction over CUDA streams that do not lose their
 * identity while being captured. Regular streams have the deficiency that the
 * captured form of a stream in one graph launch has no relation to the
 * uncaptured stream or to the captured form in other graph launches. This makes
 * streams unfit for the use of serializing access to a persistent resource.
 * Strong streams have been introduced to address this need.
 *
 * Constraints of using strong streams:
 *
 * - Operations that enqueue work to the strong stream need to be enclosed by
 *   ncclStrongStream[Acquire/Release] pairs. Acquire/release act like fences,
 *   the strong stream is not stateful so there is no harm in redundant acquire
 *   or releases.
 *
 * - An {Acquire; ...; Release} sequence must not be concurrent with any
 *   other operations against the strong stream including graph launches which
 *   reference this stream.
 *
 * - All strong stream functions take a "graph" parameter which must reference
 *   the currently capturing graph, or null if none.
 */
struct ncclStrongStream;

ncclResult_t ncclStrongStreamConstruct(struct ncclStrongStream* ss);
ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream* ss);

// Has this strong stream ever been captured in a graph.
bool ncclStrongStreamEverCaptured(struct ncclStrongStream* ss);

// Acquire-fence the strong stream.
ncclResult_t ncclStrongStreamAcquire(
  struct ncclCudaGraph graph, struct ncclStrongStream* ss
);

// Acquire-fence the strong stream assuming no graph is capturing. This permits
// the caller to enqueue directly to the `ss->stream` member using native CUDA
// calls. Strong stream must be released via:
//   ncclStrongStreamRelease(ncclCudaGraphNull(), graphRefs, ss);
ncclResult_t ncclStrongStreamAcquireUncaptured(struct ncclStrongStream* ss);

// Release-fence of the strong stream.
ncclResult_t ncclStrongStreamRelease(struct ncclCudaGraph graph, struct ncclStrongStream* ss);

// Add a host launch to the stream.
ncclResult_t ncclStrongStreamLaunchHost(
  struct ncclCudaGraph graph, struct ncclStrongStream* ss,
  cudaHostFn_t fn, void* arg
);
// Add a kernel launch to the stream.
ncclResult_t ncclStrongStreamLaunchKernel(
  struct ncclCudaGraph graph, struct ncclStrongStream* ss,
  void* fn, dim3 grid, dim3 block, void** args, size_t sharedMemBytes
);
// Cause `a` to wait for the current state `b`. Both `a` and `b` must be acquired.
ncclResult_t ncclStrongStreamWaitStream(
  struct ncclCudaGraph graph, struct ncclStrongStream* a, struct ncclStrongStream* b
);
// `b` must be capturing within `graph`.
ncclResult_t ncclStrongStreamWaitStream(
  struct ncclCudaGraph graph, struct ncclStrongStream* a, cudaStream_t b
);
// `a` must be capturing within `graph`.
ncclResult_t ncclStrongStreamWaitStream(
  struct ncclCudaGraph graph, cudaStream_t a, struct ncclStrongStream* b
);

// Synchrnoization does not need the strong stream to be acquired.
ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream* ss);

////////////////////////////////////////////////////////////////////////////////

struct ncclStrongStream {
  cudaStream_t stream;
  #if CUDART_VERSION >= 11030
  cudaEvent_t event;
  cudaGraphNode_t node; // null if never captured, otherwise never null again
  uint64_t graphId:63, eventIsLagging:1;
  #endif
};

inline ncclResult_t ncclStrongStreamConstruct(struct ncclStrongStream* ss) {
  CUDACHECK(cudaStreamCreateWithFlags(&ss->stream, cudaStreamNonBlocking));
  #if CUDART_VERSION >= 11030
    CUDACHECK(cudaEventCreateWithFlags(&ss->event, cudaEventDisableTiming));
    ss->node = nullptr;
    ss->graphId = (1ull<<(8*sizeof(long long)-1))-1;
    ss->eventIsLagging = 0;
  #endif
  return ncclSuccess;
}

inline ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    CUDACHECK(cudaEventDestroy(ss->event));
  #endif
  CUDACHECK(cudaStreamDestroy(ss->stream));
  return ncclSuccess;
}

inline bool ncclStrongStreamEverCaptured(struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    return ss->node != nullptr;
  #else
    return false;
  #endif
}

inline ncclResult_t ncclStrongStreamAcquire(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      if (ncclStrongStreamEverCaptured(ss)) {
        CUDACHECK(cudaStreamWaitEvent(ss->stream, ss->event));
        ss->eventIsLagging = 0;
      }
    } else {
      if (ss->graphId != graph.graphId) {
        if (ss->eventIsLagging) {
          // Can only be here if previous release was for uncaptured work that
          // elided updating the event because no capture had yet occurred.
          CUDACHECK(cudaStreamWaitEvent(ss->stream, ss->event));
          CUDACHECK(cudaEventRecord(ss->event, ss->stream));
        }
        ss->graphId = graph.graphId;
        ss->eventIsLagging = 0;
        //CUDACHECK(cudaGraphAddEmptyNode(&ss->node, graph.graph, nullptr, 0));
        CUDACHECK(cudaGraphAddEventWaitNode(&ss->node, graph.graph, nullptr, 0, ss->event));
      }
    }
  #endif
  return ncclSuccess;
}

inline ncclResult_t ncclStrongStreamAcquireUncaptured(struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    if (ncclStrongStreamEverCaptured(ss)) {
      CUDACHECK(cudaStreamWaitEvent(ss->stream, ss->event));
    }
    ss->eventIsLagging = 1; // Assume the caller is going to add work to stream.
  #endif
  return ncclSuccess;
}

inline ncclResult_t ncclStrongStreamRelease(struct ncclCudaGraph graph, struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    if (ss->eventIsLagging) {
      if (graph.graph == nullptr) {
        if (ncclStrongStreamEverCaptured(ss)) {
          CUDACHECK(cudaEventRecord(ss->event, ss->stream));
          ss->eventIsLagging = 0;
        }
      } else {
        CUDACHECK(cudaGraphAddEventRecordNode(&ss->node, graph.graph, &ss->node, 1, ss->event));
        ss->eventIsLagging = 0;
      }
    }
  #endif
  return ncclSuccess;
}

inline ncclResult_t ncclStrongStreamLaunchHost(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss, cudaHostFn_t fn, void* arg
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      CUDACHECK(cudaLaunchHostFunc(ss->stream, fn, arg));
    } else {
      cudaHostNodeParams p;
      p.fn = fn;
      p.userData = arg;
      CUDACHECK(cudaGraphAddHostNode(&ss->node, graph.graph, &ss->node, 1, &p));
    }
    ss->eventIsLagging = 1;
  #else
    CUDACHECK(cudaLaunchHostFunc(ss->stream, fn, arg));
  #endif
  return ncclSuccess;
}

inline ncclResult_t ncclStrongStreamLaunchKernel(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss,
    void* fn, dim3 grid, dim3 block, void* args[], size_t sharedMemBytes
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      CUDACHECK(cudaLaunchKernel(fn, grid, block, args, sharedMemBytes, ss->stream));
    } else {
      cudaGraphNode_t tip = ss->node;
      cudaKernelNodeParams p;
      p.func = fn;
      p.gridDim = grid;
      p.blockDim = block;
      p.kernelParams = args;
      p.sharedMemBytes = sharedMemBytes;
      p.extra = nullptr;
      CUDACHECK(cudaGraphAddKernelNode(&ss->node, graph.graph, &tip, 1, &p));
    }
    ss->eventIsLagging = 1;
  #else
    CUDACHECK(cudaLaunchKernel(fn, grid, block, args, sharedMemBytes, ss->stream));
  #endif
  return ncclSuccess;
}

inline cudaEvent_t ncclCudaScratchEvent() {
  thread_local cudaEvent_t event = 0x0;
  if (event == 0x0) {
    ncclResult_t result = ncclSuccess;
    CUDACHECKGOTO(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), result, completed);
  completed:
    if (result != ncclSuccess) {
      event = reinterpret_cast<cudaEvent_t>(0x1);
    }
  }
  return event;
}

inline ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, struct ncclStrongStream* a, struct ncclStrongStream* b
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      if (b->eventIsLagging) {
        b->eventIsLagging = 0;
        CUDACHECK(cudaEventRecord(b->event, b->stream));
      }
      CUDACHECK(cudaStreamWaitEvent(a->stream, b->event));
      a->eventIsLagging = 1;
    } else {
      cudaGraphNode_t pair[2] = {a->node, b->node};
      CUDACHECK(cudaGraphAddEmptyNode(&a->node, graph.graph, pair, 2));
    }
  #else
    cudaEvent_t scratch = ncclCudaScratchEvent();
    CUDACHECK(cudaEventRecord(scratch, b->stream));
    CUDACHECK(cudaStreamWaitEvent(a->stream, scratch));
  #endif
  return ncclSuccess;
}

inline ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, struct ncclStrongStream* a, cudaStream_t b
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      CUDACHECK(cudaEventRecord(a->event, b));
      CUDACHECK(cudaStreamWaitEvent(a->stream, a->event));
      // We used a->event to record b so it no longer reflects anything about a.
      a->eventIsLagging = 1;
    } else {
      cudaStreamCaptureStatus status;
      unsigned long long gid1;
      cudaGraphNode_t const* deps;
      size_t depN = 0;
      CUDACHECK(cudaStreamGetCaptureInfo_v2(b, &status, &gid1, nullptr, &deps, &depN));
      if (status != cudaStreamCaptureStatusActive || graph.graphId != gid1) {
        WARN("Stream is not being captured by the expected graph.");
        return ncclInvalidUsage;
      }
      if (depN > 0 && (depN > 1 || deps[0] != a->node)) {
        cudaGraphNode_t tie;
        if (depN == 1) {
          tie = deps[0];
        } else {
          CUDACHECK(cudaGraphAddEmptyNode(&tie, graph.graph, deps, depN));
        }
        cudaGraphNode_t pair[2] = {a->node, tie};
        CUDACHECK(cudaGraphAddEmptyNode(&a->node, graph.graph, pair, 2));
      }
      // a->eventIsLagging doesn't change since we are just updating the
      // dependencies of a->node.
    }
  #else
    cudaEvent_t scratch = ncclCudaScratchEvent();
    CUDACHECK(cudaEventRecord(scratch, b));
    CUDACHECK(cudaStreamWaitEvent(a->stream, scratch));
  #endif
  return ncclSuccess;
}

inline ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, cudaStream_t a, struct ncclStrongStream* b
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      if (b->eventIsLagging) {
        b->eventIsLagging = 0;
        CUDACHECK(cudaEventRecord(b->event, b->stream));
      }
      CUDACHECK(cudaStreamWaitEvent(a, b->event));
    } else {
      CUDACHECK(cudaStreamUpdateCaptureDependencies(a, &b->node, 1, cudaStreamAddCaptureDependencies));
    }
  #else
    cudaEvent_t scratch = ncclCudaScratchEvent();
    CUDACHECK(cudaEventRecord(scratch, b->stream));
    CUDACHECK(cudaStreamWaitEvent(a, scratch));
  #endif
  return ncclSuccess;
}

inline ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    CUDACHECK(cudaStreamWaitEvent(ss->stream, ss->event));
  #endif
  CUDACHECK(cudaStreamSynchronize(ss->stream));
  return ncclSuccess;
}

#endif
