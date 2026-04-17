/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _REDUCE_COPY_TEST_FACTORIES_H_
#define _REDUCE_COPY_TEST_FACTORIES_H_

#include <cuda_runtime.h>
#include <vector>
#include "nccl.h"
#include "nccl_device.h"
#include "checks.h"
#include "config.h"
#include "util.h"

// Kernel parameters (pre-computed on CPU, passed to kernel)
struct KernelParams {
  ncclWindow_t sendwin;
  size_t sendoffset;
  ncclWindow_t recvwin;
  size_t recvoffset;
  // For lambda variants: per-rank offsets are resolved on-device
  int nSrc;         // Number of sources (for lambda variants)
  int nDst;         // Number of destinations (for lambda variants)
  // Additional parameters can be added for other variants
};

// Base interface for creating API inputs (CPU-side)
template<typename T, ApiFunctionId FuncId>
class InputFactory {
public:
  virtual ~InputFactory() = default;

  // CPU-side: Prepare all inputs before kernel launch
  virtual void prepareData(int nSrc, int nDst, size_t count) = 0;

  // CPU-side: Get kernel parameters (pre-computed on CPU)
  virtual KernelParams getKernelParams() const = 0;

  // CPU-side: Get kernel parameters for specific device
  virtual KernelParams getKernelParams(int deviceId) const = 0;

  // CPU-side: Cleanup
  virtual void cleanup() = 0;

  // Get the number of sources/destinations this factory provides
  virtual int getNSrc() const = 0;
  virtual int getNDst() const = 0;
};

struct WindowAllocation {
  std::vector<ncclWindow_t> sendWindows;
  std::vector<ncclWindow_t> recvWindows;
  std::vector<void*> sendAllocs;
  std::vector<void*> recvAllocs;

  void allocateAndRegister(const std::vector<ncclComm_t>& comms,
               size_t sendSize, size_t recvSize) {
    const int nDevices = static_cast<int>(comms.size());
    sendWindows.assign(nDevices, ncclWindow_t{});
    recvWindows.assign(nDevices, ncclWindow_t{});
    sendAllocs.assign(nDevices, nullptr);
    recvAllocs.assign(nDevices, nullptr);

    if (nDevices == 0) {
      return;
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDevices; ++i) {
      CUDACHECK(cudaSetDevice(i));
      NCCLCHECK(ncclMemAlloc(&sendAllocs[i], sendSize));
      NCCLCHECK(ncclMemAlloc(&recvAllocs[i], recvSize));
      CUDACHECK(cudaMemset(sendAllocs[i], 0, sendSize));
      CUDACHECK(cudaMemset(recvAllocs[i], 0, recvSize));
      NCCLCHECK(ncclCommWindowRegister(
        comms[i],
        sendAllocs[i], sendSize,
        &sendWindows[i],
        NCCL_WIN_COLL_SYMMETRIC
      ));
      NCCLCHECK(ncclCommWindowRegister(
        comms[i],
        recvAllocs[i], recvSize,
        &recvWindows[i],
        NCCL_WIN_COLL_SYMMETRIC
      ));
    }
    NCCLCHECK(ncclGroupEnd());
  }

  void cleanup(const std::vector<ncclComm_t>& comms) {
    const int nDevices = static_cast<int>(std::min(comms.size(), sendWindows.size()));
    if (nDevices > 0) {
      NCCLCHECK_NO_THROW(ncclGroupStart());
      for (int i = 0; i < nDevices; ++i) {
        CUDACHECK_NO_THROW(cudaSetDevice(i));
        if (i < static_cast<int>(sendWindows.size()) && sendWindows[i]) {
          NCCLCHECK_NO_THROW(ncclCommWindowDeregister(comms[i], sendWindows[i]));
        }
        if (i < static_cast<int>(recvWindows.size()) && recvWindows[i]) {
          NCCLCHECK_NO_THROW(ncclCommWindowDeregister(comms[i], recvWindows[i]));
        }
        if (i < static_cast<int>(sendAllocs.size()) && sendAllocs[i]) {
          NCCLCHECK_NO_THROW(ncclMemFree(sendAllocs[i]));
        }
        if (i < static_cast<int>(recvAllocs.size()) && recvAllocs[i]) {
          NCCLCHECK_NO_THROW(ncclMemFree(recvAllocs[i]));
        }
      }
      NCCLCHECK_NO_THROW(ncclGroupEnd());
    }
    sendWindows.clear();
    recvWindows.clear();
    sendAllocs.clear();
    recvAllocs.clear();
  }

  ncclWindow_t sendWindow(int deviceId) const {
    return (deviceId >= 0 && deviceId < static_cast<int>(sendWindows.size()))
      ? sendWindows[deviceId] : nullptr;
  }

  ncclWindow_t recvWindow(int deviceId) const {
    return (deviceId >= 0 && deviceId < static_cast<int>(recvWindows.size()))
      ? recvWindows[deviceId] : nullptr;
  }

  void* sendAlloc(int deviceId) const {
    return (deviceId >= 0 && deviceId < static_cast<int>(sendAllocs.size()))
      ? sendAllocs[deviceId] : nullptr;
  }

  void* recvAlloc(int deviceId) const {
    return (deviceId >= 0 && deviceId < static_cast<int>(recvAllocs.size()))
      ? recvAllocs[deviceId] : nullptr;
  }
};

template<typename T>
class WindowFactoryBase {
protected:
  std::vector<ncclComm_t> comms_;
  int nDevices_;
  WindowAllocation windows_;
  int nSrc_ = 0;
  int nDst_ = 0;
  size_t count_ = 0;

  WindowFactoryBase(ncclComm_t* comms, int nDevices)
    : nDevices_(nDevices) {
    if (comms && nDevices_ > 0) {
      comms_.assign(comms, comms + nDevices_);
    }
  }

  void setCounts(int nSrc, int nDst, size_t count) {
    nSrc_ = nSrc;
    nDst_ = nDst;
    count_ = count;
  }

  void allocateWindows(size_t sendSize, size_t recvSize) {
    windows_.allocateAndRegister(comms_, sendSize, recvSize);
  }

  void cleanupWindows() {
    windows_.cleanup(comms_);
  }

  KernelParams makeKernelParams(int deviceId, size_t sendoffset, size_t recvoffset) const {
    KernelParams params{};
    if (deviceId >= 0 && deviceId < nDevices_) {
      params.sendwin = windows_.sendWindow(deviceId);
      params.sendoffset = sendoffset;
      params.recvwin = windows_.recvWindow(deviceId);
      params.recvoffset = recvoffset;
      params.nSrc = nSrc_;
      params.nDst = nDst_;
    }
    return params;
  }

  void* sendAlloc(int deviceId) const { return windows_.sendAlloc(deviceId); }
  void* recvAlloc(int deviceId) const { return windows_.recvAlloc(deviceId); }
};

// Template specialization for Window-based variants
template<typename T>
class WindowInputFactory : public InputFactory<T, ApiFunctionId::LsaReduceSumCopy_Windows>,
               protected WindowFactoryBase<T> {
public:
  WindowInputFactory(ncclComm_t* comms, int nDevices)
    : WindowFactoryBase<T>(comms, nDevices) {}

  ~WindowInputFactory() {
    cleanup();
  }

  void prepareData(int nSrc, int nDst, size_t count) override {
    this->setCounts(nSrc, nDst, count);

    // Allocate symmetric memory and register windows for each device
    // Each rank stores count elements in its send/recv buffers
    // For ReduceSumCopy: each rank has its own source data (count elements)
    // For ReduceSum: each rank has its own source data (count elements)
    // For Copy: each rank gets a portion of source data (count elements total)
    // So bufferSize should always be count * sizeof(T), not count * sizeof(T) * nSrc
    size_t bufferCount = (count > 0) ? count : 1;
    size_t bufferSize = bufferCount * sizeof(T);
    size_t allocSize = bufferSize;

    this->allocateWindows(allocSize, allocSize);
  }

  KernelParams getKernelParams() const override {
    // Return parameters for device 0 (can be extended for multi-device)
    return getKernelParams(0);
  }

  // Get kernel parameters for a specific device
  // For inter-rank ReduceSumCopy with symmetric windows:
  // - sendoffset: base offset (0) - work chunks handle per-rank source offsets
  // - recvoffset: base offset (0) - collective writes to all destination slots
  KernelParams getKernelParams(int deviceId) const override {
    // For inter-rank ReduceSumCopy with symmetric windows:
    // - sendoffset: base offset (0) - work chunks handle per-rank source offsets
    // - recvoffset: base offset (0) - collective writes to all destination slots
    return this->makeKernelParams(deviceId, 0, 0);
  }

  // Get base pointers for host-side data access
  void* getSendBuffer(int deviceId) const {
    return this->sendAlloc(deviceId);
  }

  void* getRecvBuffer(int deviceId) const {
    return this->recvAlloc(deviceId);
  }

  void cleanup() override {
    this->cleanupWindows();
  }

  int getNSrc() const override { return this->nSrc_; }
  int getNDst() const override { return this->nDst_; }
};

// Input factory for ReduceSum variants (ReduceScatter semantics)
// Send buffer holds nSrc chunks, recv buffer holds one chunk.
template<typename T>
class ReduceSumInputFactory : public InputFactory<T, ApiFunctionId::LsaReduceSum_Window_DevComm>,
                protected WindowFactoryBase<T> {
public:
  ReduceSumInputFactory(ncclComm_t* comms, int nDevices)
    : WindowFactoryBase<T>(comms, nDevices) {}

  ~ReduceSumInputFactory() {
    cleanup();
  }

  void prepareData(int nSrc, int nDst, size_t count) override {
    this->setCounts(nSrc, nDst, count);

    size_t effectiveCount = (this->count_ > 0) ? this->count_ : 1;
    size_t sendSize = static_cast<size_t>(this->nSrc_) * effectiveCount * sizeof(T);
    size_t recvSize = effectiveCount * sizeof(T);
    size_t sendAllocSize = sendSize;
    size_t recvAllocSize = recvSize;

    this->allocateWindows(sendAllocSize, recvAllocSize);
  }

  KernelParams getKernelParams() const override {
    return getKernelParams(0);
  }

  KernelParams getKernelParams(int deviceId) const override {
    return this->makeKernelParams(deviceId, 0, 0);
  }

  void* getSendBuffer(int deviceId) const {
    return this->sendAlloc(deviceId);
  }

  void* getRecvBuffer(int deviceId) const {
    return this->recvAlloc(deviceId);
  }

  void cleanup() override {
    this->cleanupWindows();
  }

  int getNSrc() const override { return this->nSrc_; }
  int getNDst() const override { return this->nDst_; }
};

// Template specialization for Lambda-based variants
// Now unified with WindowInputFactory - only registers windows
template<typename T>
class LambdaInputFactory : public InputFactory<T, ApiFunctionId::LsaReduceSum_Lambda>,
               protected WindowFactoryBase<T> {
private:
  bool enableOffsets_;
public:
  LambdaInputFactory(ncclComm_t* comms, int nDevices, bool enableOffsets)
    : WindowFactoryBase<T>(comms, nDevices),
      enableOffsets_(enableOffsets) {}

  ~LambdaInputFactory() {
    cleanup();
  }

  void prepareData(int nSrc, int nDst, size_t count) override {
    this->setCounts(nSrc, nDst, count);

    // Allocate arrays for windows and buffers
    // Allocate symmetric memory and register windows for each device
    // For lambda variants, we need one buffer per source/destination
    constexpr size_t kMaxLambdaOffsetElts = 7; // Per-rank misalignment in full elements
    size_t sendCount = count + (enableOffsets_ ? kMaxLambdaOffsetElts : 0);
    if (sendCount == 0) sendCount = 1;
    size_t sendSize = sendCount * sizeof(T);

    // For AllGather (nSrc ranks each contributing 'count' elements), the recv window
    // must hold ALL ranks' data: nSrc * count elements.  The kernel writes rank k's
    // chunk at byte offset (k * count * sizeof(T)) + peerBase into each peer's recv
    // window, so under-sizing the recv window causes an out-of-bounds access.
    // WindowInputFactory correctly scales by nSrc; replicate that here.
    size_t recvCount = static_cast<size_t>(nSrc) * count +
               (enableOffsets_ ? kMaxLambdaOffsetElts : 0);
    if (recvCount == 0) recvCount = 1;
    size_t recvSize = recvCount * sizeof(T);

    this->allocateWindows(sendSize, recvSize);
  }

  KernelParams getKernelParams() const override {
    return getKernelParams(0);
  }

  KernelParams getKernelParams(int deviceId) const override {
    const size_t offset = enableOffsets_ ? 1 : 0;
    return this->makeKernelParams(deviceId, offset, offset);
  }

  // Get base pointers for host-side data access
  void* getSendBuffer(int deviceId) const {
    return this->sendAlloc(deviceId);
  }

  void* getRecvBuffer(int deviceId) const {
    return this->recvAlloc(deviceId);
  }

  void cleanup() override {
    this->cleanupWindows();
  }

  int getNSrc() const override { return this->nSrc_; }
  int getNDst() const override { return this->nDst_; }
};

// Input factory for local strided variants (local lambda/strided)
// Allocates enough space for nSrc/nDst contiguous chunks on a single device.
template<typename T>
class LocalStridedInputFactory : public InputFactory<T, ApiFunctionId::LocalReduceSumCopy_Strided>,
                 protected WindowFactoryBase<T> {
public:
  LocalStridedInputFactory(ncclComm_t* comms, int nDevices)
    : WindowFactoryBase<T>(comms, nDevices) {}

  ~LocalStridedInputFactory() {
    cleanup();
  }

  void prepareData(int nSrc, int nDst, size_t count) override {
    this->setCounts(nSrc, nDst, count);

    size_t effectiveCount = (this->count_ > 0) ? this->count_ : 1;
    size_t sendSize = static_cast<size_t>(this->nSrc_) * effectiveCount * sizeof(T);
    size_t recvSize = static_cast<size_t>(this->nDst_) * effectiveCount * sizeof(T);
    this->allocateWindows(sendSize, recvSize);
  }

  KernelParams getKernelParams() const override {
    return getKernelParams(0);
  }

  KernelParams getKernelParams(int deviceId) const override {
    return this->makeKernelParams(deviceId, 0, 0);
  }

  void* getSendBuffer(int deviceId) const {
    return this->sendAlloc(deviceId);
  }

  void* getRecvBuffer(int deviceId) const {
    return this->recvAlloc(deviceId);
  }

  void cleanup() override {
    this->cleanupWindows();
  }

  int getNSrc() const override { return this->nSrc_; }
  int getNDst() const override { return this->nDst_; }
};

#endif // _REDUCE_COPY_TEST_FACTORIES_H_

