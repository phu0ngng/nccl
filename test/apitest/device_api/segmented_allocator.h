/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef SEGMENTED_ALLOCATOR_H
#define SEGMENTED_ALLOCATOR_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <cuda.h>

// Helper macros for error checking
#define CUDACHECK(cmd) do {                             \
  cudaError_t err = cmd;                                \
  if(err != cudaSuccess) {                              \
    printf("CUDA Error at %s:%d - %s\n",                \
           __FILE__, __LINE__,                          \
           cudaGetErrorString(err));                    \
    exit(1);                                            \
  }                                                     \
} while(0)

#define CUCHECK(cmd) do {                               \
  CUresult err = cmd;                                   \
  if(err != CUDA_SUCCESS) {                             \
    const char* errStr;                                 \
    cuGetErrorString(err, &errStr);                     \
    printf("CUDA Driver Error at %s:%d - %s\n",         \
           __FILE__, __LINE__, errStr);                 \
    exit(1);                                            \
  }                                                     \
} while(0)


/* Segment location: device memory or host NUMA. */
typedef enum {
  SEGMENT_LOCATION_DEVICE = 0,
  SEGMENT_LOCATION_HOST_NUMA
} segment_location_type_t;

/* Descriptor for one segment: location and size. */
typedef struct segment_descriptor {
  segment_location_type_t location_type;
  int location_id;   /* Device id or NUMA node id; -1 = use current device / current GPU's NUMA node */
  size_t segment_size;
} segment_descriptor_t;

#define SEGMENTED_ALLOC_ALIGN_SIZE(size, granularity) \
  (((size) + (granularity) - 1) / (granularity) * (granularity))

#if CUDART_VERSION >= 11030

static inline void allocateSegmentedMemory(
    void** ptr,
    const segment_descriptor_t* descriptors,
    int numSegments)
{
  size_t totalSize = 0;
  for (int i = 0; i < numSegments; i++) {
    totalSize += descriptors[i].segment_size;
  }

  int cudaDev = 0;
  int dcnt = 0;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUdevice currentDev;
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  CUDACHECK(cudaGetDeviceCount(&dcnt));

  int numa_id = 0;
  CUCHECK(cuDeviceGetAttribute(&numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));

  int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  int flag = 0;
#if CUDART_VERSION >= 12030
  cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev);
  if (flag) requestedHandleTypes |= CU_MEM_HANDLE_TYPE_FABRIC;
#endif

  CUmemGenericAllocationHandle* handles = (CUmemGenericAllocationHandle*)calloc(
      (size_t)numSegments, sizeof(CUmemGenericAllocationHandle));

  size_t* alignedSizes = (size_t*)malloc((size_t)numSegments * sizeof(size_t));

  size_t reservedSize = 0;
  CUmemAllocationProp memprop = {};
  memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  memprop.requestedHandleTypes = (CUmemAllocationHandleType)requestedHandleTypes;

  for (int s = 0; s < numSegments; s++) {
    size_t memGran = 0;
    const segment_descriptor_t* d = &descriptors[s];
    if (d->location_type == SEGMENT_LOCATION_DEVICE) {
      memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      memprop.location.id = (d->location_id >= 0) ? d->location_id : (int)cudaDev;
      CUCHECK(cuMemGetAllocationGranularity(&memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
      flag = 0;
      CUCHECK(cuDeviceGetAttribute(&flag,
          CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
          currentDev));
    } else {
      memprop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
      memprop.location.id = (d->location_id >= 0) ? d->location_id : numa_id;
      memprop.allocFlags.gpuDirectRDMACapable = 0;
      CUCHECK(cuMemGetAllocationGranularity(&memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    }
    alignedSizes[s] = SEGMENTED_ALLOC_ALIGN_SIZE(d->segment_size, memGran);
    reservedSize += alignedSizes[s];
  }

  size_t memGran0 = 0;
  memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  memprop.location.id = (int)currentDev;
  CUCHECK(cuMemGetAllocationGranularity(&memGran0, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  reservedSize = SEGMENTED_ALLOC_ALIGN_SIZE(reservedSize, memGran0);
  CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, reservedSize, memGran0, 0, 0));

  CUdeviceptr basePtr = (CUdeviceptr)*ptr;
  for (int s = 0; s < numSegments; s++) {
    const segment_descriptor_t* d = &descriptors[s];
    size_t alignedSize = alignedSizes[s];
    memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memprop.requestedHandleTypes = (CUmemAllocationHandleType)requestedHandleTypes;
    memprop.allocFlags.gpuDirectRDMACapable = 0;
    if (d->location_type == SEGMENT_LOCATION_DEVICE) {
      memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      memprop.location.id = (d->location_id >= 0) ? d->location_id : (int)cudaDev;
      flag = 0;
      CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
      if (flag) memprop.allocFlags.gpuDirectRDMACapable = 1;
    } else {
      memprop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
      memprop.location.id = (d->location_id >= 0) ? d->location_id : numa_id;
    }
    CUresult createErr = cuMemCreate(&handles[s], alignedSize, &memprop, 0);
#if CUDART_VERSION >= 12030
    if ((requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) &&
        (createErr == CUDA_ERROR_NOT_PERMITTED || createErr == CUDA_ERROR_NOT_SUPPORTED)) {
      requestedHandleTypes &= ~CU_MEM_HANDLE_TYPE_FABRIC;
      memprop.requestedHandleTypes = (CUmemAllocationHandleType)requestedHandleTypes;
      CUCHECK(cuMemCreate(&handles[s], alignedSize, &memprop, 0));
    } else
#endif
    {
      CUCHECK(createErr);
    }
    CUCHECK(cuMemMap(basePtr, alignedSize, 0, handles[s], 0));
    basePtr += alignedSize;
  }

  CUmemAccessDesc accessDesc[2] = {{}};
  accessDesc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  accessDesc[0].location.id = cudaDev;
  accessDesc[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  accessDesc[1].location.id = numa_id;
  accessDesc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  for (int i = 0; i < dcnt; i++) {
    int p2p = 0;
    if (i == cudaDev || (cudaDeviceCanAccessPeer(&p2p, i, cudaDev) == cudaSuccess && p2p)) {
      accessDesc[0].location.id = i;
      basePtr = (CUdeviceptr)*ptr;
      for (int s = 0; s < numSegments; s++) {
        if (descriptors[s].location_type == SEGMENT_LOCATION_DEVICE) {
          CUCHECK(cuMemSetAccess(basePtr, alignedSizes[s], accessDesc, 1));
        } else {
          CUCHECK(cuMemSetAccess(basePtr, alignedSizes[s], accessDesc, 2));
        }
        basePtr += alignedSizes[s];
      }
    }
  }
  free(alignedSizes);
  free(handles);
}

static inline void deallocateSegmentedMemory(
    void* ptr,
    const segment_descriptor_t* descriptors,
    int numSegments)
{
  CUdeviceptr basePtr = (CUdeviceptr)ptr;
  size_t totalFreed = 0;

  for (int s = 0; s < numSegments; s++) {
    CUmemGenericAllocationHandle handle;
    CUCHECK(cuMemRetainAllocationHandle(&handle, (void*)(basePtr + totalFreed)));
    size_t size = 0;
    CUCHECK(cuMemGetAddressRange(NULL, &size, basePtr + totalFreed));
    CUCHECK(cuMemUnmap(basePtr + totalFreed, size));
    CUCHECK(cuMemRelease(handle));
    totalFreed += size;
  }
  CUCHECK(cuMemAddressFree(basePtr, totalFreed));
}

static inline void segmentedMemcpyToHost(
    const void* segmentedPtr,
    void* hostPtr,
    size_t totalBytes,
    const segment_descriptor_t* descriptors,
    int numSegments)
{
  const char* src = (const char*)segmentedPtr;
  char* dst = (char*)hostPtr;
  size_t remaining = totalBytes;

  for (int seg = 0; seg < numSegments && remaining > 0; seg++) {
    size_t segSize = descriptors[seg].segment_size;
    size_t n = remaining < segSize ? remaining : segSize;
    if (n == 0) continue;
    if (descriptors[seg].location_type == SEGMENT_LOCATION_DEVICE) {
      CUDACHECK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost));
    } else {
      memcpy(dst, src, n);
    }
    src += segSize;
    dst += n;
    remaining -= n;
  }
  CUDACHECK(cudaDeviceSynchronize());
}

static inline void segmentedMemcpyToDevice(
    void* segmentedPtr,
    const void* hostPtr,
    size_t totalBytes,
    const segment_descriptor_t* descriptors,
    int numSegments)
{
  char* dst = (char*)segmentedPtr;
  const char* src = (const char*)hostPtr;
  size_t remaining = totalBytes;

  for (int seg = 0; seg < numSegments && remaining > 0; seg++) {
    size_t segSize = descriptors[seg].segment_size;
    size_t n = remaining < segSize ? remaining : segSize;
    if (n == 0) continue;
    if (descriptors[seg].location_type == SEGMENT_LOCATION_DEVICE) {
      CUDACHECK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
    } else {
      memcpy(dst, src, n);
    }
    dst += segSize;
    src += n;
    remaining -= n;
  }
  CUDACHECK(cudaDeviceSynchronize());
}

#endif /* CUDART_VERSION >= 11030 */

#endif /* SEGMENTED_ALLOCATOR_H */
