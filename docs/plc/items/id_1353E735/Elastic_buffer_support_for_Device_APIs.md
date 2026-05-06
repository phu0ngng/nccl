# Elastic buffer (CPU + GPU) support for Device APIs
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
This PLC covers adding support for elastic buffers (i.e. buffers spanning CPU + GPU memory) for device APIs.
<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

"Elastic" buffers in this context is the ability to have one contiguous VA space backed by multiple memory locations/segments that can include any mix between GPU and CPU memory. This is done using the cuMem APIs. The primary use-case of this is cases where a user runs out/wants to save GPU memory and instead use host-backed memory or a mix of GPU + CPU memory for device API operations.


### NVbugs / Jira Tickets
https://nvbugspro.nvidia.com/bug/5760716

### Requirements
- Must work with any combination of GPU and CPU buffers allocated using cuMem.
- Support both LSA and GIN.
- Must work with any arbitrary number of segments (potentially different across GPUs).
- Minimal impact to latency of existing operations.
- Memory must be allocated with cuMemCreate. If host segments are desired, the attribute for the host segments must be CU_MEM_LOCATION_TYPE_HOST_NUMA.

### Assumptions, constraints and dependencies
- In the case of GIN, the GPU always consumes the data that is transferred.
- No RMA/CE/symmetric collective support in the current version. No NVLS support for CPU buffers yet.
- Multi-node LSA support for elastic buffers requires EGM.

### User Experience


The support matrix covered in this PLC is shown below. Some of the unsupported columns will be supported in future versions based on requests/priority.

| Feature | GPU-only segments + DMABuf | GPU-only segments + No DMABuf | GPU + CPU segments |
|---|---|---|---|
| RMA | ❌ | ❌ | ❌ |
| GIN put | ✅ (no API changes)| ❌ | ✅ (with API changes) |
| GIN signal | ✅ | ❌ | ✅ (No DirectNIC support) |
| Other Device APIs for GIN (putValue, etc.) | ❌ | ❌ | ❌ |
| LSA | ✅ | No Multimem | No Multimem + Atomics have limitations – needs hostNativeAtomics = 1 + MNNVL needs EGM |

Moreover, regular collective host APIs will also not support elastic buffers. Apart from documenting this clearly, we will have error messages at runtime shown to users when they attempt to use elastic buffers with RMA operations, CE collective operations, and symmetric collectives with multiple segments.
</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed Design for LSA

RMA support uses the same ncclCommWindowRegister path, and RMA registration is always called since it's enabled by default. We need to conditionally disable RMA registration (which involves a collective) and for that we need a global allgather for numSegments.

The number of segments across ranks need not be the same. Moreover, transfers must work whether those segments are in GPU memory or CPU memory.
Currently, a mem object (struct ncclDevrMemory) only has one handle. The following changes are required to support LSA:

- Find the max number of segments across ranks using an allgather. This is needed so that we can allocate the right amount of memory for passing around metadata structs between ranks
- Loop over each segment, convert fds and map each segment into reserved memory. Set access flags to CU_MEM_LOCATION_TYPE_DEVICE, since only the device needs access to the segments. Track the following metadata : segment sizes, number of segments, type of buffer in each segment, and the memHandles.
- Use memHandles form the metadata tracked during registration for freeing segments.
- Modify ncclCuMemGetAddressRange to also indicate whether the VA is backed by one or more sysmem segments or not.

### Proposed Design for GIN

#### New and modified data structures

##### `ncclSegmentWindow` (new)

A new device/host-visible struct that captures per-segment GIN state. One instance per physical segment per registered window:

```c
struct ncclSegmentWindow {
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONNECTIONS]; // per-connection registered handles
  size_t segmentSize;
  CUmemLocationType memType; // CU_MEM_LOCATION_TYPE_DEVICE or HOST_NUMA
};
```

This struct is allocated via the shadow pool so that both host and device have access. The device-side pointer is stored in `ncclWindow_vidmem::ginMultiSegmentWins`.

##### `ncclWindow_vidmem` (expanded)

Two new fields are appended to `ncclWindow_vidmem`, growing the struct from 72 to 88 bytes:

```c
struct ncclWindow_vidmem {
  // ... existing fields unchanged at same offsets ...
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONNECTIONS]; // offset 40, single-segment fast path
  struct ncclSegmentWindow* ginMultiSegmentWins;      // offset 72, device pointer; valid when numSegments > 1
  int numSegments;                                    // offset 80
};
```
The existing `ginWins` and `ginOffset4K` fields are preserved at their original offsets for the single-segment fast path. The new `ginMultiSegmentWins` pointer accommodates a variable number of segments without changing existing field offsets.

##### `ncclResourceWindow_vidmem` (new)

A new fixed-size struct replaces the direct `ncclWindow_vidmem` embed in `ncclDevComm`. It is a subset of `ncclWindow_vidmem` containing only the three fields accessed during resource-buffer addressing (`lsaFlatBase`, `stride4G`, `mcOffset4K`). The remaining 56 bytes are reserved:

```c
struct ncclResourceWindow_vidmem {
  char reserved1[8];
  char* lsaFlatBase;
  char reserved2[8];
  uint32_t stride4G;
  uint32_t mcOffset4K;
  char reserved3[40];
};
static_assert(sizeof(ncclResourceWindow_vidmem) == 72);
```

This decouples `ncclDevComm::resourceWindow_inlined` from the `ncclWindow_vidmem` layout, so `ncclWindow_vidmem` can grow in the future without breaking the devcomm ABI. `ncclDevrCommCreateInternal` now copies only the three relevant fields instead of blitting the whole struct:

```c
outDevComm->resourceWindow_inlined.lsaFlatBase = winHost->lsaFlatBase;
outDevComm->resourceWindow_inlined.stride4G    = winHost->stride4G;
outDevComm->resourceWindow_inlined.mcOffset4K  = winHost->mcOffset4K;
```

##### `ncclWindow_vidmem_v22902` (new, backward compat)

A frozen snapshot of `ncclWindow_vidmem` as it existed before the elastic buffer feature (72 bytes). Used in `devcomm_v22902.cc` and `devcomm_v22907.cc` to preserve ABI for older device comm versions:

```c
struct ncclWindow_vidmem_v22902 {
  void* winHost;
  char* lsaFlatBase;
  int lsaRank;
  int worldRank;
  uint32_t stride4G;
  uint32_t mcOffset4K;
  uint32_t ginOffset4K;
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONNECTIONS];
};
static_assert(sizeof(ncclWindow_vidmem_v22902) == 72);
```

##### `multiSegmentGinInfo` (new, host-only)

Host-side bookkeeping struct stored in `ncclDevrMemory::ginSegmentInfos`. One entry per registered GIN segment:

```c
struct multiSegmentGinInfo {
  void*           ginHostWins[NCCL_GIN_MAX_CONNECTIONS]; // opaque host handles for deregistration
  ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS];  // device-visible handles
  CUmemLocationType memType;
  size_t segmentSize;
};
```

#### Changes to Registration

`symMemoryRegisterGin` is the main registration entry point for both single- and multi-segment cases:

**GPU-only buffers (`!globalHasSysmemSegment`):**

The full buffer is registered in one call using the existing `ncclGinRegister` path (with `NCCL_PTR_CUDA`). A single `ginSegmentInfos` entry is created to populate the shadow-pool segment window even for single-segment buffers. This requires support for dmabuf fds, so we add that as a requirement to use the feature for simplicity. Otherwise, GPU-only buffers would require the same logic that we added for host buffers, require API changes for applications using put/get, etc.

```c
ncclGinRegister(comm, mem->primaryAddr, mem->size,
                mem->ginHostWins, mem->ginDevWins,
                mem->winFlags, mem->maxGlobalNumSegments > 1, NCCL_PTR_CUDA);
mem->numGinSegments = 1;
```

**Buffers with CPU segments (`globalHasSysmemSegment`):**

1. All ranks must have the same number of segments (`numSegments == maxGlobalNumSegments`). If not, `ncclInvalidUsage` is returned.
2. Each segment's `CUmemLocationType` is queried via `cuMemGetAllocationPropertiesFromHandle`.
3. A `bootstrapAllGather` is used to exchange the padded segment-size array across all ranks. Each rank verifies that its segment sizes match every other rank's. If any mismatch is found, `ncclInvalidUsage` is returned.
4. Segments are registered individually using `ncclGinRegister` with the appropriate `ptrType` (`NCCL_PTR_HOST` for `CU_MEM_LOCATION_TYPE_HOST_NUMA`, `NCCL_PTR_CUDA` for device). Each call uses the sub-address `(char*)mem->primaryAddr + offset` and the corresponding segment size.
5. `mem->numGinSegments = mem->numSegments`. For a large number of segments, this results in many allgathers (multipler per regMrSym), but we decided against optimizing this in an effort to simplify the code for us and GIN plugin developers.

On registration failure, all previously registered segments are deregistered via `ncclGinDeregister` before returning the error.

`ncclGinRegister` gained a `memType` parameter (defaulting to `NCCL_PTR_CUDA`) that is forwarded to `regMrSym`.
**Segment location validation during `ncclDevrWindowRegisterInGroup`:**

The existing loop that retains `CUmemAllocationHandle`s was extended to validate location types. Each segment must be either `CU_MEM_LOCATION_TYPE_DEVICE` or `CU_MEM_LOCATION_TYPE_HOST_NUMA`; other types return `ncclInvalidArgument`.

**Fix to `ncclRegister` for HOST-typed pointers:**

`ncclRegister` previously called `ncclCuMemGetAddressRange` on all cuMem-backed pointers to detect sysmem segments. This path is invalid for HOST-type pointers since `IS_LEGACY_CUDA_IPC_CAPABLE` is not meaningful for host memory. The fix checks `CU_POINTER_ATTRIBUTE_MEMORY_TYPE` first and sets `hasSysmemSegment = true` directly when the base pointer is host memory, skipping the cuMem path.

#### Shadow-pool allocation for segment windows

A new helper `allocAndPopulateSegmentWindows` allocates the `ncclSegmentWindow` array in the shadow pool (so both CPU and GPU addresses are available) and initialises it from `ginSegmentInfos`:

```c
ncclShadowPoolAlloc(&devr->shadows,
    sizeof(struct ncclSegmentWindow) * mem->numGinSegments,
    (void**)&segmentWindowsDev, (void**)&segmentWindowsHost, stream);

for (int segment = 0; segment < mem->numGinSegments; segment++) {
  segmentWindowsHost[segment].memType     = mem->ginSegmentInfos[segment].memType;
  segmentWindowsHost[segment].segmentSize = mem->ginSegmentInfos[segment].segmentSize;
  for (int i = 0; i < NCCL_GIN_MAX_CONNECTIONS; i++)
    segmentWindowsHost[segment].ginWins[i] = mem->ginSegmentInfos[segment].ginDevWins[i];
}
cudaMemcpyAsync(segmentWindowsDev, segmentWindowsHost, ...);
```

`symWindowCreate` stores the device pointer in `winDevHost->ginMultiSegmentWins`. During `ncclDevrCommCreateInternal`, once GIN is activated, existing windows have their `ginMultiSegmentWins` reallocated if `numGinSegments > 1`. The old shadow-pool allocation is freed and replaced with a freshly populated one.

`symWindowDestroy` frees `ginMultiSegmentWins` from the shadow pool before freeing the window itself.

#### Changes to put (assume no signal)

srcWindow and dstWindow can be from two different registrations and have different number of segments. First, the offset must be appropriately translated into the start and end segments for each window, with their sizes. After that, they must be transferred appropriately. 

Overall idea : Take the minimum remaining segment size between windows, do a put and loop until all the data is transferred. 
Take the example below. Assume that x is the total size of the message, srcWindow has two segments, and dstWindow has 3 segments.
Put : [0.5x, 0.5x] -> [0.2x, 0.6x, 0.2x]
There is a local key (for src segments) and remote key (for dst segments). The puts are from srcWindow to dstWindow.
```c
put 0.2x using local key 0 and remote key 0
put 0.3x using local key 0 and remote key 1
put 0.3x using local key 1 and remote key 1
put 0.2x using local key 1 and remote key 2
```
At every step, take the minimum of the two windows, find out what's remaining and continue to loop until we transfer x bytes. We also internally do a membar.sys to ensure that all data from both CPU and GPU are visible to NCCL before performing the network transfer.

##### Device-side helper functions in `gin__funcs.h`

Three new `NCCL_DEVICE_INLINE` helpers support the multi-segment loop:

```c
// Walk win->ginMultiSegmentWins to find which segment `offset` falls into,
// and compute the intra-segment offset.
findSegmentFromWindow(ncclWindow_t win, size_t offset,
                      int* outSeg, size_t* outSegOffset);

// Return min(srcRemaining, dstRemaining, remaining) — the largest chunk
// that doesn't cross a segment boundary in either window.
getSegmentChunkSize(size_t srcRemaining, size_t dstRemaining, size_t remaining);

// Advance a (seg, segOffset) cursor by chunkSize bytes, wrapping to the
// next segment when the current one is exhausted.
advanceSegmentCursor(int* seg, size_t* segOffset,
                     size_t chunkSize, size_t segmentSize);
```

##### Implementation for different segment types for `put` and `get`

The put and get implementations share the same structure:

1. **`ncclGin_isDeviceOnly(bufType)` is true** — existing single-call fast path, unchanged. No overhead added.

2. **`bufType` is non-device but `numSegments == 1`** — single call as before, but `requiredRelease` is unconditionally escalated to `cuda::thread_scope_system` for safety.

3. **Multi-segment loop** — `findSegmentFromWindow` is called once each for src and dst. The loop iterates until `remaining == 0`:
   - `getSegmentChunkSize` computes the largest chunk that doesn't cross a boundary in either window.
   - If the src segment's `memType == CU_MEM_LOCATION_TYPE_HOST_NUMA` and no fence has been issued yet, `localRequiredRelease` is escalated to `cuda::thread_scope_system` for that put (and all subsequent puts revert to `cuda::thread_scope_thread`).
   - Signal/counter actions are performed only on the last put (`isLastPut = (remaining == putSize)`); intermediate puts use `ncclGin_None{}`.
   - `advanceSegmentCursor` advances both src and dst cursors after each put.

For `get`, there is no signal/counter concept so the loop is simpler: each chunk issues one `ncclGinApi_Get` call and advances both the remote and local cursors.

##### API for Put
Using separate APIs for all multi-segment/elastic buffer transfers would make the support matrix clearer from the NCCL side, but we would have to introduce 15+ new functions, which leads to a very poor user experience. Hence, we consider using the same API but adding a new templated SegmentType argument to put. This ensures that the performance of the existing put path is unaffected and makes this an opt-in.

Three tag types are defined in `src/include/nccl_device/gin.h`:

```c
struct ncclGin_SegmentDevice {};    // all segments are device-backed (default)
struct ncclGin_SegmentMixed {};     // mix of HOST_NUMA and device-backed segments
struct ncclGin_SegmentHostNuma {};  // all segments are HOST_NUMA (CPU-backed)
```

`ncclGin_isDeviceOnly` is a `constexpr` helper used inside `put`/`get` to select the fast path at compile time:

```c
constexpr bool ncclGin_isDeviceOnly(ncclGin_SegmentDevice)  { return true;  }
constexpr bool ncclGin_isDeviceOnly(ncclGin_SegmentMixed)   { return false; }
constexpr bool ncclGin_isDeviceOnly(ncclGin_SegmentHostNuma){ return false; }
```

The updated `put` signature (both overloads) adds `SegmentType` as the last template parameter and `bufType` as the last positional argument, defaulting to `ncclGin_SegmentDevice{}`:

```c
template<unsigned beMask>
template<
  typename RemoteAction,
  typename LocalAction,
  typename Coop,
  typename DescriptorSmem,
  typename SegmentType = ncclGin_SegmentDevice
>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::put(
    ncclTeam team, int peer,
    ncclWindow_t dstWin, size_t dstOffset,
    ncclWindow_t srcWin, size_t srcOffset, size_t bytes,
    RemoteAction remoteAction, LocalAction localAction,
    Coop coop,
    DescriptorSmem descriptor,
    cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
    uint32_t optFlags, SegmentType bufType = ncclGin_SegmentDevice{}
  );
```

`get` gains the same `SegmentType`/`bufType` trailing parameter.

##### Example usage with an alltoall kernel

The rest of the kernel remains the same. Users need to specify a new parameter at the end to indicate whether the put is targeting multi-segment memory. The memory registration call (ncclCommWindowRegister) is exactly the same, so users will only need to modify their kernel.

```c
__global__ void ginAlltoAllKernel(ncclWindow_t sendwin, size_t sendoffset,
                                  ncclWindow_t recvwin, size_t recvoffset,
                                  size_t count, struct ncclDevComm devComm) {
#if __CUDA_ARCH__ >= 700
  int ginContext = 0;
  unsigned int signalIndex = 0;
  ncclGin gin{devComm, ginContext};
  uint64_t signalValue = gin.readSignal(signalIndex);

  ncclGinBarrierSession<ncclCoopCta> bar{ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::None);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;

  const size_t size = count * sizeof(float);
  for (int r = tid; r < devComm.nRanks; r += nthreads) {
    gin.put(ncclTeamWorld(devComm), r,
            recvwin, recvoffset + devComm.rank * size,
            sendwin, sendoffset + r * size,
            size, ncclGin_SignalInc{signalIndex}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
            cuda::thread_scope_thread, cuda::thread_scope_device, ncclGinOptFlagsDefault, ncclGin_SegmentMixed{});
  }

  gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + devComm.nRanks);
  gin.flush(ncclCoopCta());

  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::None);
#endif
}
```

#### Required changes to signal for DirectNIC platforms - not supported in this release, but included in this PLC for completeness.
A GIN signal must guarantee that all prior puts are visible/committed to memory. For mixed memory, an IB atomic operation on GPU memory does not always guarantee visibility of puts on CPU memory. This can be divided into two broad cases:

- Platforms without DirectNIC : In this case, not changes to signal are required. Since the GPU consumes the data (with a read), any previous writes will be made visible and this ensures correctness on such platforms.
- Platforms with DirectNIC : The PCIe data path from the NIC to the CPU and GPU are different. This requires a read over PCIe to flush previous writes.

The sender must perform three operations for the signal in the DirectNIC case (in the specified order):

1. Signal to CPU buffer (or RDMA read from CPU buffer)
2. IB atomic fence (Implicit for GDAKI), or another signal (atomic) to GPU buffer in the case of proxy since there's no fence API defined there.  
3. Signal to GPU buffer

#### Testing

A new API test (`test/apitest/device_api/ncclDevApi_multi_segment_test.cu`) was added with the following new test cases:

| Test case | Description |
|---|---|
| `LSA_allreduce_16_segments` | LSA allreduce with 16 alternating GPU+host-NUMA segments |
| `GIN_alltoall_16_gpu_segments` | put-based alltoall with 16 GPU-only segments |
| `GIN_alltoall_16_segments` | put-based alltoall with 16 alternating GPU+host-NUMA segments |
| `GIN_get_alltoall_16_segments` | get-based alltoall with 16 alternating GPU+host-NUMA segments |
| `GIN_alltoall_16_host_segments` | put-based alltoall with 16 HOST_NUMA-only segments |
| `GIN_get_alltoall_16_host_segments` | get-based alltoall with 16 HOST_NUMA-only segments |
| `registration_fails_mismatched_segment_sizes` | `ncclGroupEnd` must return `ncclInvalidUsage` when segment sizes differ across ranks |

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Bharath Ramesh

</details>
