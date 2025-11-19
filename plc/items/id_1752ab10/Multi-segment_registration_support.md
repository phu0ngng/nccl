# Multi-segment registration support
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract

This feature enables NCCL registration to support multiple segments of physical memory (done through cuMemMap) mapped to one contigious VA space for the p2p, ib and nvls transports.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->
The primary motivation for this feature is to more effectively support the "expandable segments" mode in Pytorch. This mode enables pytorch to allocate segments of memory and extend them as needed, while providing a contiguous VA space for buffers. Expandable segments reduces memory fragmentation for important workloads (LLAMA pre-training, DeepSeek inference, DLRM) as it enables finer grained re-use of memory. Previous versions of NCCL only supported registering one physical segment. Passing multiple segments resulted in NCCL skipping registration to avoid IMA issues. 

### NVbugs / Jira Tickets
https://nvbugspro.nvidia.com/bug/5529614

### User Experience
Transparent to the user.

### Assumptions, constraints and dependencies
CuMem support is required.

### Use Cases
As mentioned above, this feature is intended to support the "expandable segments" mode in Pytorch.

<!-- ### Platform Requirements -->

### Functional Requirements
1. NCCL must automatically detect multi-segment memory given a virtual address.
2. Must work with NVLS, IB and P2P transports.
3. Support both graph registration and local registration.

<!-- ### System Requirements -->
<!-- ### Interface Requirements -->
<!-- ### KPI Requirements -->
<!-- ### Security Requirements -->
<!-- ### Legal and Standards Requirements -->
<!-- ### Telemetry Requirements -->
<!-- ### Backward Compatibility Requirements -->
<!-- ### Virtualization Requirements -->

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed Design

#### 1. Utility functions (added to src/include/alloc.h)

Add multi-segment specific functions for freeing reserved memory (ncclCuMemFreeAddrMultiSegment, ncclCuMemFreeMultiSegment, ncclCudaFreeMultiSegment) and address range querying (ncclCuMemGetAddressRange). Stub fallbacks included for older CUDA versions. The implementation for ncclCuMemGetAddressRange is shown below. The other functions follow a similar pattern of using cuMemGetAddressRange.

```c
// Get the base and size of all segments that span a given user buffer
static inline ncclResult_t ncclCuMemGetAddressRange(CUdeviceptr userBuff, size_t userBuffSize, CUdeviceptr* mappedPtrBase, size_t* totalMappedBufferSize, int* numSegments) {
  *totalMappedBufferSize = 0;
  *mappedPtrBase = 0;
  if (numSegments) *numSegments = 0;
  CUdeviceptr userBuffStart = userBuff;
  CUdeviceptr userBuffEnd = userBuffStart + userBuffSize;
  CUdeviceptr mappedPtrEnd = userBuffStart;
  CUdeviceptr baseSend;
  size_t baseSendSize;

  while (mappedPtrEnd < userBuffEnd) {
    CUCHECK(cuMemGetAddressRange(&baseSend, &baseSendSize, mappedPtrEnd));

    if (*totalMappedBufferSize == 0) {
      *mappedPtrBase = baseSend;
    }
    *totalMappedBufferSize += baseSendSize;
    mappedPtrEnd = baseSend + baseSendSize;

    if (numSegments) *numSegments = *numSegments + 1;
  }
  return ncclSuccess;
}
```

#### 2. Modifications for NVLS and IB

For local registration, no changes are required since NVLS and IB registration already support multiple segments internally. For graph registration, we use ncclCuMemGetAddressRange to get the base and size of all segments that span a given user buffer so that we get the total size of all segments combined.

#### 3. Modifications for P2P

For P2P, more modifications are needed as each segment has its own handle.
We introduce a new function "ipcHandleMultiSegmentRegistration" to handle multi-segment registration for P2P. This separates the logic for single-segment and multi-segment registration. In this function, we iterate over all the segments, export necessary handles/metadata and store in the ipcInfo struct for the segment. We now have "numSegments" instances of struct p2pIpcExpInfo (this could be suboptimal for the packet size for each segment as we have additional metadata on legacyIpc that multi-segment does not support, but this approach is simpler to implement)

For handles of type CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, we introduce a new function ncclProxyClientBatchQueryFdBlocking to convert the local file descriptors for each segment so that it can be sent to the remote process using UDS.

Changes to p2pProxyRegister - Modified to handle "numSegments" instances struct p2pIpcExpInfo. We reserve address for the total size of all segments (using cuMemAddressReserve), map each segment individually (using cuMemMap), and then setaccess for p2p for the entire buffer at the end. 

Changes to p2pProxyDeregister - Modified to call the multi-segment versions of the free "utility" functions defined above. We add a numSegments field to the ncclIpcImpInfo struct to indicate the number of segments in the buffer, so that it can be used here to free the correct number of segments.


<!-- ### Interface Architecture -->
<!-- ### System KPIs & Metrics -->
<!-- ### Data Architecture -->
<!-- ### Security Design -->
<!-- ### Debugging & Troubleshooting -->
<!-- ### Logging and Instrumentation -->
<!-- ### Operational Considerations -->

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Coding</h2></summary>
<!-- ============================================================================================-->

### Commit list or MR
https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/1523
</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Validation
Tested with a local test program that allocates multiple segments of memory and registers them with NCCL. The Pytorch team also did some basic testing with the expandable segments mode.


<!-- #### Code Coverage Goal Defined -->
<!-- #### KPI Coverage Goals Defined (performance, stress, stability, throughput, latency) -->
<!-- #### Requirement Coverage Goal Defined -->
<!-- #### Quality Thresholds Defined -->
<!-- #### Test Timeline -->
<!-- #### SW Verification and Test Plan -->
<!-- ### Test plan -->
<!-- #### Requirements Tests -->
<!-- #### Interface Tests -->
<!-- #### Fault-injection Tests -->
<!-- #### Resource Usage Tests -->
<!-- #### Design Coverage Testing -->
<!-- #### Boundary Tests -->
<!-- #### Certification Tests -->
<!-- #### Stress Tests -->
<!-- #### Stability Tests -->
<!-- #### Perf and Power KPI Tests -->
<!-- #### Usability & OOBE Tests -->
<!-- #### Manufacturing Diagnostic(Factory) Tests -->



</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Bharath Ramesh

</details>
