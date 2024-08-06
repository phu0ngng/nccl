# CUMEM Host Allocation
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->
NCCL is using legacy cudaHostAlloc to allocate pinned host memory for GPU access. However, this does not
consider the export and import reference count so that once the buffer is freed by the local GPU, the
remote GPU can no longer access it. Due to this limitation, NCCL has to make ncclCommAbort a intra-node
collective function to wait until all jobs on all on-node GPUs retire and avoid illegal inter-GPU buffer
accesses. However, this would increase the chances of hang when calling ncclCommAbort since applications
might have multiple communicators and cannot call ncclCommAbort with the right sequence for each rank.
To solve this issue, this feature change all pinned host memory allocation to CUMEM based allocation
so that we can lift the legacy allocation limitation.

### NVbugs / Jira Tickets
https://nvbugspro.nvidia.com/bug/4345173
https://jirasw.nvidia.com/browse/NCCL-1633

### User Experience
Calling ncclCommAbort will no longer cause hang in any case.

### Assumptions, constraints and dependencies
Require CUDA toolkit version >= 12.2, which supports CUMEM host allocation

### Use Cases
All pinned host memory allocation in NCCL.

### Platform Requirements
NA

<!-- ### Functional Requirements -->
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
New allocation function `ncclCuMemHostAlloc` is added to allocate pinned host memory with CUMEM API.
The function will automatically detect the CPU id that is closest to the current GPU and allocate
the host memory on that CPU.

All cudaHostAlloc related functions ared replaced with `ncclCuMemHostAlloc()`. In addition, for shm
transport, we have pinned host memory that needs to be exported and imported by different ranks.
To this reason, two new functions `ncclShmAllocateShareableBuffer()` and `ncclShmImportShareableBuffer()`
are created, and original function `ncclShmOpen()` is obsolete if CUMEM host allocation is available for
buffer export and import. Assume write based communication, both sender and receiver side call
`ncclShmAllocateShareableBuffer()` to allocates header buffer as well as receiver's data buffer, and export
it as a handle. After exchange the handle, both of them import the buffer by `ncclShmImportShareableBuffer()`.
Now shm transport follows the allocation style where all buffer allocation must go through proxy thread.

About proxy thread, once the CUMEM host allocation is enabled and abort is called by user. It directly close
all connections, free buffers, and exit without waiting.

### Interface Architecture

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
 https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/504
</details>
 
<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation

#### Where to run?

#### What to run?
Add a new FT test to perftest:
1. Disable runtime connect NCCL_RUNTIME_CONNECT=0 to allow each rank connect to each other's proxy.
2. Run on at least 2 nodes.
3. After `ncclCommInitRank()`, on each node, local rank 0 first call `ncclCommAbort()`.
4. Then, all ranks call `MPI_Barrier()`.
5. Finally, all ranks except local rank 0 call `ncclCommAbort()`.

#### Expected output?
The new FT test should not hang

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

### Performance
The NCCL collective/sendrecv performance will not be impacted. But it can increase the init or connection setup time.

#### What is measured?

#### Results


</details>
 
<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s): 
  - Unmesh Deodhar
  - Kaiming Ouyang

</details>
