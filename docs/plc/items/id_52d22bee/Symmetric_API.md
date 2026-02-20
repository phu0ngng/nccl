# Symmetric API
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->
Symmetric memory is a key to low latency and high bandwidth in NCCL. This feature provides the window API to allow users
and NCCL developers to register memory among ranks into NCCL window.

### NVbugs / Jira Tickets

### User Experience
Users need to call `ncclCommWindowRegister` to register the memory.
Passing window memory to NCCL ops would enable NCCL symmetric-based optimizations.

### Assumptions, constraints and dependencies
Window registration API only accepts VMM-allocated memory.

### Use Cases
Window memory registration.

### Platform Requirements

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

Window registration APIs are proposed as follows:
```
ncclResult_t  ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags);
ncclResult_t  ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win);
```

The `ncclCommWindowRegister` API is used to register the user-allocated memory to the NCCL communicator. The memory must be allocated by VMM API. The API returns the win handle for user's reference.

The layout of the window memory is as follows (also shown in visual graph):
1. Each communicator has a global unicast address space reserved at init stage for all peers.
2. Stride is determined by the maximal physical memory size of all peers aligned to 128GB. So the overall address space is STRIDE * NUM_PEERS.
3. Multicast address space is separate from unicast address space, and its size is STRIDE. Multicast handle is imported by every peer in the nvlink domain and mapped without bind actual physical memory.
4. In addition, user can tweak STRIDE manually by setting `NCCL_WIN_STRIDE`.

```
+-----------------------------------------------------------------------------------+
|                        Total Address Space (STRIDE * NUM_PEERS)                   |
+-----------------------------------------------------------------------------------+

+----------------+----------------+----------------+----------------+----------------+
|    Peer 0      |    Peer 1      |    Peer 2      |     ...        |   Peer N-1     |  <- Unicast
|   Segment      |   Segment      |   Segment      |                |   Segment      |     Address
| (STRIDE size)  | (STRIDE size)  | (STRIDE size)  |     ...        | (STRIDE size)  |     Space
+----------------+----------------+----------------+----------------+----------------+

+----------------+
|   Multicast    |  <- Separate multicast address space
|   Segment      |     (Single STRIDE size, shared by all peers)
| (STRIDE size)  |
+----------------+
```

During the window registration, it will call local registration (i.e., `ncclRegister`) to create a registration handle. For a user buffer, it can only be symmetrically registered once. Once the registration handle is created, we call into the corresponding transport to further create a symmetric memory space. For IPC transport, peers on the same node or NVLINK domain export its buffer handle and exchange with each other, and then every rank imports its peers' buffer handles to the global unicast address space. Offset of import for each peer is determined by existing symmetric heap state (i.e., `comm->symAllocHead`). In order to refer to the corresponding unicast symmetric buffer of a peer, user can simply perform arithmetic operation like:
```
void* peer_addr = (void*)(comm->baseUCSymPtr + STRIDE * peer_rank + offset);
```

For NVLS transport, it just needs to bind the physical memory to the global multicast handle with appropriate offset (see `ncclNvlsSymmetricMap`).
Considering multicast address, the equation is even simpler:
```
void* peer_addr = (void*)(comm->baseMCSymPtr + offset);
```

Here `offset` is the offset between my rank's symmetric starting address (i.e., `symPtr - (comm->baseUCSymPtr + STRIDE * my_rank)`).

In addition, since registration are collective and blocking, so it requires group ops when it comes to one thread managing multiple GPUs. If users do not want to use
symmetric registration, setting NCCL_WIN_ENABLE to 0 can disable it.
<!-- note: the following HTML code is also valid -->
<!-- <img src="images/example.png" width="900" height="500" /> -->

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
https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/810

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation

#### Where to run?

#### What to run?

#### Expected output?

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

#### What is measured?

#### Results


</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Kaiming Ouyang
</details>
