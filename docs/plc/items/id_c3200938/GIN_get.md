# GIN get
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract

This PLC introduces `gin.get` to the Device API.
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

https://nvbugspro.nvidia.com/bug/5844585

### User Experience

Users can call non-blocking `get` to any peer:

```
void get(ncclTeam team, int peer, ncclWindow_t remoteWnd, size_t remoteOffset,
    ncclWindow_t localWnd, size_t localOffset, size_t bytes,
    Coop coop = ncclCoopThread{}, uint32_t optFlags = ncclGinOptFlagsDefault)
```

Consecutive calls to `get` are not guaranteed to complete in order. To check for completion, users can call `gin.flush` which ensures completion/visibility of all previous `get`s. We will also introduce a per-peer `flushAsync` function asynchronously waits for the completion of all puts/gets up until the point `flushAsync` is called:

```
void flushAsync(int peer, Coop coop, cuda::memory_order ord = cuda::memory_order_acquire, ncclGinRequest_t* request);
void wait(ncclGinRequest_t& request);
```

### Assumptions, constraints and dependencies

`gin.get` works in all environments that support GIN.

For `gin.get` CPU Proxy, the minimum supported plugin version is v13 (introduced in this PLC).

### Use Cases

One major customer use case is "batches" of gets. A user wants to submit batch 1, then submit batch 2, then check for completion of batch 1, then check for completion of batch 2.

<!-- ### Platform Requirements -->

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

### Proposed API Design

#### Interactions with existing GIN APIs

Interaction with barrier:

* Barrier with `ncclGinFenceLevel::None`: the completion of the barrier has no connection to the completion/visibility of previous calls to `get`

Interaction with flush:

* The completion of `flush` ensures all previous calls to `get` are complete/visible.
* With this change, `flush` is very similar to NVShmem `quiet`.

Interaction with railed GIN:

* When GIN is railed, a call to `gin.get` on a non-railed peer is undefined behavior. This is the same behavior as `gin.put`.

Interaction with elastic buffers:

* TBD after both features are stable. `get` would require segmented logic that is similar to put. `flush`/`barrier` require extra logic to ensure CPU data visibility from the GPU.
* Calling `get` for data on a remote CPU is a major customer use case.

Interaction with put or signal:

* Officially, the completion of a `get` has no connection to previous calls to `put` or `signal`.

Interaction with abort:

* Currently, NCCL only considers `abort` in the DeviceAPI if the user is NCCL itself. Since NCCL does not use `gin.get/wait`, `gin.wait` does not need `abort` support.

#### Comparison to PUT

The most similar API is `put`. Here are the differences:

```c
void put(
    ncclTeam,
    int peer,
    ncclWindow_t dstWnd, // get: renamed to remoteWnd for clarity
    size_t dstOffset,
    ncclWindow_t srcWnd, // get: renamed to localWnd for clarity
    size_t srcOffset,
    size_t bytes,
    RemoteAction remoteAction = ncclGin_None{}, // get: omitted. Get+Signal does not make sense
    LocalAction localAction = ncclGin_None{},  // get: omitted. Get+Counter may be of interest, but is not needed right now
    Coop coop = ncclCoopThread{},
    DescriptorSmem descriptor = ncclGin_None{},
    cuda::thread_scope givenRelease = cuda::thread_scope_thread, // get: omitted. Does not make sense.
    cuda::thread_scope requiredRelease = cuda::thread_scope_device, // get: omitted. Does not make sense.
    uint32_t optFlags = ncclGinOptFlagsDefault,
  )
```

#### Definition of ncclGinRequest_t

`ncclGinRequest_t` is a generic 16B opaque type. Each backend defines its own data layout in its internal namespace. The layout is easily extensible/refactorable; it is not subject to any backwards compatibility constraints.

```c
typedef char ncclGinRequest_t[16];

struct ncclGinGdakiRequest {
  int peer;                               // 4B
  doca_gpu_dev_verbs_ticket_t docaTicket; // 8B
};

struct ncclGinCpuProxyRequest {
  int peer;          // 4B
  uint32_t gfdIdx;   // 4B
};

static_assert(sizeof(ncclGinGdakiRequest) <= sizeof(ncclGinRequest_t));
static_assert(sizeof(ncclGinCpuProxyRequest) <= sizeof(ncclGinRequest_t));
```

#### GIN v13

The GIN Plugin API requires a new version with 2 new functions. These functions are only needed by CPU Proxy.

```c
typedef struct {
  // ... Existing functions ...

  // New functions
  ncclResult_t (*iget)(void* collComm, uint64_t remoteOff, void* remoteMhandle, size_t size,
      uint64_t localOff, void* localMhandle, uint32_t rank, int connectionId, void** request);

  ncclResult_t (*iflush)(void* collComm, int connectionId, void* mhandle, void** request);
} ncclGin_v13_t; // New version.
```

`iflush` is similar to the NET definition of `iflush`. For comparison, here is the NET definition of `iflush`:

```c
ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
```

OPEN: Why does NET give the specific mhandles, when 1 read to any memory is sufficient? Does AWS require this? Does GIN need to give the specific handles? (This could make gin.flush difficult)

### Implementation

#### GDAKI Implementation

`gin.get` is a thin wrapper around gpunetio:


```c
struct ncclGinApi_Get<NCCL_NET_DEVICE_GIN_GDAKI> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                      ncclGinWindow_t localWin, size_t localOff, size_t bytes, uint32_t optFlags) {
      doca_gpu_dev_verbs_addr raddr; // properly initialized raddr
      doca_gpu_dev_verbs_addr laddr; // properly initialized laddr
      doca_gpu_verbs_addr daddr; // properly initialized with some fixed previously-registered address. TODO
      doca_gpu_dev_verbs_get<DOCA_GPUNETIO_VERBS_MCST_ENABLED>(laddr, raddr, daddr);
  }
}
```

The implementation of `gin.flush` is extended to ensure completion&visibility of previous gets:

```c
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_GDAKI> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag) {
    // NEW: add visibility guarantee
    doca_gpu_verbs_addr daddr; // properly initialized with some fixed previously-registered address.
    doca_gpu_dev_verbs_get_wait<DOCA_GPUNETIO_VERBS_MCST_ENABLED>(qp, daddr); // new

    // Existing. Ensure all wqe are complete.
    // ...
  }
}
```

#### CPU Proxy Implementation

**GFD extension**

`ncclGinProxyOp_t` needs 2 new ops:
```
typedef enum {
  ncclGinProxyOpPut = 1 << 0,
  ncclGinProxyOpBaseMask = 1 << 0,
  ncclGinProxyOpWithInline = 1 << 1,
  ncclGinProxyOpWithCounter = 1 << 2,
  ncclGinProxyOpWithSignalInc = 1 << 3,
  ncclGinProxyOpWithSignalAdd = 1 << 4,
  ncclGinProxyOpVASignal = 1 << 5,
  ncclGinProxyOpGet = 1 << 6, // NEW
  ncclGinProxyOpFlush = 1 << 7, // NEW
} ncclGinProxyOp_t;
```

With this change, `ncclGinProxyOp_t` exceeds the 6 bits allocated in the GFD. Some reserved bits will be used for the extension:

```c
typedef enum {
  ncclGinProxyGfdHeader = 0, // Original ops are stored here
  // ... omitted qwords 1-6 ...
  // ncclGinProxyGfdReserved = 7,
  ncclGinProxyGfdHeaderExt = 7, // NEW. QWord 7 is un-reserved.
  ncclGinProxyGfdQwords = 8,
} ncclGinProxyGfdQwordIdx_t;

typedef union {
  // ...
  struct {
    uint64_t flag : 1;
    uint64_t opLow : 6; // Original 6 bits stay as-is for backwards compat.
    uint64_t size : 57;
  } __attribute__((packed)) header;
  struct {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint16_t opHigh; // extend op by 16 bits.
    uint8_t resv2;
    uint32_t resv3;
  } __attribute__((packed)) headerExt; // NEW. Used by QWord 7.
  // ...
}
```

This is a backwards compatible change as long as we process old ops (and exit) before processing new ops.

**iget and iflush**

`iget` is straightforward: it issues a READ RDMA operation.

`iflush` issues a READ RDMA operation on the loopback qp. This QP exists on the `recvComm`, not the `sendComm`. The logic for `test` needs to be updated accordingly:

```c
ncclResult_t ncclGinIbProxyTest(void *collComm, void *request, int *done) {
  // ... code omitted

  // pseudocode
  struct ibv_cq* cq = cComm->fullSendComm[rank]->base->cq; // existing logic uses sendComm cq
  if (req->type == NCCL_NET_IB_REQ_FLUSH) { // NEW. If flush, use recvComm cq
    cq = cComm->fullRecvComm[rank]->base->cq;
  }
  NCCLCHECK(wrap_ibv_poll_cq(cq, 4, wc, &wrDone));

  // ... code omitted
}
```

An alternate design adds a loopback qp to `sendComm`. This would allow `test` to poll just 1 CQ, but requires GIN-specific changes to the NetIB definition of SendComm.

**gin.get and gin.test**

`gin.get` submits a GFD with `ncclGinProxyOpGet`, which calls `iget` when the GFD is processed.

`gin.flush` has three steps:

1) wait for all completions. This step already exists
2) submit a `ncclGinProxyOpFlush` GFD
3) wait for all completions


### Additional optimizations

**ncclGinOptFlagsNoCST**

On some platforms, the completion of a GDAKI Get WQE implies the visibility of the data. This is true for platforms later than Ampere without DirectNIC. Additional logic to ensure data visibility is needless overhead. We give users the option to skip this logic with an additional `ncclGinOptFlag` called `ncclGinOptFlagsNoCST`.

The new `gin.get` looks something like:

```c
struct ncclGinApi_Get<NCCL_NET_DEVICE_GIN_GDAKI> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                      ncclGinWindow_t localWin, size_t localOff, size_t bytes, ncclGinRequest_t* outRequest, uint32_t optFlags) {
      if (optFlags & ncclGinOptFlagsNoCST) {
        doca_gpu_dev_verbs_get<DOCA_GPUNETIO_VERBS_MCST_DISABLED>(...);
      } else if (outRequest == nullptr) {
        // we can safely skip cst because wait will not be called on this get
        doca_gpu_dev_verbs_get<DOCA_GPUNETIO_VERBS_MCST_DISABLED>(...);
      } else {
        doca_gpu_dev_verbs_get<DOCA_GPUNETIO_VERBS_MCST_ENABLED>(...);
      }
  }
}
```

`gin.flush` is extended to take `optFlags` as an argument:

```c
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_GDAKI> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag,
                  uint32_t optFlags = ncclGinOptFlagsDefault) { // New arg
    if (!(optFlags & ncclGinOptFlagsNoCST)) { // new condition to support optimization
      // new logic to support get
      doca_gpu_verbs_addr daddr; // initialize properly
      doca_gpu_dev_verbs_get_wait<DOCA_GPUNETIO_VERBS_MCST_ENABLED>(qp, daddr);
    }

    // Existing logic. Ensure all wqe are complete.
    // ... code omitted ...
  }
}
```

**ctx.getCalledSinceLastFlush**

If a user never calls get, the logic to ensure data visibility in `gin.flush` is needless overhead.

We maintain a per-context atomic bool called `getCalledSinceLastFlush`. This value is set to true each time `get` is called, and set to false after `flush` is complete. If `false`, `gin.flush` skips any data visibility logic.

### Other designs considered

We considered the following other designs:
1) get returns a ncclGinRequest_t. This is of limited use for the customer's batched get use case; the user would need to check the completion of every get
2) batched get API where the user can sumbit multiple requests at once. This adds API bloat and is of minimal use; the only optimization we would do is delay ringing the doorbell, which the user can do themselves
3) get+counter. Since counters do not imply completion of all previous operations, we would need to do get+counter for every single get in the batch. This has high overhead.


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

There will be 5 MRs:

1) unoptimized implementation of GDAKI and CPU Proxy (everything except `gin.test`)
2) `ncclGinOptFlagsNoCST` optimization
3) `ctx.getCalledSinceLastFlush` optimization
4) CPU Proxy get+flush optimization
5) `gin.test`: the implementation of `gin.test` is different before and after (4) is implemented. So we wait until after (4).

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation

#### Where to run?

Environments:
* Ampere or earlier -> CST required
* DirectNIC -> CST required
* Later than Ampere, no DirectNIC -> no CST required for GDAKI

#### What to run?

New DeviceAPI apitests that use gin.get:

* `gin.get` -> The local buffer _eventually_ has the correct value
* `gin.get` + `gin.wait` -> The local buffer has the correct value
* `gin.get` + `gin.flush` -> The local buffer has the correct value
* `gin.get` + `barrier` with `ncclGinFenceLevel::Release` -> The local buffer has the correct value


#### Expected output?

Tests pass

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

New `alltoall` perftest implementation that uses gin.get instead of gin.put

#### What is measured?

Standard latency-throughput curve

#### Results

The performance of the `gin.get` implementation is comparable to the performance of the `gin.put` implementation.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Katie

</details>
