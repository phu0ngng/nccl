# GPI for GIN
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract

This PLC describes the GPI (GPU-Push Interface) backend for the GIN (GPU-Initiated Networking)
Device API in NCCL. GPI enables GPU threads to push network descriptors (GFDs) directly to a
NIC-visible MMIO queue without any CPU involvement and no membars, providing low-latency, high-throughput
scale-out communication for custom device kernels. The backend implements the full set of GIN
device API operations — `put`, `putValue`, `get`, `flush`, `flushAsync`, `wait`, `fence`,
`resetSignal`, `resetCounter`, `getSignalPtr`, and `getCounterPtr` — on top of the GPI
hardware queue abstraction.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

NCCL's GIN Device API needs a backend that allows GPU threads to directly drive network
operations without kernel preemption or CPU proxy involvement. The existing GDAKI backend
targets DOCA GPUNetIO (InfiniBand Verbs), while the CPU Proxy backend routes GFDs through a
CPU-side proxy thread. GPI provides a third, GPU-NIC centric path that posts compact
GPU Freindly Descriptors (GFDs) to a shared MMIO ring queue that is consumed by the NIC
HW (DPA), Providin glow latency and zero occupancy impact.

### NVbugs / Jira Tickets
https://nvbugspro.nvidia.com/bug/5669180

### User Experience

Users interact with GPI through the same `ncclGin` Device API surface as all other backends.
No source-level changes are required compared to GDAKI- or Proxy-based GIN usage. The GPI
backend is selected automatically when `NCCL_NET_DEVICE_GIN_GPI` is the active backend in
the communicator's backend mask.

Resource-sharing granularity is configurable at construction time via `ncclGinResourceSharingMode`:

| Mode | Scope | Use case |
|------|-------|----------|
| `NCCL_GIN_RESOURCE_SHARING_GPU` | Whole GPU (device-scope atomics) | General multi-CTA kernels |
| `NCCL_GIN_RESOURCE_SHARING_CTA` | Single CTA (block-scope atomics) | CTA-owned channels |

### Assumptions, constraints and dependencies

- GPI backend works only with SPCX NCCLNet Plugin.


### Use Cases

- GPI is just a new backend for GIN, any user of GIN can take advnatag eof loa-latency GPI interface
- Custom device kernels (DeepEP, DeepSeek-style MoE dispatch/combine) that need to issue
  network PUT/GET operations from arbitrary GPU threads without CPU involvement.
- Symmetric collective kernels (`AllGather_GinHier`, `ReduceScatter_GinHier`) that need low-
  overhead scale-out alongside NVLink operations.
- Any workload that needs explicit fence-level memory ordering guarantees over system scope
  between GPU-initiated network operations.

### Platform Requirements

GPI-capable NIC with GPU-accessible MMIO ring queue. CUDA toolkit ≥ 12.2. SM ≥ 7.0 for basic
operation;
- GPI reuires NV NICs CX8+
- GPI requires that the PeerMemOverwite or coherent systems



</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed Design

#### GPU Channel Layout

Each GPI context maps to one `gpi_gpu_channel_t`, which is laid out in GPU-accessible memory
as follows:

```
[ gpi_gpu_channel_t ] [ flush_tickets[nRanks] ]
```

The channel contains a ring queue (`queue_`) with:
- `pi` — producer index (atomically incremented by the GPU to claim a slot)
- `ci` / `ci_shadow` — consumer index updated by the NIC; shadowed in GPU memory
- `gpu_memic_ptr` — pointer to the MMIO-mapped GFD ring buffer
- `log_depth` — log₂ of the ring size

Counter and signal arrays (`gpu_counter_ptr_`, `gpu_signal_ptr_`) are also mapped into GPU
memory. The last `nRanks` counters (starting at `flush_counter_idx`) are reserved for flush
tracking.

#### GFD Posting

A GFD (GPU Functional Descriptor) is a 64-byte descriptor that encodes the operation to be
performed by the NIC. There are two posting paths:

1. **Thread mode** (`gpi_gpu_channel_post_gfd_thread`): Each thread writes 128 bits at a time
   using `st.relaxed.sys.global.b128` PTX stores into the MMIO ring.
2. **TMA mode** (`gpi_gpu_channel_post_gfd_tma`, SM ≥ 9.0): Uses `cp.async.bulk` to copy the
   GFD from shared memory to global MMIO in a single bulk transfer, reducing the number of
   memory transactions. (while available, currently not used, under evaluation to see when to enable it)

The posting path is selected at compile time via `GPI_POST_MODE_THREAD` /
`GPI_POST_MODE_TMA` template parameters.

#### Resource Sharing Modes

Producer-index arbitration uses different atomic primitives depending on sharing mode:

| Mode | Atomic scope | CI shadow scope |
|------|-------------|-----------------|
| `EXCLUSIVE` | Plain store (single owner) | Plain load |
| `CTA` | `cuda::thread_scope_block` fetch_add | `ld.relaxed.cta` |
| `GPU` | `cuda::thread_scope_device` fetch_add | `ld.relaxed.gpu` |

#### Operation Implementations

| GIN API | GPI implementation |
|---------|--------------------|
| `put` / `putValue` | Build a `GPI_GFD_DATA_OP_WRITE` or `GPI_GFD_DATA_OP_WRITE_INLINE` GFD; optionally attach counter/signal fields; post via `gpi_gpu_channel_post_gfd`. |
| `get` | Build a `GPI_GFD_DATA_OP_READ` GFD; post on thread 0 within a `coop.sync()` barrier. |
| `flush` | For each peer, atomically increment a per-peer flush ticket, post a `GPI_GFD_DATA_OP_PE_FLUSH` GFD, and spin-wait on the corresponding flush counter; finish with `cuda::atomic_thread_fence(ord, cuda::thread_scope_system)`. |
| `flushAsync` | Same as flush but records the current counter value in an `ncclGinGpiRequest` and returns immediately without spinning. |
| `wait` | Spin on `req.flushCounterPtr[0]` until the counter exceeds `req.waitValue`. |
| `resetSignal` / `resetCounter` | Zero the corresponding entry in the GPU-mapped signal/counter array. |
| `getSignalPtr` / `getCounterPtr` | Return a typed pointer into the GPU-mapped signal/counter array indexed by signal/counter ID. |


#### Flush Counter Index Computation

The flush counter indices are derived at runtime from the distance between the signal and
counter base pointers:

```cpp
int16_t flush_counter_idx =
    (int16_t)(((uint64_t*)loadConst(&gpi_ctx->gpu_signal_ptr_) -
               (uint64_t*)loadConst(&gpi_ctx->gpu_counter_ptr_)) /
              sizeof(uint64_t)) - ctx.nRanks;
```

This allows the counter and signal arrays to be sized independently by the host without
embedding hard-coded offsets in the device code.

### API Integration and Call Flow

This subsection traces the complete path from NCCL host-side setup through the device-side
dispatch down to the GPI backend function, illustrated with the `put` operation.

#### Phase 1 — Host-side initialisation

The user calls a sequence of host (CPU) NCCL APIs before launching any kernel:

```
ncclCommInitRankConfig(comm, ...)          // create the NCCL communicator
  └─ setLocalGinType()                     // detect GPI/GDAKI/Proxy capability
ncclGinConnectOnce(comm)                   // open NIC connections, allocate QPs / GPI channels
ncclGinDevCommSetup(comm, reqs, &devComm)  // allocate GPU-visible channel memory, fill devComm
ncclGinRegister(comm, ptr, size, ...)      // register user buffer → ncclGinWindow_t handle
```

`ncclGinDevCommSetup` populates `ncclDevComm` with:

| Field | Content |
|-------|---------|
| `devComm.ginHandles[connId]` | GPU-visible pointer to the flat array of `gpi_gpu_channel_t` structs |
| `devComm.ginNetDeviceTypes[connId]` | `NCCL_NET_DEVICE_GIN_GPI` (or GDAKI / Proxy) |

`ncclGinRegister` returns an opaque `ncclGinWindow_t` that encodes the NIC memory handle
index, used later as the `srcWin` / `dstWin` argument to `put`.

`reqs.ginContextCount`, `reqs.ginQueueDepth`, and `reqs.ginSignalCount` control how many
GPI channels are allocated and how large their ring queues and signal/counter arrays are.

#### Phase 2 — Context construction (user kernel setup)

Before launching the kernel the user builds an `ncclGinCtx` from `devComm`:

```cpp
ncclGinCtx_M<-1u> ctx;           // -1u = all backends compiled in (runtime dispatch)
ctx.backend    = (ncclNetDeviceType)devComm.ginNetDeviceTypes[connectionId];
ctx.handle     = devComm.ginHandles[connectionId];
ctx.rank       = myRank;
ctx.nRanks     = nRanks;
ctx.contextId  = contextId;      // index within the connection's channel array
ctx.resourceSharingMode = NCCL_GIN_RESOURCE_SHARING_GPU; // or CTA
ctx.backendMask = -1u;           // encoded in the template parameter
```

`ncclGinCtx_M<beMask>` is a thin subclass of `ncclGinCtx` that carries the backend mask as
a compile-time constant, enabling the compiler to eliminate dead branches in `ncclGinCall`
when only one backend bit is set.

#### Phase 3 — Kernel-side dispatch (`ncclGinCall`)

Inside the device kernel the user calls:

```cpp
ncclGinCall<ncclGinApi_Put>(ctx, coop, peer, hasWins,
    dstWin, dstOff, srcWin, srcOff, bytes,
    signal, signalOp, signalOpArg,
    hasCounter, counterId,
    hasDescriptor, &desc,
    required, given, optFlags);
```

This resolves through two layers:

```
ncclGinCall<ncclGinApi_Put>(ctx, ...)
  └─ ncclGinCallImpl<ncclGinApi_Put>(ctx.backendMask, ctx, ...)
       └─ switch (singleton ? popcount(beMask-1) : ctx.backend)
            case NCCL_NET_DEVICE_GIN_GPI:
              ncclGinApi_Put<NCCL_NET_DEVICE_GIN_GPI>::call(ctx, ...)
```

The `singleton` fast-path: if the backend mask has exactly one bit set (i.e., only GPI is
compiled in or the user specialised the context to a single backend), the switch becomes a
compile-time constant branch and the compiler emits a direct call with no runtime overhead.
Otherwise `ctx.backend` is read at runtime to select the correct specialisation.

#### Phase 4 — GPI backend execution (`ncclGinApi_Put<GPI>::call`)

Inside the GPI specialisation, the operation is executed in four steps:

```
ncclGinApi_Put<NCCL_NET_DEVICE_GIN_GPI>::call(ctx, coop, peer, ...)
  │
  ├─ 1. Recover channel pointer
  │      gpi_gpu_channel_t *ch = gpi_gpu_channel_get_ptr(ctx)
  │         ctx_offset = contextId × (sizeof(gpi_gpu_channel_t) + nRanks×8)
  │         ch = (gpi_gpu_channel_t*)((char*)ctx.handle + ctx_offset)
  │
  ├─ 2. Claim a ring slot  [thread 0 only, inside coop.sync()]
  │      pi = gpi_atomic_add<uint64_t, resource_sharing_mode>(&ch->queue_.pi_, 1)
  │      idx = pi & (ring_size - 1)
  │      // spin-wait if ring is full: pi - ci_shadow > ring_size
  │
  ├─ 3. Build the GFD in registers / shared memory
  │      gpi_gpu_build_data_transfer_gfd(gfd,
  │          GPI_GFD_DATA_OP_WRITE, op_flags,
  │          bytes, peer,
  │          src_handle, srcOff, dst_handle, dstOff,
  │          counterId, signalId, signalValue)
  │      // optional counter / signal fields appended
  │
  └─ 4. Write GFD to MMIO ring (post mode selected at compile time)
         Thread mode:  gpi_gpu_channel_post_gfd_thread<resource_sharing_mode>(ch, gfd, optFlags)
           └─ st.relaxed.sys.global.b128 [ring[idx]], gfd   // 2× 128-bit MMIO stores
         TMA mode (SM≥9.0): gpi_gpu_channel_post_gfd_tma<resource_sharing_mode>(ch, gfd, optFlags)
           └─ cp.async.bulk.global.shared::cta [gmem], [smem], sizeof(gfd)
```

The NIC HW polls the ring, reads the GFD when its ownership flag matches the current
generation, and executes the RDMA write to the remote peer.

#### End-to-end sequence diagram (ping-pong example)

```
Host (CPU)                        GPU kernel (rank 0)               GPU kernel (rank 1)
─────────────────────────────     ─────────────────────────────     ─────────────────────
ncclCommInitRankConfig
ncclGinConnectOnce
ncclGinDevCommSetup ──────────── ctx = {handle, rank, ...}          ctx = {handle, rank, ...}
ncclGinRegister ──────────────── srcWin, dstWin                     srcWin, dstWin
cudaLaunchKernel ────────────────────────────────────────────────────────────────────────►
                                  ncclGinCall<ncclGinApi_Put>       spin on signal ptr
                                    → GFD posted to MMIO ring
                                                                ───► NIC delivers data
                                                                     ncclGinCall<ncclGinApi_GetSignalPtr>
                                                                     signal arrives → ncclGinCall<ncclGinApi_Put>
                                  spin on signal ptr            ◄───  NIC delivers data + signal
cudaStreamSynchronize ◄──────────────────────────────────────────────────────────────────
ncclGinDevCommFree / Deregister
```

### Interface Architecture

The GPI backend implements the following template specializations from `gin_device_common.h`:

```
ncclGinApi_Put          <NCCL_NET_DEVICE_GIN_GPI>
ncclGinApi_PutValue     <NCCL_NET_DEVICE_GIN_GPI>
ncclGinApi_Get          <NCCL_NET_DEVICE_GIN_GPI>
ncclGinApi_Flush        <NCCL_NET_DEVICE_GIN_GPI>
ncclGinApi_FlushAsync   <NCCL_NET_DEVICE_GIN_GPI>
ncclGinApi_Wait         <NCCL_NET_DEVICE_GIN_GPI>
ncclGinApi_ResetSignal  <NCCL_NET_DEVICE_GIN_GPI>
ncclGinApi_ResetCounter <NCCL_NET_DEVICE_GIN_GPI>
ncclGinApi_GetSignalPtr <NCCL_NET_DEVICE_GIN_GPI>
ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_GPI>
```

All are dispatched through the `ncclGinCall<ApiFn>(ctx, ...)` helper which selects the
correct specialization at runtime (or compile time when the backend mask has a single bit set).

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

### Key files

| File | Description |
|------|-------------|
| `src/include/nccl_device/gin/gpi/gin_gpi.h` | All GPI device-side implementations |
| `src/include/nccl_device/gin/gpi/gin_gpi_device_host_common.h` | Shared GPI data structures (`gpi_gpu_channel_t`, GFD layout, enums) |
| `src/include/nccl_device/gin/gin_device_common.h` | GIN API template declarations and `ncclGinCall` dispatcher and GPI backend type |

### Known limitations / Things to consider

- The TMA posting path (`GPI_USE_TMA_`) requires the GFD to be staged in shared memory
  before the bulk copy. Currently the caller allocates a local array on the stack (or reuses
  a descriptor smem slot) and passes it to `gpi_gpu_channel_post_gfd_tma`.
- `ncclGinApi_Get` currently enforces that only thread 0 posts the GFD (inside a
  `coop.sync()` guard). Multi-thread GFD posting for get (analogous to put) is not yet
  supported.

### Commit list or MR

<!-- Add MR links when available -->

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

Validate correctness of all GPI operations across the two resource-sharing modes and across
the two posting modes (thread and TMA) on supported hardware.

### Validation

#### Where to run?

- SM 7.0 / 8.0 system (e.g. Volta/Ampere DGX node with GPI-capable NIC) — thread posting,
  no TMA.
- SM 9.0 system (e.g. Hopper DGX H100 with GPI-capable NIC) — both posting modes.
- Multi-node (≥ 2 nodes) configuration to exercise actual network traffic.

#### What to run?

- Existing GIN Device API apitests and unit test with `NCCL_NET_DEVICE_GIN_GPI` forced as the backend.
- `alltoall` perftest using `gin.put` (baseline) and `gin.get` (new path).
- Test each GIN operation individually: put, putValue, get, flush, flushAsync+wait, fence,
  resetSignal, resetCounter.
- Validate both `NCCL_GIN_RESOURCE_SHARING_GPU` and `NCCL_GIN_RESOURCE_SHARING_CTA` modes.

#### Expected output?

All apitests pass. No data corruption observed. Performance comparable to GDAKI backend for
equivalent operations.



### Performance

#### What is measured?

- latency of putSIgnal (INC) is 50% lower than GDAKI path Bus bandwidth (GB/s)
- NCCL_EP and vLLM shows up to 8% performance compared to GDAKI and NVSHMEM
- zero impact on occupancy for fused kernels

#### Results

<!-- To be filled after benchmarking -->

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Khaled Hamidouche

</details>
