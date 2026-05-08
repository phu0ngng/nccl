# Graph Stream Ordering Bypass
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract

NCCL’s **strong stream** abstraction (`ncclStrongStream`; see PLC 17,
[Enqueue Refactor for CUDA Graphs](../17/Enqueue_Refactor_for_CUDA_Graphs.md))
serializes device-side collective kernels so communication queues are used
mutually exclusively and launches follow a deterministic order across
ranks—including when launches are **captured into CUDA graphs**, where the
user’s capture-time stream is not NCCL’s private runtime stream at replay. NCCL
therefore rebuilds stream-like ordering with CUDA external event wait/record
pairs. **Graph ordering** on that path is implemented by default with a private
**captureStream** per distinct CUDA graph ID: those streams host the
external-event nodes that chain NCCL kernels so work from different graphs (and
from interleaved uncaptured NCCL launches) remains correctly ordered. A related
per-device **launch order** strong stream orders kernels across communicators on
the same GPU ([PLC Launch Order Implicit](../id_ea571e4f/Launch_Order_Implicit.md)).

Each time a new CUDA graph ID is seen during capture, NCCL’s strong stream
abstraction allocates a captureStream via `cudaStreamCreate`.
For workloads that re-capture frequently (many distinct graph IDs, or graphs with
many sub-graphs), this stream creation and subsequent teardown via `cudaStreamDestroy`
accumulates measurable overhead on the capture-time critical path.

This feature adds **`NCCL_GRAPH_STREAM_ORDERING=0`** (environment variable) and
**`graphStreamOrdering`** (`ncclConfig_t` communicator config field) to let the caller opt out of
NCCL’s internal kernel-ordering captureStream. When opted out, NCCL runs its CUDA
kernels directly on the user’s captured stream (`graph.origin`) instead of on a
private captureStream, eliminating the stream create/destroy overhead for every
new graph ID. Cross-graph-launch ordering of a communicator's kernels is still
provided through the device strong stream's `serialEvent`, recorded and waited on
as explicit graph nodes on `graph.origin`. The user accepts responsibility for
ensuring NCCL kernels are launched in a correct order for their application (see
[Application responsibilities](#application-responsibilities)). The proxy (host)
strong streams are unaffected and continue to provide the same ordering
guarantees as before.

The primary use case is stream ordered NVLink collectives, where
there is no network proxy involvement and where the application already ensures
correct ordering of NCCL communication kernels without relying on NCCL’s internal
capture-time serialization. **`ncclConfig_t.graphStreamOrdering`** exists because some runtimes (**JAX/XLA** and similar) configure NCCL through
`ncclCommInitRankConfig` / communicator config but **cannot** reliably set
process-wide environment variables, so they need a communicator-scoped way to
opt in.

**Mixed communicators on the same GPU:** Opt-out is per communicator and
breaks cross-communicator serialization. A `graphStreamOrdering = 0`
communicator leaves the shared ordering path; its kernels may interleave
with those of any peer communicator on the same GPU, including ones that
keep the default. Safe only when the application guarantees ordering across
all NCCL communicators on the GPU (see [Application responsibilities](#application-responsibilities)).

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

https://nvbugspro.nvidia.com/bug/5633215

### User Experience

NCCL’s graph capture path creates one internal CUDA stream (captureStream) per
communicator per distinct CUDA graph ID encountered during `ncclLaunchPrepare`.
These streams are how the strong stream abstraction enforces mutually exclusive,
deterministically ordered kernel execution across graphs and uncaptured work
(see PLC 17).

When a framework captures NCCL operations into a large graph that contains many
nested (child) subgraphs — each subgraph typically needing its own stream
identity for ordering — the `cudaStreamCreate` / `cudaStreamDestroy` calls
associated with strong stream acquisition and release add up on the capture-time
path. That cost is especially visible when such a graph has many sub-graphs.

This feature exposes a knob so that applications which can guarantee correct
ordering of NCCL communication kernels without NCCL’s capture-time serialization
of those kernels across graphs and communicators on the GPU (typically
single-communicator NVLink-only workloads) can eliminate this overhead entirely,
trading NCCL’s internal serialization guarantee for application-side responsibility
for that ordering.

Both an environment variable and a per-communicator config field are provided:

- **`NCCL_GRAPH_STREAM_ORDERING=0`** (env var, default `1`) — applies globally to all
  communicators in the process.

- **`ncclConfig_t.graphStreamOrdering = 0`** — per-communicator override. This path
  exists because some runtimes (**JAX/XLA** and similar) configure NCCL through
  `ncclCommInitRankConfig` / communicator config but **cannot** reliably set
  process-wide environment variables, so they need a communicator-scoped way to
  opt in.

  **Mixed communicators on the same GPU:** Opt-out is per communicator and
  breaks cross-communicator serialization. A `graphStreamOrdering = 0`
  communicator leaves the shared ordering path; its kernels may interleave
  with those of any peer communicator on the same GPU, including ones that
  keep the default. Safe only when the application guarantees ordering across
  all NCCL communicators on the GPU.

### Application responsibilities

With `NCCL_GRAPH_STREAM_ORDERING=0` or `ncclConfig_t.graphStreamOrdering = 0`, NCCL stops enforcing device-side serialization of **communication** kernels on the graph-capture path. The **application** must then guarantee:

1. **Serialization on the GPU.** NCCL operations must not overlap: at most one NCCL operation may execute on a given GPU at a time.

2. **Scope.** The rule applies across **communicators**, across **different captured graphs**, and between **captured and uncaptured** NCCL work. The ordering must hold **at replay / execution**, not only as expressed at capture time.

3. **How to satisfy it.** The simplest approach is to enqueue **all** NCCL operations on the **same CUDA stream**. Equivalent serialization can be achieved with **device-wide synchronization** and/or **CUDA event** dependencies between streams.

4. Requesting `NCCL_GRAPH_MIXING_SUPPORT=1` or equivalent `graphUsageMode=2` in combination with `NCCL_GRAPH_STREAM_ORDERING=0` or equivalent is not supported.

**Network (proxy) transports:** NCCL continues to provide its normal **host-side** ordering guarantees for those transports regardless of this knob.

### Assumptions, constraints and dependencies

- **Capture scope.** The setting applies only during CUDA graph capture and has no
  effect outside capture. It disables NCCL’s internal **serialization of
  communication kernels** on the capture path.

- **Per-communicator override.** The same bypass can be selected with
  `ncclConfig_t.graphStreamOrdering`; when `0` or `1`, it overrides
  `NCCL_GRAPH_STREAM_ORDERING` for that communicator. Mixed values on one GPU are
  covered in the User Experience and Application responsibilities sections.

- **Network (proxy) transports.** They are unaffected: NCCL continues to provide
  normal host-side ordering guarantees for those transports regardless of
  `NCCL_GRAPH_STREAM_ORDERING` / `graphStreamOrdering`.

#### Implementation notes (PLC-only detail)

- **Device strong streams.** The bypass changes how captured **communication**
  kernels are ordered on the device strong-stream path (`deviceStream`,
  `launchOrder` in `src/enqueue.cc`).

- **`launchOrder` and implicit launch order.** With the bypass active during
  capture, `ncclLaunchPrepare` / `ncclLaunchFinish` can alias `launchOrder` to
  `graph.origin` instead of the shared per-context strong stream, which breaks
  cross-communicator kernel ordering when multiple communicators capture into the
  same or overlapping graphs on one GPU. **`NCCL_LAUNCH_ORDER_IMPLICIT=1` does not
  restore NCCL’s cross-communicator serialization in that situation.**

- **CUDA.** CUDA 11.3 or later (required for `cudaEventWaitExternal` / `cudaEventRecordWithFlags`
  used by the underlying strong stream infrastructure).

### Use cases

- **JAX / XLA:** XLA **encapsulates each NCCL collective in its own nested CUDA subgraph**,
  which creates many distinct graph IDs during capture and amplifies
  `cudaStreamCreate` / `cudaStreamDestroy` overhead on the capture-time path
  (including when graphs are re-captured often). XLA relies on
  `ncclConfig_t.graphStreamOrdering` rather than the env var; apply the mixed-communicator
  caveat when any other NCCL communicator can run concurrently on the same GPU.

- **Single-communicator NVLink-only graph workloads** where the user can assert
  that at most one NCCL graph is live at a time (or ordering is otherwise guaranteed).

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed design

#### Bypass path: run kernels on `graph.origin`

When `GRAPH_STREAM_ORDERING=0` is active during CUDA graph capture, both `ncclLaunchPrepare`
and `ncclLaunchFinish` skip all strong stream acquisition for the kernel-side streams
described below. With `GRAPH_STREAM_ORDERING=0` (or `graphStreamOrdering=0`), **graph
mixing must be disabled** (`graphUsageMode` `0` or `1`); see
[Mixing mode and graph stream ordering](#mixing-mode-and-graph-stream-ordering).
Non-mixing graph usage (`graphUsageMode` `0` or `1`) is where the stream-create bypass
below applies. If this is not done, NCCL issues a warning at initialization time.

**`ncclLaunchPrepare`** (stream wiring before kernel launch):

- `deviceStream` is set to `launchStream` (`graph.origin`) directly, skipping
  `ncclStrongStreamAcquire` for `comm->sharedRes->deviceStream`. No captureStream
  is created or looked up.
- An ExternalWait node is added on `graph.origin` for
  `comm->sharedRes->deviceStream.serialEvent` via
  `cudaStreamWaitEvent(launchStream, serialEvent, cudaEventWaitExternal)`. The
  wait is unconditional so graph structure is identical across captures (required
  for `cudaGraphExecUpdate`). On the first capture, `serialEvent` is bootstrapped
  by recording it on `liveStream` so the wait fires immediately.
- The `launchStream → deviceStream` wait node is suppressed (same stream, would
  be a graph self-loop).
- When `LAUNCH_ORDER_IMPLICIT` is enabled: `launchOrder` is set to
  `planner->capturingGraph.origin` directly, skipping `ncclStrongStreamAcquire` for
  `comm->context->launchOrder`. The `launchStream → launchOrder` wait node is
  similarly suppressed. With the default `LAUNCH_ORDER_IMPLICIT=0`, this block does
  not execute at all.

**`ncclLaunchFinish`** (stream wiring after kernel launch):

- The `ncclStrongStreamAcquiredWorkStream` + `ncclStreamAdvanceToEvent` fast-forward
  path for `deviceStream` is skipped entirely.
- When `LAUNCH_ORDER_IMPLICIT` is enabled: `ncclStrongStreamAcquiredWorkStream` for
  `launchOrder` is skipped; `launchOrder` is again `graph.origin`. With the default
  `LAUNCH_ORDER_IMPLICIT=0`, this block does not execute.
- `ncclStrongStreamRelease` for `deviceStream` is skipped (nothing was acquired in
  non-mixing mode); in its place `ncclCudaGraphRecordEvent` records `serialEvent`
  on `graph.origin` as an explicit `cudaGraphAddEventRecordNode` whose dependencies
  are the origin stream's current capture frontier. The frontier itself is **not**
  advanced (no `cudaStreamUpdateCaptureDependencies` on the origin stream),
  preserving the topology that the caller will continue to extend.
  `ncclStrongStreamRelease` for `launchOrder` is likewise skipped when
  `LAUNCH_ORDER_IMPLICIT` is active with `useOriginStream`.

The Wait/Record pair on `serialEvent` chains successive captured launches of this
communicator (and bridges captured launches with prior uncaptured work via the
first-capture bootstrap), without creating a captureStream and without modifying
the origin stream's capture frontier. `cudaStreamCreate` is never called for a
new graph ID.

#### Mixing mode and graph stream ordering

**`graphUsageMode==2` (graph mixing) together with `NCCL_GRAPH_STREAM_ORDERING=0` or
`ncclConfig_t.graphStreamOrdering = 0` is not supported.** When stream ordering is
disabled for a communicator, **graph mixing must be disabled** for that communicator:
use `graphUsageMode` `0` or `1` (and avoid `NCCL_GRAPH_MIXING_SUPPORT=1` overriding
config to mixing), or do not opt out of stream ordering for that communicator.

**Logging / enforcement:** warn about the combination at communicator init.
After that, proceed as if `NCCL_GRAPH_STREAM_ORDERING=1`.

#### Interface design

**Environment variable:**

```
NCCL_GRAPH_STREAM_ORDERING=0   # disable (bypass captureStream for kernel launches)
NCCL_GRAPH_STREAM_ORDERING=1   # enable (default, existing behavior)
```

Implemented via `NCCL_PARAM(GraphStreamOrdering, "GRAPH_STREAM_ORDERING", 1)` in
`src/enqueue.cc`. Consulted in both `ncclLaunchPrepare` and `ncclLaunchFinish`.

**Communicator config field:**

```c
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.graphStreamOrdering = 0;  // 0=bypass, 1=normal (default: NCCL_CONFIG_UNDEF_INT -> env var)
ncclCommInitRankConfig(&comm, nranks, id, rank, &config);
```

### Interface architecture

The bypass lives in `src/enqueue.cc` (`ncclLaunchPrepare` and `ncclLaunchFinish`).
It changes which strong-stream calls are made (`ncclStrongStreamAcquire`,
`ncclStrongStreamAcquiredWorkStream`, `ncclStrongStreamRelease`,
`ncclStreamWaitStream`) and adds a `serialEvent` Wait/Record pair on `graph.origin`
to provide cross-graph ordering. `src/misc/strongstream.cc` exposes
`ncclCudaGraphRecordEvent`, which records an event on a captured stream as an
explicit graph node using the stream's current capture frontier as dependencies
without advancing that frontier; the existing event-record-node logic in
`ncclStrongStreamRelease` is factored into a shared `recordEventOnStream` helper
that takes an `updateFrontier` flag (true for the captureStream path, false for
the origin-stream path).

### System KPIs & metrics

- Metric: number of `cudaStreamCreate` / `cudaStreamDestroy` calls per
  capture (should drop to zero for pure-capture non-mixing mode).
- Regression check: collective correctness and bandwidth on NVLink-only intra-node
  benchmarks (all-reduce, all-gather) with and without the flag.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Coding</h2></summary>
<!-- ============================================================================================-->

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and timeline

Validate correctness with graph capture first, then quantify capture-time savings
from skipping per–graph-ID `captureStream` creation. **nccl-tests** (`all_reduce_perf -G`)
is the primary vehicle; A/B runs toggle **`NCCL_GRAPH_STREAM_ORDERING`** via the
environment variable.

### Validation

#### Where to run?

Use an **NVLink Domain** for the main bypass story. Multi-GPU nodes that also run **IB** are still useful for regression
coverage, but IB collectives do not use the NVLink-only stream-create path this
feature targets.

#### What to run?

1. **nccl-tests (`-G`) + `NCCL_GRAPH_STREAM_ORDERING`:** Run collectives with **`-G`**
   (CUDA graph capture) and toggle `NCCL_GRAPH_STREAM_ORDERING=0` vs the default for:
   - **Correctness:** all ranks capture and replay without errors or hangs, with
     communicators in **non-mixing** graph usage (`graphUsageMode` `0` or `1`) when
     ordering is `0`, per [Application responsibilities](#application-responsibilities).
   - **Capture performance:** same workload with high distinct graph-ID counts;
     compare CUDA API call counts with `NCCL_GRAPH_STREAM_ORDERING=0` vs `1`.
   **Mixing** (`graphUsageMode=2`, captured + uncaptured interleaving) remains a
   **default-ordering-only** (`NCCL_GRAPH_STREAM_ORDERING=1`) scenario; NCCL warns
   and falls back to ordering `1` when the unsupported combination is requested.

2. **Stress:** Many capture/destroy cycles with a large set of distinct graph IDs
   (ordering `0`, non-mixing) to confirm no stream-handle leak and stable correctness.

#### Expected output?

- No errors or hangs in correctness runs for both flag values where supported.
- With ordering `0` and non-mixing graphs: fewer `cudaStreamCreate` /
  `cudaStreamDestroy` calls during capture vs ordering `1`, in line with one saved
  pair per distinct graph ID per communicator.
- Replay (non-capture) bandwidth and latency unchanged vs baseline.

### Performance

#### What is measured?

- **CUDA API call counts** (via Nsight Systems `--cuda-api-trace`): number of
  `cudaStreamCreateWithFlags` and `cudaStreamDestroy` calls during a graph capture
  workload. Target: zero with `NCCL_GRAPH_STREAM_ORDERING=0` in non-mixing mode.
- **Distinct CUDA stream count** visible in the Nsight Systems timeline: a proxy for
  memory/descriptor overhead created by captureStream allocations.
- **Graph capture wall time**: total time spent in the capture phase (before any replay).
  Measured over N iterations of capture+destroy to amortize noise.
- **Collective bandwidth and latency at replay**: all-reduce or all-gather bandwidth
  measured with `nccl-tests -G` across replay iterations. Should be identical to the
  baseline (ordering bypass does not affect the replay path).

#### Results

PoC results from XLA use-case (counts from CUDA API profiling; workload-dependent):

| API call | `ORDER=0` | `ORDER=1` |
|:---------|----------:|----------:|
| `cudaStreamCreateWithFlags` | 0 | 1740 |
| `cudaStreamDestroy` | 0 | 1740 |
| `cudaStreamIsCapturing` | 1740 | 1740 |
| `cudaStreamUpdateCaptureDependencies` | 3480 | 5220 |
| Distinct CUDA streams in profile | 156 | 6536 |

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Ludwig Schneider (lschneider@nvidia.com)

</details>
