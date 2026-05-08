# GIN Fence Levels

## Abstract

Introduce fence levels (`None` / `Put` / `Get` / `All`) for GIN
device-side barriers, so users can specify which prior network
operations must be complete after the barrier returns. Extend the
hybrid `Barrier` with a two-pass LSA design so it provides world-scope
fence semantics on railed setups (matching `GinBarrier(World)`). Add a
sentinel context value `NCCL_GIN_CONTEXT_ALL` for users who spread
operations across multiple GIN contexts (multi-NIC) and need a single
barrier call that drains every context.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

- https://nvbugs/6020323 [RFE] Design/implement ncclGinFenceLevel::Release
- https://nvbugs/5895580 [RFE] [Device API] Barrier convenience function

### User Experience

Today's `ncclGinFenceLevel` only has `Relaxed` which provides a
pure-synchronization barrier with no completion guarantee. Users who
need fence semantics must manually call `gin.flush` before the
barrier, and that only covers one context. The original "Release"
attempt (commit `66abcb7b8`) was reverted because, (1) "Release" does
not have a precise meaning in the context of GIN and (2) self-signal
does not provide the guarantees that we need.

After this work:

- `Barrier(team, fence=Put)`: all my prior puts on this context have settled at their destinations.
- `Barrier(team, fence=Get)`: all my prior gets on this context have landed in local memory.
- `Barrier(team, fence=All)`: both.
- `Barrier(team, fence=None)`: pure synchronization (today's `Relaxed`).
- A barrier constructed from `ncclGin(comm, mask, NCCL_GIN_CONTEXT_ALL)` extends the fence to every GIN context the comm has, not just the bound one.
- Convenience wrappers (`ncclBarrierSync`, `ncclGinBarrierSync`) default to `fence=All` and `ord=acq_rel` so users get the strongest guarantee by default.

### Assumptions, constraints and dependencies

- Both shipping GIN backends use per-peer endpoints:
  - **GDAKI**: per-peer QPs (`gdqp[peer]`) plus a self-loopback QP.
  - **Proxy**: per-peer GFD queues.
  Self-signal cannot drain other peers' endpoints. However, the existing `ncclGin::flush()` primitive (which loops every per-peer QP/queue) can be used to conveniently provide the guarantees that we need.
- The hybrid `Barrier` continues to use LSA (intra-node, shared-memory atomics) for the inner team and rail-GIN for the outer team. A second LSA pass propagates rail-GIN completion across rails.

### Use Cases

- **Producer/consumer with GIN**. Producer writes via `gin.put`, then `Barrier(World, fence=Put)`. Consumer reads after barrier exit and sees the data.
- **Cross-rail visibility**. Rank A on rail 0 puts to a rail-0 peer; rank B on rail 1 puts to a rail-1 peer. Both call `Barrier(World, fence=Put)`. After exit, A's node-mate on rail 1 transitively knows A's puts have settled.
- **Multi-context load balancing**. A rank issues puts across multiple GIN contexts for bandwidth. `Barrier(World, fence=All)` constructed from `ncclGin(..., NCCL_GIN_CONTEXT_ALL)` drains every context with one call.
- **Replacement for `GinBarrier(World)`**. The hybrid `Barrier(World, fence=All)` with two-pass LSA matches `GinBarrier(World, fence=All)` semantics at lower cost (fewer signals on a railed setup), making it the recommended path at world scope.

### Platform Requirements

- NCCL v2.30+ with GIN device-API support.
- For the GDAKI backend: ConnectX-class IB NIC + DOCA GPU NetIO.
- For the proxy backend: any platform NCCL supports (CPU-mediated dispatch).
- CUDA toolkit with `cuda::memory_order` device-side API.

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

Three orthogonal changes plus a convenience-API tweak:

**1. Fence levels.** Replace `enum class ncclGinFenceLevel { Relaxed }` with `{ None, Put, Get, All }`. Semantics on the bound GIN context, after the barrier returns:

| Level | Guarantee |
|---|---|
| `None` | Pure synchronization. No drain. |
| `Put`  | All prior puts from this rank have settled at their destinations. |
| `Get`  | All prior gets from this rank have landed in local memory. |
| `All`  | Both `Put` and `Get`. |

Mechanism: reuse the existing `ncclGin::flush` primitive in `ncclGinBarrierSession_internal::syncInternal`:

```
if fence == Put || fence == All:    flush(release)   // before signaling -> peer observes signal meaning my data settled
[per-peer signal/wait loop, unchanged]
if fence == Get || fence == All:    flush(acquire)   // after waiting -> local RDMA-Read completions drained
```

**2. Hybrid `Barrier` two-pass LSA.** In `ncclBarrierSession`, change the topology from `LSA(release) -> rail-GIN(acquire)` to:

```
LSA #1 (release-of-ord)   // intro-node release
rail-GIN (with fence)     // per-rail cross-node sync. fence drains rail context
LSA #2 (acquire-of-ord)   // intra-node acquire. propagates rail-GIN-completion across rails
```

LSA #2 is gated on `fence != None` (the cross-rail completion property is only meaningful when fence semantics are in play). Mirror in the timeout-aware overload.

**3. Cross-context fence.**:

```cpp
#define NCCL_GIN_CONTEXT_ALL  (-1)   // implemented in this PR
```

`NCCL_GIN_CONTEXT_ALL` only expands the *fence* — the barrier's per-peer signal/wait loop still runs on a single context (the fallback context 0 chosen at construction). `put`/`get`/`signal` on a `ncclGin` constructed with `NCCL_GIN_CONTEXT_ALL` operate on context 0 by virtue of the same fallback.

**4. Convenience-API defaults.** Default `fence = ncclGinFenceLevel::All` and `ord = cuda::memory_order_acq_rel` on `ncclBarrierSync` and `ncclGinBarrierSync`. `ncclLsaBarrierSync` does not take a fence argument (LSA has no GIN ops to drain).

### Interface Architecture

User-facing API surface added or changed:

```cpp
// gin_barrier.h
enum class ncclGinFenceLevel { None, Put, Get, All };

// core.h
#define NCCL_GIN_CONTEXT_ALL  (-1)

// sync signatures (existing, fence semantics now real)
void ncclGinBarrierSession::sync(Coop, cuda::memory_order, ncclGinFenceLevel);
void ncclBarrierSession::sync   (Coop, cuda::memory_order, ncclGinFenceLevel);

// convenience wrappers with new defaults
NCCL_DEVICE_INLINE void
ncclBarrierSync(Coop coop, ncclGin gin, uint32_t index,
                cuda::memory_order ord = cuda::memory_order_acq_rel,
                ncclGinFenceLevel  fence = ncclGinFenceLevel::All,
                bool multimem = false);

NCCL_DEVICE_INLINE void
ncclGinBarrierSync(Coop coop, ncclGin gin, ncclTeam team, uint32_t index,
                   cuda::memory_order ord = cuda::memory_order_acq_rel,
                   ncclGinFenceLevel  fence = ncclGinFenceLevel::All);
```

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

_To be populated as MRs are merged_

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

Targeting the v2.30 release cycle. Each fence level must be exercised end-to-end on both shipping backends (GDAKI and Proxy), single-node and multi-node, single-context and multi-context.

### Validation

#### Where to run?

- NCCL test harness, multi-rank single-node and multi-node where available.
- Multi-NIC node for the cross-context test (`ginContextCount >= 2`).
- Both GDAKI (DOCA-capable IB) and Proxy backends.

#### What to run?

| Test file | What it verifies |
|---|---|
| `test/unit/devapi_barrier.cu` | For each fence level: kernel issues `gin.put`s, calls `Barrier(World, fence=X)`, then reads the destination. Confirms `fence=All` actually fences and `fence=None` does not. |
| `test/unit/devapi_mixed_barriers.cu` | Cross-rail visibility: rank on rail 0 puts to a rail-0 peer; node-mate on rail 1 puts to a rail-1 peer; both call `Barrier(World, fence=All)`; cross-rail node-mates must see each others' puts after exit. Validates two-pass LSA. |
| `test/unit/devapi_barrier_with_timeout.cu` | Each fence level still respects `timeoutCycles`. |
| `test/unit/devapi_barrier_lsa.cu` | LSA-only path unchanged. Existing convenience-API test still passes. |
| `test/unit/devapi_barrier_multictx.cu` (NEW) | Multi-context fence: each rank issues puts on context 0 and context 1; `Barrier(World, fence=All)` constructed from `ncclGin(..., NCCL_GIN_CONTEXT_ALL)` drains both. Negative case: same setup with single-context gin (`contextIndex=0`) only drains context 0. |
| Smoke tests | `ncclBarrierSync` / `ncclGinBarrierSync` with default args compile and behave correctly. |

Build:
```bash
make -C test/unit devapi_barrier devapi_mixed_barriers \
                  devapi_barrier_with_timeout devapi_barrier_lsa \
                  devapi_barrier_multictx
```

#### Expected output?

- All tests pass at every fence level on both backends.
- Multi-context test: with `NCCL_GIN_CONTEXT_ALL` gin, every put on every context is visible after barrier exit. Negative case verifies the single-context gin only fences its own context.
- Cross-rail test: every node-mate transitively sees the other rail's puts after exit.
- Smoke tests confirm defaults compile cleanly (no overload-resolution surprises).

### Performance

#### What is measured?

`Barrier(World)` latency on a railed multi-node setup at four points:

1. `fence=None`: baseline (today's behavior).
2. `fence=All`, single-context gin: two-pass LSA + single-context `gin.flush`.
3. `fence=All`, `NCCL_GIN_CONTEXT_ALL` gin: multi-context flush (cost scales linearly in `ginContextCount`).
4. `GinBarrier(World, fence=All)`: full-mesh GIN signaling, for comparison.

Microbenchmark separately:
- Two-pass LSA vs. single-pass LSA cost (extra intra-node round-trip; LSA is shared-memory atomics, expected negligible).
- `ncclGin::flush` cost per peer count (O(nRanks) loop on per-peer endpoints).

Acceptance bounds:
- Single-context `fence=All` within a small constant factor of `GinBarrier(World, fence=All)`, and cheaper on a railed setup.
- Multi-context cost scales linearly in `ginContextCount` (typically 1–4).

#### Results

_To be populated after measurements._

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Ahsan Pervaiz <apervaiz@nvidia.com>

</details>
