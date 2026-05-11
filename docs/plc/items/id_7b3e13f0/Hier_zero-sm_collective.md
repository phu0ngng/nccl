# Hier zero-sm collective
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

Adds a hierarchical zero-SM collective to NCCL. When triggered by `NCCL_CTA_POLICY_ZERO` on a multi-node communicator that supports RMA proxy and CE, the collective is decomposed into inter-node RMA put-signal-group operations dispatched to the network proxy and intra-node Copy Engine scatters over NVLink.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

https://nvbugspro.nvidia.com/bug/5897251

### User Experience

The feature is transparent to applications. Existing `ncclAllGather` calls automatically take the hierarchical zero-SM path when:
- The communicator spans more than one node (`comm->nNodes > 1`).
- The user-supplied buffers are registered for symmetric / RMA access.
- The configured CTA policy is `NCCL_CTA_POLICY_ZERO`.

When any of these conditions does not hold, the existing collective paths are used unchanged. No new public API is introduced.

### Assumptions, constraints and dependencies

**Path-selection preconditions** (all must hold; otherwise the existing collective paths are used unchanged):

- Multi-node communicator (`comm->nNodes > 1`). Single-node communicators continue to use the existing flat CE-coll path; the dispatch in `ncclLaunchCeColl` checks node count before routing.
- RMA proxy initialized — the inter-node phase relies on the existing zero-SM put/wait infrastructure (GIN / IB).
- CE coll enabled — the intra-node phase reuses NVLink-based CE memcpy through `ce_coll.cc`.
- Send and receive buffers RMA-registered (`-R 2` / symmetric registration). Non-symmetric buffers fall back to existing collective paths.

**Current-scope constraints** (intentional simplifications in this drop, expected to relax in follow-ups):

- Only `ncclAllGather` is wired up. Other collectives (AllToAll, ReduceScatter, …) are out of scope.
- Single hierarchical layout: rail-aligned inter-node puts plus full intra-node LSA scatter. Algorithm/protocol selection for the hierarchical zero-SM family is not yet hooked into the search heuristics.
- Single chunking policy: uniform per-peer chunks capped at `HIER_COLL_MAX_CHUNK_SIZE` (64 MiB), with the last chunk per peer absorbing the remainder. The `ncclHierChunkPlan` API already supports per-peer non-uniform chunking; the producer is intentionally simple here and is the natural extension point.
### Use Cases

- Workloads that overlap collective communication with compute kernels and need to keep all SMs available for the application work.
- Multi-node training / inference where AllGather SM-utilization sensitivity dominates.

### Platform Requirements

- Multi-node setup with InfiniBand (or any RMA proxy-supported network).
- NVLink connectivity between local-node ranks.

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

The hierarchical AllGather (`ncclHierCeAllGather`) issues a single per-rank slice in two stages glued together by stream memops on the user stream. The chunked variant pipelines per-(peer, chunk) inter-node arrival with intra-node CE scatter so the receive side overlaps with later inbound chunks.

Phase ordering follows the principle "after every cross-node sync, issue net ops first; defer intra-node sync to the latest point that still satisfies its precondition (just before the first intra-node write into shared symmetric memory)". The proxy fans out network puts in parallel with the GPU stream's CE work, so issuing intra-node ops first wastes the overlap window.

```
DAG on the user stream:
  Phase 1 RailSync             // cross-node entry barrier (net put + wait)
  Phase 2 PutGroupSubmit       // single memop fires all nRemoteNodes * nChunks puts
  Phase 3 IntraNodeBarrier     // gates LSA peer recvbuf writes; runs while
                               // proxy is in flight
  Phase 4 SelfBcast            // CE scatter of own slice to LSA peers
  Phase 5 for (peer, chunk) in shift order:
            wait for chunk's signal
            CE-scatter the chunk to local peers via LSA
  Phase 6 PutGroupDone         // one memop blocks until all network puts complete
  Phase 7 IntraNodeBarrier     // gates user code reading recvbuf
```

Key building blocks introduced or refactored:

- **LSA-aware CE infrastructure**. `ncclCeInit`, `ncclMemOpSync`, `ncclPrepMCSync`, `ncclPrepUCSync`, and the existing `ncclCeAllGather` operate on `lsaSize` and `myLsaRank` rather than `nRanks` and `comm->rank`. This lets the existing CE code be reused as the intra-node building block of the hierarchical algorithm without disturbing the single-node CE path.
- **RMA proxy descriptor API (4-step protocol)**. `Build → Params → EnqueueDesc → cuStreamBatchMemOp`. One goal of refactoring is to decouple the put trigger and put completion. So in this way, between the network put operation and the put done, we could also enqueue a bunch of NVL transactions.
- **Put-signal-group descriptor (`ncclRmaDescTypePutSignalGroup`)**. Bundles N puts under a single (readySeq, doneSeq) pair. The GPU stream emits exactly one start memop and one (or two for graph mode) done memop regardless of N, instead of O(N).
- **Forward-compatible chunking plan**. CSR-style flat arrays in `ncclHierChunkPlan` (per-peer prefix-sum, per-chunk size and offset). Supports per-peer non-uniform chunking so future strategies (e.g., zero-sm-allgather's two-phase + geometric tail) can be plugged into `ncclHierAllGatherBuildChunk` without touching `ncclHierCeAllGather`.


### Interface Architecture

No new public API. Internally:

- `enqueue.cc` routes `ncclAllGather` to the CE collective task path when the existing CE-collective availability checks succeed, plus the new `ncclHierCeAvailable(comm, ...)` check (multi-node, RMA-ready, LSA-local).
- `ncclLaunchCeColl` dispatches to `ncclHierCeAllGather` when `comm->nNodes > 1`.
- `ncclHierCeAllGather` itself uses:
  - `ncclRailSync` (helper) for the cross-node entry barrier.
  - `ncclRmaProxyPutGroupBuildDesc` / `PutGroupStartParams` / `PutGroupDoneParams` / `EnqueueDesc` for the inter-node put group.
  - `ncclProxyWaitOnePeer` (helper) for per-(peer, chunk) inbound waits.
  - `ncclCeInitBatchOpsParams` / `ncclCeLaunchBatchOps` / `ncclMemOpSync` for intra-node CE scatter and barriers.

Failure-path cleanup is uniform: the canonical caller pattern always calls `DestroyDesc(comm, &desc)` only when `desc != nullptr`; success paths null out the local pointer via the `T**` API.

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

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

- Verify functional correctness of `ncclHierCeAllGather` on multi-node configurations under both non-graph and CUDA-graph capture.
- Verify no regression on the existing single-node CE collective path (LSA-aware refactor of `ncclCeAllGather` should be behavior-preserving).
- Verify no regression on existing RMA put / wait paths after the proxy descriptor API refactor.

### Validation

#### Where to run?

- Multi-node SLURM clusters (e.g., pre-nyx), 2+ nodes, 8 GPUs per node.
- Single-node nodes also covered by the existing CE-coll perf jobs.

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
  - Zhenhao He

</details>
