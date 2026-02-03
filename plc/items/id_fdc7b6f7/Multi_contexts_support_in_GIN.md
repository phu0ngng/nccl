# Multi contexts support in GIN
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

NCCL Device API is currently limited to 4 GIN contexts, which correspond to
network QPs.  This limit is motivated by an existing NCCL assumption of at most
4 local NICs being available per rank.

However, having more QPs than NICs (and also more than 4) is desirable for
performance reasons.  The DeepEP kernel can use up to 32 for best performance; the
DeepSeek v2 kernel is expected to need many more than that.

A workaround currently being employed by DeepEP is to create redundant NCCL
communicators.  This multiplies the number of available GIN contexts, since GIN
state is local to each communicator.  However, this method is clunky and
inefficient (both in terms of initialization time and memory overhead).

### NVbugs / Jira Tickets

https://nvbugspro.nvidia.com/bug/5502732

### User Experience

A user should be able to specify a larger number of GIN contexts via the
`ginContextCount` field of the `ncclDevCommRequirements` structure, when
creating a device communicator using `ncclDevCommCreate`.  While the field
already exists, it is currently ignored.

### Assumptions, constraints and dependencies

Our NICs can support virtually unlimited number of QPs (certainly many
thousands) so we should support thousands of contexts.

GIN contexts are passed to the device kernels via the `ncclDevComm` structure
initialized by `ncclDevCommCreate`.  As that structure is normally passed to
the kernel as one of the function arguments, it should be kept compact, so
(over-)allocating a massive static array for contexts "just in case" would be
undesirable.

### Use Cases

Custom device kernels using the Device API GIN support to communicate between
nodes.  In particular, both DeepEP and DeepSeek requested the lifting of the
limit.

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

The new design effectively renames the 4 existing contexts to _connections_ and
creates a user-configurable number of contexts underneath the connections.
This has the advantage of keeping the `ncclDevComm` structure compact.
- The number of connections is chosen automatically by NCCL and is normally
  equal to the number of local NICs (actually, min of that across all the
  ranks).
- Users can override it via `NCCL_GIN_NCONNECTIONS`.  It can be set to any
  value, but NCCL will create at most 4 and if needed will round-robin physical
  NICs over the available connections.
- `ncclDevCommRequirements.ginContextCount` specifies the _total_ number of
  contexts.  They are assigned to connections in a round-robin fashion
  (`connectionId = contextId % NCCL_GIN_NCONNECTIONS`).
- GDAKI creates one QP per peer for each context.  Different contexts use
  different QP pools.
- Not really a new thing but worth mentioning:
  Per GIN semantics, incrementing one signal via multiple contexts is
  unsupported.  _Contexts can map to different NICs, which would result in
  races when conducting RMW operations on signals_.

The NET API (specifically the GIN part of it) needs to be extended:
- `connect` gains an extra `nConnections` argument
- `createContext` gains an extra `nContexts` argument
- `iput` and `iputSignal` gain an extra `connectionId` argument

An alternative approach to the NET API bump that is being considered is to
decouple GIN from NET into its own plugin so that they can evolve separately.

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

### Known limitations / Things to consider

- Renaming contexts to communicators introduces a breaking change to the
  existing API: the `ginContextCount` field in `ncclDevComm` gets renamed to
  `ginConnectionCount`.  The field has been documented, because without it how
  would the kernel code know how many contexts are available?  Still, none of
  NCCL's device kernels currently use it.  **Update**: this is actually easily
  solvable.  Nothing prevents us from having _both_ `ginContextCount` and
  `ginConnectionCount` in `ncclDevComm`.

- Extensions to the NET API require a version bump of the API, which would be
  unusual for a point release (although we do provide full backward
  compatibility).

- GinBarrier implementation assumes a fixed number of signals per barrier (4)
  and probably needs to be updated.  **Update**: fixed by John in the meantime.

- The number of contexts is tracked as part of the `ncclGinState` structure,
  which is part of `ncclComm`, not `ncclDevComm`.  The requested
  number of communicators is respected only during the first invocation of
  `ncclDevCommCreate` (when GIN is being initialized); subsequent invocations
  for a given communicator ignore it, which violates user expectations.  In
  particular, if the user creates a symmetric window before creating a device
  communicator, the requested number of contexts will be ignored, because
  creating a symmetric window creates an internal device communicator as well
  (for use by the built-in symmetric kernels), so the user invocation of
  `ncclDevCommCreate` ends up not being the first.  **Update**: we can solve
  this for now via an environment variable that requests a minimum number of
  contexts to create (`NCCL_GIN_MIN_NCONTEXTS`? It should be set to the max of
  what the user expects to ever need in all their devComms)

- CE support relies on symmetric memory support, as does GIN support.  We should verify that it's
  not negatively affected by these changes.

### Commit list or MR

https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/1779

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation

- Confirm that symmetric and device API continue to work.
- Confirm that CE continues to work.
- Confirm that the expected number of contexts is passed to the custom kernels
  and that all those contexts are operational.
- Confirm that the scenario of creating a symmetric window first, then creating
  a devComm with a large number of contexts, works as expected for different
  values of the environment variable.

#### Where to run?

The usual platforms.  We should ensure that at least one system has 2 or more
NICs per rank (Bia, for instance).

#### What to run?

The usual.

We'll need to add a multi-context variant of a2a custom kernel.  Shane Snyder
has an implementation; Theo has some examples as well.

To confirm that the expected number of contexts is being passed, we may just
add debug output to `ncclDevCommCreate` that prints that property of the newly
created device communicator; then we could use the same test as above, just
with `NCCL_DEBUG=INFO`.

The perf tests meet the requirement of registering a symmetric window before
creating a devComm so they should be suitable to confirm the last validation
scenario as well.

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

We should measure the performance with different numbers of contexts to see if
we can observe an effect.

#### What is measured?

#### Results


</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Pak Markthub

</details>
