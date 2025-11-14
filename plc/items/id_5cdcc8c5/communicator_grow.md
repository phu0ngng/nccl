# Communicator Grow

## Abstract

The Communicator Grow feature allows NCCL to dynamically expand existing communicators by adding new ranks and creating a new communicator that spans all devices, enabling collectives across the expanded group (without restarting the app). This feature enables elastic scaling of distributed workloads by seamlessly integrating new compute resources into an existing communication group. The implementation uses a unified API where all ranks (existing and new) call `ncclCommGrow()` with different parameters to create a new expanded communicator. The feature introduces a new per-communicator unique ID API (`ncclCommGetUniqueId`) and leverages existing infrastructure for maximum code reuse. This capability is essential for elastic training, dynamic resource scaling, and adaptive distributed computing scenarios.

<details>
<summary><h2>Motivation and requirements</h2></summary>

### NVbugs / Jira Tickets

- NVBug: [New ncclCommGrow API](https://nvbugspro.nvidia.com/bug/5288166)

### Feature context and focus

In modern distributed computing environments, workloads often need to adapt to changing resource availability. Traditional static communicator allocation limits the ability to scale applications dynamically. The Grow feature addresses these needs by:

- Enabling elastic training that can scale up when additional resources become available
- Supporting dynamic resource allocation in cloud and cluster environments
- Allowing applications to add compute capacity without restarting entire jobs
- Providing seamless integration of new ranks into existing communication patterns
- Maintaining optimal performance in expanded communicators

This feature significantly enhances NCCL's flexibility for modern distributed computing scenarios where resource availability changes during execution.

### Efficiency Highlights

- Only NEW ranks receive the grow ID (minimal OOB). Existing ranks do not receive it; existing root passes `&uniqueId`, non-root existing ranks pass `NULL`.
- Only NEW ranks connect to the root for coordination; existing ranks re-establish connections using existing connections without connecting to root. On the existing side, only boundary ranks interact with the root during grow.
- `ncclCommGetUniqueId` encapsulates coordination details per communicator so applications avoid broadcasting to all ranks or directly addressing boundary ranks (reduces OOB traffic and avoids exposing internals).

### User Experience

To use the Communicator Grow feature, follow these steps:

1. **Select root rank**: Choose a root rank from the existing communicator to coordinate the grow operation.

2. **Generate unique ID**: The root rank calls `ncclCommGetUniqueId()` to create a per-communicator coordination identifier (reusable for multiple grows).

3. **Distribute ID to new ranks only**: Send the unique ID to NEW ranks only using application-level communication (MPI, sockets, etc.). Existing ranks don't need the ID since they already have the communicator.

4. **All ranks call ncclCommGrow**: Both existing and new ranks call the unified `ncclCommGrow()` function to create a new expanded communicator.

```c
// === ROOT RANK (EXISTING) ===
ncclUniqueId uniqueId;
if (myRank == rootRank) {
    NCCLCHECK(ncclCommGetUniqueId(comm, &uniqueId));
    // Send uniqueId to new ranks only (OOB)
}

// === EXISTING RANKS ===
ncclComm_t newComm;
const ncclUniqueId* uidPtr = (myRank == rootRank) ? &uniqueId : NULL;
NCCLCHECK(ncclCommGrow(comm, /*nRanks=*/totalRanks, uidPtr, -1, &newComm, NULL));

// === NEW RANKS ===
ncclComm_t newComm;
ncclUniqueId uniqueId;
// Receive uniqueId from root (OOB)
int myNewRank = originalNRanks + myIndexAmongNewRanks;
NCCLCHECK(ncclCommGrow(NULL, /*nRanks=*/totalRanks, &uniqueId, myNewRank, &newComm, NULL));
```

5. **Use expanded communicator**: After grow completion, all ranks can use collective operations on the new expanded communicator.

6. **Clean up**: Destroy the original communicator (existing ranks only) and use the new expanded communicator.

```c
// Existing ranks can destroy original communicator
if (originalComm != NULL) {
    NCCLCHECK(ncclCommDestroy(originalComm));
}

// Verify expansion
int newNRanks;
NCCLCHECK(ncclCommCount(newComm, &newNRanks));
printf("New communicator has %d ranks\n", newNRanks);
```

### Example

Grow 2→4 ranks (role-specific snippets):

```c
// Existing root
ncclUniqueId growId;
NCCLCHECK(ncclCommGetUniqueId(comm, &growId));
// Send growId to new ranks only (OOB)
ncclComm_t newComm;
NCCLCHECK(ncclCommGrow(comm, /*nRanks=*/4, &growId, /*rank=*/-1, &newComm, /*config=*/NULL));
```

```c
// Existing non-root
ncclComm_t newComm;
NCCLCHECK(ncclCommGrow(comm, /*nRanks=*/4, /*uniqueId=*/NULL, /*rank=*/-1, &newComm, /*config=*/NULL));
```

```c
// New rank
ncclUniqueId growId;
// Receive growId from root (OOB)
int myNewRank = /* assigned rank: originalNRanks + newIndex */;
ncclComm_t comm = NULL;
NCCLCHECK(ncclCommGrow(comm, /*nRanks=*/4, &growId, myNewRank, &comm, /*config=*/NULL));
```

### Out-of-band (OOB) distribution examples for grow

Ways to send the grow ID to new ranks only:

1) MPI dynamic processes (intercommunicator): spawn or connect, then `MPI_Bcast` over the intercommunicator to the new ranks.
2) TCP sockets: root listens and sends the ID to each new rank client.
3) Shared file or key-value store: root writes the ID; new ranks read it.

### Assumptions, constraints and dependencies

- The Grow operation requires NCCL version 2.29 and above
- **Unified API**: All ranks (existing and new) call `ncclCommGrow()` with different parameters
- **New communicator creation**: Creates new expanded communicator
- **Existing ranks**: Pass their existing comm; rank=-1; get new expanded comm
- **New ranks**: Pass NULL comm, received uniqueId, their assigned rank, get new comm
- **Root coordination**: Only root calls `ncclCommGetUniqueId()`, sends uniqueId to NEW ranks only
- **Root identification**: Root existing rank passes &uniqueId; non-root existing ranks pass NULL
- **Per-communicator unique ID**: Uses `ncclCommGetUniqueId()` API for per-communicator coordination
- **UID usage constraints**:
  - Each UID can only be used once - cannot reuse a UID after it has been consumed by a grow operation
  - Cannot generate a second UID via `ncclCommGetUniqueId()` while a previous UID is still unconsumed
  - Must wait for the grow operation to complete before calling `ncclCommGetUniqueId()` again
- Rank numbering: new ranks are assigned sequential numbers after existing ranks
- OOB communication is required to distribute the `uniqueId` to NEW ranks only
- The feature works with NCCL's group API for synchronized operations
- **Resource coordination**: The grow operation coordinates bootstrap and transport setup across all ranks
- **Topology integration**: New ranks are integrated into existing communication topologies
- **Performance**: Post-grow communication performance should match fresh communicator creation
- **No artificial limits**: Removed maximum rank limits for unlimited scalability

### Use Cases

1. **Elastic Training**: Scale up training workloads when additional GPUs become available.

```c
// During training, additional resources become available
if (additionalResourcesAvailable()) {
  ncclUniqueId uniqueId;
  if (myRank == rootRank) {
    NCCLCHECK(ncclCommGetUniqueId(trainingComm, &uniqueId));
    // Send uniqueId to new ranks only (OOB)
  }

  ncclComm_t newTrainingComm;
  // newTotalRanks = currentRanks + newWorkerCount
  NCCLCHECK(ncclCommGrow(trainingComm, /*nRanks=*/newTotalRanks, &uniqueId, -1, &newTrainingComm, NULL));

  // Destroy old communicator and use new one
  NCCLCHECK(ncclCommDestroy(trainingComm));
  trainingComm = newTrainingComm;

  // Continue training
  continueTrainingWithExpandedComm(trainingComm);
}
```

2. **Dynamic Resource Scaling**: Add compute capacity to running applications based on workload demands.

```c
// Monitor workload and scale up when needed
if (workloadRequiresMoreResources()) {
  int additionalRanks = calculateRequiredRanks();
  ncclUniqueId uniqueId;

  if (myRank == rootRank) {
    NCCLCHECK(ncclCommGetUniqueId(computeComm, &uniqueId));
    requestAdditionalResources(additionalRanks, &uniqueId);
    // Send uniqueId to new ranks only (OOB)
  }

  ncclComm_t newComputeComm;
  // newTotalRanks = currentRanks + additionalRanks
  NCCLCHECK(ncclCommGrow(computeComm, /*nRanks=*/newTotalRanks, &uniqueId, -1, &newComputeComm, NULL));

  // Switch to expanded communicator
  NCCLCHECK(ncclCommDestroy(computeComm));
  computeComm = newComputeComm;

  redistributeWorkload(computeComm);
}
```

3. **Cloud Auto-scaling**: Automatically add instances when demand increases.

4. **Fault Recovery with Additional Resources**: Recover failed resources and add new resources for improved resilience.

### Platform Requirements

- CUDA-capable GPUs
- NCCL version 2.29 or higher
- Multi-process execution environment (MPI, multi-node, etc.)
- Out-of-band communication capability for ID distribution
- Sufficient network bandwidth for expanded topology

</details>

<details>
<summary><h2>Design</h2></summary>

### Proposed Design

#### API Extension

The Grow feature introduces two key API functions with a unified design:

```c
// Generate unique ID for grow coordination (called by root rank only)
ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId* uniqueId);

// Unified grow function for all ranks (existing and new)
ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, ncclUniqueId uniqueId, int rank, ncclComm_t* newcomm, ncclConfig_t* config);
```

**ncclCommGetUniqueId Parameters:**
- `comm`: Existing communicator to generate unique ID from
- `uniqueId`: Output parameter for the unique coordination ID

**ncclCommGrow Parameters (Unified API):**
- `comm`: Input communicator (existing: existing comm, new: NULL)
- `nRanks`: Total number of ranks in the expanded communicator
- `uniqueId`: Pointer to unique ID (existing root: &uniqueId, existing non-root: NULL, new: &uniqueId)
- `rank`: Rank identifier (existing: -1, new: assigned rank number)
- `newcomm`: Output parameter for new expanded communicator
- `config`: Optional configuration for expanded communicator (NULL inherits from original)

**Parameter Usage by Rank Type:**
- **Existing root**: `ncclCommGrow(existingComm, nRanks, &uniqueId, -1, &newComm, config)`
- **Existing non-root**: `ncclCommGrow(existingComm, nRanks, NULL, -1, &newComm, config)`
- **New ranks**: `ncclCommGrow(NULL, nRanks, &uniqueId, myNewRank, &newComm, config)`

#### New Communicator Creation Strategy

Using the same code paths for reuse, Grow creates a new expanded communicator:

1. **New Communicator Creation**: Creates expanded communicator with full topology
2. **Complete Initialization**: Performs all bootstrap, transport, and connection setup
3. **Return New Communicator**: Returns new expanded communicator via newcomm parameter
4. **Manual Cleanup**: Application destroys original communicator and uses new one

This approach follows established code reuse patterns for consistency and resource management.

#### Code Reuse Architecture

The implementation maximizes reuse of existing NCCL infrastructure:

**Bootstrap Reuse:**
- OPTIMIZATION: Only new ranks connect to root during coordination; all ranks participate in creating the new communicator
- Reuses existing bootstrap patterns (see MR)
- Uses existing proxy setup and UDS handling patterns

**Transport Reuse:**
- Uses the same `initTransportsRank()` call
- Reuses topology computation, graph generation, connection setup
- Follows same AllGather patterns for peer info exchange

**Resource Management:**
- Reuses parent-rank mapping logic
- Follows identical cleanup and error handling paths
- Uses the same async job infrastructure (see MR)

#### Internal Architecture

**Coordination Flow:**
1. **Root Selection**: Application chooses root from existing communicator
2. **ID Generation**: `ncclCommGetUniqueId()` creates coordination identifier
3. **OOB Distribution**: Application distributes ID to new ranks
4. **Parallel Initialization**: All ranks call respective functions simultaneously
5. **Bootstrap Coordination**: Unified AllGather coordinates all ranks
6. **Transport Setup**: Standard topology and connection initialization
7. **Post-Grow Usage**: All ranks use the expanded communicator

**New Rank Integration:**
- New ranks use unified `ncclCommGrow()` API with received uniqueId
- Bootstrap automatically detects grow operation from ID pattern
- New ranks participate in same AllGather coordination as existing ranks
- Seamless integration into expanded topology

**Rank Assignment:**
- Existing ranks maintain their original rank numbers
- New ranks assigned sequential numbers: `originalNRanks + newRankIndex`
- Example: 2-rank comm + 2 new ranks → ranks 0,1,2,3

See MR for implementation details.
</details>

<details>
<summary><h2>Coding</h2></summary>

### Commit list or MR

The implementation of the Grow feature includes:

**Core Implementation Files:**
- `src/nccl.h.in`: API declarations for `ncclCommGrow` and `ncclCommGetUniqueId`
- `src/init.cc`: Core grow logic and async job handling (see MR)
- `src/bootstrap.cc`: Bootstrap coordination via existing patterns
- `src/include/bootstrap.h`: Bootstrap function declarations

**Key Implementation Components:**

1. **API Functions** (`src/init.cc`):
   - `ncclCommGetUniqueId()`: Generate coordination ID from existing communicator
   - `ncclCommGrow()`: Expand communicator with new ranks

2. **Bootstrap Coordination** (`src/bootstrap.cc`):
   - Reuses existing bootstrap flows (see MR)
   - Reuses collective exchange and connection setup patterns

3. **Async Job Infrastructure** (`src/init.cc`):
   - Uses the existing async job flow (see MR)
   - See MR for initialization flow

4. **Testing** (`test_ncclCommGrow.cu`):
   - Unit test demonstrating grow operation usage
   - Validation of communicator expansion

**Code Reuse Statistics:**
- Bootstrap: High reuse via existing flows (see MR)
- Transport Setup: Full reuse via `initTransportsRank()`
- AllGather Coordination: Full reuse of existing patterns
- Resource Management: High reuse from existing infrastructure
- Root Coordination: Reuses existing coordination paths

</details>

<details>
<summary><h2>Testing and Validation</h2></summary>

### Tests

1. API tests (`test/apitest/ncclCommGrow_test.cu`):
   - Run: `./build/test/apitest/apitest --gtest_filter=ncclCommGrow*`

2. Fault Tolerance tests (`test/perf/ft_test.cu`):
   - Run: `./build/test/perf/all_reduce_perf -B 0 -F 1 -L "grow"`

#### Expected output?

**Unit Test Success:**
```
ncclCommGrow Test Results:
✓ Existing rank initialization: PASS
✓ Root ID generation: PASS
✓ Grow operation completion: PASS
✓ Expanded communicator verification: PASS
✓ New communicator size: 4 ranks (expected: 4)
✓ Collective operation test: PASS
✓ Resource cleanup: PASS
Test completed successfully!
```

</details>

<details>
<summary><h2>Future Enhancements</h2></summary>

- In-place grow: add ranks without creating a new communicator (no comm swap)
- Rank assignment semantics: explicit user-driven re-assignment (remove -1 sentinel); add policy support (append-to-end, topology-aware auto placement) selectable via config

</details>

<details>
<summary><h2>Signoff List</h2></summary>

Author(s):
  - Bruce Chang

</details>
