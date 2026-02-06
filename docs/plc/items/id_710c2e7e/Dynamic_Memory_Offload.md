# Dynamic Memory Offload

## Abstract
Implement GPU memory management functionality in NCCL to allow applications to suspend/resume GPU memory when not actively performing collectives. This feature enables communicator suspension for RL training workflows without external plugins. NCCL tracks memory allocations and supports different memory categories: persistent (never released), scratch (freed without saving), and offload (backed up to CPU before suspend and restored on resume).

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets
https://nvbugspro.nvidia.com/bug/5760696

### User Experience

Users can call `ncclCommSuspend(comm, NCCL_SUSPEND_MEM)` to release dynamic GPU memory when NCCL is idle (e.g., during inference or environment steps in RL training). When collectives are needed again, users call `ncclCommResume(comm)` to restore the memory. The `ncclCommMemStats()` API allows querying memory usage statistics.

Users should expect:
- GPU memory savings during idle periods
- Overhead for suspend/resume operations
- No impact on collective performance when memory is active
- Ability to query memory statistics at any time

### Assumptions, constraints and dependencies

1. Suspend/resume operations are collective - all ranks must call them together
2. Applications must not perform NCCL collectives while communicator is suspended
3. CUDA VMM support is required for stable virtual addresses
4. Memory type is determined at allocation time by NCCL internally
5. Peer-imported buffers require coordination: all ranks must unmap peer imports before owners can suspend local memory
6. Split-share communicators do not support suspend/resume (manager refCount > 1)
7. We focus on suspension of intra-node P2P buffers as they occupy the most significant memory consumption. Suspension of other buffers is left as future improvement
8. Memory allocated directly with cudaMalloc/cuMemCreate is not tracked and cannot be suspended
9. Memory tracking is automatic when using NCCL allocators (ncclCudaMalloc, ncclCuMemAlloc); direct use of CUDA APIs bypasses tracking

### Use Cases

1. **RL Training Workflows**: During environment steps, policy rollouts, or inference phases where GPU memory is needed for other computations, NCCL memory can be temporarily released.

2. **Multi-tenant GPU Sharing**: Applications that share GPUs with other workloads can release NCCL memory during idle periods.

3. **Memory-constrained Training**: Large model training where every MB of GPU memory matters can benefit from releasing NCCL buffers during non-collective phases.

### Platform Requirements

- CUDA Toolkit >= 11.3 for VMM APIs
- CUDA P2P support between communicating ranks for buffer sharing
- Sufficient CPU pinned memory for offloading buffers

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

The implementation consists of a Memory Manager (`ncclMemManager`) attached to each communicator that tracks memory allocations (including persistent memory for statistics).

**Memory Types:**
```c
typedef enum {
  ncclMemPersist  = 0,  // Persistent memory - track stats only, never suspended
  ncclMemScratch  = 1,  // Scratch - freed without backup during suspend
  ncclMemOffload  = 2   // Offload - backed up to CPU during suspend, restored on resume
} ncclMemType_t;
```

**Key Data Structures:**

We introduce `ncclDynMemEntry` structure that tracks scratch and offload allocations:
- GPU virtual address (stable across suspend/resume)
- Physical memory handle and allocation properties
- Shareable handle for P2P (POSIX FD or FABRIC handle)
- CPU backup pointer for OFFLOAD type
- Peer tracking for imported/exported buffers

```c
// Memory entry state (tracks lifecycle during suspend/resume)
typedef enum {
  ncclDynMemStateActive   = 0,  // Memory is mapped and usable
  ncclDynMemStateReleased = 1  // Physical memory released, VA reserved
} ncclDynMemState_t;

// Individual tracked memory entry
typedef struct ncclDynMemEntry {
  // Core allocation info
  void*                          ptr;           // GPU virtual address (stable across release/restore)
  size_t                         size;          // Allocation size
  CUmemGenericAllocationHandle   handle;        // Physical memory handle
  CUmemAllocationHandleType      handleType;    // POSIX_FILE_DESCRIPTOR or FABRIC
  ncclMemType_t                  memType;       // Scratch or Offload (Persist not in list)
  ncclDynMemState_t              state;         // Current lifecycle state
  int                            cudaDev;       // CUDA device this allocation belongs to

  // Shareable handle for P2P (used for cross-rank buffer sharing)
  union {
    int                          fd;            // For CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    CUmemFabricHandle            fabricHandle;  // For CU_MEM_HANDLE_TYPE_FABRIC
  } shareableHandle;
  bool                           shareableHandleValid;

  // CPU backup for OFFLOAD type memory (data preserved across release/restore)
  void*                          cpuBackup;     // Host pinned memory for offloaded data

  // Peer export tracking (this buffer is shared TO other ranks)
  int                            numExportedPeers;
  int                            exportedPeerRanks[NCCL_DYN_MEM_MAX_PEERS];

  // Peer import tracking (this buffer is imported FROM another rank)
  bool                           isImportedFromPeer;
  int                            ownerRank;           // Rank that owns the original buffer
  int                            ownerDev;            // CUDA device of the owner
  void*                          ownerPtr;            // Owner's virtual address
  uint64_t                       ownerShareableHandle; // Handle received from owner

  struct ncclDynMemEntry*        next;          // Linked list pointer
} ncclDynMemEntry;

// Memory manager
typedef struct ncclMemManager {
  ncclDynMemEntry*  entries;        // Linked list of scratch/offload allocations
  int               numEntries;     // Number of entries in list
  pthread_mutex_t   lock;           // Thread safety
  int               released;       // 1 if currently suspended
  int               initialized;    // Initialization flag
  int               refCount;       // Reference count for sharing

  // Statistics (atomically updated)
  size_t            totalPersist;         // Persistent memory owned (bytes)
  size_t            totalPersistImported; // Persistent memory imported (bytes)
  size_t            totalScratch;         // Scratch memory owned (bytes)
  size_t            totalScratchImported; // Scratch memory imported (bytes)
  size_t            totalOffload;         // Offload memory owned (bytes)
  size_t            totalOffloadImported; // Offload memory imported (bytes)
  size_t            cpuBackupUsage;       // CPU backup usage (bytes)

  int               commCudaDev;    // CUDA device associated with this comm
} ncclMemManager;
```

**Public APIs:**
```c
// Suspend communicator (releases memory based on flags)
ncclResult_t ncclCommSuspend(ncclComm_t comm, int flags);

// Resume communicator (restores memory based on flags)
ncclResult_t ncclCommResume(ncclComm_t comm);

// Query communicator statistics
ncclResult_t  ncclCommMemStats(ncclComm_t comm, ncclCommMemStat_t stat, uint64_t* value);

// Flags
#define NCCL_SUSPEND_MEM 0x01  // Suspend memory allocations


/* Communicator statistics */
typedef enum {
  ncclStatGpuMemTotal        = 0,  // Total allocated GPU memory tracked by NCCL (bytes)
  ncclStatGpuMemPersist      = 1,  // Allocated GPU memory that cannot be suspended (bytes)
  ncclStatGpuMemSuspend      = 2,  // Allocated GPU memory that can be suspended (bytes)
  ncclStatGpuMemSuspended    = 3   // GPU memory suspended? (0=active, 1=suspended)
} ncclCommMemStat_t;
```

**Suspend Operation Order:**
1. Unmap all peer-imported buffers
2. Barrier to ensure all ranks complete step 1
3. For Offload buffers: copy GPU data to CPU pinned memory
4. Unmap and release physical memory (cuMemUnmap + cuMemRelease)
5. Keep virtual address reservations for resume
6. Final barrier

**Resume Operation Order:**
1. Re-allocate physical memory (cuMemCreate)
2. Re-map to same virtual addresses (cuMemMap)
3. For Offload buffers: restore data from CPU backup
4. Barrier to ensure all ranks complete local resume
5. Exchange new handle info via bootstrap
6. Re-import peer buffers using new handles
7. Final barrier

**P2P Handle Types:**
- `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`: Single-node P2P, shared via Unix domain sockets
- `CU_MEM_HANDLE_TYPE_FABRIC`: Multi-node MNNVL, shared via bootstrap data exchange

**Environment Variables:**
- `NCCL_DYN_MEM_ENABLE`: Enable/disable dynamic memory tracking (default: enabled)

### Interface Architecture

The memory manager (`ncclMemManager`) is stored as a pointer in both `ncclComm` and `ncclProxyState`, enabling sharing across split-share communicators through reference counting. This design ensures proxy-allocated buffers are properly tracked even when communicators are destroyed.

**Allocator Interface:**
- All NCCL allocators accept a `manager` parameter for tracking
- Passing `manager = NULL` disables tracking for that allocation
- The `memType` parameter controls suspend/resume behavior:
  - `ncclMemPersist`: Track for statistics only, never suspended
  - `ncclMemScratch`: Suspend without backup (re-allocate on resume)
  - `ncclMemOffload`: Backup to CPU before suspend, restore on resume

**Modified Allocator Signatures:**
```c
ncclResult_t ncclCuMemAlloc(void **ptr, CUmemGenericAllocationHandle *handlep,
                            CUmemAllocationHandleType type, size_t size,
                            struct ncclMemManager* manager, ncclMemType_t memType);

ncclResult_t ncclCuMemFree(void *ptr, struct ncclMemManager* manager);

ncclResult_t ncclCudaMalloc(void **ptr, size_t size,
                            struct ncclMemManager* manager, ncclMemType_t memType);

ncclResult_t ncclCudaFree(void *ptr, struct ncclMemManager* manager);
```

**Internal Tracking Functions:**
```c
// Called by allocators to register new allocations
ncclResult_t ncclMemTrack(struct ncclMemManager* manager, void* ptr, size_t size,
                         CUmemGenericAllocationHandle handle,
                         CUmemAllocationHandleType handleType,
                         ncclMemType_t memType);

// Called by free functions to remove entries
ncclResult_t ncclMemUntrack(struct ncclMemManager* manager, void* ptr, size_t size);
```

**P2P Buffer Sharing (called during transport setup):**
```c
// Mark buffer as exported to a peer rank
ncclResult_t ncclDynMemMarkExportToPeer(struct ncclMemManager* manager, void* ptr, int peerRank);

// Mark buffer as imported from a peer rank (requires ownerPtr for restore matching)
ncclResult_t ncclDynMemMarkImportFromPeer(struct ncclMemManager* manager, void* ptr,
                                          int ownerRank, int ownerDev, void* ownerPtr,
                                          uint64_t shareableHandle);
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

**New Files:**
- `src/include/mem_manager.h` - Header with data structures and API declarations
- `src/mem_manager.cc` - Implementation of memory manager

**Modified Files:**
- `src/include/comm.h` - Add `ncclMemManager* memManager` to `struct ncclComm`
- `src/include/proxy.h` - Add `ncclMemManager* memManager` to `struct ncclProxyState`
- `src/include/alloc.h` - Change parameter from `comm` to `manager` in allocation functions
- `src/init.cc` - Initialize/destroy memory manager, implement public APIs
- `src/proxy.cc` - Set `proxyState->memManager` during proxy creation
- `src/transport/p2p.cc` - Track P2P buffers, mark exports/imports
- `src/transport/net.cc` - Track NET buffers as persistent (not suspended)
- `src/transport/*.cc` - Update all transport allocations to pass manager
- `src/nccl.h.in` - Add public API declarations (`ncclCommSuspend/Resume/Stats`)

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation

Verify correct memory release/restore behavior with P2P coordination across multiple ranks.

#### Where to run?

H100x8, B200x{4,36,72}, multi-node configurations

#### What to run?

1. **Basic Suspend/Resume Test:**
   - Initialize communicator and run several collectives
   - Call `ncclCommSuspend(comm, NCCL_SUSPEND_MEM)`
   - Verify GPU memory is freed using `ncclCommStats()`
   - Call `ncclCommResume(comm, NCCL_SUSPEND_MEM)`
   - Run collectives again and verify correctness

2. **Offload Data Integrity Test:**
   - Allocate buffers marked as Offload type
   - Fill with known patterns
   - Suspend and resume
   - Verify data integrity after resume

3. **Repeated Cycle Test:**
   - Perform multiple suspend/resume cycles
   - Verify no memory leaks using `ncclCommStats()`
   - Check refCount handling in split-share scenarios

4. **Multi-node Test:**
   - Test across multiple nodes
   - Verify P2P intra-node buffers are suspended/resumed correctly
   - Verify NET inter-node buffers remain persistent (not suspended)

5. **Error Handling Test:**
   - Call collective while suspended (should fail gracefully)
   - Call suspend twice (expect error)
   - Call resume when not suspended (expect error)
   - Test with split-share (should reject suspend/resume)

#### Expected output?

- All collectives produce correct results after resume
- Memory statistics (`ncclCommStats`) show expected suspend/resume amounts
- No memory leaks after repeated cycles
- Proper error messages for invalid usage
- Split-share communicators correctly reject suspend/resume operations

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

NCCL collective/sendrecv performance should not be impacted when memory is in active state.
Suspend/resume operations have overhead proportional to the amount of memory being managed.

#### What is measured?

1. Suspend latency: Time to complete `ncclCommSuspend(comm, NCCL_SUSPEND_MEM)`
2. Resume latency: Time to complete `ncclCommResume(comm, NCCL_SUSPEND_MEM)`
3. Memory savings: GPU memory freed during suspend (via `ncclCommStats`)
4. CPU memory overhead: Pinned memory used for Offload buffers
5. Collective performance: Bandwidth/latency with tracking enabled (should be unaffected)

#### Results

Example results on single node Theia cluster

=== Dynamic Memory Offload Unit Test ===
Ranks: 4
Element count: 268435456
Options:
  Repeat cycles: 3
  Symmetric window: no
  Verify integrity: no
  Verify collectives: yes
=========================================
[Cycle 1] Release time: 248.419 ms
[Cycle 1] Restore time: 598.103 ms
[Cycle 2] Release time: 260.545 ms
[Cycle 2] Restore time: 602.965 ms
[Cycle 3] Release time: 268.976 ms
[Cycle 3] Restore time: 607.087 ms
--- Memory Status After Restore ---
=== NCCL Dynamic Memory Manager Statistics ===
Device: 0
State: ACTIVE
Dynamic memory (512 tracked entries + persistent):
  Owned:
    PERSIST: 257949696 bytes (246.00 MB)
    SCRATCH: 0 bytes (0.00 MB)
    OFFLOAD: 1610612736 bytes (1536.00 MB)
    SUBTOTAL: 1868562432 bytes (1782.00 MB)
  Imported from peers:
    PERSIST: 0 bytes (0.00 MB)
    SCRATCH: 0 bytes (0.00 MB)
    OFFLOAD: 1610612736 bytes (1536.00 MB)
    SUBTOTAL: 1610612736 bytes (1536.00 MB)
  TOTAL:   3479175168 bytes (3318.00 MB)
CPU backup usage: 0 bytes (0.00 MB)
==============================================
--- Timing Summary ---
Average release time: 259.314 ms
Average restore time: 602.718 ms

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Zhenhao He

</details>
