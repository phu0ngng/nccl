# CE_Profiler_Interface

## Abstract

Copy Engine (CE) based collectives introduce a new execution model in NCCL that bypasses traditional GPU SMs and utilizes dedicated hardware copy engines for data movement. The current NCCL Profiler Plugin Interface **cannot capture timing and performance of CE collectives**, lacking visibility into CE collective operations, their synchronization mechanisms, and performance characteristics. This feature extends the profiler interface to capture CE-specific events, enabling comprehensive profiling, debugging, and performance analysis of CE collectives.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and Requirements</h2></summary>
<!-- ============================================================================================-->

CE collectives represent a fundamentally different execution model compared to traditional SM-based collectives. Some of the key features:

1. **Multi-stage synchronization operations** including host-initiated, intra-batch, multicast, and unicast communications using CUDA stream memory operations (`CU_STREAM_MEM_OP_WAIT_VALUE_32`, `CU_STREAM_MEM_OP_WRITE_VALUE_32`)
2. **Batched copy engine operations** with different latency and throughput profiles.
3. **CE operation overlap** with computation.

With proper profiling support, users can:
- Understand CE collective performance bottlenecks
- Correlate CE operations with application workloads
- Debug synchronization issues specific to CE collectives
- Compare CE vs SM-based collective performance in detail

### Key Functional Requirements:

1. **Capture CE collective lifecycle events**
   - CE collective initiation and completion.
   - Synchronization phase timings (readiness and completion syncs).
   - Batched operations execution timing.

2. **Expose CE-specific operation details**
   - Synchronization strategy (multicast vs unicast)
   - Batch operation parameters (number of operations, intra-batch sync frequency)
   - Copy engine utilization metrics
   - Sequence number tracking for tallying across ranks.

3. **Enable correlation with existing events**
   - Link CE events to their originating API calls
   - Associate CE operations with their parent group events

### Performance/Optimization Requirements:

1. **Low-overhead event capture**
   - Minimize impact on CE collective performance
   - Use activation masks to selectively enable CE profiling

2. **Hierarchical event organization**
   - CE collective events parent their constituent synchronization and copy operations
   - Clear correlation between API events and CE runtime events
   - Support for CUDA graph captured CE operations

### NVbugs / Jira Tickets

- 5469910(NVbug) (CE Collectives implementation)

### Assumptions, constraints and dependencies for CE profiling

- Changes for CE events are backward compatible with existing profiler plugin versions.
- CE profiling supports both graph-captured and non-graph CE operations.
- The profiler interface maintains the existing versioning strategy.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed Design

The profiler interface is extended with new event types specific to CE collective operations.
The design follows a **plugin-controlled architecture** where the profiler plugin has full control over timing implementation,
while NCCL core provides simple event hooks, which can be used to add cudaEvent beacons to mark various events.
This is different from other events where the time of callbacks is more controlled by core NCCL lib.

**Design Principle: Event Granularity**

The CE profiler captures three levels of events:
1. **CeColl**: High-level CE collective operation (e.g., AllGather, AlltoAll)
   - Scheduling of high level collective op on the user provided stream)
2. **CeSync**: Synchronization operations (readiness and completion using stream memory ops)
   - Within a collective op these capture barrier like sync events.
3. **CeBatch**: Batch copy operations (cudaMemcpyAsync/cudaMemcpyBatchAsync)
   - These capture multiple batch operations within higher level collective.

**Key Architectural Decision: Plugin-Controlled Timing**

The implementation delegates all timing infrastructure to the profiler plugin:

- **NCCL Core**: Simple hooks that call `plugin->startEvent()` and `plugin->stopEvent()` only
- **Plugin**: Creates timing events internally, manages poller thread, calculates all timing metrics
- **NO recordEventState() calls**: NCCL core does NOT call `recordEventState()` for CE events
- **Plugin-managed timing**: Poller thread running on the CPU does all timing recording
- **No record input interface changes needed**: `ncclProfilerEventStateArgs_v6_t` is identical to v5

This provides maximum flexibility for different profiler implementations while keeping NCCL core minimal and maintainable. The plugin has complete control over:
- Choice of timing mechanism (e.g. CUDA events, simple memory writes queued in stream etc.)
- When and how to capture timing data
- How to store and report timing information

**Interface Simplicity:**
1. New CE event types in `ncclProfilerEventDescr_v6_t` (ceColl, ceCollSync, ceCollBatch)
2. New event states in the enum (ncclProfilerCeCollStart/Complete, etc.)
3. Same state args as v5 (no changes needed)
4. Event Hierarchy and Correlation

#### 1. New CE event types in `ncclProfilerEventDescr_v6_t` (ceColl, ceCollSync, ceCollBatch)

**Event Mask Values:**
```C
enum {
  // Existing events (v5)
  ncclProfileGroup          = (1 << 0),  // group event type
  ncclProfileColl           = (1 << 1),  // host collective call event type
  ncclProfileP2p            = (1 << 2),  // host point-to-point call event type
  ncclProfileProxyOp        = (1 << 3),  // proxy operation event type
  ncclProfileProxyStep      = (1 << 4),  // proxy step event type
  ncclProfileProxyCtrl      = (1 << 5),  // proxy control event type
  ncclProfileKernelCh       = (1 << 6),  // kernel channel event type
  ncclProfileNetPlugin      = (1 << 7),  // network plugin-defined, events
  ncclProfileGroupApi       = (1 << 8),  // Group API events
  ncclProfileCollApi        = (1 << 9),  // Collective API events
  ncclProfileP2pApi         = (1 << 10), // Point-to-Point API events
  ncclProfileKernelLaunch   = (1 << 11), // Kernel launch events
  // New CE events
  ncclProfileCeColl         = (1 << 12), // CE collective operation
  ncclProfileCeSync         = (1 << 13), // CE synchronization operation
  ncclProfileCeBatch        = (1 << 14), // CE batch operation
};
```


```C
typedef struct {
  uint64_t type;                // event type descriptor: ncclProfileColl, ...
  void* parentObj;              // pointer to the profiler parent object (for coll is the group)
  int rank;                     // originating rank
  union {
    // ... existing event descriptors ...

    struct {
      uint64_t seqNumber;
      const char* func;         // "AllGather", "AlltoAll", etc.
      void const* sendBuff;
      void* recvBuff;
      size_t count;
      int root;
      const char* datatype;
      const char* syncStrategy; // "multicast", "unicast"
      bool intraBatchSync;      // whether intra-batch sync is enabled
      uint32_t batchSize;       // number of operations per batch
      uint32_t numBatches;      // number of batches
      uint32_t ceSeqNum;        // CE sequence number for sync tracking
      void* stream;             // CUDA stream for CE collective
    } ceColl;

    struct {
      bool isComplete;           // false = Ready sync, true = Complete sync
      int nRanks;                // number of ranks participating
      // Note: seqNumber and stream inherited from parent ceColl event
    } ceCollSync;

    struct {
      int numOps;               // number of copy operations in batch
      size_t totalBytes;        // total bytes to transfer in batch
      bool useIntraSync;        // whether intra-batch sync is used
      // Note: stream inherited from parent ceColl event
    } ceCollBatch;
  };
} ncclProfilerEventDescr_v6_t;
```

#### 2. New event states in the enum (ncclProfilerCeCollStart/Complete, etc.)

```C
typedef enum {
  ncclProfilerProxyOpSendPosted        = 0,  // deprecated in v4
  ncclProfilerProxyOpSendRemFifoWait   = 1,  // deprecated in v4
  ncclProfilerProxyOpSendTransmitted   = 2,  // deprecated in v4
  ncclProfilerProxyOpSendDone          = 3,  // deprecated in v4
  ncclProfilerProxyOpRecvPosted        = 4,  // deprecated in v4
  ncclProfilerProxyOpRecvReceived      = 5,  // deprecated in v4
  ncclProfilerProxyOpRecvTransmitted   = 6,  // deprecated in v4
  ncclProfilerProxyOpRecvDone          = 7,  // deprecated in v4
  ncclProfilerProxyOpInProgress_v4     = 19,

  /* Legacy proxy profiler states */
  ncclProfilerProxyStepSendGPUWait     = 8,
  ncclProfilerProxyStepSendPeerWait_v4 = 20,
  ncclProfilerProxyStepSendWait        = 9,
  ncclProfilerProxyStepRecvWait        = 10,
  ncclProfilerProxyStepRecvFlushWait   = 11,
  ncclProfilerProxyStepRecvGPUWait     = 12,

  /* Legacy proxy control states */
  ncclProfilerProxyCtrlIdle            = 13,
  ncclProfilerProxyCtrlActive          = 14,
  ncclProfilerProxyCtrlSleep           = 15,
  ncclProfilerProxyCtrlWakeup          = 16,
  ncclProfilerProxyCtrlAppend          = 17,
  ncclProfilerProxyCtrlAppendEnd       = 18,

  /* Network defined event states */
  ncclProfilerNetPluginUpdate          = 21,

  /* Kernel event states */
  ncclProfilerKernelChStop             = 22,

  /* Group API States */
  ncclProfilerGroupStartApiStop        = 23,
  ncclProfilerGroupEndApiStart         = 24,

  // New CE-specific states (v6)
  ncclProfilerCeCollStart = 25,        // CE collective operation begins
  ncclProfilerCeCollComplete = 26,     // CE collective operation completes
  ncclProfilerCeSyncStart = 27,        // CE synchronization begins
  ncclProfilerCeSyncComplete = 28,     // CE synchronization completes
  ncclProfilerCeBatchStart = 29,       // CE batch operation begins
  ncclProfilerCeBatchComplete = 30,    // CE batch operation completes
} ncclProfilerEventState_v6_t;
```

#### 3. Same state args as v5 (no changes needed)

**IMPORTANT**: CE events do NOT use `recordEventState` callbacks, so v6 uses the exact same state args as v5:

```C
// v6 uses same state args as v5 (no CE-specific state args needed)
// CE events don't use recordEventState - plugin manages all timing internally
typedef ncclProfilerEventStateArgs_v5_t ncclProfilerEventStateArgs_v6_t;
```

**v5 State Args (inherited by v6):**
```C
typedef union {
  struct {
    size_t transSize;
  } proxyStep;

  struct {
    int appendedProxyOps;
  } proxyCtrl;

  struct {
    void* data;
  } netPlugin;

  struct {
    uint64_t pTimer;
  } kernelCh;
} ncclProfilerEventStateArgs_v5_t;  // v6 = v5
```

#### 4. Event Hierarchy and Correlation

The CE events follow this hierarchy:

```
GroupApi Event
└── CollApi Event (CE-enabled)
    └── CeColl Event
        ├── CeSync Event (readiness)
        │   └── Multiple sync operations (stream memory ops)
        ├── CeBatch Event
        │   └── Multiple copy operations (cudaMemcpyAsync/cudaMemcpyBatchAsync)
        │       └── Intra-batch syncs included but not separately profiled
        └── CeSync Event (completion)
            └── Multiple sync operations (stream memory ops)
```

### Interface Architecture

#### Profiler Interface v6

```C
typedef struct {
  const char* name;

  // init - initialize the profiler plugin
  ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask, const char* commName, int nNodes, int nranks, int rank, ncclDebugLogger_t logfn);

  // startEvent - initialize and start a new event
  ncclResult_t (*startEvent)(void* context, void** eHandle, ncclProfilerEventDescr_v6_t* eDescr);

  // stopEvent - stop/finalize an event
  ncclResult_t (*stopEvent)(void* eHandle);

  // recordEventState - record event state transitions and updates
  ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v6_t eState, ncclProfilerEventStateArgs_v6_t* eStateArgs);

  // finalize - finalize the profiler plugin
  ncclResult_t (*finalize)(void* context);
} ncclProfiler_v6_t;
```

#### Activation Mask for CE Events

Users can selectively enable CE profiling using the activation mask. The profiler implements **hierarchical event enabling** where child events automatically enable their parent events:

```C
// Enable only CE collective events (high-level overview)
activationMask |= ncclProfileCeColl;
// → Auto-enables: CollApi, GroupApi events

// Enable detailed CE profiling including sync operations
activationMask |= ncclProfileCeSync;
// → Auto-enables: CeColl (parent), CollApi, GroupApi events

// Enable full CE profiling: collective, sync, and batch operations
activationMask |= (ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch);
// → Auto-enables: CollApi, GroupApi events

// Note: Intra-batch synchronization is captured as metadata in the
// ceCollBatch descriptor (intraBatchSync flag), not as separate events
```

**Hierarchical Event Enabling:**
- Enabling `ncclProfileCeSync` or `ncclProfileCeBatch` automatically enables their parent `ncclProfileCeColl`
- Enabling any CE event (`CeColl`, `CeSync`, or `CeBatch`) automatically enables `CollApi` and `GroupApi` events
- This ensures complete event chains are captured for correlation and analysis
- Implemented via activation masks in NCCL core:
  - `groupApiMask` includes CE events (line 250 in profiler.cc)
  - `collApiMask` includes CE events (line 323 in profiler.cc)
  - `ceCollMask` checks for any CE child events (line 761 in profiler.cc)

#### Key Implementation Decisions

**1. Default Profiler Interface Changed to v6**
```c
// In nccl_profiler.h:
typedef ncclProfiler_v6_t ncclProfiler_t;           // Changed from v5 to v6
typedef ncclProfilerEventDescr_v6_t ncclProfilerEventDescr_t;
typedef ncclProfilerEventStateArgs_v6_t ncclProfilerEventStateArgs_t;  // Same as v5

```
This allows CE event descriptors to be used without casting. State args remain unchanged from v5. Older profiler versions (v1-v5) are still supported via version-specific loaders with appropriate casts.

**2. Stop Events Always Created**
Regardless of timing mode, stop events are always created. The timing mode only controls:
- Event creation flags (`cudaEventDisableTiming` for CPU mode)
- How elapsed time is calculated (GPU events vs CPU timestamps)

**3. No recordEventState() Callbacks for CE Events**
- NCCL core does NOT call `recordEventState()` for CE events
- All timing is managed internally by the plugin
- Plugin poller thread detects event completion and records timing
- Plugin can report timing via its own mechanisms (e.g., JSON output, print functions)
- Therefore, `ncclProfilerEventStateArgs_v6_t` is identical to v5 (no new fields needed)

**4. Activation Mask Controls Event Generation**
Event generation is controlled by `ncclProfilerEventMask` (set by profiler plugin), not by separate `NCCL_PROFILER_CE_TIME_*` environment variables (which were removed for consistency).

#### Backward Compatibility

- v6 profiler interface is fully backward compatible with v5 and earlier
- Existing profiler plugins continue to work without modification
- CE events are only emitted when explicitly enabled in activation mask
- Older profiler versions can be loaded with appropriate type casting

### CE Profiler Architecture

#### NCCL Core Responsibilities (Minimal)

**File: `src/plugin/profiler.cc`** - Simple wrapper functions only:

```C
// NO CE-specific context or init functions in NCCL core
// Plugin owns all configuration via its opaque profilerContext

// Simple wrapper functions that call plugin callbacks
ncclResult_t ncclProfilerStartCeCollEvent(..., void** ceCollHandle) {
  if (!ncclProfiler) return ncclSuccess;

  // Check if any CE event is enabled (CeColl, CeSync, or CeBatch)
  // If child events are enabled, we auto-enable parent CeColl
  int ceCollMask = ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch;
  if (!(ncclProfilerEventMask & ceCollMask)) return ncclSuccess;

  ncclProfilerEventDescr_t eDescr = { ... };  // Fill descriptor
  ncclProfiler->startEvent(comm->profilerContext, ceCollHandle, &eDescr);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartCeSyncEvent(..., void** ceSyncHandle) {
  if (!ncclProfiler || !args->ceCollProfHandle) return ncclSuccess;

  // Only check if CeSync is enabled; parent CeColl is auto-enabled via ceCollMask
  if (!(ncclProfilerEventMask & ncclProfileCeSync)) return ncclSuccess;

  ncclProfilerEventDescr_t eDescr = { ... };  // Fill descriptor
  ncclProfiler->startEvent(comm->profilerContext, ceSyncHandle, &eDescr);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCeCollEvent(..., void* ceCollHandle) {
  if (!ncclProfiler || !ceCollHandle) return ncclSuccess;
  ncclProfiler->stopEvent(ceCollHandle);
  return ncclSuccess;
}
// Similar simple wrappers for CeBatch
```

**Key Point**: NCCL core has NO CE-specific state, NO poller thread, NO CUDA event management, NO timing calculation. Plugin owns everything via its opaque `profilerContext`.

#### Plugin Responsibilities (Example Profiler)

**What Plugin's `init()` Callback Does:**

1. **One-time Global Setup** (first communicator only, via `pthread_once`):
   - Reads plugin-specific environment variables (`NCCL_PROFILER_CE_TIMING`, `NCCL_PROFILER_CE_POLLER_INTERVAL_MICROSECONDS`)
   - Allocates global context registry to track all communicators
   - Starts the global poller thread (shared across all communicators)

2. **Per-Communicator Setup** (each communicator):
   - Allocates and initializes per-communicator context
   - Returns opaque context pointer to NCCL core
   - Sets activation mask to tell NCCL which events to send

3. **Environment Variables Read by Plugin** (not NCCL core):
   - `NCCL_PROFILER_CE_TIMING`: "gpu" or "cpu" (default: cpu)
   - `NCCL_PROFILER_CE_POLLER_INTERVAL_MICROSECONDS`: Polling interval (default: 500)
   - Any other plugin-specific configuration

**Plugin-Managed State (Example):**

Plugin owns all CE-specific state including:
- Global poller thread (shared across all communicators)
- Timing mode configuration (GPU vs CPU)
- Context registry to track all communicators
- Event queues and pending event tracking

**Plugin Internal Event Tracking:**

Plugin maintains internal event structures for:
- Hierarchical event organization (parent-child relationships)
- Dual tracking: hierarchy queue + poller tracking list
- Pending event queues with completion status
- Timing data (CPU timestamps, elapsed time, etc.)

#### Stream Field Usage

The `stream` field in CE event descriptors enables the plugin to:
- Record timing events in the correct CUDA stream
- Maintain stream ordering for async operations
- Support stream-ordered profiling without host synchronization

#### Timing Modes

Plugin uses a poller thread to detect async operation completion. Timing mode affects:
1. Event creation flags (`cudaEventDisableTiming` for CPU mode)
2. Elapsed time calculation (GPU events vs CPU timestamps)

**GPU Mode**: Uses `cudaEventElapsedTime()` for precise timing
**CPU Mode**: Uses host timestamps (lower overhead)

#### Timing Fields (Example Profiler Implementation)

All CE events (`ceColl`, `ceSync`, `ceBatch`) contain the following timing fields:

**Clock and Units:**
- **Clock source**: `CLOCK_MONOTONIC` via `gettime()` (monotonic clock, immune to system time changes)
- **Time units**: All fields measured in **microseconds**

**Field Definitions:**

1. **`cpuStartTime` / `cpuStopTime`** (type: `double`):
   - Captured using `CLOCK_MONOTONIC` when CUDA events complete
   - Always available regardless of timing mode
   - Units: microseconds

2. **`cpuDuration`** (type: `double`):
   - Always CPU-measured: `cpuStopTime - cpuStartTime`
   - Provides fallback timing if GPU timing fails
   - Units: microseconds

3. **`elapsedTime`** (type: `uint64_t`):
   - **Final reported timing** used in trace output
   - Calculation depends on `timingMode`:
     - **GPU Mode** (`CE_TIMING_GPU`): Obtained from `cudaEventElapsedTime()` (converted from milliseconds to microseconds)
     - **CPU Mode** (`CE_TIMING_CPU`): Same as `cpuDuration`
   - Falls back to `cpuDuration` if GPU timing fails
   - Units: microseconds
   - Type: `uint64_t` for consistency with trace output format

**Design Rationale:**
- **`cpuDuration`** always provides a CPU-measured baseline, useful for debugging and fallback
- **`elapsedTime`** provides the final reported timing (GPU or CPU) based on user configuration
- Separate fields allow comparison between CPU and GPU timing when debugging
- GPU timing provides higher precision for CE operations but adds overhead
- CPU timing has lower overhead and is the default

**Plugin Callback Implementations:**

Plugin implements `startEvent()` and `stopEvent()` callbacks:
- Allocate event from pool
- Set up parent-child hierarchy
- Create and record timing events
- Add to poller tracking

#### Plugin Poller Thread

Plugin manages a poller thread that:
- Checks event completion using `cudaEventQuery()`
- Records CPU timestamps when events complete
- Calculates elapsed time
- Stores/reports timing via plugin's own mechanisms
- Cleans up completed events

#### Environment Variables

**Plugin-specific** (read by plugin, not NCCL core):
- `NCCL_PROFILER_CE_TIMING`: "gpu" or "cpu" (default: cpu)
- `NCCL_PROFILER_CE_POLLER_INTERVAL_MICROSECONDS`: Polling interval (default: 500)

**Event Activation**:
- `NCCL_PROFILE_EVENT_MASK`: Enable CE events (CeColl=0x1000, CeSync=0x2000, CeBatch=0x4000)

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Coding</h2></summary>
<!-- ============================================================================================-->

### Implementation Components

#### 1. Profiler Interface Extensions

**File: `src/include/plugin/profiler/profiler_v6.h`**
- Define new event types, descriptors, and states
- Extend event state arguments for CE-specific data
- Maintain compatibility with previous versions

#### 2. CE Profiler Integration

**File: `src/ce_coll.cc`** - CE collectives call profiler hooks:
- Call `ncclProfilerStartCeCollEvent()` at operation start
- Call `ncclProfilerStopCeCollEvent()` at operation end
- Sync and batch operations similarly instrumented

#### 3. NCCL Core CE Profiler Functions

**File: `src/plugin/profiler.cc`** - Simple wrapper functions:
- Build event descriptor from CE operation parameters
- Call `plugin->startEvent()` / `plugin->stopEvent()`
- No CUDA event creation, no poller thread, no timing calculation

### File Organization

**NCCL Core** (Minimal CE profiler code):
- `src/include/profiler.h` - CE profiler event wrapper function declarations
- `src/plugin/profiler.cc` - Simple CE profiler wrapper functions (6 functions, ~50 lines total)
- `src/include/plugin/profiler/profiler_v6.h` - v6 interface definition with CE event types
- `src/plugin/profiler/profiler_v6.cc` - v6 plugin loader
- `src/ce_coll.cc` - Instrumented with CE profiler start/stop calls

**Note**: No CE-specific context, init functions, or env var reading in NCCL core

**Example Profiler Plugin** (Full timing implementation):
- `ext-profiler/example/nccl/profiler_v6.h` - v6 interface copy for plugins
- `ext-profiler/example/event.h` - CE event structures with taskEventBase and poller tracking
- `ext-profiler/example/plugin.cc` - Main plugin with v5 events (1030 lines, reduced from 1513)
- `ext-profiler/example/profiler_plugin_ce.h` - CE profiler API declarations (38 lines)
- `ext-profiler/example/profiler_plugin_ce.cc` - CE poller thread, CUDA event management, timing calculation (536 lines)
- `ext-profiler/example/print_event.cc` - CE event printing functions

**Code Organization Note**: CE-specific profiling logic has been refactored into separate `profiler_plugin_ce.{h,cc}` files to improve modularity and maintainability. The plugin automatically includes CE profiling when built.

### Configuration and Usage

Plugin-specific environment variables (examples):
- `NCCL_PROFILER_CE_TIMING`: "gpu" or "cpu" (default: cpu)
- `NCCL_PROFILER_CE_POLLER_INTERVAL_MICROSECONDS`: Polling interval (default: 500)

Event activation via mask:
- `NCCL_PROFILE_EVENT_MASK`: Enable CE events (CeColl=0x1000, CeSync=0x2000, CeBatch=0x4000)

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation

#### Where to run?

#### What to run? (example + inspector profiler plugin)
1. **Functionality Tests:**
   - CE collective profiling with different collective types (AllGather, AlltoAll, Scatter, Gather)
   - Mixed CE/SM workloads in single traces
   - CUDA graph captured CE operations
   - Different synchronization strategies (multicast(NVLS) vs unicast)
   - Batch operations with and without intra-batch sync (verified via metadata, not separate events)

2. **Performance Tests:**
   - Measure profiler overhead with CE events enabled/disabled
   - Compare CE collective performance with and without profiling
   - Validate profiler activation mask selective enabling

3. **Compatibility Tests:**
   - Run existing profiler plugins with CE-enabled NCCL
   - Verify v6 profiler plugins work with legacy collective operations
   - Test profiler plugin version negotiation

4. **Integration Tests:**
   - Use profiler data to analyze CE collective performance
   - Correlate CE events with application-level events
   - Test with real DL framework workloads using CE collectives

#### Expected output?

1. **Profiler traces containing:**
   - CE collective events with accurate timing and metadata
   - Proper event hierarchy and correlation
   - Synchronization operation details
   - Batch operation metrics

2. **Performance validation:**
   - Profiler overhead < 2% of CE collective execution time
   - No degradation in CE collective performance when profiling is disabled

3. **Compatibility validation:**
   - All existing profiler plugins continue to function
   - Mixed event traces contain both traditional and CE events correctly

### Performance

#### What is measured?
- CE collective execution time with/without profiling enabled
- Profiler event generation overhead
- Memory usage impact of CE event tracking
- Event correlation accuracy and completeness

#### Results
Expected results demonstrate:
- Negligible performance impact when CE profiling is enabled
- Complete capture of CE collective operations and their constituent phases
- Accurate correlation between API calls and CE runtime events
- Successful integration with existing profiler infrastructure

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Sirshak Das (CE Profiler Interface design)

Reviewers:
  - Giuseppe Congiu (Profiler Plugin Interface expert)
  - Zhenhao He (CE Collectives implementation)
  - Bharath Ramesh (Profiler Infrastructure)

</details>
