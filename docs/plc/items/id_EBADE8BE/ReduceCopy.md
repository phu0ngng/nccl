# ReduceCopy API
<!-- NCCL Device API for Reduce, Copy (Broadcast), and ReduceCopy Operations -->

## Abstract

The ReduceCopy API provides 39 device-side functions for reduce, copy (broadcast), and combined reduce-copy operations in NCCL, simplifying custom kernel development while maintaining flexibility and performance.

### Key Features

- **Three operation types**: ReduceSum (N -> 1), Copy (Broadcast) (1 -> N), and ReduceSumCopy (N -> M)
- **Multiple memory models**: LSA (Load Store Accessible), Multimem, and Local
- **Flexible abstractions**: Lambda-based generic functions and concrete convenience wrappers
- **Performance control**: `UNROLL` template parameter for performance/register tradeoff

### Target Audience

Kernel engineers developing custom NCCL collectives and less experienced engineers using simplified convenience APIs.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

https://nvbugspro.nvidia.com/bug/5455848

### User Experience

Users of the NCCL device API should have a simplified experience writing kernels with good performance.
That follow the reduce - copy pattern.


#### Problem Statement

The existing NCCL Device API requires significant expertise: manual memory addressing across ranks, verbose boilerplate for common patterns, and deep knowledge for optimal performance.

#### Solution

The ReduceCopy API provides simplified high-level functions for common patterns, flexible lambda abstractions for custom layouts, automatic optimization through templates, and progressive complexity where simple cases remain simple while complex cases are possible.

### Use Cases

1. **Custom AllReduce + Computation patterns**: ReduceSum -> Computation -> Copy (Broadcast)
2. **Custom ReduceScatter & AllGather**: ReduceSum & Copy with different destination ranks
3. **Hybrid memory operations**: Mix LSA and Multimem in same kernel (for performance)
4. **Strided data layouts**: Reduce/copy with custom stride patterns

### Assumptions, constraints and dependencies

**Assumptions:**
- CUDA device code environment
- NCCL communicator properly initialized
- Symmetric memory allocations (LSA/Multimem)

**Constraints:**
- Operations are device-side only
- All participating threads/CTAs must call collectively, using existing ncclCoop abstraction
- Memory must be properly allocated and accessible

**Dependencies:**
- NCCL core primitives (ncclSymPtr, ncclTeam, ncclDevComm, etc.)
- C++14 or later (for lambda support)

### Platform Requirements

- Multi-GPU platform with LoadStore accessible GPUs (incl. PCI or NVLINK) for pure LSA versions
- Multimem support (i.e. NVLink) for variations with Multimem
- single GPU system without further restrictions is sufficient for the local reduce variants

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed Design

#### Architecture Overview

The API uses hierarchical layering:

```
Level 1 (Generic): reduceCopy<T, RedOp, ...>
                   |
Level 2 (Typed):   ncclLsaReduceSumLsaCopy<T, ...>
                   |
Level 3 (Convenience): ncclLsaReduceSum(...)
                       ncclLsaCopy(...)
                       ncclLsaReduceSumCopy(...)
```

There is also a generic wrapper abstracting LSA and Multimem with templates, since the optimization patterns are similar.
However, that version is not part of the API, since we want the memory explictly part of the API signature.

**Design Principles:**
1. **One optimized implementation**: Generic `reduceCopy<>` kernel handles all cases
2. **Type-safe wrappers**: Strong typing prevents common errors
3. **Progressive disclosure**: Simple cases are simple, complex cases possible
4. **Zero abstraction overhead**: All wrappers inline to core implementation

Note: Memory model (LSA/Multimem) is explicit in API signatures despite similar internal optimization patterns.

#### API Structure (39 Total Functions)

**Series 1.x - Generic ReduceCopy** (2 functions)
- Custom reduction operations via explicit `RedOp` parameter
- Only LSA sources supported (Multimem sources require `ncclOpSum`)
- Foundation for all specialized operations
- **Optimization note**: Vectorized optimization currently only in Sum-specialized variants. Future versions may add vectorization support to the `RedOp` abstraction.

**Series 2.x - Sum ReduceCopy** (4 functions)
- Sum operation (most common case), no `RedOp` parameter
- All LSA/Multimem combinations
- Foundation for convenience APIs

**Series 3.x - ReduceSum** (10 functions)
- N->1 reduction: LSA (5 variations), Multimem (3), Local (2)

**Series 4.x - Copy (Broadcast)** (10 functions)
- 1->N copy: Same structure as ReduceSum for consistency

**Series 5.x - ReduceSumCopy** (13 functions)
- N->M combined: LSA (5), Multimem (3), Mixed (4), Local (1)

### Interface Architecture

#### Key Template Parameters

```cpp
template<
  typename T,              // Element type (float, half, int, etc.)
  typename Coop,           // Cooperation level (ncclCoopCta, ncclCoopThread)
  typename IntCount,       // Count type (int, size_t, etc.)
  int UNROLL=4*16/sizeof(T) // Performance tuning parameter
>
```

**UNROLL Parameter:**
- Default: `4*16/sizeof(T)` = 64 bytes worth of elements (similar to internal implementation)
- Higher values: Better performance, more registers
- Lower values: Lower register pressure, better occupancy
- Users can override for specific use cases

#### Memory Models

- **LSA (Load Store Accessible)**: Symmetric allocations across ranks, accessed via `ncclSymPtr<T>` and `ncclTeam`
- **Multimem**: CUDA Multimem for cross-GPU access via `ncclMultimemHandle` or raw pointers
- **Local**: Thread/CTA-local memory, no inter-rank communication

#### Cooperation Levels

- **ncclCoopCta**: Entire CTA cooperates for higher performance (requires user to chunk data to block level)
- **ncclCoopThread**: Individual threads, most flexible for custom patterns. No specialization: generic

### Key Design Decisions

#### 1. Lambda vs Concrete Pointers

Both provided: lambda versions for maximum flexibility with custom layouts, concrete versions for simplicity in common cases.

```cpp
// Lambda (flexible)
ncclLsaReduceSum(coop, srcLambda, nSrc, dstPtr, count);

// Concrete (simple)
ncclLsaReduceSum(coop, src, dstPtr, count, team);
```

#### 2. No Return Codes

Functions return `void` for API simplicity and consistency with existing NCCL device primitives. Error handling via external mechanisms. Future: Optional error checking version.

#### 3. Barriers Separate

Barriers are NOT included in API calls. Users control synchronization explicitly for clarity.

```cpp
bar.sync()
ncclLsaReduceSum(...);
coop.sync()
processData(...);
ncclLsaCopy(...);
bar.sync()
```

#### 4. Combined Unrolling and Vectorization

Single `UNROLL` parameter controls both for API simplicity. Internally optimized for vector loads/stores with sensible default (64 bytes).

#### 5. Precision Handling

Fixed precision relationships matching platform multimem support. No explicit precision parameter yet. Future: May add template parameter for custom precision.

**FP4 support:** FP4 is not supported in the current ReduceCopy API. The design will be revisited to support fp4x2 types in a future revision.

### Data Flow Diagrams

#### ReduceSum Operation (N->1)
```
Rank 0: [A0] \
Rank 1: [A1]  \
Rank 2: [A2]   >--[ Reduce (Sum) ]--> [Result] on Rank 2
Rank 3: [A3]  /
Rank N: [AN] /
```

#### Copy (Broadcast) Operation (1->N)
```
                     /--> [B0] Rank 0
                    /---> [B1] Rank 1
[Source] --[ Copy ]+----> [B2] Rank 2
                    \---> [B3] Rank 3
                     \--> [BN] Rank N
```

#### ReduceSumCopy Operation (N->M)
```
Rank 0: [A0] \                        /--> [C0] Rank 0
Rank 1: [A1]  \                      /---> [C1] Rank 1
Rank 2: [A2]   >--[ Reduce -> Copy ]+----> [C2] Rank 2
Rank 3: [A3]  /                      \---> [C3] Rank 3
Rank N: [AN] /                        \--> [CN] Rank N
```

### API Usage Examples

#### Example 1: Simple ReduceSum
```cpp
// Reduce from all ranks, store on rank 0
__shared__ float reduced[SIZE];
auto mySymPtr = srcSymPtr + globalBlockIdx * SIZE;
ncclLsaReduceSum<float>(
  ncclCoopCta(),
  reduced,                    // destination
  mySymPtr,                  // source
  SIZE,
  myTeam
);
```

#### Example 2: Custom Strided Reduction
```cpp
// Reduce 4 buffers with custom stride
ncclLocalReduceSum<float>(
  ncclCoopThread(),
  4,                         // 4 sources
  basePtr,                   // base pointer
  stride,                    // displacement between sources
  outputPtr,                 // destination
  SIZE
);
```

#### Example 3: ReduceSum + Process + Copy (Broadcast)
```cpp
// AllReduce pattern
ncclLsaReduceSum<float>(coop, srcPtr, tempBuf, count, team);

// Custom processing
processData(tempBuf, count);

ncclLsaCopy<float>(coop, tempBuf, dstPtr, count, team);
```

#### Example 4: Combined ReduceSumCopy
```cpp
// Equivalent to above, but more efficient
ncclLsaReduceSumCopy<float>(
  coop,
  srcPtr,
  dstPtr,
  count,
  team
);
// No intermediate storage or sync needed!
```

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Testing Strategy

#### Unit Tests
- Individual function correctness
- Edge cases (count=0, count=1, single rank)
- All memory model variations

#### Performance Tests
- Bandwidth measurements
- Comparison with NCCL Host API (symmetric kernels)
- Implementation in perf tests as custom kernels for CI & QA to test

#### Integration Tests
- Different cooperation levels
- Mixed operations in single kernel
- Implementation in perf tests as custom kernels for CI & QA to test

### Validation Plan

1. **Functional correctness**: All combinations tested
2. **Performance parity**: Match existing implementations in most cases
3. **Documentation**: Complete API reference with examples

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Implementation Notes</h2></summary>
<!-- ============================================================================================-->

### Development Decisions

Based on design review discussions, the following decisions were made:

#### API Coverage
- [x] Team/DevComm/lambda abstractions provided
- [x] Generic reduceCopy with lambdas and RedOp parameter
- [x] Specialized reduceSumCopy (publicly exposed)
- [x] All 4 LSA/Multimem combinations
- [x] Coop specialization (ncclCoopCta and ncclCoopThread)
- [x] IntCount template argument
- [x] Windows & SymPtr design support
- [x] Multimem pure pointer versions
- [x] Local reduceCopy convenience functions

#### What's NOT Included (Initial Version)
- [ ] Return codes (kept void for now)
- [ ] Short-cut/SOL to avoid if statements alltogether (fixed count & aligned ptrs) API (maybe later)
- [ ] Type whitelist enforcement (compiler handles)
- [ ] Configurable precision (fixed relationships for now)
- [ ] Low-latency version
- [ ] Barriers intertwined with API
- [ ] Use case where ncclCoopCta stores non-continguous memory per thread on stack (RMS out of order)

#### Notes on Architecture Support and Error Handling
This implementation prioritizes compile-time validation where possible, but multi-architecture builds impose limits on how far `static_assert` can be used. In particular, if any compiled target lacks a required feature (notably FP8 multimem), a strict compile-time error would fail the entire build.
Core NCCL mitigates this via separate kernel builds and runtime dispatch based on availability; the device API cannot impose that build structure on downstream users.
Given that constraint, unsupported feature paths use runtime `assert` so multi-arch builds remain valid and supported configurations compile successfully. If a non-supported operation is executed at runtime, the kernel will terminate with a device-side assert.
This is not ideal from a UX perspective, but it is currently the most practical tradeoff without controlling the user build.
For perf tests, the build flow was adjusted to record which architectures were compiled and perform runtime dispatch based on that knowledge, allowing graceful skips on unsupported systems. The same approach is planned for the API tests. When device-side error reporting becomes available, this should be revisited to replace runtime asserts with structured error handling.

#### Naming Conventions
- `ReduceCopy` -> Combined operation
- `Copy` -> 1->N operation (renamed from "Bcast")
- `Lsa` -> Load Store Accessible memory
- `Multimem` -> Multimem memory
- `Local` -> Thread-local operations

### Implementation Macros

The implementation uses the following macro for conditional C++17 `if constexpr` support:

```cpp
// Macro for conditional constexpr support
#if defined(__cpp_if_constexpr) && __cpp_if_constexpr >= 201606
#define NCCL_IF_CONSTEXPR constexpr
#else
#define NCCL_IF_CONSTEXPR
#endif
```

This macro is used throughout the implementation to enable compile-time branch elimination when C++17 is available, while maintaining compatibility with earlier C++ standards. The macro is defined in `reduce_copy__impl.h` and undefined at the end of the file to avoid polluting the global namespace.

The implementation also uses an explicit opt-in macro for unsafe device code:

```cpp
// Permit unsafe device code in NCCL device API
// Users must define this at compile time, e.g. -DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1
#define NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE 1
```

This macro is required because two behaviors rely on ISA characteristics that may change in future architectures:
1) Loading outside of the user-specified range and discarding the extra bytes (acceptable today due to known cache line sizes).
2) Issuing non-multimem instructions on multimem addresses for small (<32-bit) stores, relying on identical SASS encodings.

### Future Possible Enhancements

1. **Additional reduction operations**: Min, Max, Prod, custom ops
2. **Precision control**: Template parameter for accumulation precision
3. **Error handling**: Optional return code versions
4. **Low-latency variants**: For small message sizes
5. **Hybrid memory models**: GIN integration
6. **fp4x2 support**: Revisit design to support fp4x2 types

</details>

---

## API Reference

See the comprehensive API reference table and function definitions below.

---

<details open>
<summary><h2>Complete API Specification (RAW - Will be moved to header files)</h2></summary>

<!-- This section contains the raw API definitions and will be removed once moved to actual header files -->

### API Function Reference Table

| ID | Operation | Memory Type | Lambda/Concrete | Key Signature Parameters |
|----|-----------|-------------|-----------------|--------------------------|
| **[1.x](#series-1x)** | **Generic ReduceCopy (with RedOp)** | | | |
| 1.1 | ReduceCopy | LSA<->LSA | Lambda | `srcLambda, nSrc, dstLambda, nDst, redOp` |
| 1.2 | ReduceCopy | LSA<->Multimem | Lambda | `srcLambda, nSrc, dstLambda, nDst, redOp` |
| ~~1.3~~ | ~~ReduceCopy~~ | ~~Multimem<->LSA~~ | ~~Lambda~~ | *Not available - Multimem source doesn't support generic RedOp* |
| ~~1.4~~ | ~~ReduceCopy~~ | ~~Multimem<->Multimem~~ | ~~Lambda~~ | *Not available - Multimem source doesn't support generic RedOp* |
| **[2.x](#series-2x)** | **Sum-specific ReduceCopy** | | | |
| 2.1 | ReduceSumCopy | LSA<->LSA | Lambda | `srcLambda, nSrc, dstLambda, nDst` |
| 2.2 | ReduceSumCopy | LSA<->Multimem | Lambda | `srcLambda, nSrc, dstLambda, nDst` |
| 2.3 | ReduceSumCopy | Multimem<->LSA | Lambda | `srcLambda, nSrc, dstLambda, nDst` |
| 2.4 | ReduceSumCopy | Multimem<->Multimem | Lambda | `srcLambda, nSrc, dstLambda, nDst` |
| **[3.x](#series-3x)** | **ReduceSum (N->1)** | | | |
| 3.1 | ReduceSum | LSA | Lambda | `srcLambda, nSrc, dstPtr` |
| 3.2a | ReduceSum | LSA | ncclSymPtr | `src, dstPtr, team` |
| 3.2b | ReduceSum | LSA | ncclDevComm_t | `src, dstPtr, devComm` |
| 3.2c | ReduceSum | LSA | ncclWindow_t + ncclTeam | `window, offset, dstPtr, team` |
| 3.2d | ReduceSum | LSA | ncclWindow_t + ncclDevComm_t | `window, offset, dstPtr, devComm` |
| 3.3a | ReduceSum | Multimem | ncclSymPtr | `src, dstPtr, multimemHandle` |
| 3.3b | ReduceSum | Multimem | Raw pointer | `mcSrcPtr, dstPtr` |
| 3.3c | ReduceSum | Multimem | ncclWindow_t | `window, offset, dstPtr, multimemHandle` |
| 3.4 | ReduceSum | Local | Lambda | `srcLambda, nSrc, dstPtr` |
| 3.5 | ReduceSum | Local | Strided | `nSrc, basePtr, displ, dstPtr` |
| **[4.x](#series-4x)** | **Copy (1->N)** | | | |
| 4.1 | Copy | LSA | Lambda | `srcPtr, dstLambda, nDst` |
| 4.2a | Copy | LSA | ncclSymPtr | `srcPtr, dst, team` |
| 4.2b | Copy | LSA | ncclDevComm_t | `srcPtr, dst, devComm` |
| 4.2c | Copy | LSA | ncclWindow_t + ncclTeam | `srcPtr, window, offset, team` |
| 4.2d | Copy | LSA | ncclWindow_t + ncclDevComm_t | `srcPtr, window, offset, devComm` |
| 4.3a | Copy | Multimem | ncclSymPtr | `srcPtr, dst, multimemHandle` |
| 4.3b | Copy | Multimem | Raw pointer | `srcPtr, mcDstPtr` |
| 4.3c | Copy | Multimem | ncclWindow_t | `srcPtr, window, offset, multimemHandle` |
| 4.4 | Copy | Local | Lambda | `srcPtr, dstLambda, nDst` |
| 4.5 | Copy | Local | Strided | `srcPtr, nDst, basePtr, displ` |
| **[5.x](#series-5x)** | **ReduceSumCopy (N->M)** | | | |
| 5.1a | ReduceSumCopy | LSA | Same team | `src, dst, team` |
| 5.1b | ReduceSumCopy | LSA | ncclDevComm_t | `src, dst, devComm` |
| 5.1c | ReduceSumCopy | LSA | ncclWindow_t + ncclTeam | `srcWindow, srcOffset, dstWindow, dstOffset, team` |
| 5.1d | ReduceSumCopy | LSA | ncclWindow_t + ncclDevComm_t | `srcWindow, srcOffset, dstWindow, dstOffset, devComm` |
| 5.1e | ReduceSumCopy | LSA | Different teams | `src, srcTeam, dst, dstTeam` |
| 5.2a | ReduceSumCopy | Multimem | ncclSymPtr | `src, srcHandle, dst, dstHandle` |
| 5.2b | ReduceSumCopy | Multimem | Raw pointer | `mcSrcPtr, mcDstPtr` |
| 5.2c | ReduceSumCopy | Multimem | ncclWindow_t | `srcWindow, srcOffset, srcHandle, dstWindow, dstOffset, dstHandle` |
| 5.3a | ReduceSumCopy | LSA->Multimem | ncclSymPtr | `src, srcTeam, dst, dstHandle` |
| 5.3b | ReduceSumCopy | LSA->Multimem | Raw dst | `src, srcTeam, mcDstPtr` |
| 5.3c | ReduceSumCopy | Multimem->LSA | ncclSymPtr | `src, srcHandle, dst, dstTeam` |
| 5.3d | ReduceSumCopy | Multimem->LSA | Raw src | `mcSrcPtr, dst, dstTeam` |
| 5.4 | ReduceSumCopy | Local | Strided | `nSrc, srcBasePtr, srcDispl, nDst, dstBasePtr, dstDispl` |

**Notes:**
- All functions take `Coop coop`, `IntCount count` parameters (omitted from table for brevity)
- Template parameters: `T, Coop, IntCount, UNROLL=4*16/sizeof(T)` (with additional lambda types where applicable)
- Generic versions (1.x) also include `RedOp const& redOp` parameter
- Lambda versions provide maximum flexibility; concrete versions are convenience wrappers

### Parameter Naming and Ordering Conventions

The API follows consistent conventions aligned with C++ standard library practices:

#### Parameter Ordering

All functions follow this canonical order:
1. **Cooperation context**: `Coop coop` (always first)
2. **Source parameters**: All source-related parameters in this order:
   - Source data: `srcLambda`, `srcPtr`, `src`, `mcSrcPtr`, `window`/`offset`, `basePtr`
   - Source count: `nSrc` (for lambda-based APIs)
   - Source context: `srcTeam`, `srcHandle` (when different from destination)
3. **Destination parameters**: All destination-related parameters in this order:
   - Destination data: `dstLambda`, `dstPtr`, `dst`, `mcDstPtr`
   - Destination count: `nDst` (for lambda-based APIs)
   - Destination context: `dstTeam`, `dstHandle` (when different from source)
4. **Count**: `IntCount count` (operation size)
5. **Shared context**: `team`, `devComm`, `multimemHandle` (when source and destination share the same context)
6. **Reduction operation**: `RedOp const& redOp` (only in generic 1.x APIs, omitted in specialized APIs)

**Rationale**: This ordering follows the std::copy and std::transform convention where source precedes destination, making the data flow direction immediately clear.

#### Naming Conventions

**Pointers:**
- `srcPtr`, `dstPtr`: Local raw pointers (`T*`)
- `mcSrcPtr`, `mcDstPtr`: Multimem raw pointers (`T*`) - the `mc` prefix distinguishes multimem pointers
- `src`, `dst`: Symmetric pointer abstractions (`ncclSymPtr<T>`)
- `basePtr`: Base pointer for strided operations

**Lambdas:**
- `srcLambda`, `dstLambda`: Lambda functions `[int rank/index] -> T*`

**Context:**
- `team`: Shared team for both source and destination
- `srcTeam`, `dstTeam`: Separate teams when source and destination differ
- `devComm`: Device communicator (`ncclDevComm_t`)
- `multimemHandle`: Shared multimem handle for both source and destination
- `srcHandle`, `dstHandle`: Separate multimem handles when source and destination differ

**Counts:**
- `nSrc`, `nDst`: Number of source/destination pointers in lambda-based APIs
- `count`: Number of elements to process (always `IntCount` type)

**Windows:**
- `window`: Shared window for source and destination
- `srcWindow`, `dstWindow`: Separate windows when source and destination differ
- `offset`, `srcOffset`, `dstOffset`: Offsets within windows

**Displacement:**
- `displ`, `srcDispl`, `dstDispl`: Stride between consecutive chunks in strided operations

**Rationale**: Consistent naming makes the API predictable. The `src`/`dst` prefix pattern and the `mc` prefix for multimem raw pointers help developers quickly understand parameter roles without consulting documentation.

### Function Definitions

#### Implementation Detail (Not Public API)

```cpp
// Generic everything reduce-copy: not public API (implementation detail)
template <typename T, typename Coop, bool srcMultimem, bool dstMultimem,
          typename SrcLambda, typename DstLambda, typename RedOp,
          typename IntCount, int UNROLL>
__device__ __inline__ void reduceCopy(Coop coop,
                                      SrcLambda srcLambda, int nSrc,
                                      DstLambda dstLambda, int nDst,
                                      RedOp const& redOp,
                                      IntCount count);
```

---

<a id="series-1x"></a>
### Series 1.x - Generic ReduceCopy (with RedOp)

Lambda-based versions with explicit `RedOp` parameter. Only LSA sources supported.

```cpp
// [ID 1.1] LSA <-> LSA version
template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename RedOp, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceLsaCopy(Coop coop,
                                             SrcLambda srcLambda, int nSrc,
                                             DstLambda dstLambda, int nDst,
                                             RedOp const& redOp, IntCount count) {
  reduceCopy<T, Coop, /*srcMultimem*/ false, /*dstMultimem*/ false, SrcLambda, DstLambda, RedOp, IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, redOp, count);
}

// We have specialization of this for for example fp8 with the precision logic baked in
template <typename T>
struct ncclOpSum {
    constexpr T operator()(const T& a, const T& b) const {
        return a + b;
    }
};

// Note: Multimem SOURCE does not support generic RedOp (only ncclOpSum<T>)
// Therefore, only LSA source versions are provided here for generic ReduceCopy

// [ID 1.2] LSA <-> Multimem version
template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename RedOp, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceMultimemCopy(Coop coop,
                                                  SrcLambda srcLambda, int nSrc,
                                                  DstLambda dstLambda, int nDst,
                                                  RedOp const& redOp, IntCount count) {
  reduceCopy<T, Coop, /*srcMultimem*/ false, /*dstMultimem*/ true, SrcLambda, DstLambda, RedOp, IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, redOp, count);
}

// NOTE: Multimem <-> LSA and Multimem <-> Multimem versions (IDs 1.3, 1.4) are NOT provided
// for generic RedOp because Multimem source only supports ncclOpSum.
// Use the Sum-specific versions (IDs 2.3, 2.4) instead.
```

---

<a id="series-2x"></a>
### Series 2.x - Sum-Specific ReduceCopy

Lambda-based versions with `ncclOpSum<T>` baked in. All LSA/Multimem combinations supported.

```cpp
// [ID 2.1] LSA <-> LSA ReduceSum version
template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumLsaCopy(Coop coop,
                                                SrcLambda srcLambda, int nSrc,
                                                DstLambda dstLambda, int nDst,
                                                IntCount count) {
  ncclLsaReduceLsaCopy<T, Coop, SrcLambda, DstLambda, ncclOpSum<T>, IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, ncclOpSum<T>{}, count);
}

// [ID 2.2] LSA <-> Multimem ReduceSum version
template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop coop,
                                                     SrcLambda srcLambda, int nSrc,
                                                     DstLambda dstLambda, int nDst,
                                                     IntCount count) {
  ncclLsaReduceMultimemCopy<T, Coop, SrcLambda, DstLambda, ncclOpSum<T>, IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, ncclOpSum<T>{}, count);
}

// [ID 2.3] Multimem <-> LSA ReduceSum version
template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop coop,
                                                     SrcLambda srcLambda, int nSrc,
                                                     DstLambda dstLambda, int nDst,
                                                     IntCount count) {
  ncclMultimemReduceLsaCopy<T, Coop, SrcLambda, DstLambda, ncclOpSum<T>, IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, ncclOpSum<T>{}, count);
}

// [ID 2.4] Multimem <-> Multimem ReduceSum version
template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumMultimemCopy(Coop coop,
                                                          SrcLambda srcLambda, int nSrc,
                                                          DstLambda dstLambda, int nDst,
                                                          IntCount count) {
  ncclMultimemReduceMultimemCopy<T, Coop, SrcLambda, DstLambda, ncclOpSum<T>, IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, ncclOpSum<T>{}, count);
}
```

---

<a id="series-3x"></a>
### Series 3.x - ReduceSum (N->1)

Specialized for reducing from N sources to 1 destination. Includes LSA, Multimem, and Local variations.

```cpp
// [ID 3.1] LSA ReduceSum: N sources -> 1 local destination (lambda-based)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop,
                                         SrcLambda srcLambda, int nSrc,
                                         T* dstPtr,
                                         IntCount count) {
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };
  constexpr int nDst = 1;  // Reduce has single destination
  ncclLsaReduceSumLsaCopy<T, Coop, SrcLambda, decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// Concrete pointer versions

// [ID 3.2a] LSA ReduceSum: N sources from team -> 1 local destination (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop,
                                         ncclSymPtr<T> src,
                                         T* dstPtr,
                                         IntCount count,
                                         ncclTeam team) {
  // Create lambda: N sources from team
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(team, i);
  };

  ncclLsaReduceSum<T, Coop, decltype(srcLambda), IntCount, UNROLL>(coop, srcLambda, team.nRanks, dstPtr, count);
}

// [ID 3.2b] LSA ReduceSum: N sources from devComm -> 1 local destination (with ncclDevComm_t)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop,
                                         ncclSymPtr<T> src,
                                         T* dstPtr,
                                         IntCount count,
                                         ncclDevComm_t devComm) {
  // Extract team from devComm
  ncclTeam team = ncclTeamLsa(devComm);

  ncclLsaReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, team);
}

// [ID 3.2c] LSA ReduceSum: N sources from window+offset -> 1 local destination (with ncclWindow_t + ncclTeam)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop,
                                         ncclWindow_t window,
                                         size_t offset,
                                         T* dstPtr,
                                         IntCount count,
                                         ncclTeam team) {
  // Construct ncclSymPtr from window and offset
  ncclSymPtr<T> src{window, offset};

  ncclLsaReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, team);
}

// [ID 3.2d] LSA ReduceSum: N sources from window+offset -> 1 local destination (with ncclWindow_t + ncclDevComm_t)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop,
                                         ncclWindow_t window,
                                         size_t offset,
                                         T* dstPtr,
                                         IntCount count,
                                         ncclDevComm_t devComm) {
  // Construct ncclSymPtr from window and offset
  ncclSymPtr<T> src{window, offset};

  ncclLsaReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, devComm);
}

// [ID 3.3a] Multimem ReduceSum: 1 multimem source -> 1 local destination (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop coop,
                                              ncclSymPtr<T> src,
                                              T* dstPtr,
                                              IntCount count,
                                              ncclMultimemHandle multimemHandle) {
  // Create lambdas for 1 source and 1 destination
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return src.multimemPtr(multimemHandle);
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };

  constexpr int nSrc = 1;  // Single multimem source
  constexpr int nDst = 1;  // Single destination
  ncclMultimemReduceSumLsaCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 3.3b] Multimem ReduceSum: 1 multimem source -> 1 local destination (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop coop,
                                              T* mcSrcPtr,
                                              T* dstPtr,
                                              IntCount count) {
  // Create lambdas for 1 source and 1 destination
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcSrcPtr;
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };

  constexpr int nSrc = 1;  // Single multimem source
  constexpr int nDst = 1;  // Single destination
  ncclMultimemReduceSumLsaCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 3.3c] Multimem ReduceSum: 1 multimem source -> 1 local destination (with ncclWindow_t)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop coop,
                                              ncclWindow_t window,
                                              size_t offset,
                                              T* dstPtr,
                                              IntCount count,
                                              ncclMultimemHandle multimemHandle) {
  // Construct ncclSymPtr from window and offset
  ncclSymPtr<T> src{window, offset};

  ncclMultimemReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, multimemHandle);
}

// 3) Local ReduceSum: N local chunks -> 1 local destination

// [ID 3.4] Lambda-based version
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(Coop coop,
                                           SrcLambda srcLambda, int nSrc,
                                           T* dstPtr,
                                           IntCount count) {
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };
  constexpr int nDst = 1;  // Reduce has single destination
  ncclLsaReduceSumLsaCopy<T, Coop, SrcLambda, decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 3.5] Concrete pointer version: reduce n chunks separated by displacement
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(Coop coop,
                                           int nSrc,
                                           T* basePtr,
                                           size_t displ,
                                           T* dstPtr,
                                           IntCount count) {
  // Create lambda: n local sources separated by displ
  auto srcLambda = [=] __device__ (int i) -> T* {
    return basePtr + i * displ;
  };

  ncclLocalReduceSum<T, Coop, decltype(srcLambda), IntCount, UNROLL>(coop, srcLambda, nSrc, dstPtr, count);
}
```

---

<a id="series-4x"></a>
### Series 4.x - Copy (Broadcast) (1->N)

Specialized for copying from 1 source to N destinations. Includes LSA, Multimem, and Local variations.

```cpp
// [ID 4.1] Lambda-based version
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop,
                                    T* srcPtr,
                                    DstLambda dstLambda, int nDst,
                                    IntCount count) {
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  constexpr int nSrc = 1;  // Copy has single source
  ncclLsaReduceSumLsaCopy<T, Coop, decltype(srcLambda), DstLambda, IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// Concrete pointer versions

// [ID 4.2a] LSA Copy: 1 local source -> N destinations (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop,
                                    T* srcPtr,
                                    ncclSymPtr<T> dst,
                                    IntCount count,
                                    ncclTeam team) {
  // Create lambda: N destinations to team
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(team, i);
  };

  ncclLsaCopy<T, Coop, decltype(dstLambda), IntCount, UNROLL>(coop, srcPtr, dstLambda, team.nRanks, count);
}

// [ID 4.2b] LSA Copy: 1 local source -> N destinations (with ncclDevComm_t)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop,
                                    T* srcPtr,
                                    ncclSymPtr<T> dst,
                                    IntCount count,
                                    ncclDevComm_t devComm) {
  // Extract team from devComm
  ncclTeam team = ncclTeamLsa(devComm);

  ncclLsaCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, team);
}

// [ID 4.2c] LSA Copy: 1 local source -> N destinations (with ncclWindow_t + ncclTeam)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop,
                                    T* srcPtr,
                                    ncclWindow_t window,
                                    size_t offset,
                                    IntCount count,
                                    ncclTeam team) {
  // Construct ncclSymPtr from window and offset
  ncclSymPtr<T> dst{window, offset};

  ncclLsaCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, team);
}

// [ID 4.2d] LSA Copy: 1 local source -> N destinations (with ncclWindow_t + ncclDevComm_t)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop,
                                    T* srcPtr,
                                    ncclWindow_t window,
                                    size_t offset,
                                    IntCount count,
                                    ncclDevComm_t devComm) {
  // Construct ncclSymPtr from window and offset
  ncclSymPtr<T> dst{window, offset};

  ncclLsaCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, devComm);
}

// 2) Multimem Copy: 1 local source -> 1 multimem destination

// [ID 4.3a] Multimem Copy: 1 local source -> 1 multimem destination (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop coop,
                                         T* srcPtr,
                                         ncclSymPtr<T> dst,
                                         IntCount count,
                                         ncclMultimemHandle multimemHandle) {
  // Create lambdas for 1 source and 1 destination
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dst.multimemPtr(multimemHandle);
  };

  constexpr int nSrc = 1;  // Single source
  constexpr int nDst = 1;  // Single multimem destination
  ncclLsaReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 4.3b] Multimem Copy: 1 local source -> 1 multimem destination (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop coop,
                                         T* srcPtr,
                                         T* mcDstPtr,
                                         IntCount count) {
  // Create lambdas for 1 source and 1 destination
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcDstPtr;
  };

  constexpr int nSrc = 1;  // Single source
  constexpr int nDst = 1;  // Single multimem destination
  ncclLsaReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 4.3c] Multimem Copy: 1 local source -> 1 multimem destination (with ncclWindow_t)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop coop,
                                         T* srcPtr,
                                         ncclWindow_t window,
                                         size_t offset,
                                         IntCount count,
                                         ncclMultimemHandle multimemHandle) {
  // Construct ncclSymPtr from window and offset
  ncclSymPtr<T> dst{window, offset};

  ncclMultimemCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, multimemHandle);
}

// 3) Local Copy: 1 source -> N local destinations

// [ID 4.4] Lambda-based version
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalCopy(Coop coop,
                                      T* srcPtr,
                                      DstLambda dstLambda, int nDst,
                                      IntCount count) {
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  constexpr int nSrc = 1;  // Copy has single source
  ncclLsaReduceSumLsaCopy<T, Coop, decltype(srcLambda), DstLambda, IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 4.5] Concrete pointer version: copy to n chunks separated by displacement
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalCopy(Coop coop,
                                      T* srcPtr,
                                      int nDst,
                                      T* basePtr,
                                      size_t displ,
                                      IntCount count) {
  // Create lambda: n local destinations separated by displ
  auto dstLambda = [=] __device__ (int i) -> T* {
    return basePtr + i * displ;
  };

  ncclLocalCopy<T, Coop, decltype(dstLambda), IntCount, UNROLL>(coop, srcPtr, dstLambda, nDst, count);
}
```

---

<a id="series-5x"></a>
### Series 5.x - ReduceSumCopy (N->M)

Combined reduce and copy operations. Includes LSA, Multimem, Mixed, and Local variations.

**Note:** Lambda-based versions are in Series 2.x (`ncclLsaReduceSumLsaCopy`, etc.)

```cpp
// [ID 5.1a] LSA ReduceSumCopy: same team for src and dst (most common case)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop,
                                             ncclSymPtr<T> src,
                                             ncclSymPtr<T> dst,
                                             IntCount count,
                                             ncclTeam team) {
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(team, i);
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(team, i);
  };

  ncclLsaReduceSumLsaCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, team.nRanks, dstLambda, team.nRanks, count);
}

// [ID 5.1b] LSA ReduceSumCopy: with ncclDevComm_t (extract team)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop,
                                             ncclSymPtr<T> src,
                                             ncclSymPtr<T> dst,
                                             IntCount count,
                                             ncclDevComm_t devComm) {
  ncclTeam team = ncclTeamLsa(devComm);
  ncclLsaReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, dst, count, team);
}

// [ID 5.1c] LSA ReduceSumCopy: with windows + ncclTeam
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop,
                                             ncclWindow_t srcWindow, size_t srcOffset,
                                             ncclWindow_t dstWindow, size_t dstOffset,
                                             IntCount count,
                                             ncclTeam team) {
  ncclSymPtr<T> src{srcWindow, srcOffset};
  ncclSymPtr<T> dst{dstWindow, dstOffset};
  ncclLsaReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, dst, count, team);
}

// [ID 5.1d] LSA ReduceSumCopy: with windows + ncclDevComm_t
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop,
                                             ncclWindow_t srcWindow, size_t srcOffset,
                                             ncclWindow_t dstWindow, size_t dstOffset,
                                             IntCount count,
                                             ncclDevComm_t devComm) {
  ncclSymPtr<T> src{srcWindow, srcOffset};
  ncclSymPtr<T> dst{dstWindow, dstOffset};
  ncclLsaReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, dst, count, devComm);
}

// [ID 5.1e] LSA ReduceSumCopy: different teams for src and dst (advanced use case)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop,
                                             ncclSymPtr<T> src, ncclTeam srcTeam,
                                             ncclSymPtr<T> dst, ncclTeam dstTeam,
                                             IntCount count) {
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(srcTeam, i);
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(dstTeam, i);
  };

  ncclLsaReduceSumLsaCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, srcTeam.nRanks, dstLambda, dstTeam.nRanks, count);
}

// 2) Multimem ReduceSumCopy: 1 multimem source -> 1 multimem destination

// [ID 5.2a] Multimem ReduceSumCopy (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop coop,
                                                  ncclSymPtr<T> src, ncclMultimemHandle srcHandle,
                                                  ncclSymPtr<T> dst, ncclMultimemHandle dstHandle,
                                                  IntCount count) {
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return src.multimemPtr(srcHandle);
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dst.multimemPtr(dstHandle);
  };

  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  ncclMultimemReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 5.2b] Multimem ReduceSumCopy (with raw pointers)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop coop,
                                                  T* mcSrcPtr,
                                                  T* mcDstPtr,
                                                  IntCount count) {
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcSrcPtr;
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcDstPtr;
  };

  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  ncclMultimemReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 5.2c] Multimem ReduceSumCopy (with ncclWindow_t)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop coop,
                                                  ncclWindow_t srcWindow, size_t srcOffset, ncclMultimemHandle srcHandle,
                                                  ncclWindow_t dstWindow, size_t dstOffset, ncclMultimemHandle dstHandle,
                                                  IntCount count) {
  // Construct ncclSymPtr from window and offset
  ncclSymPtr<T> src{srcWindow, srcOffset};
  ncclSymPtr<T> dst{dstWindow, dstOffset};

  ncclMultimemReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, srcHandle, dst, dstHandle, count);
}

// 3) Mixed LSA/Multimem ReduceSumCopy variations

// [ID 5.3a] LSA source -> Multimem destination
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop coop,
                                                     ncclSymPtr<T> src, ncclTeam srcTeam,
                                                     ncclSymPtr<T> dst, ncclMultimemHandle dstHandle,
                                                     IntCount count) {
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(srcTeam, i);
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dst.multimemPtr(dstHandle);
  };

  constexpr int nDst = 1;
  ncclLsaReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, srcTeam.nRanks, dstLambda, nDst, count);
}

// [ID 5.3b] LSA source -> Multimem destination (with raw dst pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop coop,
                                                     ncclSymPtr<T> src, ncclTeam srcTeam,
                                                     T* mcDstPtr,
                                                     IntCount count) {
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(srcTeam, i);
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcDstPtr;
  };

  constexpr int nDst = 1;
  ncclLsaReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, srcTeam.nRanks, dstLambda, nDst, count);
}

// [ID 5.3c] Multimem source -> LSA destination
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop coop,
                                                     ncclSymPtr<T> src, ncclMultimemHandle srcHandle,
                                                     ncclSymPtr<T> dst, ncclTeam dstTeam,
                                                     IntCount count) {
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return src.multimemPtr(srcHandle);
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(dstTeam, i);
  };

  constexpr int nSrc = 1;
  ncclMultimemReduceSumLsaCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, dstTeam.nRanks, count);
}

// [ID 5.3d] Multimem source -> LSA destination (with raw src pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop coop,
                                                     T* mcSrcPtr,
                                                     ncclSymPtr<T> dst, ncclTeam dstTeam,
                                                     IntCount count) {
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcSrcPtr;
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(dstTeam, i);
  };

  constexpr int nSrc = 1;
  ncclMultimemReduceSumLsaCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, dstTeam.nRanks, count);
}

// 4) Local ReduceSumCopy: N local sources -> M local destinations
// (Lambda-based version is ncclLsaReduceSumLsaCopy at the top)

// [ID 5.4] Concrete pointer version: reduce n src chunks and copy to m dst chunks
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalReduceSumCopy(Coop coop,
                                               int nSrc, T* srcBasePtr, size_t srcDispl,
                                               int nDst, T* dstBasePtr, size_t dstDispl,
                                               IntCount count) {
  auto srcLambda = [=] __device__ (int i) -> T* {
    return srcBasePtr + i * srcDispl;
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dstBasePtr + i * dstDispl;
  };

  ncclLsaReduceSumLsaCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(coop, srcLambda, nSrc, dstLambda, nDst, count);
}
```

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Coding</h2></summary>
<!-- ============================================================================================-->

### Commit list or MR
https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/1564

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Ludwig Schneider

</details>
