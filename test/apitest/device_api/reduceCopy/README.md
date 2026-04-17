# reduceCopy device API tests

Tests for the NCCL device-side collective API (`ncclLsa*`, `ncclLocalReduce*`, `ncclMultimem*`).
The suite covers every combination of API function × data type × cooperation level × unroll factor × element count, and fails if any expected combination was never executed.

---

## Directory layout

```
reduceCopy/
├── README.md                      # this file
├── run_reducecopy_types.sh        # top-level driver (full run or single type)
├── generate_tests.py              # code generator: emits suite_*.cu and inst_*.cu
│
├── api_function_traits.h          # X-macro registry of 39 API functions + their flags
├── test_type_traits.h             # X-macro registry of 12 data types + their flags
├── config.h                       # element counts, cooperation levels, unroll helpers
│
├── common.cuh                     # GTest fixture base (SetUpTestCase / TearDownTestCase)
├── checks.h                       # NCCLCHECK / CUDACHECK macros (throw on error)
├── support.h                      # runtime feature detection (multimem, P2P, device API)
├── util.h                         # env-var helpers (verbose, seed, missing-tests list)
│
├── test_params.h                  # TestParams struct + INSTANTIATE parameter builders
├── test_types.h                   # TYPED_TEST_CASE instantiation (all 12 types)
├── test_functions.h               # thin wrappers around api_function_traits.h predicates
├── test_matrix.h                  # coverage tracker (7-dimensional expected vs. executed)
├── test_matrix_listener.cu        # GTest listener: prints coverage report + fails on gaps
│
├── factories.h                    # memory allocation helpers (Window, Lambda, Local)
├── data.h                         # deterministic test data initialisation
├── reference.h                    # CPU reference implementations (sum, copy, gather, …)
├── reference_gpu.cuh / .cu        # GPU reference kernels (used for FP8)
├── test_compare.h                 # ULP-based output comparison
├── test_nccl_reference.h          # optional NCCL-level reference (allreduce / scatter / …)
├── acc_type_trait.cuh             # AccumulatorType mapping (half→float, fp8→half, …)
│
├── test_reduce_copy.h             # runFullTestImpl<T>: orchestrates one test end-to-end
├── test_dispatch.h                # three-layer template dispatch (unroll → coop → funcId)
├── kernels.cuh                    # CUDA kernel launchers for every API variant
│
├── nm_reduce_sum_copy_kernels.cuh # explicit N→M kernel implementations
└── nm_reduce_sum_copy_test.cu     # 15 hand-written N→M test cases (Split A / B / Multimem)
```

Generated at build time into `build/test/apitest/device_api/reduceCopy/generated/`:

```
suite_<TYPE>.cu                    # INSTANTIATE_TEST_CASE_P blocks for one type
inst_sum_<TYPE>_<UNROLL>_<COOP>.cu       # explicit kernel instantiations
inst_allreduce_<TYPE>_<UNROLL>_<COOP>.cu
inst_gather_<TYPE>_<UNROLL>_<COOP>.cu
```

---

## Building

```sh
make -C test/apitest/device_api
```

To build for a subset of types:

```sh
NCCL_TEST_REDUCE_COPY_TYPES=Float make -C test/apitest/device_api
```

Binary: `build/test/apitest/device_api/device_api_test`

---

## Running tests

Full run — loops over every type, stops at first failure:

```sh
./test/apitest/device_api/reduceCopy/run_reducecopy_types.sh
# or with srun:
srun -n 1 --label ./test/apitest/device_api/reduceCopy/run_reducecopy_types.sh
```

Single type (`-t TYPE`):

```sh
./test/apitest/device_api/reduceCopy/run_reducecopy_types.sh -t Float
```

Valid type tags: `Int Uint Int8 Uint8 LongLong ULongLong Float Double Half Bf16 Fp8E4M3 Fp8E5M2`

Narrow to a GTest filter (first positional arg after any flags):

```sh
./run_reducecopy_types.sh -t Float '*Thread*'
./run_reducecopy_types.sh -t Float '*LsaReduceMultimemCopy_Generic_Thread_UNROLL1_count997_gpus3*'
```

Run the binary directly:

```sh
NCCL_TEST_REDUCE_COPY_TYPES=Float \
  ./build/test/apitest/device_api/device_api_test \
  --gtest_filter='*ReduceCopy_Float.*' --gtest_color=yes
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `NCCL_TEST_REDUCE_COPY_TYPES` | *(all)* | Comma-separated type tags controlling which types are generated at build time and expected by the coverage tracker at runtime. `all` selects all types explicitly. |
| `NCCL_TEST_VERBOSE` | *(off)* | `1` or `true` to enable per-test `[TEST] …` progress output. |
| `NCCL_TEST_REDUCE_COPY_SEED` | `0x1234567890ABCDEF` | RNG seed for test data generation. |
| `NCCL_TEST_REDUCE_COPY_LIST_MISSING` | *(off)* | `1` or `true` to print every expected-but-not-executed combination in the coverage report. |

---

## Adding a new API function

1. Add an entry to `NCCL_REDUCE_COPY_API_FUNC_LIST` in `api_function_traits.h` and increment `NCCL_REDUCE_COPY_API_FUNC_COUNT`.
2. Add a corresponding RST anchor in `docs/userguide/source/api/device_reducecopy.rst` and add the mapping to `_RST_LABEL_TO_API_IDS` in `generate_tests.py`.
3. Add a kernel launcher case in `kernels.cuh`.
4. Rebuild — `generate_tests.py` regenerates the suite and instantiation files automatically.

## Adding a new data type

1. Add an entry to `NCCL_REDUCE_COPY_TYPE_LIST_ALL` in `test_type_traits.h`.
2. Add accumulator mapping in `acc_type_trait.cuh` if needed.
3. Add comparison tolerance in `test_compare.h` (`OutputMantissaBits<T>`).
4. Add the type tag to `_TEST_TAG_TO_RST_TYPE` in `generate_tests.py`.
5. Rebuild.

---

## Implementation: how a test executes

This section walks through the full call chain from binary startup to result verification.

### 1. Code generation (build time)

Before compilation, `generate_tests.py` reads two C++ headers as its single source of truth:

- `api_function_traits.h` — parses `NCCL_REDUCE_COPY_API_FUNC_LIST` to get the ordered list
  of 39 API functions and their per-function flags (lambda, local, multimem source/dest,
  mul-variant, category).
- `test_type_traits.h` — parses `NCCL_REDUCE_COPY_TYPE_LIST_ALL` to get the 12 types and
  their per-type flags (multimem-source supported, mul supported).
- `config.h` — reads `NCCL_REDUCE_COPY_COUNTS_LIST` for the five element counts.

For each selected type it emits:

- **`suite_<TYPE>.cu`** — one `INSTANTIATE_TEST_CASE_P` block per (function, cooperation level)
  pair, with every (count × unroll × maxGpus × lambdaOffsets) combination as a parameter row.
  Unsupported combinations (e.g. multimem functions for types without multimem, mul variants
  for FP8) are silently omitted, mirroring the runtime exclusion logic in
  `TestMatrix::getExpectedTests()`.
- **`inst_sum_<TYPE>_<UNROLL>_<COOP>.cu`**, **`inst_allreduce_…`**, **`inst_gather_…`** — one
  explicit template instantiation each for `launchReduceSumKernel`, `launchAllReduceKernel`,
  and `launchAllGatherKernel`.  Splitting instantiations into separate translation units
  keeps individual compile times manageable.

### 2. GTest fixture lifecycle

Each type gets its own GTest test suite (`ReduceCopy_Float`, `ReduceCopy_Half`, …).  All
suites inherit from `ReduceCopyTestBase<T>` which manages shared state as static members:

```
SetUpTestCase()          — runs once per type
  cudaGetDeviceCount()   → nVis
  ncclCommInitAll()      → comms[nVis]
  cudaStreamCreate()     → streams[nVis]

  for each parameterised test:
    SetUp()              → sync all streams (PerTestSetUp)
    TEST_P body          → runFullTest(params)
    TearDown()           → sync all streams + devices (PerTestTearDown)

TearDownTestCase()       — runs once per type
  stream destroy, ncclCommDestroy
```

`devComms` (device communicators) are created fresh inside each test because different tests
require different `ncclDevCommRequirements` (e.g. multimem tests need `lsaMultimem = true`;
barrier count matches the grid size for that test).

### 3. Test parameter structure

Each parameterised test row is a `TestParams`:

```
funcId          — which of the 39 ApiFunctionId variants to call
count           — element count (0, 1, 2, 997, 32768)
coopLevel       — Thread / Warp / Cta
unroll          — 1 / getDefaultUnroll<T>() / getDoubleUnroll<T>()
maxTestGpus     — 0 (use all visible GPUs), 1 (local), or 3
enableLambdaOffsets — whether lambda variants use per-rank byte offsets
gridSize        — number of CUDA blocks (default: nVis)
blockSize       — threads per block (depends on coopLevel)
```

`getDefaultUnroll<T>()` is `512 / bitSizeOfElement<T>()` (64 bytes of elements).
`getDoubleUnroll<T>()` is `2 × default`.

### 4. `runFullTestImpl<T>` — test orchestration

`test_reduce_copy.h` implements the main test body:

**Step 1 — support checks.**  Skip the test (no failure) if:
- the type is not supported on this architecture
- the function requires mul but the type doesn't support it
- the device API is not enabled on this communicator
- inter-rank test but P2P is not connected
- multimem is required but not supported, or the type lacks multimem specialisation

**Step 2 — adapt nSrc / nDst.**  The raw params may say `nSrc=0` (meaning "all GPUs").
The function resolves this:
- ReduceSum semantics: `nSrc = nDst = numDevices` (ReduceScatter-style, each rank receives its chunk)
- AllGather: `nSrc = nDst = numDevices`
- ReduceSumCopy with multimem source: `nSrc = 1` (multicast handle counts as one source)
- ReduceSumCopy with multimem dest: `nDst = 1`

**Step 3 — create device communicators.**

```cpp
ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.lsaBarrierCount = gridSize;   // one barrier slot per CTA
reqs.lsaMultimem     = requiresMultimem;
ncclGroupStart();
for each device: ncclDevCommCreate(comms[i], &reqs, &devComms[i]);
ncclGroupEnd();   // blocks until all ranks have joined
```

**Step 4 — allocate memory via factory.**  `createInputFactory<T>()` selects the factory
class based on the function's traits:

| Factory class | Used for |
|---|---|
| `LocalStridedInputFactory<T>` | `local` functions (single-device, strided chunks) |
| `ReduceSumInputFactory<T>` | ReduceSum category (N-to-1: nSrc chunks in send, 1 chunk in recv) |
| `LambdaInputFactory<T>` | lambda functions in GenericReduceCopy / ReduceSumCopy / AllGather |
| `WindowInputFactory<T>` | everything else (window / symptr variants, N→M) |

Each factory calls `ncclWindowCreate` + `ncclWindowRegister` to create symmetric
(`ncclWindow_t`) allocations backed by CUDA device memory, and fills them with
deterministic test data from `data.h`.

**Step 5 — compute reference output (CPU).**  `reference.h` runs the expected operation on
host memory using the accumulator type for the element type (`float` for `half`/`bf16`,
`half` for FP8, the type itself otherwise):
- *ReduceSum*: sum across nSrc source arrays → one output array
- *Copy*: memcpy from source to nDst destinations
- *ReduceSumCopy*: reduce nSrc → one buffer, copy to nDst destinations
- *AllGather*: concatenate nSrc chunks into one large buffer

For FP8 types the reference is computed on GPU (via `reference_gpu.cu`) because the
conversion and rounding semantics must match the CUDA hardware exactly.

An optional NCCL-level reference (allreduce / reduce-scatter / allgather via the high-level
NCCL API) runs for standard numeric types without per-rank offsets as an independent
cross-check.

**Step 6 — launch test kernel.**  The host launches one CUDA kernel per device via the
three-layer dispatch in `test_dispatch.h`:

```
unrollDispatch<T>(unroll, coopLevel, funcId, …)
  → coopDispatch<T, UNROLL>(coopLevel, …)       // switch on Thread/Warp/Cta
    → apiFunctionDispatch<T, UNROLL, COOP>(…)   // if/else on ReduceSum/ReduceSumCopy/AllGather
      → launchReduceSumKernel / launchAllReduceKernel / launchAllGatherKernel
```

All three template parameters (`T`, `UNROLL`, `CooperationLevel`) must be compile-time
constants.  The dispatch converts the runtime values to compile-time constants using the
`if (unroll == unroll1)` / `switch (coopLevel)` pattern.  The explicit instantiation files
(`inst_*.cu`) ensure that every `(T, UNROLL, COOP)` triple is compiled into the binary even
though it is only referenced at runtime.

**Step 7 — kernel execution.**  Inside each kernel (`kernels.cuh`) the pattern is:

```
ncclLsaBarrierSession<ncclCoopCta> bar{ctaCoop, devComm, fullTeam, devComm.lsaBarrier, blockIdx.x}
computeWorkDistribution<coopLevel>(countUnits, rank, nRanks, startOffset, localCount)
// call the appropriate ncclLsa* / ncclMultimem* / ncclLocal* API function
bar.sync()
```

Work is distributed across ranks and CTAs.  `storageCountForElements<T>` and
`elementCountForStorageRange<T>` handle the element-to-storage-unit conversion.

**Step 8 — validate output.**  After kernel completion and stream synchronisation, for each
destination rank the test copies the GPU output back to host and calls `expectValuesEqual`
element-by-element:

- For single-source copy operations: exact equality.
- For sums of ≥ 2 sources: ULP-based tolerance via `withinSumUlpTolerance`.  Tolerance
  scales with `sqrt(nSrc)` ULPs and is clamped by `maxAbsSource` (the maximum magnitude
  seen across all source values) to handle near-zero cancellation.
- Per-type mantissa bits in `OutputMantissaBits<T>` set the ULP bucket size.

Source ranks verify that their send buffer is unchanged.  Ranks not involved in a
particular N→M split verify that their receive buffer still holds the sentinel value
(`-999.0f`) written before the kernel.

**Step 9 — mark test executed.**  `MARK_TEST_EXECUTED(params, typeName)` inserts a key
into the `TestMatrix` singleton so the coverage listener can account for it.

### 5. Coverage tracking

`TestMatrix::getExpectedTests()` generates the full set of expected (funcId, coopLevel,
typeName, unroll, count, maxGpus, enableLambdaOffsets) tuples at startup, applying the same
exclusion logic as `generate_tests.py`.  After all tests finish, `MatrixTestListener::
OnTestProgramEnd()` diffs the executed set against the expected set and:

- prints a coverage percentage per type and overall
- optionally (with `NCCL_TEST_REDUCE_COPY_LIST_MISSING=1`) lists every missing combination
- exits the process with a non-zero code if anything is missing

The `run_reducecopy_types.sh` script sets `NCCL_TEST_REDUCE_COPY_TYPES` to the current
type before each binary invocation so that `getExpectedTests()` only expects that type's
tests — this prevents false "missing" reports from tests that were legitimately not run
because they belong to a different type.

### 6. N→M tests

`nm_reduce_sum_copy_test.cu` contains 15 hand-written `TEST_F` cases that exercise split
scenarios (some ranks are source-only, some destination-only, some both).  They use
`nm_reduce_sum_copy_kernels.cuh` which implements three N→M kernel variants:

- **Lambda** — `ncclLsaReduceSumLsaCopy` with per-rank index lambdas
- **DifferentTeams** — `ncclLsaReduceSumCopy` with manually constructed `ncclTeam` structs
- **Multimem** — `ncclMultimemReduceSumCopy` with separate multicast handles for source and
  destination partial teams

The team arithmetic for DifferentTeams uses:

```
srcTeam = {nSrc, devComm.lsaRank,           1}
dstTeam = {nDst, devComm.lsaRank - nDstStart, 1}
```

which makes `peerPtr(srcTeam, i) = i` and `peerPtr(dstTeam, i) = i + nDstStart` for all
calling ranks without requiring sub-communicators.
