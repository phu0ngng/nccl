# alltoallv perf test
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
This new alltoallv performance test enables evaluation of irregular alltoall traffic patterns using NCCL.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

https://nvbugspro.nvidia.com/bug/5754719

### User Experience
The alltoallv test integrates into the standard NCCL perf tests benchmark harness, enabling users to benchmark irregular alltoall traffic patterns using familiar test commands. The test provides accurate performance estimates that properly account for workload imbalance.

The test supports two modes for specifying traffic patterns:
- **Generated pattern mode (default)**: Uses a built-in distance-weighted distribution (no configuration required)
- **Matrix file mode**: Optionally provide an explicit traffic matrix file describing rank-to-rank byte counts

### Assumptions, constraints and dependencies
- **Traffic pattern specification**:
  - **Generated pattern mode (default)**: Generates a configurable distance-weighted distribution where more distant ranks (in rank ID space) receive proportionally more traffic than nearby ranks. This creates imbalanced workloads useful for testing irregular communication patterns.
  - **Matrix file mode** (when `ALLTOALLV_MATRIX_FILE` is provided): Uses an explicit traffic matrix file to define rank-to-rank byte counts, enabling testing of arbitrary communication patterns.
    - Cell (row i, col j) indicates the amount of bytes sent from rank i -> j.
    - File format: whitespace-separated numeric matrix; the entire file is read and must be square (rows == cols).
    - Matrix dimension must be ≥ `nranks` (only the first `nranks` rows/cols are used).
    - All per-peer traffic amounts are aligned to 16-byte boundaries for consistency with other perf tests.
- **Data size sweeps**:
  - **Generated pattern mode**: Supports standard perf test sweep behavior via `-b`/`-e` parameters.
  - **Matrix file mode**: Does not support sweeps, as the traffic matrix defines the exact amount of data to exchange; `-b` and `-e` must be equal.
- Sufficient GPU memory to allocate send/recv buffers sized to the per-rank maximum implied by the pattern.
  - In matrix mode: `-e` must be set to at least the maximum send or receive bytes across all ranks (i.e., `-e` ≥ `max(max_send_bytes, max_recv_bytes)` across all ranks). The test harness uses this maximum value to allocate buffers for all ranks.
- Depends on NCCL ≥ 2.7 (uses `ncclSend`/`ncclRecv`).

### Use Cases

Evaluating NCCL performance for irregular alltoall communication patterns typical in MoE (mixture of experts) and other sparse communication use cases.

### Platform Requirements

Standard NCCL performance test requirements.

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

The alltoallv test integrates into the existing perf test harness by implementing two function
pointer structs:
- **testEngine**: test-level functions (`getBuffSize`, `runTest`)
- **testColl**: per-collective functions (`getCollByteCount`, `initData`, `getBw`, `runColl`)

No modifications to the common test harness infrastructure were required.

**Traffic Pattern Specification**:

The test supports two modes for specifying per-peer traffic patterns, selected based on whether `ALLTOALLV_MATRIX_FILE` is provided:

- **Generated pattern mode (default)**: When no matrix file is provided, per-peer byte counts are computed on-the-fly using a distance-weighted formula that distributes traffic proportionally to rank distance. By default, more distant ranks receive proportionally more traffic.

  For rank `i` sending to rank `j`, the distance-weighted calculation is:
  - `distance = (j - i + nranks) % nranks` (circular distance in rank space)
  - `sum_distances = nranks * (nranks - 1) / 2` (sum of all possible distances)
  - `dist = distance * total_bytes / sum_distances` (distance-weighted distribution)

  The `ALLTOALLV_SPREAD` environment variable (default 1.0) can blend this distance-weighted distribution with a uniform distribution:
  - `uniform = total_bytes / nranks` (equal distribution)
  - `bytes = (1.0 - spread) * uniform + spread * dist` (blended result)
  - Final value is aligned to 16-byte boundaries: `bytes & -(size_t)16`

  Where `spread` is the `ALLTOALLV_SPREAD` value:
  - `spread = 1.0` (default): Fully distance-weighted (more traffic to distant ranks)
  - `spread = 0.0`: Uniform distribution (equal traffic to all peers)
  - `0.0 < spread < 1.0`: Blends uniform and distance-weighted distributions

- **Traffic matrix mode**: When `ALLTOALLV_MATRIX_FILE` is provided, the test reads a whitespace-separated matrix file where:
  - Each row represents a source rank and each column represents a destination rank
  - Matrix cell `[i][j]` contains the number of bytes sent from rank `i` to rank `j`
  - The matrix must be square (rows == cols) and the dimension must be ≥ `nranks`
  - Values are parsed as floating-point numbers, scaled by `ALLTOALLV_MATRIX_SCALE`, then aligned to 16-byte boundaries
  - If the matrix dimension is larger than `nranks`, only the first `nranks` rows and columns are effectively used (the test only accesses cells where `i < nranks` and `j < nranks`)
  - All values must be non-negative

  Example matrix file for 3 ranks:
  ```
  100.0  200.0  150.0
  50.0   100.0  75.0
  300.0  150.0  200.0
  ```
  This represents: rank 0 sends 100B to rank 0, 200B to rank 1, 150B to rank 2; rank 1 sends 50B to rank 0, etc.

  Note: When converting matrix values to element counts (by dividing by `element_size`), truncation occurs per cell. This means if a matrix cell contains 17 bytes and `element_size` is 4, the resulting element count is 4 (17 / 4 = 4.25, truncated to 4), not 5.

**Function Implementations**:

The following sections describe how each function in the `testEngine` and `testColl` structs is implemented for alltoallv.

**`getBuffSize` (`testEngine`)**:

This function determines send/recv buffer sizes given the maximum message size used by the test (i.e., the `-e` argument, `maxBytes`, which is passed to this function as the `count` parameter).

```
void (*getBuffSize)(size_t *sendcount, size_t *recvcount, size_t count, int nranks);
```

For alltoallv, this function calculates send/recv byte counts from either a generated pattern or from a traffic matrix file, depending on the test configuration. The general process is as follows:
 1. **Initialize alltoallv config**: `getBuffSize` is the first test function called, so it is where alltoallv-specific bootstrapping happens. It reads/validates test environment variables and, if `ALLTOALLV_MATRIX_FILE` is specified, the given traffic matrix file. This is where the test mode is determined (generated pattern mode vs. traffic matrix mode) based on whether a matrix file is provided. This initialization step is made thread-safe to handle multi-thread NCCL setup (i.e., the parallel init option).
 2. **Compute max counts**: Calls `getCollByteCount` which computes the maximum send/recv byte counts across all ranks. The source of these counts differs by mode:
    - **Generated mode**: Derived directly from `maxBytes`, just as in traditional test cases.
    - **Traffic matrix mode**: Read directly from the matrix file contents, summing per-rank send/recv volumes across all peers to compute the maximum buffer size requirements. Since the test harness allocates buffers using `maxBytes` rather than these computed requirements, validation is performed to ensure `maxBytes` is sufficient for the traffic pattern.

**Error handling**: Since this function has no ability to report errors directly (`void` return value) and performs non-trivial tasks (reading/validating matrix input), it uses an error flag system. When errors are detected, they are recorded in a flag and descriptive error messages are printed. The error flag is checked by the subsequent `runTest` call, which can appropriately error out of the test. Validations performed include:
  - Matrix file existence and readability
  - Matrix file format (whitespace-separated numeric values, square matrix requirement)
  - Matrix dimension ≥ `nranks`
  - `maxBytes` sufficient for maximum send/recv volumes indicated by the traffic matrix

**`runTest` (`testEngine`)**:

This function is primarily responsible for executing the benchmark across the requested datatypes.

```
testResult_t (*runTest)(struct threadArgs* args, int root, ncclDataType_t type,
      const char* typeName, ncclRedOp_t op, const char* opName);
```

Similar to other test cases, alltoallv implements this function as a thin wrapper to the `TimeTest` call, but with additional validation of some preconditions:
 - Abort if any errors were encountered in the preceding call to `getBuffSize`, as discussed above.
 - When using a traffic matrix file, abort if user specifies different values for `-b` and `-e`.
   - Message size sweeps are not supported because sizes come directly from the matrix file, not from the `-b`/`-e` parameters.

**`getCollByteCount` (`testColl`)**:

This function computes send/recv element counts and in-place offsets given info on the input buffer (total number of elements and their size) and the number of ranks.

```
void (*getCollByteCount)(
      size_t *sendcount, size_t *recvcount, size_t *paramcount,
      size_t *sendInplaceOffset, size_t *recvInplaceOffset,
      size_t count, size_t eltSize, int nranks);
```

For alltoallv, output values are set as follows:
  - Param count traditionally represents the element count passed to the NCCL collective operation (e.g., the per-peer count in alltoall). For alltoallv, variable per-peer counts from the pattern are used instead of a single uniform count. However, param count is still calculated for compatibility, internal use, and benchmark output (the "count" column):
    - **Generated mode**: Computed as `(count / nranks)` aligned to 16-byte boundaries, matching the traditional alltoall calculation for per-peer send/recv counts. Used internally to reconstruct the total bytes used for the distance-weighted formula.
    - **Traffic matrix mode**: Set to 0 (not used for internal calculations, as sizes come directly from the matrix).
  - Send/recv counts represent the maximum total element count across all ranks, calculated as follows for each mode:
    - **Generated mode**: The distance-weighted pattern is symmetric and balanced, so compute one rank's total (sum of per-peer element counts computed on-the-fly using the distance-weighted formula). Each per-peer byte value is aligned to 16-byte boundaries, then divided by `eltSize` to get element counts. Use that rank's total for both send and receive counts across all ranks.
    - **Traffic matrix mode**: For each rank, sum its per-peer element counts. Each matrix cell contains raw bytes (already 16-byte aligned when read from the file), which are divided by `eltSize` to get element counts. Then take the maximum send and receive counts across all ranks.
 - In-place operations are not supported (just as in alltoall), so output in-place offsets are just set to `0`.

**`initData` (`testColl`)**:

This function prepares per-rank device buffers for both send data and expected receive data. The expected data buffer will be compared against the actual receive buffer to ensure data arrives as expected.

```
testResult_t (*initData)(struct threadArgs* args, ncclDataType_t type,
      ncclRedOp_t op, int root, int rep, int in_place);
```

For alltoallv, this function handles variable-length data across ranks as follows:
 - **Recv and expected buffers**: Zeroed out entirely to ensure untouched regions match during validation.
   - Buffer sizes are based on max requirements across all ranks, so some ranks may not use their full buffer.
 - **Send buffer**: Built as concatenated segments for each destination rank.
   - Total send count is the sum of elements sent to all peers according to the pattern (generated or traffic matrix).
   - Fills with predictable test data seeded by local rank ID.
 - **Expected buffer**: Built as concatenated segments from each source rank, processed in rank order.
   - For each source rank, computes the per-peer element count this rank will receive from that source (according to the pattern).
   - For each segment, computes the offset where this rank's segment begins within that source's send stream (sum of element counts for all preceding destination ranks).
   - Initializes each segment separately with predictable test data seeded by that source's rank ID and the computed offset within that source's send buffer, ensuring validation succeeds.

As in alltoall, in-place data validation is not supported.

**`getBw` (`testColl`)**:

This function provides algorithmic and bus bandwidth estimates for a given operation.

```
  void (*getBw)(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks);
```

For alltoallv, performance estimates are computed as follows:

 - Algorithmic bandwidth: `max(max_send_bytes, max_recv_bytes) / time`, where `max_send_bytes` and `max_recv_bytes` are the maximum total bytes sent or received by any rank (computed via `AlltoAllvComputeMaxCounts`).
   - This effectively captures the bottleneck rank's workload, accurately reflecting the imbalanced nature of alltoallv patterns.
   - Most accurate results come from using max time across ranks (`-a 3` option), rather than the default time value which is just the average across all ranks. Averages can be heavily skewed by inactive ranks, while max time estimates can accurately capture workload imbalance.
 - Bus bandwidth: algorithmic bandwidth scaled by `((nranks - 1) / nranks)`, just as in alltoall.
 - Optional per-rank bandwidth summary: If `ALLTOALLV_PRINT_SUMMARY` is set, prints per-rank send/recv bytes and bandwidth estimates for all ranks (computed and printed by rank 0).

**`runColl` (`testColl`)**:

This function is responsible for actually issuing the collective communication calls for the benchmark.

```
  testResult_t (*runColl)(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset,
      size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int implIndex);
```

For alltoallv, this function implements variable-length all-to-all using point-to-point operations:

 - Wraps all operations in `ncclGroupStart()` / `ncclGroupEnd()`.
 - Iterates through all peer ranks, and for each peer rank:
   - Determines the byte counts for sending to that peer and receiving from that peer:
     - **Traffic matrix mode**: Byte counts come directly from the matrix file.
     - **Generated mode**: Byte counts are computed on-the-fly using the distance-weighted formula. The `count` parameter (from `paramcount`) is used to reconstruct `total_bytes = count * nranks * element_size` for the formula.
   - Converts bytes to element counts by dividing by `element_size`.
   - Issues `ncclSend` to that peer and `ncclRecv` from that peer if the element count is non-zero (skipping zero-count edges), advancing buffer pointers after each operation.

### Interface Architecture

The alltoallv-specific interface relies on environment variables to avoid adding CLI options that only apply to a single test.

- Environment variables:
  - `ALLTOALLV_MATRIX_FILE` (optional): path to a whitespace-separated matrix file. If not provided, the test uses the default generated pattern mode. The entire file is read and must be square (rows == cols); matrix dimension must be ≥ `nranks`. Line i column j represents bytes sent from rank i to rank j. All values are aligned to 16-byte boundaries.
  - `ALLTOALLV_MATRIX_SCALE` (optional): scale factor applied to all traffic matrix values. Default: 1.0 (no scaling). Applied before 16-byte alignment. Useful for testing different traffic volumes with the same pattern.
  - `ALLTOALLV_SPREAD` (optional): controls the distance-weighted spread factor for generated patterns. Range: 0.0 to 1.0. Default: 1.0 (fully distance-weighted).
    - 0.0 = uniform distribution (equal traffic to all peers)
    - 1.0 = fully distance-weighted (more traffic to distant ranks)
    - Intermediate values blend uniform and distance-weighted distributions
  - `ALLTOALLV_PRINT_SUMMARY` (optional): when set to non-zero, prints detailed per-rank bandwidth statistics including send_bytes, recv_bytes, alg_bw, and bus_bw for each rank. Useful for analyzing imbalanced workloads.

Beyond this, alltoallv honors the same command line arguments parsed by the existing perf test harness (e.g., for data types, iterations, warmups, etc.), with some notable caveats when using traffic matrix mode:
  - `-b` and `-e` must be set to the same value (e.g., `-b 32M -e 32M`). Sweeping is not supported because per-peer sizes come from the matrix file, which defines fixed values with no range to sweep over.
  - `-e` must be at least the maximum total send or receive bytes across all ranks in the matrix (i.e., `-e` ≥ `max(max_send_bytes, max_recv_bytes)`). The test harness uses `-e` to allocate buffers for all ranks, so it must accommodate the largest workload.

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

https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/2015

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation

#### Where to run?

Standard NCCL perf test environments. No specific hardware requirements.

#### What to run?

Test both traffic pattern modes (generated and matrix file) to verify:
- Generated pattern mode works correctly with different spread values
- Matrix file mode correctly reads and applies traffic patterns
- Environment variables (ALLTOALLV_SPREAD, ALLTOALLV_MATRIX_FILE, ALLTOALLV_MATRIX_SCALE) function as expected

#### Expected output?

Standard NCCL perf test output format with bandwidth metrics (algorithmic and bus bandwidth).

The test should correctly handle both generated and matrix file modes, properly distribute traffic according to the specified patterns, and report performance metrics based on the bottleneck rank's workload.

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

Algorithmic bandwidth and bus bandwidth for irregular alltoall communication patterns. Metrics are based on the bottleneck rank's workload (max send/recv bytes across all ranks) and maximum time across ranks to accurately reflect imbalanced workloads.

#### Results


</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Shane Snyder

</details>
