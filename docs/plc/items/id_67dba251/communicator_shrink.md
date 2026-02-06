# Communicator Shrink
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract

The Communicator Shrink feature allows NCCL to dynamically remove devices from a communicator while maintaining communication between the remaining devices. This feature provides two operational modes: a default mode for controlled reconfiguration during normal operation, and an error mode for fault recovery scenarios. In default mode, applications can modify their device topology while ensuring no outstanding operations exist, while error mode automatically aborts pending operations to recover from failures. The feature supports configurable resource sharing options, allowing for efficient reuse of resources when appropriate. This capability is a critical component of NCCL's resilience features that helps maintain system reliability in distributed training environments.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

- [NVBugs](https://nvbugspro.nvidia.com/bug/5007856): NCCL API extension to support Shrink communicators

### Feature context and focus

In distributed training environments, hardware failures can occur unexpectedly, especially during long-running workloads. When a device fails, the entire training process might fail if the system cannot adapt to the failure. Additionally, in some scenarios, applications may want to dynamically reconfigure their device topology during normal operation. The Shrink feature addresses these needs by:

- Allowing applications to detect and isolate failed devices in error recovery scenarios
- Supporting dynamic reconfiguration of device topologies during normal operation
- Creating a new communicator that excludes specified devices with appropriate resource handling
- Maintaining the correct rank ordering for the remaining devices
- Ensuring that collective operations continue to function correctly

This feature significantly enhances the resilience and flexibility of NCCL, enabling applications to handle both planned reconfiguration and unexpected device failures without having to restart the entire job.

### User Experience

To use the Communicator Shrink feature, follow these steps:

1. **Detect devices to exclude**: Identify which ranks need to be excluded from the communicator (due to failure or planned reconfiguration).

2. **Create a list of ranks to exclude**: Prepare an array of the rank indices that should be excluded.

3. **Choose the appropriate mode**:
   - For planned reconfiguration (default mode), ensure no outstanding operations exist
   - For error recovery, use error mode to automatically abort ongoing operations

4. **Call ncclCommShrink within a group**: Use NCCL's group API to ensure synchronized creation across all remaining ranks.

```c
// For planned reconfiguration during normal operation:
NCCLCHECK(ncclGroupStart());
for (int i = 0; i < nGpus; i++) {
  if (i != excludedRank) {
    ncclCommShrink(comm[i], &excludeRanks, excludeCount, ncclShrinkModeDefault, &newcomm[i], NULL);
  }
}
NCCLCHECK(ncclGroupEnd());

// Or for error recovery scenarios:
NCCLCHECK(ncclGroupStart());
for (int i = 0; i < nGpus; i++) {
  if (i != excludedRank) {
    ncclCommShrink(comm[i], &excludeRanks, excludeCount, ncclShrinkModeError, &newcomm[i], NULL);
  }
}
NCCLCHECK(ncclGroupEnd());
```

5. **Configure resource sharing** (optional): Control resource sharing behavior using configuration settings.

```c
// Enable resource sharing for non-error mode shrink operations
ncclConfig_t shrinkConfig = NCCL_CONFIG_INITIALIZER;
shrinkConfig.shrinkShare = 1;
NCCLCHECK(ncclCommShrink(comm, excludeRanks, excludeCount, ncclShrinkModeDefault, &newcomm, &shrinkConfig));
```

6. **Use the new communicator**: The new communicator will have ranks re-ordered to maintain contiguous numbering. For example, if you exclude rank 1 from a 4-rank communicator, the new communicator will have ranks 0, 1, and 2 (corresponding to original ranks 0, 2, and 3).

7. **Destroy communicators**: When done, destroy the new communicator with ncclCommDestroy.

### Example

The following example demonstrates how to exclude a specific rank from a communicator:

```c
// Rank 1 is experiencing errors - exclude it
int excludeList[] = {1};
ncclComm_t newcomm;

// For error recovery mode:
if (myRank != 1) {
  ncclCommShrink(comm, excludeList, 1, ncclShrinkModeError, &newcomm, NULL);

  // Use the new communicator
  ncclAllReduce(...);

  // Destroy when done
  ncclCommDestroy(newcomm);
}

// Or for planned reconfiguration (default mode):
if (myRank != 1) {
  // Ensure no outstanding operations
  cudaStreamSynchronize(stream);

  // Create new communicator with resource sharing enabled
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.shrinkShare = 1;
  ncclCommShrink(comm, excludeList, 1, ncclShrinkModeDefault, &newcomm, &config);

  // Use the new communicator
  ncclAllReduce(...);

  // Destroy when done
  ncclCommDestroy(newcomm);
}
```

### Assumptions, constraints and dependencies

- The Shrink operation is only supported for NCCL version 2.27 and above
- The Shrink operation requires all remaining processes to call the function with the same exclude list
- The function cannot be used to remove all devices from a communicator
- Ranks are reordered in the new communicator to maintain contiguous numbering
- Shrink works with the NCCL group API for synchronized creation
- **Important**: Ranks listed in the exclusion list should not call the Shrink function
- For optimal fault tolerance, applications should use non-blocking communicators
- **Mode-specific behavior**:
  - **Default mode**: No outstanding operations should exist on the parent communicator
  - **Error mode**: Can handle outstanding operations, as it will abort them automatically
- **Resource sharing**:
  - Resource sharing for shrink operations can be enabled via `shrinkShare` config
  - Resource sharing is never allowed in error mode
  - When enabled, some resources can be reused for better efficiency

### Use Cases

1. **Handling device failures during training**: When a GPU fails during a long-running training job, the application can detect the failure, create a new communicator excluding the failed GPU, and continue training.

   ```c
   // Example: Handling a failed GPU during training
   if (detectDeviceFailure()) {
     int failedRank = getFailedRank();
     int excludeList[] = {failedRank};

     // Create new communicator without the failed rank, using error mode
     ncclGroupStart();
     ncclCommShrink(comm, excludeList, 1, ncclShrinkModeError, &newcomm, NULL);
     ncclGroupEnd();

     // Continue training with the new communicator
     continueTraining(newcomm);
   }
   ```

2. **Dynamic topology reconfiguration**: Applications can adjust their device topology during execution based on workload demands, freeing up resources for other tasks or adding resources as they become available.

   ```c
   // Example: Reconfiguring topology by removing a device
   if (needToReleaseResources()) {
     int rankToRelease = selectRankToRelease();
     int excludeList[] = {rankToRelease};

     // Ensure no ongoing operations
     cudaStreamSynchronize(stream);

     // Create new communicator with default mode and resource sharing
     ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
     config.shrinkShare = 1;
     ncclGroupStart();
     ncclCommShrink(comm, excludeList, 1, ncclShrinkModeDefault, &newcomm, &config);
     ncclGroupEnd();

     // Continue with reduced resources
     continueWithReducedResources(newcomm);
   }
   ```

3. **Graceful degradation**: In production systems, maintaining service availability is critical. When a device fails, the system can continue to operate with reduced capacity rather than experiencing total failure.

4. **Fault-tolerant distributed applications**: Applications can implement recovery mechanisms that handle device failures automatically, improving overall system reliability.

### Platform Requirements

- CUDA-capable GPUs
- NCCL version 2.27 or higher
- Support for NCCL fault tolerance features
- Applications must use non-blocking communicators for optimal fault tolerance

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

#### API Extension

The Shrink feature introduces a new API function to the NCCL library:

```c
ncclResult_t ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclShrinkMode_t mode, ncclComm_t *newcomm, ncclConfig_t *config);
```

Parameters:
- `comm`: The original communicator
- `excludeRanksList`: Array of ranks to exclude (from the original communicator)
- `excludeRanksCount`: Number of ranks to exclude
- `mode`: Mode of the shrink operation (ncclShrinkModeDefault or ncclShrinkModeError)
- `newcomm`: Pointer to the new communicator that will be created
- `config`: Optional configuration parameters for the new communicator (NULL to inherit from the original)

#### Operation Modes

The Shrink operation supports two operational modes:

1. **Default Mode (ncclShrinkModeDefault)**:
   - Used for planned reconfiguration during normal operation
   - Requires no outstanding operations on the parent communicator
   - Can use resource sharing if configured (via shrinkShare)
   - Better performance and resource utilization

2. **Error Mode (ncclShrinkModeError)**:
   - Used for error recovery scenarios
   - Automatically aborts ongoing operations on the parent communicator
   - Never shares resources with the parent communicator
   - Prioritizes clean recovery over performance

#### Resource Sharing Configuration

Resource sharing for Shrink operations can be controlled via the config parameter:

```c
// Enable resource sharing for default mode (has no effect in error mode)
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.shrinkShare = 1;
```

This setting controls whether certain resources are shared between the parent and child communicators, which can improve efficiency during normal operation. Resource sharing is never allowed in error mode.

#### Internal Architecture

The implementation leverages existing infrastructure for creating child communicators, similar to `ncclCommSplit`. The key differences are:

1. **Rank Selection**: Instead of using a color/key mechanism like `ncclCommSplit`, the Shrink operation explicitly specifies which ranks to exclude.

2. **Fixed Parameters**: Shrink uses fixed color (0) and key (comm->rank) values to simplify the process.

3. **Rank Remapping**: A mapping function translates ranks from the parent communicator to the child communicator, excluding the specified ranks.

4. **Mode-Specific Handling**: Different code paths for default vs. error mode, with error mode including additional steps to abort operations.

The workflow for creating a shrunken communicator is:

1. **Mode Check**: First, determine if this is a default or error mode operation
2. **Error Mode Handling**: If in error mode, set abort flags, wait for operations to complete, and reset flags
3. **Input Validation**: Check that the exclude list is valid and that the current rank is not in the list
4. **Resource Allocation**: Create a new communicator structure and initialize resources based on mode
5. **Rank Calculation**: Determine new ranks by filtering out excluded ranks
6. **Bootstrap**: Set up communication channels between the remaining ranks
7. **Finalization**: Complete initialization of the new communicator

#### Rank Calculation

The rank calculation is handled by the `getParentRanks` function, which efficiently computes the new ranks by skipping excluded ranks:

```c
static ncclResult_t getParentRanks(int parentRanks, int parentRank, int* excludeRanksList,
                                  int excludeRanksCount, int* nRanksRet, int* myRankRet,
                                  int* parentRanksRet) {
  int count = 0;
  int j = 0;

  for (int i = 0; i < parentRanks; i++) {
    if (j < excludeRanksCount && excludeRanksList[j] == i) {
        j++;
        continue;
    }
    if (i == parentRank) *myRankRet = count;
    parentRanksRet[count++] = i;
  }
  *nRanksRet = parentRanks - excludeRanksCount;

  return ncclSuccess;
}
```

This function:
1. Iterates through all ranks in the parent communicator
2. Skips ranks that are in the exclude list
3. Assigns new rank numbers to the remaining ranks
4. Calculates the calling rank's new rank in the shrunken communicator

#### Resource Management

The Shrink operation handles resources differently from Split, with the resource sharing logic determined by the following code:

```c
// Set the shareResource field, this is used throughout the init and must be reset every time.
// If we shrink, we only reuse resources if we are not shrinking due to an error.
comm->shareResources = !comm->revokedFlag && (isShrink ? (mode != ncclShrinkModeError && comm->config.shrinkShare) : comm->config.splitShare);
if (comm->shareResources) {
  childComm->abortFlag = comm->abortFlag;
  childComm->abortFlagDev = comm->abortFlagDev;
  childComm->abortFlagRefCount = comm->abortFlagRefCount;
  comm->childAbortFlag = NULL;
  ncclAtomicRefCountIncrement(comm->abortFlagRefCount);
} else {
  NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlag, 1), res, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc(&childComm->abortFlagDev, 1), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlagRefCount, 1), res, fail);
  /* temporarily used to abort everything during child comm init. */
  comm->childAbortFlag = childComm->abortFlag;
  comm->childAbortFlagDev = childComm->abortFlagDev;
  *childComm->abortFlagRefCount = 1;
}
```

The resource sharing behavior depends on several factors:

1. **Operation type and mode**:
   - For Split operations: Resource sharing is controlled by `comm->config.splitShare`, but disabled if the parent is revoked
   - For Shrink operations: Resource sharing is only allowed when:
     - Not in error mode (`mode != ncclShrinkModeError`)
     - Sharing is explicitly enabled via `comm->config.shrinkShare`
     - The parent communicator is not revoked

2. **Resource sharing configuration**:
   - `splitShare`: Controls resource sharing for Split operations
   - `shrinkShare`: Controls resource sharing for Shrink operations, but only when not in error mode

3. **Error mode handling**:
   - When using `ncclShrinkModeError`, resource sharing is always disabled
   - This ensures clean resource separation during error recovery
   - The implementation sets abort flags on the parent communicator first, waits for kernels to complete, then resets the flags before creating the new communicator

When resource sharing is disabled (which is always the case for error mode):
- New abort flags are allocated
- Child communicator maintains independent resources
- The parent temporarily holds references to child's abort flags during initialization

This ensures that during error recovery scenarios, the shrunk communicator can operate independently of any potential issues with the parent communicator.

#### Input Validation

The implementation handles several validation checks:

```c
// excludeRanksList may not be sorted, need to sort it
qsort(excludeRanksList, excludeRanksCount, sizeof(int), compareInts);
// ranks in excludeRanksList should not call into this function
NCCLCHECKGOTO(bsearch(&comm->rank, excludeRanksList, excludeRanksCount, sizeof(int), compareInts)
              ? ncclInvalidArgument : ncclSuccess, res, exit);
```

Key validations:
1. Sort the exclude list to optimize the rank calculation
2. Verify that the current rank is not in the exclude list
3. Check that the exclude list is not empty
4. Validate that the communicator and newcomm pointers are valid

#### Implementation Flow

The main entry point for the Shrink operation:

```c
ncclResult_t ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount,
                           ncclShrinkMode_t mode, ncclComm_t *newcomm, ncclConfig_t* config) {
  NVTX3_RANGE(NcclNvtxParamsCommShrink)
  ncclResult_t res = ncclSuccess;
  NCCLCHECK(ncclGroupStartInternal());

  // Handle error mode by setting abort flags and waiting for kernels to complete
  if (mode == ncclShrinkModeError) {
    NCCLCHECKGOTO(setCommAbortFlags(comm, 1), res, exit);
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream), res, exit);
    NCCLCHECKGOTO(setCommAbortFlags(comm, 0), res, exit);
  }

  // Use fixed color and keys when shrinking the communicator
  NCCLCHECKGOTO(ncclCommInitChildComm(comm, newcomm, /*isShrink=*/true,
                   /*shrinkAfterError=*/mode == ncclShrinkModeError,
                   /*color=*/0, /*key=*/comm->rank, excludeRanksList,
                   excludeRanksCount, config, __func__), res, exit);

  NVTX3_RANGE_ADD_PAYLOAD(CommShrink, NcclNvtxParamsCommShrinkSchema,
    NVTX3_PAYLOAD(comm->commHash, comm->nRanks, comm->rank, comm->cudaDev, excludeRanksCount));

exit:
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
}
```

The key aspects of the implementation:

1. **Error Mode Handling**: When `mode` is set to `ncclShrinkModeError`, the function:
   - Sets abort flags on the communicator
   - Waits for all active kernels to complete
   - Resets the abort flags to prevent bootstrap issues

2. **Child Communicator Creation and Telemetry**:
   - Delegates to `ncclCommInitChildComm` with appropriate flags:
     - Indicates this is a shrink operation (`isShrink=true`)
     - Passes error recovery information based on mode
     - Uses fixed color (0) for all participating ranks
     - Uses original rank as key to preserve ordering
   - Adds payload telemetry with communicator hash, rank count, and other metrics for monitoring and debugging

### Design Considerations

1. **Performance Impact**: The Shrink operation is designed to be efficient, with minimal overhead during normal operation.

2. **Robustness**: The implementation handles edge cases such as:
   - Invalid exclude lists
   - Unsorted exclude lists
   - Attempts to exclude all ranks
   - Attempts by excluded ranks to call the function

3. **Synchronization**: The function works with NCCL's group API to ensure synchronized creation across all ranks.

4. **Inheritance**: The new communicator inherits configuration settings from the parent communicator unless explicitly overridden through the config parameter.

5. **Resource Management**: The implementation carefully manages resources based on mode and configuration:
   - Default mode with sharing enabled: Reuses resources for efficiency
   - Default mode with sharing disabled: Allocates new resources
   - Error mode: Always allocates new resources regardless of configuration

<!-- ### Interface Architecture -->
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

The implementation of the Shrink feature can be found in:

- NCCL Merge Request: [MR 830](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/830/commits)


</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

The testing objectives for the Shrink feature are to:

1. Verify the correct functionality of the `ncclCommShrink` API in both default and error modes
2. Ensure that collective operations work correctly on shrunken communicators
3. Validate fault tolerance scenarios where devices fail
4. Test performance impacts and resource management with different sharing configurations
5. Verify correct behavior when switching between modes

### Validation

#### Where to run?

The feature has been tested on DGX systems

#### What to run?

1. **Unit Tests** (`test/apitest/ncclCommShrink_test.cu`):
   - Run tests for both default and error modes
   - Verify basic functionality excluding a single rank
   - Verify handling of unsorted exclude lists
   - Test error handling cases
   - Test resource sharing configurations

```bash
# Example: Running the Shrink unit tests
./build/test/apitest/apitest --gtest_filter=ncclCommShrink*
```

2. **Fault Tolerance Tests** (`test/perf/ft_test.cu`):
   - Run the distributeFTShrinkTest to simulate failures and validate recovery
   - Verify that collective operations work correctly after shrinking

```bash
# Example: Running the fault tolerance tests with Shrink
./build/test/perf/all_reduce_perf -B 0 -F 1 -L "shrink"
```

#### Expected output?

For the unit tests, successful execution should show tests passing for both modes:
```
Running main() from gtest_main.cc
Note: Google Test filter = ncclCommShrink*
[==========] Running 9 tests from 1 test case.
[----------] Global test environment set-up.
[----------] 9 tests from ncclCommShrink_test
[ RUN      ] ncclCommShrink_test.basic
[       OK ] ncclCommShrink_test.basic (8064 ms)
[ RUN      ] ncclCommShrink_test.basic_error_mode
[       OK ] ncclCommShrink_test.basic_error_mode (8124 ms)
[ RUN      ] ncclCommShrink_test.not_sorted
[       OK ] ncclCommShrink_test.not_sorted (2423 ms)
[ RUN      ] ncclCommShrink_test.not_sorted_error_mode
[       OK ] ncclCommShrink_test.not_sorted_error_mode (2453 ms)
[ RUN      ] ncclCommShrink_test.rank_null
[       OK ] ncclCommShrink_test.rank_null (1294 ms)
[ RUN      ] ncclCommShrink_test.rank_null_error_mode
[       OK ] ncclCommShrink_test.rank_null_error_mode (1305 ms)
[ RUN      ] ncclCommShrink_test.shrink_null
[       OK ] ncclCommShrink_test.shrink_null (1290 ms)
[ RUN      ] ncclCommShrink_test.shrink_null_error_mode
[       OK ] ncclCommShrink_test.shrink_null_error_mode (1302 ms)
[ RUN      ] ncclCommShrink_test.shrink_share
[       OK ] ncclCommShrink_test.shrink_share (8214 ms)
[----------] 9 tests from ncclCommShrink_test (34469 ms total)

[----------] Global test environment tear-down
[==========] 9 tests from 1 test case ran. (34469 ms total)
[  PASSED  ] 9 tests.
```

For the fault tolerance tests, successful execution should show:
```
        ================ Test fault tolerance for NCCL shrink ================
Distributed FT: Sleep 10us, abort 1 communicators in shrink     [SUCCESS]
Distributed FT: Sleep 100us, abort 1 communicators in shrink    [SUCCESS]
Distributed FT: Sleep 10000us, abort 1 communicators in shrink  [SUCCESS]
Distributed FT: Sleep 1000000us, abort 1 communicators in shrink        [SUCCESS]
Distributed FT: Sleep 2000000us, abort 1 communicators in shrink        [SUCCESS]
Test fault tolerance for NCCL shrink    [SUCCESS]
```

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Bruce Chang

</details>


