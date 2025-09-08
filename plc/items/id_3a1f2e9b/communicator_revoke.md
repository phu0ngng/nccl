# Communicator Revoke
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract

The Communicator Revoke feature allows NCCL to promptly stop ongoing device-side NCCL activities on a communicator without destroying it or reclaiming its resources. Revoke immediately sets abort flags to signal device work to exit, marks the communicator as revoked to prevent new collective operations, and asynchronously handles cleanup of in-flight kernels. After completion, the communicator remains available for management operations such as finalize, destroy, split, or shrink, but rejects new collective operations. Revoke behavior (blocking vs non-blocking) is controlled by the communicator's `ncclConfig_t::blocking` setting.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

- NVBugs: NCCL API extension to support Revoke communicators

### Feature context and focus

Long-running distributed jobs may need to promptly stop outstanding NCCL operations on a communicator without tearing it down. Typical scenarios include fault mitigation, fast recovery before a `ncclCommShrink`, or ensuring clean progress before `ncclCommFinalize`/`ncclCommDestroy`. Revoke provides a lightweight, explicit mechanism to cancel in-flight work and restore the communicator to a consistent, usable state.

**Key difference from `ncclCommAbort`**: Unlike abort, revoke preserves the communicator and its resources for potential reuse. Specifically, revoke allows bootstrap resources to be reused in subsequent operations like `ncclCommShrink`, whereas abort destroys the communicator entirely, requiring complete reinitialization. This makes revoke particularly valuable for fault recovery scenarios where you want to remove failed ranks via shrink operations without the overhead of full communicator recreation.

### User Experience

Revoke is called on an existing communicator:

```c
// Local revoke (device sync only)
ncclResult_t r = ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
```

Behavior is governed by the communicator configuration:
- When `config.blocking == 1`: the call blocks until revoke completes.
- When `config.blocking == 0`: the call returns `ncclInProgress` and the revoke proceeds asynchronously. Applications should poll `ncclCommGetAsyncError` until it returns `ncclSuccess` before issuing operations that depend on revoke completion.

#### Common flows and examples

- Revoke then Split (blocking):

```c
// Quiesce, then split safely
ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
ncclGroupStart();
ncclCommSplit(comm, color, key, &newcomm, NULL);
ncclGroupEnd();
```

- Revoke then Split (non-blocking):

```c
ncclResult_t r = ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
// r is ncclInProgress or ncclSuccess
ncclResult_t st;
do { ncclCommGetAsyncError(comm, &st); } while (st == ncclInProgress);
ncclGroupStart();
ncclCommSplit(comm, color, key, &newcomm, NULL);
ncclGroupEnd();
```

- Prohibited: Revoke after Abort

```c
ncclCommAbort(comm);
// Next call is invalid; communicator is finalizing/destroying
// returns ncclInvalidArgument
ncclResult_t err = ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
```

- Prohibited: Collective operations on revoked communicator

```c
ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
// Next call is invalid; communicator is revoked
// returns ncclInvalidUsage
ncclResult_t err = ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);
```

### Assumptions, constraints and dependencies

- **State Requirements**: The communicator must not be finalizing, destroyed, or already revoked. Attempts to revoke in these states return `ncclInvalidArgument`.
- **Flag Validation**: Only `NCCL_REVOKE_DEFAULT` (0) is supported for `revokeFlags`. Other values return `ncclInvalidArgument`.
- **Resource Preservation**: Revoke does not free or reinitialize communicator resources; it only quiesces ongoing operations.
- **Post-Revoke State**: Once revoked, the communicator permanently rejects new collective operations with `ncclInvalidUsage`. The communicator can still be used for management operations like finalize, destroy, split, or shrink.

### Use Cases

1. Ensure no in-flight NCCL operations before a critical phase transition.
2. Quiesce a communicator before `ncclCommShrink` to remove failed or undesired ranks.
3. Stabilize a communicator after detecting transient errors, then continue or finalize.

### Platform Requirements

- CUDA-capable GPUs
- Supported NCCL version containing revoke implementation

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed Design

#### API

```c
ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags);
```

Parameters:
- `comm`: Target communicator.
- `revokeFlags`: Must be `NCCL_REVOKE_DEFAULT` (0). Reserved for future use.

Return Values:
- `ncclSuccess`: Revoke completed successfully (blocking mode) or revoke initiated successfully (non-blocking mode).
- `ncclInProgress`: Revoke is proceeding asynchronously (non-blocking mode only).
- `ncclInvalidArgument`: Invalid revokeFlags, communicator is finalizing/destroyed, or communicator already revoked.

#### Behavior

1. Validates arguments and communicator state (returns error if finalizing, destroyed, or already revoked).
2. Sets abort flags to signal device-side NCCL activities to stop immediately.
3. Marks the communicator as revoked to prevent new collective operations.
4. Initiates asynchronous cleanup that synchronizes with in-flight kernels and clears abort flags.

**Blocking vs Non-blocking behavior** (controlled by `ncclConfig_t::blocking` on the communicator):
- **Blocking mode** (`config.blocking = 1`): Function returns `ncclSuccess` when revoke operation completes.
- **Non-blocking mode** (`config.blocking = 0`): Function returns immediately with `ncclSuccess` or `ncclInProgress`. Applications must poll `ncclCommGetAsyncError` until it returns `ncclSuccess` to confirm completion.

#### Design Architecture

- **Immediate Response**: The function validates state and sets abort signals immediately, allowing fast cancellation of ongoing operations.
- **Asynchronous Cleanup**: Stream synchronization and final cleanup occur asynchronously to avoid blocking the calling thread unnecessarily.
- **State Protection**: Strict state validation prevents revoke operations on communicators that are already being destroyed or have been previously revoked.
- **Post-Revoke Enforcement**: Once revoked, the communicator rejects new collective operations with `ncclInvalidUsage` to ensure clean state management.

#### Interactions

- Revoke + Shrink: A common flow is to call revoke to quiesce the communicator, then call `ncclCommShrink` to remove failed ranks and continue.
- Revoke + Split: After revoke completes, splitting the communicator via `ncclCommSplit` is safe (no outstanding ops).
- Revoke + Finalize/Destroy: After revoke completes, applications can safely proceed to finalize/destroy if desired.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Coding</h2></summary>
<!-- ============================================================================================-->

### Commit list or MR

- MR1226: [https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/1226](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/1226)

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

Validate both blocking and non-blocking modes; verify local revoke path; ensure correct rejection during finalize/destroy.

### Validation

#### Where to run?

- Standard NCCL test environments (e.g., DGX-class systems) with multiple GPUs.

#### What to run?

1. Unit tests: `test/apitest/ncclCommRevoke_test.cu`
   - Blocking and non-blocking modes (controlled by `ncclConfig_t::blocking`)
   - NULL communicator no-op
   - Rejection when finalizing or after abort/destroy
   - Verification that collective operations return `ncclInvalidUsage` after revoke

2. Fault-tolerance tests: `test/perf/ft_test.cu`
   - Revoke to cancel ongoing work; optionally follow with shrink (`revoke_shrink`)


```bash
# Example: run revoke API tests
./build/test/apitest/apitest --gtest_filter=ncclCommRevoke*

# Example: run FT tests focused on revoke flows
./build/test/perf/all_reduce_perf -B 0 -F 1 -L "revoke,revoke_shrink,revoke_split"
```

#### Expected output?

- API tests should pass for all revoke modes.
- FT tests should show successful cancellation of work and ability to proceed with shrink or finalize afterward.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Bruce Chang

</details>


