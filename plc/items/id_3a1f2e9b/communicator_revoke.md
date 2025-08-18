# Communicator Revoke
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract

The Communicator Revoke feature allows NCCL to cancel or quiesce ongoing device-side NCCL activities on a communicator without destroying it or reclaiming its resources. Revoke sets abort flags so device work exits promptly, waits for in-flight kernels to finish, and then clears the flags so the communicator remains in a quiesced state for further management operations such as finalize, destroy, split, or shrink. It is not intended for launching new collective operations on the revoked communicator. Revoke supports optional cross-rank coordination via a global barrier and works in both blocking and non-blocking modes based on `ncclConfig_t::blocking`.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

- NVBugs: NCCL API extension to support Revoke communicators

### Feature context and focus

Long-running distributed jobs may need to promptly stop outstanding NCCL operations on a communicator without tearing it down. Typical scenarios include fault mitigation, fast recovery before a `ncclCommShrink`, or ensuring clean progress before `ncclCommFinalize`/`ncclCommDestroy`. Revoke provides a lightweight, explicit mechanism to cancel in-flight work and restore the communicator to a consistent, usable state.

### User Experience

Revoke is called on an existing communicator:

```c
// Local revoke (device sync only)
ncclResult_t r = ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);

// Or coordinated revoke with a cross-rank barrier
ncclResult_t r2 = ncclCommRevoke(comm, NCCL_REVOKE_GLOBAL);
```

Behavior is governed by the communicator configuration:
- When `config.blocking == 1`: the call blocks until revoke completes (including the optional barrier).
- When `config.blocking == 0`: the call returns `ncclInProgress` and the revoke proceeds asynchronously. Applications should poll `ncclCommGetAsyncError` until it returns `ncclSuccess` before issuing operations that depend on revoke completion.

#### Common flows and examples

- Revoke then Split (blocking):

```c
// Quiesce, then split safely
ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT); // or NCCL_REVOKE_GLOBAL
ncclGroupStart();
ncclCommSplit(comm, color, key, &newcomm, NULL);
ncclGroupEnd();
```

- Revoke then Split (non-blocking):

```c
ncclResult_t r = ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT); // or GLOBAL
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

### Assumptions, constraints and dependencies

- The communicator must not be in the middle of finalize or destroy; revoke will return `ncclInvalidArgument` if `finalizeCalled` or `destroyFlag` is set.
- When using `NCCL_REVOKE_GLOBAL` in blocking mode, all ranks must call concurrently; otherwise, the global barrier may deadlock. In non-blocking mode, the barrier is handled by a background job.
- Revoke does not free or reinitialize resources; it only stops ongoing work and clears abort flags.

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
- `revokeFlags`: Bitmask of behavior flags. Supported flags:
  - `NCCL_REVOKE_DEFAULT`: Local revoke (device sync only).
  - `NCCL_REVOKE_GLOBAL`: Include cross-rank barrier after local revoke.

#### Behavior

1. Set abort flags to request device-side NCCL activities to stop.
2. Ensure communicator is ready, then synchronize the internal device stream so in-flight kernels complete.
3. Clear abort flags so the communicator becomes usable again.
4. Optionally, perform a cross-rank barrier when `NCCL_REVOKE_GLOBAL` is set.

Blocking vs Non-blocking:
- Controlled by `ncclConfig_t::blocking` on the communicator.
- Non-blocking returns `ncclInProgress` immediately and performs the revoke steps in a background job; applications should poll `ncclCommGetAsyncError`.

#### Internal Architecture

- A shared implementation (`ncclCommRevokeImpl`) performs the core steps: set abort flags, sync device stream, clear abort flags, optional barrier.
- For non-blocking mode, an async job encapsulates that implementation and reports completion via `ncclCommSetAsyncError`.
- The function rejects calls if a finalize or destroy is in progress to avoid undefined states.

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

- Implementation is in `src/init.cc` (`ncclCommRevoke`, async job, and `ncclCommRevokeImpl`).

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

Validate both blocking and non-blocking modes; verify local and global revoke paths; ensure correct rejection during finalize/destroy.

### Validation

#### Where to run?

- Standard NCCL test environments (e.g., DGX-class systems) with multiple GPUs.

#### What to run?

1. Unit tests: `test/apitest/ncclCommRevoke_test.cu`
   - Blocking/non-blocking with `NCCL_REVOKE_DEFAULT` and `NCCL_REVOKE_GLOBAL`
   - NULL communicator no-op
   - Rejection when finalizing or after abort/destroy

2. Fault-tolerance tests: `test/perf/ft_test.cu`
   - Revoke to cancel ongoing work; optionally follow with shrink (`revoke_shrink`)

```bash
# Example: run revoke API tests
./build/test/apitest/apitest --gtest_filter=ncclCommRevoke*

# Example: run FT tests focused on revoke flows
./build/test/perf/all_reduce_perf -B 0 -F 1 -L "revoke,revoke_shrink"
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


