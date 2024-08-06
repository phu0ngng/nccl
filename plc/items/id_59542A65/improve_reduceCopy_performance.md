# improve_reduceCopy_performance
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->
This feature improves performance of reduceCopy with a single source buffer (1:N) for unaligned buffer.
Time taken for Allgather in an unaligned buffer was 80% more than the aligned case.
This improvement takes this delta down to 5%.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
ReduceCopy with a single source buffer (1:N) is a broadcasting memcpy. Many
algorithms use this primitive, especially broadcast, allgather, and the second
round of allreduce+ring. Performance for non 4-byte aligned buffers was
lacking. Since we type erase our non-reduction operations (allgather, etc) and
handle them as single byte types, this slow path was being incurred even when
it shouldn't (e.g. allgather of f32 when not 16-byte aligned gets treated as
fully unaligned instead of 4-byte aligned). This feature improves the perf of
arbitrarily aligned byte buffers which especially helps the type erased case.
Meta is observing performance drop in allgather with addresses that are not divisible by 16.
<!-- ============================================================================================-->
 
### NVbugs / Jira Tickets
Nvbug: https://nvbugspro.nvidia.com/bug/4675018
Jira ticket: https://jirasw.nvidia.com/browse/NCCL-1640

### User Experience
This feature should be invisible to the user except improvement in unaligned performance.

### Assumptions, constraints and dependencies

### Use Cases

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
Align the "src" pointer to next multiple of 16 by processing all bytes before
that using unaligned element wise load and store. Since there can be at most 15
such elements this can be done with 1 warp and 1 load+store instruction pair.
Aligning the loads to 16 is probably the most important aspect for perf which is
why we do that first allowing the following code to only concern itself with
unaligned "dsts". Now with a 16-byte aligned "src" pointer we:

1. Read from "src" using 16-byte loads containing the data.
2. Store those to smem via 16-byte stores.
3. __syncwarp() so each thread can see scratch populated by other threads of warp.
4. Form 16-byte regs for storing to dst by loading consecutive 4-byte vals from
smem and funnel-shifting them into correct byte position as dictated by the
alignment of "dst" address.
5. Store to dst using 16-byte stores.

<!-- note: the following HTML code is also valid -->
<!-- <img src="images/example.png" width="900" height="500" /> -->

### Interface Architecture
The implementation leverages per-warp smem dubbed "scratch" (which already
existed for LL128). A pointer to the scratch for "this warp" must now be
supplied when calling reduceCopy.
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
 https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/493
</details>
 
<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation

#### Where to run?
Any platform.

#### What to run?
Allgather perf tests with `-u1` command line option to enable unaligned addresses.

#### Expected output?
Time required to complete Allgather operation with unaligned buffer is only ~5% more than aligned buffers.

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
Time required to perform unaligned Allgather operation.

#### Results
Without this change time required for Allgather unaligned buffers was ~80% more than the aligned case.
With these changes time required for Allgather unaligned buffers is only ~5% more than the aligned case.

</details>
 
<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s): 
  - Unmesh Deodhar
  - John Bachan

</details>
