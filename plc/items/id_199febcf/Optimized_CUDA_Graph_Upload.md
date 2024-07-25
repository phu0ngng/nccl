# Optimized CUDA Graph Upload
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
Graph capture time of NCCL operations was inefficient because of:

1. Use of cudaMalloc for device side buffer to hold metadata.
2. Use of a synchronous cudaMemcpy to populate that buffer.

This feature replaces cudaMalloc with cudaMallocAsync against a cudaMemPool which is considerably faster. We then use cudaMemcpyAsync on the same stream as the kernel launches.

Measured Impact: OpenXLA gets 3X times the total throughput with this optimizatoin.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets
nvbug 4449210

### User Experience

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

Abstract nearly says it all. The part remaining is that we need to detect when
the cudaMemcpyAsync completes to cleanup the host side source buffer. That is
done by managing a queue of cudaEvent's and associated callbacks. This is a
general mechanism we could reuse for other purposes, but in our case the
callbacks we push call `free(srcbuf)`.

### Interface Architecture

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
https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/512
</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation

#### Where to run?

#### What to run?

#### Expected output?

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

#### Results


</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - John Bachan

</details>
