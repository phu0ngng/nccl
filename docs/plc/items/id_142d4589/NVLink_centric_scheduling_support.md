# NVLink centric scheduling support

## Abstract
GPU floor sweeping can cause non-uniform scheduling of NCCL CTAs across available CPCs. Since every
CPC has the same number of XBAR ports, this results in NCCL CTAs not sharing CPCs to complete faster
than those sharing one, causing a performance tailing effect.

To solve this problem a new extended kernel launch attribute will be added by CUDA 13.0. This attribute
is effectively a hint to the GPU driver that NCCL CTAs should be distributed uniformely across available
CPCs. For details concerning the exact mechanism used by the driver to achieve this result refer to the
linked NVBug.

<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets
[NVLink Centric Scheduling](https://nvbugspro.nvidia.com/bug/5203334)

### User Experience

Users can set an env var to control the enablement of the NVLink centric scheduling feature in CUDA during
kernel launch.

### Assumptions, constraints and dependencies
The new kernel launch attribute is only intended for performance characterization and not to be used
with real workloads. The feature requires CUDA 13.0 and is disabled for lower versions, leaving them
uneffected by the change.

### Use Cases
NCCL performance characterization benchmarks.

### Platform Requirements
Blackwell platforms (B200 and GB200)

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
The new CUDA 13.0 kernel launch attribute can be enabled in NCCL through the ``NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE``.
Setting the var to ``1`` enables the feature.

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

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation
NCCL perftests: AllGather/AllReduce/ReduceScatter

#### Where to run?
B200 and GB200 systems: prenyx and pretyche containing nodes that exhibit the CPC camping issue.

#### What to run?
Single/Multi node with ``NCCL_NVLS_ENABLE=0``, ``NCCL_ALGO=RING``, ``NCCL_CGA_CLUSTER_SIZE=2``, ``NCCL_MIN_CTAS=32``, ``NCCL_MAX_CTAS=32``.

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
Bus Bandwidth

#### Results
Higher BusBw when ``NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE=1``

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Giuseppe Congiu <gcongiu@nvidia.com>

</details>
