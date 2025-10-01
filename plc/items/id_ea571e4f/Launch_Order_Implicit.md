# Launch Order Implicit
Implicitly create a launch order per-device so that kernels from different communicators can overlap without risking deadlock.

## Abstract
Users want to launch NCCL ops in parallel on different communicators. This can deadlock since CUDA can insert false dependencies differently by rank. By issuing CUDA 12.3 launch order events we can permit NCCL kernel overlap by maintaining a per-device ordering of kernel launches. Users are required to launch kernels on different communicators in a uniform order over all ranks. Assertions have been added to detect if racing threads are attempting to launch at the same time.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets
https://nvbugspro.nvidia.com/bug/5002430
https://nvbugspro.nvidia.com/bug/4999399

### User Experience
Relaxed NCCL semantics to make more programs deadlock free. Many users were using CUDA_DEVICE_MAX_CONNECTIONS=1 as an unsupported way to achieve the same thing.

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

We now track state per CUcontext_t so the new struct ncclCudaContext has been added. This struct holds
info about each actively capturing CUDA graph, and a ncclStrongStream used to order launches. This struct may not be accessed concurrently. At the moment it is only accessed when launching kernels, and since we
require that kernel launch order be deterministic we don't have to lock the context struct. There is a mechanism for asserting that only one thread accesses the context struct at a time, the env var NCCL_LAUNCH_RACE_FATAL (default=1) controls this.

Getting ncclStrongStream to work with launch order events involved a refactor that led to a better design. The previous design had strong streams tracking graph nodes explicitly (cudaGraphNode_t's) if the stream had been captured. The new design uses a captured stream instead of graph nodes and exposes that stream to the consumer so they may do arbitrary CUDA work with it and have CUDA capture it into the graph implicitly. This led to a shrinking of the ncclStrongStream API. ncclStrongStreamAcquire now returns a stream handle for the consumer to use during the Acquire/Release section.

![Example](images/example.png)
<!-- note: the following HTML code is also valid -->
<!-- <img src="images/example.png" width="900" height="500" /> -->

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

A new unit test has been added: test/unit/overlap_test. This test creates 32 communicators and launches collectives round robin and overlapped mixing graph captured and non-captured.

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
  - Tintin
  - Captain Haddock

</details>
