# Relax Symmetric Buffer Registration Requirements for Collectives
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->
In order to utilize symmetric kernels, NCCL currently requires both src and dst buffer to be registered for
collectives such as AR, AG and RS. This requirement is too strict to follow for some users where they can
only guarantee either src or dst buffer to be symmetrically registered. To solve the issue, this feature
relaxes the symmetric registration requirement for collectives and automatically pick feasible symmetric
kernels internally based on whether src or dst buffers have been symmetrically registered.

### NVbugs / Jira Tickets
https://nvbugspro.nvidia.com/bug/5441348

### User Experience
Users only need to symmetrically register src or dst buffer, and NCCL will automatically pick optimal
and feasible symmetric kernels for them. 

### Assumptions, constraints and dependencies
NA

### Use Cases
Either symmetrically register src or dst in the applications to enable symmetric kernels in NCCL.

### Platform Requirements
NA

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
NCCL currently only supports AR, AG and RS to pick symmetric kernels. We have relax the registration requirements
so that some of symmetric kernels can still be picked even though not both src and dst buffer are registered.

For now, we have the following symmetric kernels:

* `AllReduce_AGxLL_R`
* `AllReduce_AGxLLMC_R`
* `AllReduce_RSxLD_AGxST`
* `AllReduce_RSxLDMC_AGxSTMC`
* `AllGather_LL`
* `AllGather_LLMC`
* `AllGather_ST`
* `AllGather_STMC`
* `ReduceScatter_LL`
* `ReduceScatter_LD`
* `ReduceScatter_LDMC`

For LL-based symmetric kernels:

* `AllReduce_AGxLL_R`
* `AllReduce_AGxLLMC_R`
* `AllGather_LL`
* `AllGather_LLMC`
* `ReduceScatter_LL`

We don't need to register *src and dst* buffers at all, but allocate LL symmetric buffers during init stage.
As long as the following conditions are satisfied:

1. Message size is small.
2. User calls single NCCL op instead of grouped NCCL ops.
3. Symmetric LL kernel performs better than legacy kernel.

Then, we can pick the corresponding LL symmetric kernels automatically for users.

For any other non-LL symmetric kernels:
(1) for AR, both *src and dst* buffer need to be registered, otherwise legacy kernels will be picked;
(2) for AG, when *dst* buffer is registered, symmetric kernels can be picked;
(3) for RS, when *src* buffer is registered, symmetric kernels can be picked.

For sendrecv-based collectives:
(1) for alltoall and scatter, only *dst* buffer needs to be registered;
(2) for gather, only *src* buffer needs to registered.

<!-- ![Example](images/example.png) -->
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
  - Kaiming Ouyang

</details>
