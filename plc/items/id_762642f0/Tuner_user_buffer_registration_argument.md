# Tuner user buffer registration argument
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
External tuner plugins are given a user buffer registration hint indicating whether
NCCL can register the user buffer or not. NCCL takes the final decision after algo
and proto have been picked by the tuner and after verifying all ranks have registered
their buffer.

<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->
 
### NVbugs / Jira Tickets
[NVBugs - RFE](https://nvbugswb.nvidia.com/NVBugs5/redir.aspx?url=/4852002)

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
The tuner plugin `getCollInfo` function is extended with a user buffer registration hint. The hint is
set to one if the following conditions is met: either both send and recv buffer have been registered
locally using `ncclCommRegister`, or NCCL is under graph capturing and `NCCL_GRAPH_REGISTER` is 1.

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
[MR 598](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/598)

</details>
 
<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation
Previous versions of tuner plugin should still work (compat layer is provided) and unit tests should pass.

#### Where to run?
No specific platform

#### What to run?
Unit tests.
Run NCCL with an older v2/v3 tuner plugin.

#### Expected output?
Unit tests for the ext-mixed plugin should pass.
NCCL should load and use the v2/v3 tuner plugin as before the feature was added.
 
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
  - Giuseppe Congiu <gcongiu@nvidia.com>

</details>
