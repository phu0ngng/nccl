# Graph Search for NVL72
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

### User Experience
NCCL should be functional on NVL72 systems.

### Assumptions, constraints and dependencies
None.

### Use Cases
Computing on NVL72.

### Platform Requirements
A GB200 system.
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

Since the introduction of MNNVL systems, the topology graph contains the entire NVLink domain.
Each rank discovers its own GPU and attachment in the system, then we exchange the XML graphs
between ranks within the same OS or NVLink domain (whichever is larger), then we fuse all those
XML structures together to create the big topology graph, including all GPUs.

The major challenge which comes with this increase in size is the graph search. The graph search
can grow exponentially with the number of GPUs in the graph, and exploring paths within a 72-GPU
graph would not converge.

The graph search implements a timeout mechanism which will stop a graph search after a given time.
This ensures we never take an infinite amount of time to compute the graphs, but if not properly
directed, it can also make the search fail to find the optimal solution. Great care is therefore
devoted to nudging the algorithm in the right direction.

In previous NCCL versions we already restricted the NVSwitch search so that each GPU would only
talk to its direct neighbors.

That was not enough however on NVL72 systems, so NCCL 2.25 optimizes the search further, to only
explore GPU/GPU and GPU/NIC distances which may actually work, by computing the min and max of
GPU-GPU path types to control the typeIntra values, and GPU-NIC path types to control the typeInter
values.

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
MR [739](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/739)
</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline
Testing this code would require a cluster of NVL72 systems, connected through IB or RoCE.
We should verify the correct operation on such a cluster. Performance may not be optimal yet
but should be decent; this is a first version.

### Validation
N/A.

#### Where to run?
On a cluster of NVL72 systems.

#### What to run?
Regulat NCCL test suite.

#### Expected output?
No errors.

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
  - Sylvain Jeaugey

</details>
