# collCostTable

<details>
<summary><h2>Requirements</h2></summary>

### Project Context and Purpose
The tuner API allows external tuner plugins to supply a combination of algorithm, protocol and number of channels that NCCL will use for the configuration of collectives. The problem is that NCCL does not provide the external plugins with enough context (e.g. combinations of algorithm/protocol/channel that are incompatible with the detected topology) allowing them to make an informed decision.

### NVbugs / Jira Tickets
Aha link: https://nvaiinfa.aha.io/features/MIONCCL-114

<!-- ### User Experience -->
<!-- ### Assumptions, constraints and dependencies -->

### Use Cases
External tuner plugins need to provide NCCL with a algorithm/protocol combination that makes sense to NCCL (is compatible with the NCCL detected topology).

### Functional Requirements
The external tuner plugins should receive enough information from NCCL about the allowed algorithm/protocol combinations supported for the detected topology. The tuner can then supply NCCL with a algorithm/protocol combination that is consistent with the NCCL provided information.

<!-- ### System Requirements -->

### Interface Requirements
The `getCollInfo` call in the external tuner plugin API needs to be updated so that NCCL can provide the required topology context. The version of the external tuner API need to be updated from v2 to v3 to account for the change.

<!-- ### KPI Requirements -->
<!-- ### Platform Requirements -->
<!-- ### Security Requirements -->
<!-- ### Legal and Standards Requirements -->
<!-- ### Telemetry Requirements -->

### Backward Compatibility Requirements
A compatibility layer needs to be provided to support the new v3 interface on top of the old v2 interface.

<!-- ### Virtualization Requirements -->

### Signoff list

Author :
Giuseppe Congiu <gcongiu@nvidia.com>

<details>
<summary><h2>Design</h2></summary>

### Proposed Design
NCCL supplies the external tuner plugin with a per-collective 2D cost table, reporting the estimated time needed to carry out the operation for every combination of algorithm and protocol. NCCL sets the table entries that are not compatible with the detected topology to –1, to indicate to external tuners that these combinations are not supported/allowed to be overwritten. To select one specific combination, the external tuners update the value for that combination to 0 (or the min value across the whole table). After the plugin has updated the cost table, NCCL can use it to select the final collective configuration for the given collective.

External Tuner v3 interface: `ncclResult (*getCollInfo)(void *context, ncclFunc_t collType, size_t nBytes, int numPipeOps, float **costTable, int numAlgo, int numProto, int *nChannels);`

The `costTable` is a 2D table containing estimated execution time for different combinations of algorith/protocol as entries. Invalid entries are set to -1.
Additionally, because the collective cost table now embeds topology information, nvlsSupport and collNetSupport are removed from the `getCollInfo` argument list.

<!-- ### Interface Architecture -->
<!-- ### System KPIs & Metrics -->
<!-- ### Data Architecture -->
<!-- ### Security Design -->
<!-- ### Debugging & Troubleshooting -->
<!-- ### Logging and Instrumentation -->
<!-- ### Operational Considerations -->

### Signoff list

Author :
Giuseppe Congiu <gcongiu@nvidia.com>
</details>

<details>
<summary><h2>Coding</h2></summary>

### Commits

commit 688483f8082e52df7ba6a82e3b123e53a72509e2

commit cc29ebd2b1cf9675cceaa7e4c867e9910ab97cc7

commit c85f8176cd6bf49137def835d8ed31f057034254

commit 24210208bde9b2006e31a222756e6f156337a413

</details>

<details>
<summary><h2>Testing</h2></summary>

<!-- ### Objectives and Timeline -->
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

</details>

<details>
<summary><h2>Signoff List</h2></summary>

Author(s):

</details>
