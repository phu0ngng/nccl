# Proxy Affinity
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

Being able to control the proxy affinity is helpful in some debugging scenarios, or to comply with the OMP placement.
In this feature we provide a new env variable: `NCCL_PROXY_CPUSET` that can be used to bind the proxy threads to a specific set of cores.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

- [NVBUG 4860089](https://nvbugs/4860089) RFE: Provide option to control NCCL proxy thread affinity.

<!-- ### User Experience

### Assumptions, constraints and dependencies

### Use Cases

### Platform Requirements -->

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

The proxy threads are created in the following order:
- init thread call `ncclProxyCreate`, which will create both `proxyService` and `proxyServiceUDS` threads
- `proxyService` thread calls `ncclProxyProgressCreate` that will create the `proxyProgress` thread.

By default, each of the thread created with `pthread_create` inherits the affinity maks of the paren thread.
Therefore, only changing the affinity of the `proxyService` and `proxyServiceUDS` is required.

The environment variable `NCCL_PROXY_CPUSET` accepts a (comma-separated) list of core indexes.
The indexes can be given either in decimal, or in hexadecimal with the prefix `0x`, see `ncclStrListToCpuset`.



<!-- note: the following HTML code is also valid -->
<!-- <img src="images/example.png" width="900" height="500" /> -->

<!-- ### Interface Architecture -->

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
  - Thomas Gillis

</details>
