# IB modify QP retry
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract

At connection, the function `ibv_modify_qp` can fail for various reasons (link flap, authentication issue, etc)
This feature provides a retry capability in case of failure, together with important information to debug if needed

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

### User Experience

### Assumptions, constraints and dependencies

### Use Cases

### Platform Requirements


</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed Design


- the changes are contained within `ibvwrap.cc` to provide a common way to handle that issue across NCCL IB plugins
- by default, only timeout are retried.
- the total number of retries is controled by `NCCL_IB_MQP_RETRY_CNT` (default `34`). Before each retry, we sleep for a time given by `NCCL_IB_MQP_RETRY_SLEEP_MSEC` (default `100`). As we increase the sleep time each retry, the total sleep time across all retries is given by `(NCCL_IB_MQP_RETRY_CNT+1)/2 * NCCL_IB_MQP_RETRY_CNT * NCCL_IB_MQP_RETRY_SLEEP_MSEC`.
- on some systems, retrying other errors than the timeout is needed as a temporary measure. This can be done using `IB_MQP_RETRY_ALL=1` (default `0`).

<!-- ### Interface Architecture -->
<!---->
<!-- <!-- ### System KPIs & Metrics --> -->
<!-- <!-- ### Data Architecture --> -->
<!-- <!-- ### Security Design --> -->
<!-- <!-- ### Debugging & Troubleshooting --> -->
<!-- <!-- ### Logging and Instrumentation --> -->
<!-- <!-- ### Operational Considerations --> -->

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Coding</h2></summary>
<!-- ============================================================================================-->

### Commit list or MR

- [MR 592](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/592)
- [MR 655](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/655)

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
  - Marina Varshaver
  - Sreeram Potluri

</details>
