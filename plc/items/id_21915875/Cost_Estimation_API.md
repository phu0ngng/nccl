# Cost Estimation API
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->
Cost estimation API exposes NCCL's internal model. Users can use this API to query different
attributes of simulation.
<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->
Google requested this feature in NCCL to optmize XLA compiler.
This compiler will invoke the newly created API to determine the configuration for the best performance.
In the original request Google asked to expose NCCL internal model to estimate time for an operation.

### NVbugs / Jira Tickets
Aha link: https://nvaiinfa.aha.io/features/MIONCCL-107

### User Experience
None.

### Assumptions, constraints and dependencies
In the case of two or more aggregated operations, NCCL will not properly simulate the multiple
operations, and instead return the time of the last operation?

### Use Cases
Any user who wants to optimize their operations by querying how much time an operation could take.
As of v2.22 XLA team will be using this to optimize the compiler.

### Platform Requirements
None.

<!-- ### Functional Requirements -->
<!-- ### System Requirements -->
### Interface Requirements
Added a new API: `ncclResult_t  ncclGroupSimulateEnd(ncclSimInfo _t* simInfo);`

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
NCCL's internal tuner determines the best algo/proto combination and the estimated time for the operation.
For group operations, instead of calling `ncclGroupEnd()` as before, the caller will call `ncclGroupSimulateEnd(ncclSimInfo_t*)` and send
a `ncclSimInfo_t*` through the new API. This will be passed down the call stack to NCCL’s internal tuner. Estimated time will be saved
in the `ncclSimInfo_t`’s `estimatedTime` field.
We will skip the code that launches kernel because we only want the estimation.

New API: `ncclResult_t  ncclGroupSimulateEnd(ncclSimInfo _t* simInfo);`
This parameter will be passed through the following call stack (as of v2.22):
`ncclGroupSimulateEnd -> ncclGroupEndInternal -> groupLaunch -> ncclPrepareTasks -> getAlgoInfo -> topoGetAlgoInfo`
In the `topoGetAlgoInfo`, the variable will be populated with a min value.

### Interface Architecture
User should create a `ncclSimInfo` struct and initialize it with `NCCL_SIM_INFO_INITIALIZER`.
This will be validated by the magic field in the structure.
This structure will be passed to `ncclGroupSimulateEnd`
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
6d946ffd2bf2d33fd00d37c46e9d5435885612c6
e7a14e6b64cf0ad8756c04a6079dc25049190449
f074465f96af4a5addaa98ab5c5ed2b3939b823a
49c27838a0a8a442fe5b0fafcb6d49a32c1d2257
89045ed7876b5d1d33f12f2799fb259b79c3405b
690aea32eaab0479225ea975f3037bac278a00e9

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline
Test that it does not break any existing flows.
Test that when enabled, estimated time is displayed.

### Validation

#### Where to run?
Any cluster.

#### What to run?
When testing the feature itself, gitlab perf tests should be run with the new flag `-E1`.

#### Expected output?
When simulate mode is enabled in tests, additional column is displayed:
```
#       size         count      type   redop    root     time   algbw   busbw  #wrong   esttime     time   algbw   busbw  #wrong   esttime  #iters
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)              (us)     (us)  (GB/s)  (GB/s)              (us)
        1024           256     float     sum      -1    30.86    0.03    0.06       0     20.96    30.04    0.03    0.06       0     20.96      20
        2048           512     float     sum      -1    30.23    0.07    0.12       0     21.32    30.09    0.07    0.12       0     21.32      20
        4096          1024     float     sum      -1    30.99    0.13    0.23       0     22.03    30.68    0.13    0.23       0     22.03      20
        8192          2048     float     sum      -1    32.28    0.25    0.44       0     23.47    32.15    0.25    0.45       0     23.47      20
       16384          4096     float     sum      -1    35.88    0.46    0.80       0     26.33    35.58    0.46    0.81       0     26.33      20
       32768          8192     float     sum      -1    43.61    0.75    1.31       0     32.07    43.28    0.76    1.33       0     32.07      20
       65536         16384     float     sum      -1    46.86    1.40    2.45       0     43.54    46.15    1.42    2.49       0     43.54      20
      131072         32768     float     sum      -1    69.22    1.89    3.31       0     66.48    69.11    1.90    3.32       0     66.48      20
      262144         65536     float     sum      -1    118.2    2.22    3.88       0     112.4    118.1    2.22    3.88       0     112.4      20
      524288        131072     float     sum      -1    185.9    2.82    4.94       0     180.0    186.0    2.82    4.93       0     180.0      20
     1048576        262144     float     sum      -1    280.0    3.74    6.55       0     271.7    278.9    3.76    6.58       0     271.7      20
     2097152        524288     float     sum      -1    472.4    4.44    7.77       0     455.2    469.1    4.47    7.82       0     455.2      20
     4194304       1048576     float     sum      -1    864.1    4.85    8.49       0     822.2    864.7    4.85    8.49       0     822.2      20
     8388608       2097152     float     sum      -1   1641.9    5.11    8.94       0    1556.2   1641.3    5.11    8.94       0    1556.2      20
    16777216       4194304     float     sum      -1   3197.0    5.25    9.18       0    3024.2   3199.5    5.24    9.18       0    3024.2      20
    33554432       8388608     float     sum      -1   6314.6    5.31    9.30       0    5960.2   6306.8    5.32    9.31       0    5960.2      20
    67108864      16777216     float     sum      -1    12576    5.34    9.34       0     11832    12584    5.33    9.33       0     11832      20
   134217728      33554432     float     sum      -1    25137    5.34    9.34       0     23576    25171    5.33    9.33       0     23576      20
   268435456      67108864     float     sum      -1    50141    5.35    9.37       0     47064    50132    5.35    9.37       0     47064      20
   536870912     134217728     float     sum      -1    99918    5.37    9.40       0     94041    99939    5.37    9.40       0     94041      20
  1073741824     268435456     float     sum      -1   199661    5.38    9.41       0    187993   199634    5.38    9.41       0    187993      20
```
<!-- #### Code Coverage Goal Defined -->
<!-- #### KPI Coverage Goals Defined (performance, stress, stability, throughput, latency) -->
<!-- #### Requirement Coverage Goal Defined -->
<!-- #### Quality Thresholds Defined -->
<!-- #### Test Timeline -->
<!-- #### SW Verification and Test Plan -->
<!-- ### Test plan -->
<!-- #### Requirements Tests -->
#### Interface Tests
Added 2 API tests:
1. Added a test that validates the simulate mode. (No crash)
2. Added a negative test that sends an uninitialized struct, ncclInternalError expected.
<!-- #### Fault-injection Tests -->
<!-- #### Resource Usage Tests -->
<!-- #### Design Coverage Testing -->
<!-- #### Boundary Tests -->
<!-- #### Certification Tests -->
<!-- #### Stress Tests -->
<!-- #### Stability Tests -->
#### Perf and Power KPI Tests
Changed Perf tests to add a column for estimated time and add a command line option to enable simulate mode.

<!-- #### Usability & OOBE Tests -->
<!-- #### Manufacturing Diagnostic(Factory) Tests -->

### Performance

#### What is measured?
Estimated time taken for an operation is measured and displayed.

#### Results


</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Unmesh Deodhar <udeodhar@nvidia.com>

</details>
