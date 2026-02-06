# Network Defined Event Update

## Abstract
Network defined events were added in the NCCL profiler interface in version v2.26. Such events support start and stop semantics but lack update, which is useful in many cases. One example is capturing network supplied information and passing this to the profiler plugin. This can not be done with start/stop semantics. Therefore, update semantics, throught the _recordEventState_ profiler interface, is added for network defined events.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets
[NetworkEventUpdates](https://jirasw.nvidia.com/browse/NCCL-1763)

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

We define a new network event _ncclProfilerNetPluginUpdate_ state and _netPlugin_ state argument in the profiler interface:

```C
typedef enum {
  ...
  ncclProfilerNetPluginUpdate,
} ncclProfilerEventState_t;

typedef union {
  ...
  struct {
    void* data;
  } netPlugin;
} ncclProfilerEventStateArgs_v4_t;
```

The _type_ argument in the _ncclProfilerCallback_ function is extended to support update (type = 2). When the network plugin invokes the callback with the update type, the plugin supplies the callback with the update data, through the _extData_ argument. The NCCL core code does the rest, preparing the update arguments and invoking the profiler _recordEventState_ function.

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

[NetworkEventUpdate](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/835)

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
  - Giuseppe Congiu <gcongiu@nvidia.com>

</details>
