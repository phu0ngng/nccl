# Quality_of_service
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->
To enable Quality of Service (QoS) for applications with concurrent networking tasks, NCCL should introduce an interface for configuring QoS levels during communicator initialization, propagate these settings to network plugins through a flexible and extensible API, and ensure backward compatibility to maintain seamless operation for applications that do not require QoS.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

QoS in network communication is critical for many scenarios. For example, during LLM training, various types of network communications often overlap, such as pipeline parallelism (PP) and data parallelism (DP) communications. Among these, PP communication typically resides on the critical path, while DP communication does not. When these communications overlap, both experience slowdowns as they contend for shared network resources. By prioritizing communication on the critical path (e.g., allocating greater bandwidth to PP communication), the end-to-end performance of the benchmark can be significantly improved.

To enable QoS for diverse application networking tasks, NCCL should provide applications with the ability to specify higher QoS levels for specific communicators. This requires the following enhancements:

(1) QoS Configuration Interface: NCCL must introduce an interface allowing applications to set the QoS level during communicator initialization.

(2) QoS Propagation to Network Plugins: The NCCL core should propagate QoS settings to the underlying network plugins. This requires extending the NCCL network plugin API to allow for the exchange of QoS-related information.

(3) Extensibility: The design of the extended network plugin API must be modular and extensible, enabling future enhancements without breaking existing functionality.

(4) Compatibility: The QoS enhancements should be designed to remain non-disruptive for applications that do not require QoS. In such cases, the system should function as it does currently, without any changes to performance or behavior.

### NVbugs / Jira Tickets
https://nvbugspro.nvidia.com/bug/4795873

https://nvbugspro.nvidia.com/bug/4915028
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

To enable applications to configure QoS, we propose adding a new `trafficClass` field to the existing `ncclConfig_t` structure. The `trafficClass` is an integer that serves as an abstract representation of the QoS level for communicator network traffic. Applications can set this field using either a default value or a user-defined setting. The specific meaning of the `trafficClass` is determined by the network system administrator and the network stack implementor.

```
struct { 
  ... 
  int splitShare; 
  int trafficClass; // add trafficClass to communicator  
} ncclConfig_t 
```

The NCCL core passes the `trafficClass` to the network plugin, ideally without modifying it. Thus, the context of the `trafficClass` is oblivious to the NCCL core.

We propose to extend the network plugin interface by introducing a new struct called `ncclNetCommConfig_t`, which encompasses the `trafficClass`. Such `ncclNetCommConfig_t` is passed to each connections in the network plugin. For instance, the `connect` function is being appended an input argument `ncclNetCommConfig_t`* config.

```
struct {
  int trafficClass; // Plugin-specifig trafficClass value
} ncclNetCommConfig_t

ncclResult_t (*connect) (int dev, ncclNetCommConfig_t* config, void* handle, void** sendComm, ncclNetDeviceHandle_v10_t** sendDevComm);
```

Network plugin implementors can utilize the `trafficClass` value to set specific fields in their network packets to enable QoS. The extended plugin interface is designed to support both internal and external network plugins. For example, the value can be passed to the NCCL internal net_ib plugin, where it can be used to configure the `qpAttr.ah_attr.grh.traffic_class` field if the `trafficClass` is defined.

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
  - Zhenhao He <zhenhaoh@nvidia.com>

</details>
