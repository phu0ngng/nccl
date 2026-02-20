# Pre-posting Receive WQE
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract

This PLC describes the implementation of Receive Work Queue Element (WQE) pre-posting to enhance communication efficiency in NCCL and allow future support for port-failover. The pre-posting of Receive WQEs allows for reduced latency and improved throughput by ensuring that the necessary resources are available before data transmission begins.

Receive WQEs pre-posting is a pre-requisite for enabling port-failover capabilities in NCCL, which will enhance the robustness and reliability of communication in multi-path network environments. Port-failover allows NCCL to seamlessly switch between different network paths/devices in case of failures, ensuring continuous data flow and minimizing disruptions. In case of device failures, pre-posting Receive WQEs enables the receiver to accept data from other functional devices without knowing apriori on which device or devices each data transfer will be sent.

> **Note**
>
> Enabling the feature of pre-posting Receive WQEs requires the ID-based matching scheme (`NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1`) to be enabled as well, since the receiver needs to be able to match completions with pre-posted receive requests.

<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

* [[Feature Request] NCCL Failover Support on CX8 (NVBug #5256433)](https://nvbugspro.nvidia.com/bug/5256433)

### User Experience

The feature is configurable via the following environment variable:
- `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS`
  * Value of `0` (default): disables Receive WQE pre-posting. The receiver posts Receive WQEs as prior to this change, i.e., on-demand, upon a call to `ncclIbIrecv()`.
  * Value of `1`: enables Receive WQE pre-posting. The receiver pre-post receive WQEs in advance, upon connection establishment (and continues to post more as needed when receive WQEs are consumed and completed).

> **Note**
>
> Enabling the feature of pre-posting Receive WQEs requires the ID-based matching scheme to be enabled as well, since the receiver needs to be able to match completions with pre-posted receive requests.

Receive

Receive

The feature is disabled by default (i.e., `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=0`).

### Assumptions, constraints and dependencies

No system assumptions or constraints are introduced by this feature. The feature is optional and disabled by default.

This feature depends on the ID-based matching scheme being enabled (`NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1`).

### Use Cases

The main use-cases for this feature are:
1. Enabling port-failover capabilities in NCCL, allowing for seamless switching between different network paths/devices in case of failures.
2. Enhancing communication efficiency by reducing latency and improving throughput through pre-posting of Receive WQEs.

#### Port-failover use-case

Port-failover allows NCCL to seamlessly switch between different network paths/devices in case of failures, ensuring continuous data flow and minimizing disruptions. To facilitate this, when some device fails, the receiver side should be able to accept data from other functional devices (the devices that the sender fails-over to) without knowing apriori on which device or devices each data transfer will be sent, which is made possible by Receive WQE pre-posting combined with the ID-based matching scheme.

#### Imroved communication latency

By pre-posting Receive WQEs, the receiver ensures that the latency from the time `ncclIbIrecv()` is called until a doorbell is rang is minimized.

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

The feature is platform agnostic and does not have any specific platform requirements.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Design

#### Pre-posting of Receive WQEs

When enabled, the pre-posting of Receive WQEs is performed during the connection establishment phase, right before receiver declares the readiness of the connection and returns the communicator to the caller. In addition, each time the receiver issues a completion of a receive request, it posts a new receive request to "replace" the completed one, ensuring that there are always enough receive WQEs posted to handle incoming data.

##### Identical Receive WQEs

The pre-posting is posting WQEs with the exact same parameters. The address parameters are dummy adresses as the sender uses RDMA Write (with Immediate) operations to send the data that specify the exact address in memory to which the data is written so the **addresses specified by the receive WQEs on the receiver side are not in use**. The `wr_id` that every receive WQE is pre-posted with is yet again a unique value (`NCCL_IB_RECV_WR_ID_DUMMY`) that would help debug on the receiver side and in some cases could help identify errors on when device fails.

As pre-posting Receive WQEs is only possible when the ID-based matching scheme is enabled, the receiver does not rely on `wr_id` to match completions with receive requests, and hence using a dummy value for `wr_id` is safe.

##### Cleanup

The plugin does not cleanup pre-posted Receive WQEs upon connection teardown, as the underlying RDMA resources are cleaned up by the connection teardown logic of the rdma-core driver.

Under "normal" cirmustances (i.e., no device failures, no late packet arrivals, no retransmission from the sender side. etc.), upon a connection close, no packets should arrive to the receiver side (as the sender and receiver close the connection when all requests have been completed), so letting the RDMA driver cleanup the pre-posted Receive WQEs is safe.

In case there is a need for cleanup of pre-posted Receive WQEs upon connection teardown (e.g., in case of device failures, late packet arrivals, etc.), the plugin would need to implement logic to modify QPs to error state and then the device will complete all the pre-posted Receive WQEs with error completions, which the plugin would need to poll from the CQ. Afterwards, it's safe to teardown the connection and destroy the QPs.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Coding</h2></summary>
<!-- ============================================================================================-->

### Commit list or MR

* [MR!1400](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/1400)

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Validation

To validate the implementation of Receive WQE pre-posting, the following environment variables should be set:
* `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1` (required dependency)
* `NCCL_IB_RECEIVE_WQE_PRE_POSTING=1`

Note: It's not possible to test the Receive WQE pre-posting alone, as it requires the ID-based matching scheme to be enabled as well.

#### Where to run?

Systems that supports NCCL with InfiniBand transport can be used to run the tests, so the network IB plugin is used.

#### What to run?

The standard NCCL test suite can be used to validate the implementation. This includes running various collective operations (e.g., all-reduce, broadcast, reduce, etc.) with different message sizes and numbers of processes.

Specifically, need to cover zero-sized messages, small messages (less than ArThreshold), and large messages (greater than ArThreshold), as well as single and multiple send/receive requests.

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

The tests should complete successfully without any errors or crashes. The results of the collective operations should be correct and match the expected output.

### Performance

Pre-posting Receive WQEs is expected to reduce the latency of communication operations and should not negatively impact performance.

#### Results

Numbers were not collected.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Rami Nudelman
  - Asaf Schwartz

</details>
