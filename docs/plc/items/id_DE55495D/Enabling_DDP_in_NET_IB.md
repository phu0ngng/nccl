# Enabling DDP in NET IB
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

CX-8's out-of-order mode (aka DDP - Direct Data Placement) allows the NIC to deliver RDMA payloads out of order to improve throughput and latency. This capability of NIC is supported by default in both IB and Spectrum-X E2E mode, but will not be supported if running in pure RoCEv2 mode. This PLC describes how to enable DDP in the control path of internal net_ib transport.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

[NVBUG 5803369: Enable DDP (Direct Data Placement) in NET/IB transport](https://nvbugspro.nvidia.com/bug/5803369)

### Background

#### Adaptive Routing for NCCL
Adaptive Routing (AR) was first introduced with the CX-5 NIC, allowing responders to handle `RDMA_READ_RESPONSE` and `RDMA_WRITE` packets that arrive out of order.
AR-capable data transfer in NCCL is enabled by default on both InfiniBand (IB), not on RoCEv2 (including SPC-X) platforms, using `NCCL_IB_ADAPTIVE_ROUTING` variable:

```C
ncclIbDevs[ncclNIbDevs].ar = (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
if (ncclParamIbAdaptiveRouting() != -2) ncclIbDevs[ncclNIbDevs].ar = ncclParamIbAdaptiveRouting();
```

By leveraging AR, packets can be sprayed into the multiple network paths in fabric, helping to alleviate network congestion and maximize overall performance.

With Adaptive Routing (AR), `RDMA_WRITE` operations are transmitted as `RDMA_WRITE_ONLY` packets.
Each packet includes a RETH header, specifying both the destination address and payload size. This allows the NIC to perform direct data placement to the target receiver buffer, without the need for intermediate staging. As a result, these packets can be safely delivered out-of-order; the `free_ar` bit in the BTH header enables their adaptive routing. By allowing packets to travel different paths and balancing load on a per-packet basis, this mechanism enhances overall efficiency and minimizes network contention. While adding the RETH header incurs a slight decrease in bandwidth utilization, the gains in throughput and reduced congestion outweigh this minor drawback.

This packet-spray and out-of-order delivery mechanism is not applicable for `RDMA_WRITE_WITH_IMM` operations, which are typically used to signal the completion of preceding `RDMA_WRITE` messages to the peer.
The requester has to place a fence before sending `RDMA_WRITE_WITH_IMM` messages until all previous `RDMA_WRITE` messages are acked. As a result, `RDMA_WRITE_WITH_IMM` won't benefit from AR.

Below was the code snippet in `ncclIbIsend`:

```C
wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
wr.wr.rdma.remote_addr = slot->addr;
wr.wr.rdma.rkey = slot->rkey;
wr.imm_data = size; // Send the message size via imm_data
```
Historically, NCCL used only `RDMA_WRITE_WITH_IMM` for data transfers. However, relying exclusively on `RDMA_WRITE_WITH_IMM` leads to performance degradation when AR is enabled. To address this, NCCL was updated to use `RDMA_WRITE` for the data payload, enabling out-of-order delivery with AR and use `RDMA_WRITE_WITH_IMM` only for the final completion notification. This approach allows NCCL to fully leverage the benefits of AR:

```C
// RDMA_WRITE data
for (int r=0; r<nreqs; r++) {
  struct ibv_send_wr* wr = comm->wrs+r;
  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->wr.rdma.remote_addr = slots[r].addr;
  sge->addr=(uintptr_t)reqs[r]->send.data;
  // ...
}

uint32_t immData = ncclParamIbReceiverSideMatchingScheme() == BY_ID ? (uint32_t)(reqs[0]->id % UINT32_MAX) : reqs[0]->send.size;

struct ibv_send_wr* lastWr = comm->wrs+nreqs-1;
if (nreqs > 1 || (comm->ar && reqs[0]->send.size > ncclParamIbArThreshold())) {
  // When Adaptive Routing is enabled, send the bulk of the data first as an
  // RDMA Write.
  lastWr++;
  memset(lastWr, 0, sizeof(struct ibv_send_wr));
  if (nreqs > 1) {
    // Write remote sizes array
    lastWr->wr.rdma.remote_addr = comm->remCmplsRecords.addr + slot*sizeof(struct ncclIbRequestCompletionRecord);
    lastWr->num_sge = 1;
  }
}
lastWr->wr_id = wr_id;
lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
lastWr->imm_data = htobe32(immData);
lastWr->next = NULL;
lastWr->send_flags = IBV_SEND_SIGNALED;
```

The `NCCL_IB_AR_THRESHOLD` parameter controls whether the payload data and completion notification are sent together or separately when `nreqs=1` (single request). By default, it is set to 8192 bytes.

#### DDP and Its Impact on NCCL
The primary goal of DDP is to improve network efficiency of AR by allowing all types of RDMA packets to be transmitted and received out-of-order. With DDP, the responsibility for maintaining completion order shifts from the requester to the responder: the responder consumes WQEs as soon as out-of-order packets arrive, buffers and reorders data as necessary, and ensures completions are delivered strictly in order.
However, the legacy NCCL WQE matching scheme assumes in-order WQE consumption and does not support out-of-order WQE consumption, thus, it will not function correctly with DDP.
Out-of-order WQE consumption [ID-based matching scheme](../id_4bde6b8f/ID-based_matching_scheme.md) is already present in NCCL and can be used as a solution.
This scheme can be force enabled by setting `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1`

> `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME` is not documented, and users should not tweak it in normal cases.

With DDP enabled, separating the data and notification transfers is unnecessary for single request (`nreqs=1`) because the responder guarantees ordering, and the `RDMA_WRITE_WITH_IMM` packet that carries both payload and notification won't be delayed.

An extra `RDMA_WRITE_WITH_IMM` for notification is only needed in two cases:

1. `nreqs > 1`. DDP enabled or not does not change multi-send.
2. `nreqs=1`, DDP not enabled, and send size exceeds AR threshold.


### User Experience

DDP will be enabled on the QPs NCCL creates of NCCL, after querying the device sees that the device supports DDP.

### Assumptions, constraints and dependencies

#### Environment Requirements

1. The devices on both sides support DDP (by querying). If use NIC Fusion, all devices must support DDP.
2. ID-based matching scheme enabled: `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1`.
This is a must-have since out-of-order work completion matching relies on this scheme.
3. `NCCL_IB_ADAPTIVE_ROUTING=1`.
4. Pre-posting recv wqe enabled: `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1`.
This is for perf requirement. By enabling it, receiver side will pre-post WQEs during connection establishment.
It should be noted that this will break net profiler, since tracking of the wqe was done in `ncclIbIrecv`.
5. AR-threshold `NCCL_IB_AR_THRESHOLD` will be ignored if DDP enabled.


#### Performance Issues Solved by Pre-posting Receive WQEs

In legacy NCCL implementation, the receiver will post only one recv WQE for RDMA_WRITE_WITH_IMM per irecv, and since the sender has a fence before sending the second RDMA_WRITE_WITH_IMM,
there's no out-of-order arrival of RDMA_WRITE_WITH_IMM when AR enabled. With DDP the sender will not be responsible for the control of out-of-order and it's possible RDMA_WRITE_WITH_IMM
arrivs out-of-order. If the receiver does not have available receive wqes, it may return RNR ACK packet to the sender and trigger retransmission, which can cause performance drop.

By pre-posting `NET_IB_MAX_REQUESTS` receive WQEs per QP, it's guaranteed there are enough recv WQEs to reduce the chance of RNR.

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

#### 1. Behavior of `NCCL_IB_OOO_RQ`

This env variable controls whether enable DDP if possible.

- `0`: default value, disable DDP.
- `1`: Force enable DDP, fail if DDP not supported or requirements not met.
- `2`: Enable DDP if possible, fallback to normal QP creation if DDP cannot be enabled.

#### 2. Query and align DDP capabilities on both sides

Traverse NIC and query their DDP capability, record in `ncclIbDev.oooRqSize` which is a per-device indicator. If NIC Fusion is used, DDP is enabled only if all devices support DDP.
By setting `MLX5DV_CONTEXT_MASK_OOO_RECV_WRS` mask `mlx5dv_context` will include `struct mlx5dv_ooo_recv_wrs_caps` as `ooo_recv_wrs_caps`.

```C
struct mlx5dv_ooo_recv_wrs_caps {
        uint32_t max_rc;
        uint32_t max_xrc;
        uint32_t max_dct;
        uint32_t max_ud;
        uint32_t max_uc;
};
```

Since NCCL uses only RC, the whole querying process would be:

```C
dvCtx.comp_mask = MLX5DV_CONTEXT_MASK_OOO_RECV_WRS;
if (wrap_mlx5dv_query_device(ibvCtx, &dvCtx) == ncclSuccess) {
  if (dvCtx.comp_mask & MLX5DV_CONTEXT_MASK_OOO_RECV_WRS && dvCtx.ooo_recv_wrs_caps.max_rc > 0) {
    oooRqSize = dvCtx.ooo_recv_wrs_caps.max_rc;
    if (oooRqSize <= MAX_REQUESTS) {
      WARN("NET/IB : Maximal RQ size(%u) is not large enough on %s", oooRqSize, devName);
      oooRqSize = 0;
    }
  }
}
```

A device is marked as `DDP enabled` only when:
1. `oooRqSize` queried from the device is not 0
2. `oooRqSize <= MAX_REQUESTS`
3. Adaptive Routing is supported.

It should be noted that `NCCL_IB_ADAPTIVE_ROUTING=1` does not enable AR on the NIC/Switch,
it controls the way NCCL submitting WQE. However, if user set `NCCL_IB_ADAPTIVE_ROUTING=0` then DDP should be disabled as well.


When using out-of-band socket to exchange `ncclNetVDeviceProps_t`, we add a trailing data `ncclIbDevExtraProps` to align DDP capability on both sides.

```C
struct ncclIbDevExtraProps {
  bool oooRq;
};
```

Then, we check the requirements:
1. `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME`: We will ignore this behavior if DDP can be enabled. This should follow the behavior of SPC-X.
2. `oooRqSize` on both sides
3. perf related requirements, print WARN if not met. This will not lead to hang but can lead to perf degradation.

#### 3. Enable DDP during QP creatiion

DDP configuration is per QP through devx API.
`MLX5DV_QP_CREATE_OOO_DP` was already added into rdma-core and creating a QP with this attributite will be done using devx API.

```C
if (qp->oooRq) {
  dvAttr.create_flags |= MLX5DV_QP_CREATE_OOO_DP;
  dvAttr.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
}
qp->qp = wrap_mlx5dv_create_qp(ncclIbDevs[base->ibDevN].context, &qpInitAttr, &dvAttr);
```
#### 4. AR-related logic
As describe in previous sections, if DDP enabled, for `nreqs=1` there's no need to split the payload and immData notification.

```C
if (nreqs > 1 || (!(comm->base.remOooRq && comm->base.localOooRq) && comm->ar && reqs[0]->send.size > ncclParamIbArThreshold())) {
lastWr++;
// .... sent notification in a separate message
}
```

### All Related Environment Variables

**Overview**
| # | Variable                                | Role                                                                       | Exposed to User | Default                                  |
|---|-----------------------------------------|----------------------------------------------------------------------------|-----------------|------------------------------------------|
| 1 | `NCCL_IB_OOO_RQ`                        | Controls DDP enablement                                                    | Y               | Auto (enabled)                           |
| 2 | `NCCL_IB_ADAPTIVE_ROUTING`              | Controls NCCL's protocol on AR-enabled systems (`comm->ar == 1`)           | Y               | Auto (IB -> enabled, RoCEv2 -> disabled) |
| 3 | `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME` | Controls NCCL's the matching scheme                                        | N               | Auto (By-Index)                          |
| 4 | `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS` | Controls NCCL's pre-posting feature                                        | N               | Auto (disabled)                          |
| 5 | `NCCL_IB_AR_THRESHOLD`                  | Controls NCCL's protocol on AR-enabled systems (irrelevant when `nreqs>1)  | Y               | Auto (resolves to 8192)                  |

- id-matching scheme & pre-posting does not have auto mode now.  ==> we add auto-mode in this release
- NCCL_IB_AR_THRESHOLD does not have auto mode now.  ==> we don't need auto mode for this NCCL_IB_AR_THRESHOLD, no change required.

**User Toggle**
- `NCCL_IB_OOO_RQ=Disable`: DDP will be disabled, nothing will change.
- `NCCL_IB_OOO_RQ=Enable`: If any one from the above checklist does not meet, an error should be returned.
  - Functional requirements: ID-matching scheme. If not meet, return error.
  - Perf reqruirements: pre-posting. If not meet, enable with a WARN.
- `NCCL_IB_OOO_RQ=Auto`: (Default) NCCL will automatically enable DDP if all requirements were met.
  - If functional not met: silently disable DDP.
  - If perf not met: silently disable DDP as well.

| NCCL_IB_OOO_RQ |  State  |
| ---------------| --------|
| 0              | Disable |
| 1              | Enable  |
| 2              | Auto    |

In this release, `Disable` is the default value. We will change the default value to `Auto` in future releases.

#### Uniform Tri-State Scheme

All variables follow the same pattern: **auto / explicit disable / explicit enable**.

> **Note**
>
> When a variable is set to be a specific value (and not auto) explicitly but the corresponding feature cannot be activated (due to hardware limitations or other variable settings) or it conflicts **functionally** with another variable that has strict required on a different value - NCCL will **fail with an error**.

"On"/"Off" variables:

The following variables are considered "on"/"off" switches. The following table describes each variable's behavior under the tri-state scheme, and how it interacts with DDP enablement.

| # | Variable                                | Auto (default)                          | Explicit disable      | Explicit enable |
|---|-----------------------------------------|-----------------------------------------|-----------------------|-----------------|
| 1 | `NCCL_IB_OOO_RQ`                        | Enabled by default                      | DDP disabled          | DDP on or fail  |
| 2 | `NCCL_IB_ADAPTIVE_ROUTING`              | Auto per link layer (IB→on, RoCEv2→off) | AR disabled           | AR on           |
| 3 | `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS` | DDP can decide to enable       | Pre-post disabled (Perf risk under DDP, but allowed.) | Pre-post recv WQEs |

"Numeric"/"Enum" variables:

The following are variables that accept numeric or enum values.

| # | Variable                                | Auto (default)                             | Explicit value                                                |
|---|-----------------------------------------|--------------------------------------------|---------------------------------------------------------------|
| 1 | `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME` | DDP can decide to assign "By-ID"  | Legacy matching (blocks DDP) | ID-based matching |
| 2 | `NCCL_IB_AR_THRESHOLD`                  | System default (8192), ignored when DDP on | `<value>`: honored (even under DDP) or ignored when `nreqs>1` |

#### Prerequisite Classification

##### OOO_RQ prerequisites


###### Functional prerequisites (correctness — block DDP if not met)

* If HW does not support DDP (`oooRqSize > MAX_REQUESTS`), DDP cannot be enabled. (Query from NIC's capability. All devices must support DDP if it's a merged devices).
* If one of the sides does not support DDP - DDP cannot be enabled. (Out-of-band negotiation)
* Enabling DDP requires `NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME` to be `By-ID` (for correctness) and DDP cannot/must not be enabled if the matching scheme is not `By-ID`.
* DDP is allowed to be enabled only if AR is enabled. (i.e. `comm->ar == 1`).

> Why Adaptive Routing (AR) is a hard requirement for DDP?
By taking this more restrictive approach, we do not allow users to set `NCCL_IB_ADAPTIVE_ROUTING` and toggle DDP enablement. This restrictive approach can always be relaxed in the future in case needed.

Furthermore, even though there is no strict NCCL implementation that mandates this restriction, having `NCCL_IB_ADAPTIVE_ROUTING=0` and `OOO_RQ=1` is not very well defined (from the overall system perspective and whether RDMA Write with Imm packets will be subjected to AR or not) - so forbidding such configuration allows us not to deal with such questions.

Another point is that `NCCL_IB_ADAPTIVE_ROUTING` enables kind of "multipathing" in NCCL. `OOO_RQ` also, in a way add multipathing in NCCL. Having AR=0 but DDP=1 - makes it strange because when a user asks in this case whether multipathing is enabled - the answer is not clear.


###### Perf prerequisites (warn but allow DDP)
> No enforcement if not enabled, but a warning will be emitted. (Turn this on in auto-detection)

* `NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS` should be enabled. Without pre-posted recv WQEs, RNR retransmissions may degrade performance, but DDP is functionally correct

#### Behavior Matrix

##### `MATCHING_SCHEME` and `OOO_RQ`

How `MATCHING_SCHEME` and `OOO_RQ` interact:

| # | `OOO_RQ` | `MATCHING_SCHEME` | Result                                                                  |
|---|----------|-------------------|-------------------------------------------------------------------------|
| 1 | Disabled | <any>             | DDP off. Matching scheme can be determined by other factors. |
| 2 | Enabled  | Auto              | DDP overrides matching scheme to "By-ID". |
| 3 | Enabled  | By-Index          | **Error.** User says "force DDP" and "force legacy matching" — contradiction. |
| 4 | Enabled  | By-ID             | Aligned. No conflict. |
| 5 | Auto     | Auto              | If DDP is enabled DDP overrides matching scheme to "By-ID", otherwise matching scheme can be determined by other factors. |
| 6 | Auto     | By-Index          | User's explicit preference honored. DDP cannot be enabled → **silent fallback**, DDP disabled |
| 7 | Auto     | By-ID             | Aligned. If DDP is enabled by default, DDP can proceed (if other gates pass). |

##### `PREPOST`and `OOO_RQ`

How `PREPOST` and `OOO_RQ` interact:

| # | `OOO_RQ` | `PREPOST` | Result                                                                  |
|---|----------|-----------|-------------------------------------------------------------------------|
| 1 | Disabled | <any>     | DDP off. Pre-post is determined regardless of DDP. |
| 2 | Enabled  | Auto      | DDP overrides to 1. Clean path. |
| 3 | Enabled  | Disabled  | **DDP enabled + WARN.** Perf may degrade due to RNR retransmissions. Not a correctness issue. |
| 4 | Enabled  | Enabled   | Aligned. No conflict. |
| 5 | Auto     | Auto      | If DDP is enabled by default, pre-post is being enabled. Otherwise pre-post can be determined by other factors. |
| 6 | Auto     | Disabled  | DDP not enabled  **disabled silently.** |
| 7 | Auto     | Enabled   | Aligned. If DDP is enabled by default, DDP can proceed (if other gates pass). |

##### `ADAPTIVE_ROUTING` and `OOO_RQ`

How `ADAPTIVE_ROUTING` and `OOO_RQ` interact:

| # | `OOO_RQ` | `ADAPTIVE_ROUTING` | Result                                                                                              |
|---|----------|--------------------|-----------------------------------------------------------------------------------------------------|
| 1 | Disabled | <any>              | DDP off. AR is determined regardless of DDP.                                                        |
| 2 | Enabled  | Auto               | If AR is disabled --> Error (What about RoCEv2 (non-SPCX) systems? AR is disabled by default here!) |
| 3 | Enabled  | Disabled           | Error.                                                                                              |
| 4 | Enabled  | Enabled            | AR on. DDP proceeds.                                                                                |
| 5 | Auto     | Auto               | AR resolved per link layer. If AR disabled --> DDP disabled, If AR enabled --> see line 7           |
| 6 | Auto     | Disabled           | DDP will be disabled.                                                                               |
| 7 | Auto     | Enabled            | DDP is allowed to be enabled.                                                                       |

##### `AR_THRESHOLD` and `OOO_RQ`

How `AR_THRESHOLD` and `OOO_RQ` interact:

`AR_THRESHOLD` only has impact when `nreqs = 1`.

| # | `OOO_RQ` | `AR_THRESHOLD` | Result                                                                                 |
|---|----------|----------------|----------------------------------------------------------------------------------------|
| 1 | Disabled | <any>          | DDP off. Threshold used normally for AR split logic.                                   |
| 2 | Enabled  | <any>          | If DDP enabled (not gated by anything else), threshold will ignored                  . |
| 3 | Auto     | Auto           | If DDP enabled, then see line 2, if DDP disabled see line 1.                           |

> whether we need optional warnings if ar threshold set but DDP enabled and this value will be ignored.
If DDP enabled, AR threshold will be ignored, and an INFO log will contain this information.

NCCL has a default value for ar threshold, so it's fair to say we will ignore `NCCL_IB_AR_THRESHOLD`. If we find out it'd be better to change `AR_THRESHOLD` into auto mode, we can do it in another MR.

#### DDP Enablement Decision Flow

Bottom up, first NCCL needs to determine the values/behavior of the non-user exposed variables (`MATCHING_SCHEME`, `PREPOST`) and see if they have explicit constraints from user or if NCCL can override them under DDP. Then, based on the final values of all variables, NCCL can decide whether DDP can be enabled or not.

#### Summary Table

| # | `NCCL_IB_OOO_RQ` | DDP Result                                         | Functional pre. is not met | Perf. pre. is not met  |
|---|------------------|----------------------------------------------------|----------------------------|------------------------|
| 1 | Disabled         | Off                                                | Disabled (silently)        | Disabled (silently)    |
| 2 | Enabled          | Enabled or failure                                 | Failed (with error)        | Enabled (with warning) |
| 3 | Auto             | NCCL's internal choice (optimized for performance) | Disabled (silently)        | Disabled (silently)    |

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

### Validation

#### Where to run?

#### What to run?

1. Set the following env variables:

```
NCCL_IB_OOO_RQ=1
NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1
NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1
NCCL_IB_ADAPTIVE_ROUTING=1 # for SPC-X
```

2. Run nccl-tests or internal perftests

A perf gain should be observed, especially in small messages.


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
  - Junyu Ma
  - Rami Nudelman

</details>
