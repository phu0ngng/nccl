# Harmonization of Proxy Events
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
The _ncclProfileProxyOp_ and _ncclProfileProxyStep_ events were added to the NCCL profiler plugin interface in v2.23 release.
The _ncclProfileProxyStep_ events were brought in from the legacy profiler infrastructure written by Sylvain, whereas the
_ncclProfileProxyOp_ events were brought in from the Meta Proxy Trace infrastructure in NCCL-X. The idea was to provide
different levels of granularity for the network proxy code. Thus, in the event tree hierarchy, the _ncclProfileProxyStep_
events are parented by _ncclProfileProxyOp_ events.

However, as currently defined, the two events practically record the same proxy state transitions. Therefore, in the code
(src/transport/net.cc), the instrumentation points for _ProxyOp_ and _ProxyStep_ events are duplicated. As an example, in the
case of the _sendProxyProgress_ function, the proxy states captured by the two events are: _ncclProfilerProxyOpSendPosted_,
_ncclProfilerProxyOpRemFifoWait_, _ncclProfilerProxyOpSendTransmitted_, _ncclProfilerProxyOpSendDone_ and
_ncclProfilerProxyStepSendGPUWait_, _ncclProfilerProxyStepSendWait_, for _ProxyOp_ and _ProxyStep_ respectively.

The instrumentation points for _ncclProfilerProxyOpSendPosted_ and _ncclProfilerProxyStepSendGPUWait_ are in the same place in
the code and capture the posting of the host buffer to the GPU. The same happens for _ncclProfilerProxyOpSendTransmitted_ and
_ncclProfilerProxyStepSendWait_, they capture the successful return of isend.

_ncclProfilerProxyOpSendDone_ has no matching state in the _ProxyStep_ event and is captured through the stop of the event.
Above, the only state missing in the _ProxyStep_ event is _ncclProfilerProxyStepSendPeerWait_, that matches
_ncclProfilerProxyOpRemFifoWait_.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->
 
### NVbugs / Jira Tickets
* [NVBug-5040628 (item 10)](https://nvbugspro.nvidia.com/bug/5040628)
* [Jira-1762](https://jirasw.nvidia.com/browse/NCCL-1762)

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

We propose to harmonize the _ProxyOp_ and _ProxyStep_ events in the following way:

* Modify the _ProxyOp_ event states to only have one state (_ncclProfilerProxyOpInProgress_). This state indicates when the proxy op
is moved from _Ready_ (implicitly encoded by the _ProxyOp_ event start) to _Progress_ inside the progress function (e.g., _sendProxyProgress_).
The instrumentation for the start of the _ProxyOp_ event is moved to _ProxyAppend_ in src/proxy.cc. _ProxyAppend_ is where the proxy progress
thread picks the _ncclProxyOp_ from the queue and converts it into _ncclProxyArgs_ and _ncclProxySubArgs_ for the proxy progress function.
The stop of the _ProxyOp_ event stays in the proxy progress function.

* Modify the _ProxyStep_ event to include one additional state (_ncclProfilerProxyStepSendPeerWait_).

With the changes outlined, the number of _ProxyOp_ instrumentation points goes from the current six (four for state updates and 2 for start/stop)
to only three, at the cost of reduced information in the _ProxyOp_ event. Such information can be restored by enabling _ProxyStep_ event profiling,
at the cost of higher overhead.

Following is an example of what the traces from the example profiler plugin look like after the integration of this feature:

```
[
{"name": "Group", "cat": "GROUP", "ph": "b", "id": 0, "pid": 1006509, "tid": 1, "ts": 761142.687500, "args": {"groupId": 0}},
{"name": "AllReduce", "cat": "COLL", "ph": "b", "id": 0, "pid": 1006509, "tid": 1, "ts": 761146.083984, "args": {"SeqNum": 0, "CommHash": 10644631396088996035, "Rank": 7, "Count": 262144, "Datatype": "ncclFloat32", "Algorithm": "RING", "Protocol": "LL", "nMaxChannels": 2}},
{"name": "ScheduleRecv", "cat": "PROXY", "ph": "b", "id": 0, "pid": 1006509, "tid": 1, "ts": 764591.027344, "args": {"Channel": 0, "Peer": 6, "Steps": 28, "ChunkSize": 32768, "transSize": 1835008}},
{"name": "ScheduleRecv", "cat": "PROXY", "ph": "e", "id": 0, "pid": 1006509, "tid": 1, "ts": 764603.183594},
{"name": "ProgressRecv", "cat": "PROXY", "ph": "b", "id": 0, "pid": 1006509, "tid": 1, "ts": 764603.183594, "args": {"Channel": 0, "Peer": 6, "Steps": 28, "ChunkSize": 32768, "transSize": 1835008}},
{"name": "RecvWait", "cat": "NET", "ph": "b", "id": 0, "pid": 1006509, "tid": 1, "ts": 765871.857422, "args": {"Step": 16}},
{"name": "RecvWait", "cat": "NET", "ph": "e", "id": 0, "pid": 1006509, "tid": 1, "ts": 766449.435547},
{"name": "RecvFlushWait", "cat": "NET", "ph": "b", "id": 0, "pid": 1006509, "tid": 1, "ts": 766449.435547, "args": {"Step": 16}},
{"name": "RecvFlushWait", "cat": "NET", "ph": "e", "id": 0, "pid": 1006509, "tid": 1, "ts": 766462.154297},
{"name": "RecvGpuWait", "cat": "NET", "ph": "b", "id": 0, "pid": 1006509, "tid": 1, "ts": 766462.154297, "args": {"Step": 16}},
{"name": "RecvGpuWait", "cat": "NET", "ph": "e", "id": 0, "pid": 1006509, "tid": 1, "ts": 766477.525391},
{"name": "RecvWait", "cat": "NET", "ph": "b", "id": 1, "pid": 1006509, "tid": 1, "ts": 765925.568359, "args": {"Step": 17}},
{"name": "RecvWait", "cat": "NET", "ph": "e", "id": 1, "pid": 1006509, "tid": 1, "ts": 766503.626953},
{"name": "RecvFlushWait", "cat": "NET", "ph": "b", "id": 1, "pid": 1006509, "tid": 1, "ts": 766503.626953, "args": {"Step": 17}},
{"name": "RecvFlushWait", "cat": "NET", "ph": "e", "id": 1, "pid": 1006509, "tid": 1, "ts": 766510.826172},
{"name": "RecvGpuWait", "cat": "NET", "ph": "b", "id": 1, "pid": 1006509, "tid": 1, "ts": 766510.826172, "args": {"Step": 17}},
{"name": "RecvGpuWait", "cat": "NET", "ph": "e", "id": 1, "pid": 1006509, "tid": 1, "ts": 766516.093750},
...
{"name": "ProgressRecv", "cat": "PROXY", "ph": "e", "id": 0, "pid": 1006509, "tid": 1, "ts": 767230.205078},
{"name": "ScheduleSend", "cat": "PROXY", "ph": "b", "id": 1, "pid": 1006509, "tid": 1, "ts": 764594.927734, "args": {"Channel": 0, "Peer": 0, "Steps": 28, "ChunkSize": 32768, "transSize": 1835008}},
{"name": "ScheduleSend", "cat": "PROXY", "ph": "e", "id": 1, "pid": 1006509, "tid": 1, "ts": 764661.214844},
{"name": "ProgressSend", "cat": "PROXY", "ph": "b", "id": 1, "pid": 1006509, "tid": 1, "ts": 764661.214844, "args": {"Channel": 0, "Peer": 0, "Steps": 28, "ChunkSize": 32768, "transSize": 1835008}},
{"name": "SendGpuWait", "cat": "NET", "ph": "b", "id": 16, "pid": 1006509, "tid": 1, "ts": 765874.220703, "args": {"Step": 16}},
{"name": "SendGpuWait", "cat": "NET", "ph": "e", "id": 16, "pid": 1006509, "tid": 1, "ts": 766387.140625},
{"name": "SendPeerWait", "cat": "NET", "ph": "b", "id": 16, "pid": 1006509, "tid": 1, "ts": 766387.140625, "args": {"Step": 16}},
{"name": "SendPeerWait", "cat": "NET", "ph": "e", "id": 16, "pid": 1006509, "tid": 1, "ts": 766387.814453},
{"name": "SendWait", "cat": "NET", "ph": "b", "id": 16, "pid": 1006509, "tid": 1, "ts": 766387.814453, "args": {"Step": 16}},
{"name": "SendWait", "cat": "NET", "ph": "e", "id": 16, "pid": 1006509, "tid": 1, "ts": 766467.814453},
{"name": "SendGpuWait", "cat": "NET", "ph": "b", "id": 17, "pid": 1006509, "tid": 1, "ts": 765927.910156, "args": {"Step": 17}},
{"name": "SendGpuWait", "cat": "NET", "ph": "e", "id": 17, "pid": 1006509, "tid": 1, "ts": 766454.472656},
{"name": "SendPeerWait", "cat": "NET", "ph": "b", "id": 17, "pid": 1006509, "tid": 1, "ts": 766454.472656, "args": {"Step": 17}},
{"name": "SendPeerWait", "cat": "NET", "ph": "e", "id": 17, "pid": 1006509, "tid": 1, "ts": 766455.076172},
{"name": "SendWait", "cat": "NET", "ph": "b", "id": 17, "pid": 1006509, "tid": 1, "ts": 766455.076172, "args": {"Step": 17}},
{"name": "SendWait", "cat": "NET", "ph": "e", "id": 17, "pid": 1006509, "tid": 1, "ts": 766527.082031},
...
{"name": "ProgressSend", "cat": "PROXY", "ph": "e", "id": 1, "pid": 1006509, "tid": 1, "ts": 767199.630859},
...
{}]
```

<!-- note: the following HTML code is also valid -->
<!-- <img src="images/example.png" width="900" height="500" /> -->

### Interface Architecture

The NCCL profiler plugin interface is changed as following:

```C
typedef enum {
  ncclProfilerProxyOpSendPosted        = 0,  // deprecated in v4
  ncclProfilerProxyOpSendRemFifoWait   = 1,  // deprecated in v4
  ncclProfilerProxyOpSendTransmitted   = 2,  // deprecated in v4
  ncclProfilerProxyOpSendDone          = 3,  // deprecated in v4
  ncclProfilerProxyOpRecvPosted        = 4,  // deprecated in v4
  ncclProfilerProxyOpRecvReceived      = 5,  // deprecated in v4
  ncclProfilerProxyOpRecvTransmitted   = 6,  // deprecated in v4
  ncclProfilerProxyOpRecvDone          = 7,  // deprecated in v4
  ncclProfilerProxyOpInProgress_v4     = 20,

  /* Legacy proxy profiler states */
  ncclProfilerProxyStepSendGPUWait     = 8,
  ncclProfilerProxyStepSendPeerWait_v4 = 30,
  ncclProfilerProxyStepSendWait        = 9,
  ncclProfilerProxyStepRecvWait        = 10,
  ncclProfilerProxyStepRecvFlushWait   = 11,
  ncclProfilerProxyStepRecvGPUWait     = 12,

  /* Legacy proxy control states */
  ncclProfilerProxyCtrlIdle            = 13,
  ncclProfilerProxyCtrlActive          = 14,
  ncclProfilerProxyCtrlSleep           = 15,
  ncclProfilerProxyCtrlWakeup          = 16,
  ncclProfilerProxyCtrlAppend          = 17,
  ncclProfilerProxyCtrlAppendEnd       = 18,
} ncclProfilerEventState_t;
```

Two new states have been added: _ncclProfilerProxyOpInProgress_v4_ and _ncclProfilerProxyStepSendPeerWait_v4_. All the old
_ProxyOp_ states have been deprecated by the new version 4 of the profiler interface.

Additionally, the event state argument structure has been extended to accomodate for the changes in the _ProxyOp_ event:

```C
typedef union {
  struct {
    int isSend;
  } proxyOp;

  struct {
    size_t transSize;
  } proxyStep;

  struct {
    int appendedProxyOps;
  } proxyCtrl;
} ncclProfilerEventStateArgs_v4_t;
```

The _ProxyOp_ event is now started outside of the transport progress function. Meaning, the type of _ProxyOp_ event is not
known at this time. Thus, the type (send/recv) needs to be updated when the _ProxyOp_ is finally processed by the progress
function. Moreover, the _transSize_, previously, updated in the _ProxyOp_ event is now part of the _ProxyStep_ event update
info.

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
[MR !808](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/808)

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
