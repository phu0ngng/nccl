# Qp Profiling
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->
The NCCL profiler was originally designed with NCCL core events in mind. External plugin events were
not initially considered. However, profiling of those events, e.g. QP events in Infiniband and RoCE
networks, can be important to users. For this reason, the NCCL profiler interface is extended to
support external plugin defined events.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets
[Jira - QP Prof](https://jirasw.nvidia.com/browse/NCCL-1682)
[NVBugs - QP Prof](https://nvbugspro.nvidia.com/bug/4784411)

### User Experience
External plugin implementers need to define plugin specific events on top of the new "netPlugin"
NCCL event. Moreover, they need to implement support for such events in the profiler, which needs
to access the source plugin event definition.

### Assumptions, constraints and dependencies
The definition of plugin defined events in the source plugin has to be visible to the profiler.

### Use Cases
Profiling of plugin defined events, e.g., QP events in Infiniband and RoCE networks.

### Platform Requirements
No special requirements.

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
The network plugin `init` interface is extended to take a NCCL core callback following defined:

* `ncclResult_t (*ncclProfilerCallback_t)(void** eHandle, int type, void* pHandle, int64_t pluginId, void* extData)`

The callback takes a `type` (0/1) indicating the type of profiler call (start/stop), an opaque pointer
to a NCCL core object (`pHandle`), a `pluginId` and an opaque pointer to a network plugin object (`extData`).
If `type` is `0` the callback returns a new event handle in `eHandle`, otherwise `eHandle` is a pointer
to the event to stop. In this case `pHandle` and `extData` are ignored. The `pluginId` is a combination of
network type and plugin version. This id is used to keep the network and profiler plugins aligned and
able to work correctly together.

Similarly, the network plugin `isend` and `irecv` interfaces are extended to take an opaque pointer
(`pHandle`) to a NCCL core object. This pointer is later passed to the NCCL core callback when new
events are started.

* `ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void* pHandle, void** request)`
* `ncclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** pHandles, void** request)`

The profiler plugin interface is extended with a new `ncclProfileNetPlugin` event, that covers all the
net plugin defined events. A new event descriptor for the `ncclProfileNetPlugin` event is also added to
the plugin interface:

```
typedef struct {
  uint8_t type;
  void* parentObj;
  int rank;
  ...
  union {
    struct coll { ... };
    struct p2p { ... };
    ...
    struct netPlugin { int64_t id, void* data; };
  };
} ncclProfilerEventDescr_t;
```

The NCCL core callback is implemented in `src/misc/profiler.cc`. The NCCL core callback treats
net plugin events as opaque. It merely forwards them to the profiler by initializing the event
descriptor appropriately and calling `startEvent`:

```
ncclResult_t ncclProfilerCallback(void** eHandle, int type, void* pHandle, int64_t pluginId, void* extData) {
  if (type == 0) { // start
    struct ncclProxySubArgs* sub = (struct ncclProxySubArgs*)pHandle;
    ncclProfilerEventDescr_t eDescr = { 0 };
    eDescr.type = ncclProfileNetPlugin;
    eDescr.parentObj = sub->stepEventHandles[step%NCCL_STEPS];
    eDescr.rank = sub->rank;
    eDescr.netPlugin.id = pluginId;
    eDescr.netPlugin.data = extData;
    ncclProfiler->startEvent(eHandle, sub->profilerContext, &eDescr);
  } else { // stop
    ncclProfiler->stopEvent(*eHandle);
  }
}
```

The NCCL core callback allows to support plugin defined events decoupling the network and
the profiler plugin interfaces at the same time. The NCCL core callback plays the role of
intermediary, making profiler calls on behalf of the network plugin.

<!-- note: the following HTML code is also valid -->
<!-- <img src="images/example.png" width="900" height="500" /> -->

### Interface Architecture

Network plugins can add internal profiler events, apart from those defined by NCCL, by defining
them in a header, located in the `src/include/profiler/` directory. The name of the header should
have the format `nccl_profiler_net_<type>_vX.h`, where `type` is replaced by the type of network
(e.g., ib or sock) and `X` is replaced by the plugin version. An example for IB is reported below:

```
enum {
  ncclProfileQp = (1 << 0),
};

typedef struct {
  uint8_t type;        // event type (plugin defined)
  union {
    struct {
      uint64_t wr_id;  // work request id
      int opcode;      // ibv opcode
      int qpNum;       // QP number
      size_t length;   // work request data length
    } qp;
  };
} ncclProfilerNetIbDescr_v1_t;
```

The header can is copied by external plugins inside the `nccl/` directory in the ext-profiler.
The version alignment between the network and the profiler plugin is done throught the `pluginId`.

The pluginId could be further broken into more components. For example, two network plugins
implemented for the same network might have different event definitions but same pluginId (if they
have matching versions). Adding a vendor id to the pluginId could solve the problem. But a vendor
could also have more than one plugin implementation (e.g., NCCL internal plugin and IB ext). A
more functional pluginId could be a combination of network type + vendor id + plugin number +
plugin version. 

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

121cd5f37 Profiler: add QP event support
fb9806862 net_ib: instrument plugin to capture qp events
09ab0bcf1 Implement NCCL core profiler callback
e78e45281 Profiler: extend profiler interface and bump up ver
397b00bf6 net: update network plugin to support profiler

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
