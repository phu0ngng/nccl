# Kernel Profiler
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
NCCL v2.23 introduced profiling support for proxy events. This allows users to get an estimate of the execution
time of NCCL operations. However, to accurately measure execution time for each NCCL operation, kernel instrumentation
is also needed. This is because NCCL uses the proxy progress thread to assist the GPU during network transfers.
The network transfer can complete before the kernel (or viceversa). Moreover, if the operation does not require
any network transfer, the proxy events are not captured and the users are left with little to no information
about the execution time of NCCL operations. Adding profiling support solves all these problems.

<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets
[NVBug - Kernel-Prof](https://nvbugspro.nvidia.com/bug/4784751)
[Jira - Kernel-Prof](https://jirasw.nvidia.com/browse/NCCL-1681)

### User Experience
Users can enable kernel profiling in NCCL by providing a profiler plugin library, through the `NCCL_PROFILER_PLUGIN`
env variable, and by setting the event activation mask to `ncclProfileKernelCh`. After doing this, all collective
and point-to-point operations in NCCL will generate profiler callbacks for the corresponding plugin. The plugin is
responsible for gathering the NCCL provided event data and to present it to the users in the most appropriate format.

### Assumptions, constraints and dependencies

### Use Cases
Profiling of intra/inter-node NCCL operations.

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
The GPU kernel provides per-channel progress updates to the host CPU using a set of counters. These counters
are read by a host CPU thread to start and stop kernel events. Every time a new NCCL operation is started,
the enqueue code prepares ncclProxyOp(s) for the network proxy progress thread (if there is any network
activity involved) and the the profiler proxy progress thread (if the kernel events are enabled). Every proxy
operation has a sequence number starting from 1 to inf identifying the original NCCL operation. As the kernel
runs it publishes, through a shared array in host memory and for every channel, the sequence number of the
currently processed NCCL operation. The profiler proxy progress function polls on this array of counters to
check for progress of the give ncclProxyOp. When the sequence number of the current proxy op is published by
the kernel, the profiler proxy starts the kernel event.

#### GPU Code Instrumentation
The GPU code keeps a `workCounter` per channel in global memory. This counter lives for the duration of the
communicator. When a kernel begins execution, it loads the `ncclDevChannel` struct from global to shared
memory (this includes `workCounter`). While the kernel executes NCCL work items, for a given channel, it
increments the value of `workCounter` in shared memory. Eventually, when the kernel ends, it stores the
updated `workCounter` value from shared memory to global memory. This allows NCCL to monotonically increment
the value of the counter across kernels.

The updated value of `workCounter` is exposed to the host code using two arrays: `workStarted[]` and
`workCompleted[]`, allocated in page-locked host memory using `ncclCudaHostCalloc`. This allows memory
updates to overlap with kernel execution.

#### CPU Code Instrumentation
The host code leverages the existing proxy progress thread to monitor kernel progress. The kernel profiler
infrastructure adds a new transport (`src/transport/profiler.cc`) with its own progress function. The
progress function of the profiler transport checks the value of `workStarted[]` and `workCompleted[]`,
updated by the kernel code, to start and stop kernel events in the profiler plugin.

#### Profiler Proxy Connection
To support the new profiler proxy, NCCL creates additional connections with the proxy thread
using the new `TRANSPORT_PROFILER` type. This allows `uploadProxyOps`, during enqueue, to select the profiler
transport for the proxyOp. Proxy connections for the profiler are marked as `shared` so that the profiler can
progress batches of ncclSend/Recv operations together (profiler proxyOps are converted to subArgs and added to
the same proxyArgs by the progress thread).

#### ProxyOp Enqueue
When a NCCL operation is enqueued, NCCL creates a new `ncclProxyOp` for every channel and peer. If proxy assistance
is needed (e.g., for inter-node communication), the proxyOps are appended to the kernel plan `proxyOpQueue` during
`addProxyOpIfNeeded`. Similarly, if the `ncclProfileKernelCh` event in the NCCL event activation mask is set, NCCL
adds one profiler proxyOp per channel to the kernel plan `proxyOpQueue`. The profiler proxyOps poll the `workStarted[]`
and `workCompleted[]` arrays to start and stop kernel events (as described below).

The distinction between profiler proxyOps and traditional proxyOps is made through the `pattern` attribute.
During `ncclSaveProxyOp` the pattern is checked. If the pattern is set to `ncclPatternProfiler`, `SaveProxyProfiler`
is called, otherwise `SaveProxy` is called.

#### Proxy Progress
The proxy progress thread checks the `proxyOps` pool for new ops appended by `uploadProxyOps`. When it finds
one it converts it into `ncclProxyArgs` and sets `args->progress = op->connection->tcomm->proxyProgress`. Where
`op->connection` is set by `ncclLocalOpAppend` in `SaveProxyProfiler`. Afterwards, the proxy goes through the
`ncclProxyArgs` and invokes the progress function for each of them to make progress. If the proxyOp comes from
the profiler the `profilerProxyProgress` function is called.

The profiler progress function is reported below:

```C
ncclResult_t profilerProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      sub->base = sub->workCounter;
      sub->posted = sub->transmitted = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      uint64_t* workStarted = (uint64_t *)sub->sendbuff;
      uint64_t* workCompleted = (uint64_t *)sub->recvbuff;
      if (sub->posted < sub->nsteps && sub->base <= workStarted[sub->channelId]) {
        ncclProfilerStartKernelChEvent(args, s);
        sub->posted = sub->nsteps;
      }
      if (sub->transmitted < sub->nsteps && sub->base <= workCompleted[sub->channelId]) {
        ncclProfilerStopKernelChEvent(args, s);
        sub->transmitted = sub->nsteps;
        args->done++;
      }
    }
    if (args->done == args->nsubs) args->state = ncclProxyOpNone;
  }
  return ncclSuccess;
}
```

The profiler progress function reuses some of the `ncclProxySubArgs` attributes to move from event start to event
stop. More specifically, `posted` is used to indicate that the event was started, while `transmitted` is used to
indicate that the event was stopped.

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

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline
Leverage the kernel information to give users a more accurate view of the NCCL inner behavior.

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
The enqueue code generates ncclProxyOps for the proxy progress thread. Thus, the proxy thread is
used for network ProxyOps and profiler ProxyOps. If the proxy progress thread has a sequence of
network channels that it needs to progress, the profiler ProxyOps might be delayed and the
profiler will lose accuracy. This is expected in the current design and we plan to mitigate this
issue in future design iterations of the profiler.

Another limitation in the current design is that under heavy load in the CPU the proxy progress
thread might be late processing profiler ProxyOps. Since the profiler does not synchronize with
the kernel this might also result in loss of accuracy. One possible solution for this could be
to let the kernel (somehow) export its own timestamps for start/stop for the proxy progress
thread to pick up and pass to the profiler plugin.

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
