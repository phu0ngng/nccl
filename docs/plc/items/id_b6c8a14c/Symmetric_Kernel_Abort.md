# Symmetric Kernel Abort
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->
Current symmetric kernels and device APIs do not support abort when any error happens so that applications are not
able to deal with NCCL error properly. This feature supports abort for all blocking device side functions.

### NVbugs / Jira Tickets
https://nvbugspro.nvidia.com/bug/5529489
### User Experience
When users call host API ncclCommAbort, for symmetric kernels, it will return from blocking functions and complete immediately;
for device API, it returns the error code "Abort" right away.

### Assumptions, constraints and dependencies
NA

### Use Cases
Application abort

### Platform Requirements
NA

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
NCCL communicator already has an abortFlag, so we just need to pass the abortFlag to devComm and let symmetric kernels and device API to check it
during blocking calls.

The abort checking function is implemented as follows:
```
static NCCL_DEVICE_INLINE bool testAbort(uint32_t* abortFlag, uint32_t& steps) {
  const uint32_t maxSteps = 100000;
  if (++steps < maxSteps) {
    return false;
  } else {
    steps = 0;
    return abortFlag != nullptr && cuda::atomic_ref{*abortFlag}.load(cuda::memory_order_relaxed) != 0;
  }
}
```

devComm is stored in Gin, Bar and LL session objects. When upper layer calls blocking functions such as gin.waitSignal or bar.sync,
testAbort is called in each loop, and corresponding devComm.abortFlag is passed. For example, in gin.waitSignal
```
template<unsigned beMask>
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::waitSignal(Coop coop, ncclGinSignal_t signal, uint64_t least, int bits, cuda::memory_order ord) const {
  uint32_t steps = 0;
  coop.sync();
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(this->_makeCtx(), this->comm.ginSignalBase + signal);
    uint64_t got;
    #pragma unroll 1
    do got = cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!nccl::utility::rollingLessEq(least, got, bits) && !testAbort(this->comm.abortFlag, steps));
  }
  coop.sync();
}
```

Currently, we directly return from these blocking functions, it is enough for our internal symmetric kernels. However, when users
call device API and want to abort, we need a new set of APIs to return abort code in order to better serve the application.

One special case for abort is gin.flush. Now gin.flush is implemented by GDAKI and PROXY backend, and both backends call blocking
polling function to check the status of the send request. For GDAKI, it is calling `doca_gpu_dev_verbs_wait` which needs to be
changed to nonblocking call `doca_gpu_dev_verbs_poll_one_cq_at`; for PROXY, it needs to change internal function API and let upper layer
pass abortFlag to it.

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
  - Kaiming Ouyang

</details>
