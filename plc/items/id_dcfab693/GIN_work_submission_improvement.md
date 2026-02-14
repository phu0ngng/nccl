# GIN work submission improvement
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

The work submission algorithms of GIN are conservative. In the GDA-KI backend of
GIN, we perform credit checking to make sure that we can reuse WQ slots. DB is
rung mostly at every post-send call. Each post-send eventually reaches a QP lock
to enforce the atomicity of updating DBR and ringing DB. These protections are
important for normal use cases where we do not have knowledge about applications
that use GIN GDA-KI. They are also necessary for guaranteeing forward progress.
However, they add overhead.

For applications such as DeepEP where these items are guaranteed elsewhere such
as by design of the application, these protections are unnecessary. Allowing
those applications to bypass the protections may lead to higher performance
(lower latency, higher message rate).

### NVbugs / Jira Tickets

NVBUG 5765593

### User Experience

### Assumptions, constraints and dependencies

### Use Cases

Custom device kernels using the Device API GIN support to communicate between
nodes.  In particular, both DeepEP and DeepSeek requested the lifting of the
limit.

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

The proposed design composes of three areas. They are for the GDA-KI backend.
But the functionalities are exposed to users via user-facing GIN APIs.

1. Reliable DB: This is a hardware feature of NVIDIA NICs. It is available since
CX-8. This feature allows us to **NOT** update DBR in post-send. It leads to
removal of one fence-release (GPU scope) and the QP lock. This hardware feature
has a limitation. The maximum number of QPs we can create with this reliable DB
feature is 1024 QPs. The number is aggregated across all processes that are
using that NIC at that point in time. There is no way to query how many reliable
DB QPs have already been created. There is a software-emulated version of this
feature. It uses a CPU thread to monitor the QP and periodically update the DBR
buffer as well as rering the DB. This operation is outside the critical path.
The design requires GDRCopy so that GPU does not loose performance communicating
with the CPU thread. From time to time, the CPU thread will use GDRCopy to
access the QP information on the GPU memory and perform the necessity. This
feature is designed as an opt-in feature.
This is enabled through the `NCCL_GDAKI_USE_RELIABLE_DB` env var.

2. Skip credit checking: Before GDA-KI can create WQEs, it must reserve enough
slots in the WQ buffer. WQ is a circular buffer. We may end up reserving the
same slots that the NIC has not yet consumed. To create new WQEs safely into
these slots, we need to poll the CQ to see if the NIC has already returned the
ownership of these slots to the software. Users may skip this checking if they
have a way to guarantee that they will not reuse slots that the NIC have not yet
consumed. This feature leads to an ability for the users to specify the QP depth
in the Dev Comm requirements parameter as well as a bit-wise OR `optFlags` in
`gin.put`, `gin.putSignal`, and `gin.signal` device APIs.

3. Aggregate requests: Ringing DB tells the NIC to start processing all WQEs up
to the WQE index in the DB. Frequenly ringing the DB may lead to the NIC reading
just a few (or even one) WQEs at a time. If we ring one DB per n WQEs, the NIC
has a better chance to issue one (or a few) large PCI read request to get
multiple WQEs in one transactions. This feature ends up as a bit-wise OR
`optFlags` in `gin.put`, `gin.putSignal`, and `gin.signal` device APIs.


<!-- ![Example](images/example.png) -->
<!-- note: the following HTML code is also valid -->
<!-- <img src="images/example.png" width="900" height="500" /> -->

### Interface Architecture

We extend `gin.put`, `gin.putSignal`, `gin.signal` with `optFlags`.
```
enum ncclGinOptFlags {
  ncclGinOptFlagsDefault = 0,
  ncclGinOptFlagsMaySkipCreditCheck = (1 << 0),
  ncclGinOptFlagsAggregateRequests = (1 << 1),
};

NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::put(..., uint32_t optFlags);

# For C-style API
NCCL_DEVICE_INLINE void ncclGinPutEx(..., optFlags);
```

### FAQs

Why does Reliable DB an opt-in feature? Can't we always enable this feature?

Reliable DB does not always work. It requires hardware support or GDRCopy to use
the software-emulated implementation. When using the hardware support, the
number of QPs with this reliable DB feature is up to 1024 QPs. Image a system
where the NIC has hardware support but GDRCopy is not present. DeepSeek, for
example, plans to create up to ~2000 QPs. This number is greater than the
hardware can support. The software-emulated version cannot be used as a fallback
because of the unavailability of GDRCopy. Hence, some QPs will need DBR update,
which leads to QP locking. Now, the users will have mixing QPs that have
different performance characteristics. It likely shows up as a NCCL significant
performance degradation when the user scales out pass a certain number of QPs.

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

https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/1828

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation

In addition to the usual test cases, we should also test the followings.

Additional unit test arguments:
```
  --gin_reliable_db <val>    Reliable DB mode (default 0)
                             0: Disable
                             1: Enable as a required feature
                             2: Enable as an optional feature
  --gin_skip_credit_check    Skip credit check in GIN (default 0)
  --gin_aggregate_requests   Use aggregate requests in GIN (default 0)
```

Unit tests that support the new arguments:
1. put_signal_ping_pong_gin: Support `--gin_reliable_db` and `--gin_skip_credit_check`. It does not support `--gin_aggregate_requests`.
2. devapi_put_bw: Support all new test arguments.

We should run all combinations. All new features with the GIN GDA-KI backend only.

The reliable DB feature has two modes:
1. HW mode: Require CX-8 or newer.
2. SW-emulated mode: Require GDRCopy. Work with all CX NIC generations.

When using `--gin_reliable_db 1` and the system does not support either the HW mode or the SW-emulated mode, the test should run as if it is using `--gin_reliable_db 0`.
When using `--gin_reliable_db 2` and the system does not support either the HW mode or the SW-emulated mode, the test should fail.

How to check if the reliable DB feature is used:
- Run the application with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET`. Look for the following line as an example.
```
utskinnyjoe-dvt-38:156691:156691 [0] NCCL INFO [0] Created a QP group: qp_idx=0, main_qpn=0x17357, companion_qpn=0x17358, reliable_db=SW emulation
```

The skip credit check feature and the aggregate requests feature are always
respected if specified. If there is an error on these features, it will show up
as either a proper error report and a non-zero exit code, or a segfault, or a
hang. These unit test applications should take just a few seconds to finish.

Currently, these features are supported in the GDA-KI backend only. They do not work with CPU Proxy.

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
  - Pak Markthub
  - 

</details>
