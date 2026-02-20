# Symmetric Kernels

## Abstract
Implement kernels specialized to take advantage of symmetrically registered memory.
These are currently for non-network communicators only.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

https://jirasw.nvidia.com/browse/NCCL-1828

### User Experience

For buffers registered as symmetric, when passed to NCCL allreduce, allgather, or reducescatter
users should expect to see performance gains in all "important" cases, which definitely includes
small and medium sizes. Only floating point types less than 64-bit are considered, as well as only
the sum operator.

For large sizes with NVLS they will not see perf loss, but without NVLS they will see significant
perf loss (~25%) because the symmetric kernels do alltoall traffic which gets poor bandwidth compared to rings.

Floating point summation is always done in fp32 accumulators with the exception of fp8 on NVLS
where it uses fp16 inside the switch. Thus the accuracy of f8 and f16 will be much improved, and potentially
even deterministic.

### Assumptions, constraints and dependencies

Passing a symmetric buffer to NCCL comes with a symmetry constraint that all ranks are passing
buffers at the same offset in the same registration.

Gruops with more than one collective will not be considered for symmetric execution.

### Use Cases

For apps that want small/medium message perf gains and can fit themselves into the symmetry
constraint.

### Platform Requirements

<!-- ### Functional Requirements -->
<!-- ### System Requirements -->
CUDA VMM support required.
CUDA P2P between all pairs of ranks required.

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

See "src/include/symmetric.h" for definitions relevant to both host and device.
See "src/symmetric.cc" for host code that chooses the right symmetric kernel.
"src/enqueue.cc" has been taught how to identify collectives suitable for symmetric execution.
New kernels in new source files located in new directory "src/device/symmetric".

The env var `NCCL_SYM_KERNEL=<name>` can force a specific symmetric kernel.
The kernels are individually named, NCCL_ALGO/PROTO are irrelevant. In the following
names MC refers to use of multicast (NVLS).

* `AllReduce_AGxLL_R`
* `AllReduce_AGxLLMC_R`
* `AllReduce_RSxLD_AGxST`
* `AllReduce_RSxLDMC_AGxSTMC`
* `AllGather_LL`
* `AllGather_LLMC`
* `AllGather_ST`
* `AllGather_STMC`
* `ReduceScatter_LL`
* `ReduceScatter_LD`
* `ReduceScatter_LDMC`

LL kernels do not work on buffers >= 2GB because they use 32-bit integers to track indices and sizes.

The env var `NCCL_SYM_SMS=<num>` can force the number of SMs to use for the symmetric kernel.

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

Perf of small and medium (<32MB) must be superior to non-symmetric. Make sure to
restrict datatypes and reduction operation to the designated float types and sum
respectively (see User Experience). Also fp8 cannot use NVLS on H100 so do not
expect it to be on par with the other floating point types.

#### Where to run?

H100x8, B200x{4,36,72}

#### What to run?

allreduce, allgather, reduce_scatter.

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
  - John Bachan

</details>
