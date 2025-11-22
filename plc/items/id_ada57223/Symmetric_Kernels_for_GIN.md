# Symmetric Kernels for GIN
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
This feature introduces new AG and RS kernels to extend acceleration of symmetric collectives over the GIN network.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->
Currently symmetric kernels only support single nvlink domain which limits the usages and benefits of symmetric features.
Expanding symmetric kernels for scale-out use case would be important for training as well as inference performance and
resource usage.
This feature implements scale-out kernels for AG and RS to bridge the gap.

### NVbugs / Jira Tickets
https://nvbugspro.nvidia.com/bug/5247282

### User Experience

Like symmetric kernels, all the user has to do is register the buffers in a
symmetric window and the kernels will be selected automatically.

### Assumptions, constraints and dependencies

Continuing with the design of the non-network sym kernels only AllReduce,
AllGather, and ReduceScatter will be specialized for the same datatypes.

### Use Cases

### Platform Requirements

GIN support will be required for these kernels to be enabled.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

Guiding assumptions:
* Winning the 1 byte latency benchmarks at small scale requires flat alltoall
algorithms.
* Realistic workloads lean more towards medium/large message sizes which are sensitive to the
bandwidth component.
* NVL bandwidth is now around 800GB/s but a single NIC could be as low as 100Gbps
  or as high as 800Gbps (GPU:CX8 = 1:1).

## ReduceScatter Design

A large possible disparity between NIC and NVL bw necessitates hierarchical
algos for medium sized collectives. But with bandwidth parity flat algorithms
will deliver best performance. FP8 complicates this because it has to be
accumulated as fp16. A hierarchical algo needs to send accumulators on the wire
so if the bandwidth of the network is at least half of NVL then hierarchy could
hurt more than help.

Hierarchical algos will use alltoalls at both levels.

Barriers will be used over LSA, but not over the network. Signalling puts
will do all the syncs necessary for the network.

The algorithms:
* ReduceScatter_GinFlat{_LDMC}
* ReduceScatter_GinHier{_LDMC}

### Inbox buffers:

We want the alltoall algos to work at all scales without requiring memory in
excess of the latency-bandwidth product. Dedicated buffers per peer would
require unbounded memory, so instead we will use a rolling set of N buffers
where in the quiesced state your N immediate upstream neighbors start with a
credit, and as data is consumed the credit is forwarded to the next peer in the
alltoall pattern who needs that buffer. This will allow the kernel to
immediately start doing PUTs without waiting for initial handshake. Since the
credit delivery pattern is fixed, these buffers cannot be shared between
different algos, meaning that the flat and hierarchical algos will need
distinct sets of buffers.

### Outbox buffers:

Hierarchical reductions need a temporary buffer for generating intermediate
results to be sent to network. This can be a simple circular buffer with GIN
counters such that the previous kernel can still have data in flight without
blocking the next kernel from grabbing buffer space.

### NVLS + FP8

Currently NVLS LDGMC cannot return a wider fp type as output than it is given
as input. This means that for fp8 we have to convert the input to a wide type
and stage that in video memory and use that to source the LDGMC. This hurts
latency and requires bandwidth from the SM to do the copy. Pursuing LDGMC for
fp8 is deemed out of scope.

## Allgather Design

In order to use scale out AG symmetric kernels, we require users to symmetrically register
both src and dst buffers. Once buffers are registered and multi-node environment is detected,
scale-out kernels will be automatically picked. We assume each GPU has a local NIC or at least
can access a NIC in this implementation.

We provide 3 different algorithms for allgather.

The first one is Ring + NVLS (`ncclSymkRun_AllGather_GinHier_MCRing`). We split block into 2 parts;
the first part contains 1 warp which is used to issue GIN put on the rail, and the second part contains
the rest of warps to receive data from network and perform NVLS ops. The ring is built based on the
previous and next peer relative to my rank; for each round, each rank will load all its data received
from previous peer and put to next peer's dst buffer (except the first round where rank will directly
send its own src data).
Whenever data arrives, the second part of warps will start to broadcast data to all intra-node peers
by NVLS.

The second one is A2A + NVLS (`ncclSymkRun_AllGather_GinHier_MCA2A`). This algorithm is more for small
to medium message sizes. Everything is the same as Ring + NVLS except that rail communication pattern
becomes A2A-based put. The A2A is one-shot operation and each thread would issue all src data to a different
target rail peer until all peers are processed. At the same time, the second part of warps would wait on
them and broadcast to all intra-node peers.

The final one is pure A2A going through GIN (`ncclSymkRun_AllGather_GinFlat_A2A`). This is a fallback
algorithm for A2A + NVLS in case NVLS is not available. However, the best implementation should be doing
A2A on GIN for network and on NVLINK for intra-node. I would treat it as the future work.

### Performance Model

We have a simplified performance model for allgather, and it requires 3 major variables to compute the time:
network latency (netLatency), intra-node bandwidth (intraBw), and inter-node bandwidth (interBw). We borrow
the net latency value from legacy tuning model and now assign
netLatency to `comm->tunerConstants.hwLatencies[NCCL_HW_NET][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]`; intraBw is assigned
based on platform and empirical values (i.e., 360GB/s on Hopper and 720GB/s on Blackwell); interBw is assigned
to pure NIC bandwidth. Take Ring + NVLS algorithm as an example, we compute the number of steps, intra-node communication
time, and inter-node communication time; then, the overall time would be:
```
timeUs = steps * netLatency + std::max(intraTime, interTime);
```

Besides computing the time, we determine the block usage based on the number of bytes, chunkSize and platform, and now we
allocate 1 block to each chunk and cap it based on upper limit block usage. For example, on Blackwell with 8 GPUs, we only
need 4 SMs to saturate nvlink bandwidth, if we have 8 chunks, we would allocate std::min(4, 8) number of blocks.
chunkSize now might not be optimal and can be tuned in the future


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
The bandwidth on CW 8 DGX H100 nodes is as follows:

![AG Scale Out Bus Bandwidth on CW](images/AG_Scale_Out_Perf_Sweep_All.png)

The corresponding latency on CW 8 DGX H100 nodes is as follows:

![AG Scale Out Bus Bandwidth on CW](images/AG_Scale_Out_Perf_Sweep_Latency.png)

#### What is measured?

#### Results


</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - John Bachan
  - Kaiming Ouyang

</details>
