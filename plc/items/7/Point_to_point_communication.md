# Point to point communication
<details>
<summary><h2>Requirements</h2></summary>
 
### Project Context and Purpose

NCCL provides fast inter-GPU communication for parallel applications.

### NVbugs / Jira Tickets

[NVBug 12345](http://nvbugs.nvidia.com/12345)

[Jira NCCL-12345](https://jirasw.nvidia.com/browse/NCCL-12345)

### User Experience

NCCL is used through deep learning frameworks or other HPC applications
to accelerate GPU-to-GPU communication. It is supposed to be transparent
to the user.

### Assumptions, constraints and dependencies

None.

### Use Cases

Point to point communication is required to make NCCL a fully capable
communication library. It is used by most HPC codes, and started to be
also asked by DL frameworks for model parallelism (mostly recommenders).

We assume in-place point-to-point communication is not needed and is
therefore out of scope. In place point-to-point would be able to make
sendrecv implement sendrecv_replace in MPI terms, or in-place alltoall
(which is not covered by MPI).

### Functional Requirements

Codes need to be able to express any point-to-point communication and
related collective operations (scatter, gather, alltoall, neighbor
collectives/N-dimension halo exchange).

### System Requirements

None.

### Interface Requirements

None.

### KPI Requirements

#### First step (NCCL 2.7)

Performance on single node DGX2 intra-node needs to be close to SOL,
i.e. 120 GB/s.

Performance on multi-node DGX1 and DGX2 needs to be close to SOL, i.e. 6
GB/s (with a 2GPU:1NIC ratio).

#### Second step (NCCL 2.8)

Performance on single-node DGX1 needs to be close to SOL.

### Platform Requirements

N/A

### Security Requirements

NCCL is a user library and benefits from the user-mode security.

### Legal and Standards Requirements

N/A

### Telemetry Requirements

N/A

### Backward Compatibility Requirements

Changes do not need to be backward compatible.

### Virtualization Requirements

N/A

### Signoff list

Author : Sylvain Jeaugey
</details>
 
<details>
<summary><h2>Design</h2></summary>
 
### Proposed Design

#### Pairwise approach

Since NCCL operations are CUDA kernels and there is no guarantee from
CUDA on concurrent progress of different kernels, all related send/recv
operations need to be part of the same kernel.

Inside the kernel, progressing all operations is complicated since it
requires frequent changes of communication context (FIFOs and counters)
which might be slow and require a whole new set of low-level functions.

NVSwitch performance might also not be ideal when communicating
simultaneously from different peers.

For that reason, we implement point-to-point communication as a
pair-wise alltoallv. This consist of n send/recv operations where we
receive data from rank-delta and send data to rank+delta. This approach
has many advantages :

- We send/recv large chunks from a single peer, which is the most
  optimized case in NCCL, since register usage is minimal.
- Pairwise alltoall is a way to only progress some operations while
  being deadlock free.
- We can split the different send/recv operations on different channels
  (SMs) to use all NVLinks (p2p) and get closer to peak performance.

Beyond that, we also split large operations on multiple channels to get
better performance and address the nvswitch performance limitation when
communicating with too many peers.

#### P2P Channels

The number of p2pChannels is increased to the next power of two. This is
better for two reasons. First, it maps better to the nranks operations
in an alltoall, when nranks is a power of two (most common case). But
the main reason is to permit a better load balancing and scheduling on
channels.

To get gradually better bandwidth as we increase the size, we use more
channels with the same peer. If the channels we use are channels used
for other peers, then we start having issued due to scheduling. The
solution is to have one dimension (per-chunk) spread on channels in a
way that is complementary to the other dimension (per-peer) : one
dimension increases the channel number linearly (lsb bits increasing
first, then msb) while the other dimension increases the channel number
in a mirrored way (msb bits increasing first, then lsb). For example,
with 16 channels, the second dimension will start with channel 0, then
8, then 4, then 6, ...

Because channel bit mirroring is easier in a power-of-two space, we
round p2pnChannels to the next a power of two.

### Interface Architecture

Two new NCCL verbs are added : ncclSend and ncclRecv.

Composition with ncclGroupStart and ncclGroupEnd will let users create
all possible point-to-point collective operations, like
scatter/gather/alltoall and all their variants.

### System KPIs & Metrics

Pairwise alltoall is supposed to be optimal.

Performance on the different platforms meet the required KPI.

### Data Architecture

N/A

### Security Design

N/A

### Debugging & Troubleshooting

None.

### Logging and Instrumentation

None.

### Operational Considerations

None.

### Signoff list
</details>
 
<details>
<summary><h2>Coding</h2></summary>
 

- nccl@b6431d1a1cbb00c5d62a661cec93185697ef8620
- nccl@d255f667283175dca91179f299c862b540fc2934
- nccl@050b4ce572d44f2f6c947c87c3af6a1bf2db5092
- nccl@c9f36ebb8ac5a31fdd89eb600a8f22aba431a935
- nccl@4973c6dad28176e77fd75aa355d73e91f7c0293e
- nccl@bd59af9e6c6db0ab8a6f09731aa299a1f8a52481
- nccl@1ed32984db6c8aff519fb560fdfaee5a91500bec
- nccl@ba60c826e80580514a9f8fe9729927207aebf7f5
- nccl@04a02583143b63ee9ba56ab48bbb10ff71946174
- nccl@fe89e869282cc3da427356b61f87241d97a306c5
- nccl@5f70f2705676592b9fa06ee8b5fd5fc18aaf6fba
- nccl@02e470b44ccde2401351a20aa8a52d0210d1abac
- nccl@7a727a5a7117169f12415c1808723209ec9505bc
- nccl@f3a2d2c9026d3d9f6c95bb762511fbfcc915cc64
- nccl@03c455e22de131511d083ab986074bd0537801f9
- nccl@e0384bf500352e179ae1c49bc3c1cd15bdc37128
- nccl@53b88211e95bc5f383823d46d7c8504d70630da8
- nccl@8e8eda9d3cc9516497fb20fd7227aa37df87024d
- nccl@aa02f5d0f6374bd0afe84330dd02d6b14d9624c8
- nccl@d38c8a153579942b714c8878a134e2d80cdaac47
- nccl@b99defdd04091db2705e212f5ba9cd79df5812eb
- nccl@d792d131a0121c718fbd9ac9fe63c5779e12f877
- nccl@a2953add73bc51c3c1b7eba8b67f2a893b225ec7
- nccl@ed226ffd376637e36afc98ab19b359f4e4ee53c1
- nccl@e1a3022c4f03deb087c0c74d55bf40614bd931bd
- nccl@97342844deb346c23d0489fc0dcf1fd0489a273d
- nccl@e1598ba49f05fa021d63d0482971c1464c45d5e8
- nccl@a82988eb0d53b367408714227c5d00553237329a
- nccl@8df62e12d649ce0157a050329d361195e111e341
- nccl@e7f95d7357c8fcfeec025b2eebf4b03bd5ea9fd6
- nccl@dd1cd39d88b2481583f998caa1639a76fa149eea
- nccl@c67a3caa366f9f3e0fbd49efbbda865d7049862f
- nccl@ac8944aaff40b35f3feb2696b7f0cee6ecaf80f4
- nccl@41ba8934a5e934ac67d4ad732e284b1d3ba76d15
- nccl@98cdc8c9f8e030aa198cca82438c8a1ec8672794
- nccl@493519987c197a332a3acbdddf671f98f8aac9dd
- nccl@d9857c3d0df44c9edf52c529886c2a8a1cabbf79
- nccl@6c83fdc6cc1fa5ad92fe2fdd5d8cf787d2842491
- nccl@d1698b103df0b7d563c7ac83b8b12702579b3a71
- nccl@35ca9a7edf3e3608cd725337e010fab6d2eea4ab
- nccl@6a14408e2a8d04b4a09c86f692ab6e5a09dc9bdb
- nccl@60b2e6a181a46c17de00b4d3638dd79dc294bab4
- nccl@b013fcbf99388b16f388ce88f7ba443cbf975a0b
- nccl@7456243b3f0322471e4f0de26fea11e1a58b7e42
- nccl@8ee7227bbac5b7a9fc194453b5a01bec7d367696
- nccl@255a2ed6f034ecb82d100675887c9fc5b03e8d0e
- nccl@1b62975f568b7709a1327a94b13112a6c2db4c18
- nccl@b505b3bb2f1cce1c867e04a5e18df24af900ee46
- nccl@3a6a88b427f4260469387dcd7fb9e90b180d73e9
- nccl@8c18a561ed6fa084446b185331470482c14e5c99
- nccl@b27709f3d55c886fe9ae36f2d1f2e1b8fbb15021
- nccl@5ba2dc7ef8491fdda31adae80eb160bf15456bac
- nccl@c0a5be6b38c19b4772d221841a1b68bffcbfadca
- nccl@7f5091f374dcad4528f28d82755874bfe3670690
- nccl@e2dae7cb88edc8c538556f9e79b39292c5a54a43
- nccl@8fb387c5cc7ae66940ecb3d9239bf7c5bb580460
- nccl@0ff8d55eef0ec85c66df575b1411aeb9f3927e02
- nccl@7fdabd66082c1006f775723a2b122f17ef3938a6
</details>
 
<details>
<summary><h2>Testing</h2></summary>
 
### Objectives and Timeline

#### Code Coverage Goal Defined

None.

#### KPI Coverage Goals Defined (performance, stress, stability, throughput, latency)

No change.

#### Requirement Coverage Goal Defined

As a first step, we need to test point-to-point operations through an
alltoall performance test. Next steps will require running sendrecv,
halo exchange, ...

#### Quality Thresholds Defined

No error.

#### Test Timeline

Alltoall will be tested in NCCL 2.7 (Q2 2020). Other tests will be in
NCCL 2.8 (Q3 2020).

#### SW Verification and Test Plan

Run above tests on DGX platforms, PCI platforms, P9 (CPU Nvlink), both
intra-node and multi-node.

### Test plan

#### Requirements Tests

Performance tests need to be written for Alltoall, Scatter, Gather,
Halo, Sendrecv.

#### Interface Tests

New API tests needs to be written for ncclSend and ncclRecv.

#### Fault-injection Tests

None.

#### Resource Usage Tests

None.

#### Design Coverage Testing

None.

#### Boundary Tests

None.

#### Certification Tests

None.

#### Stress Tests

None.

#### Stability Tests

None.

#### Perf and Power KPI Tests

None.

#### Usability & OOBE Tests

None.

#### Manufacturing Diagnostic(Factory) Tests

None.

### Signoff list

Author : Sylvain Jeaugey
</details>
 
