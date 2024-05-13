# Rings and Tree rework for DGX H100
<details>
<summary><h2>Requirements</h2></summary>
 
### Introduction

#### Overview

NCCL aims at providing the best performance for DGX systems, so that our
users can have the best possible experience when moving from their
development setup to a DGX, potentially in the cloud.

Since the beginning of NCCL and DGX systems, allreduce operations have
heavily relied on the ring algorithm. One specificity of DGX H100 is
that each network adapter is connected directly through a PCI switch to
only one GPU (instead of 2 so far). This prevents NCCL from closing the
rings, which should enter from one NIC and go to a local GPU, then exit
from another GPU to that same NIC.

To solve that problem, NCCL added the PXN feature. The problem is that
on DGX H100, we are limited by the NVLink bandwidth, especially when
using PXN since PXN adds an extra NVLink step to go back to the first
GPU before leaving the node.

This means that the ring allreduce algorithm only runs at 300-330 GB/s
instead of 390GB/s.

#### Assumptions

None.

#### Constraints

None.

#### Dependencies

None.

#### References

[Jira NCCL-1554](https://jirasw.nvidia.com/browse/NCCL-1554)

### Use cases

#### Allreduce

Allreduce performance on multiple nodes.

### Requirements

#### REQ-1 - KPI - Allreduce SOL on DGX H100

Dual node ring allreduce on DGX H100 should work at SOL, i.e. 390GB/s.

### Signoff list

Author : Sylvain Jeaugey
</details>
 
<details>
<summary><h2>Design</h2></summary>
 
### Design

#### Alternate cross-nic rings

To avoid the final PXN step to close the rings, we can exit directly
from the last GPU using its local NIC. The problem with that design is
that it causes cross-rail traffic. For example, say we enter the node
through NIC A, then go to GPU A, around all GPUs, and finish with GPU B,
then exit through NIC B. When we connect nodes together, we would send
from NIC B to the NIC A of the next node.

The solution is to "reverse" the ring on every other node, i.e. send to
the NIC B of the next node, which will exit from NIC A, and use the
normal order on the 3rd node, and so on.

In practice, during the graph search, we will allow ourselves to exit
from a different NIC than we entered the node with, as long as there is
a corresponding ring with opposite NIC direction. With that, we build
pairs of rings, one entering on NIC A and exiting on NIC B, and the
other entering on NIC B and exiting on NIC A.

This also only works with an even number of nodes. If we have an odd
number of nodes, we may still want to use that algorithm, as the rail
crossing would only occur once, and not upon every node transition,
which is still a significant win.

#### Channel tuning

Now that we have removed the NVLink bottleneck by avoiding the extra PXN
step, we can reach higher performance. Unfortunately, on DGX H100, we
cannot reach 390 GB/s with only 16 channels.

One problem is that the tree algorithm needs an even number of channels
per tree, so we would need to bump the number of channels to 4 per
ring/tree, which would be 32 channels. That is because the tree
algorithm relies on a double binary tree, and each channel only
implements one of the two binary trees.

#### Send/Recv and Coll harmonization

To be able to use only 3 channels per ring/tree, we need to re-architect
the Tree algorithm to implement both trees in each channel. This is
doable since we now have two connections within each channel (connIndex
0 and connIndex 1). The problem is that connIndex 0 and connIndex 1 are
not used in the same way, so we cannnot share a connIndex 1 connection
between collectives and send/recv operations. In previous versions,
connIndex 0 was used by most collective operations, and connIndex 1 was
used by send/receive operations, plus intra-node by NVLS and Direct
algorithms.

But the problem is on the network side as connections on connIndex 1 use
shared buffers which work differently w.r.t. the head counter, and which
is incompatible with LL/LL128 operation.

So if we want to be able to use connIndex 1 for the tree algorithm, we
need to harmonize how shared buffers work (so that they're compatible
with LL/LL128 head operation), and also use shared buffers for both
send/recv and collective operations, when using the simple protocol.

The first step was to harmonize the use of the head counter. In shared
buffer mode, we used to only give gredits through the head counter when
we would post buffers in the offset FIFO. So solve that problem, we now
use the head counter as for non-shared buffers, but set the offsets to
-1 to indicate they are not ready to consume. The CUDA kernel will poll
on that value until the proxy sets it, and the proxy will reset it to -1
after it has been consumed.

The second step is to allocate and use a shared buffer for all network
operations on connIndex 0. For send/recv operations, the shared buffers
have been allocating 16 128KB chunks. The ring algorithm, on the other
hand, needs 4 slots of 1MB. To allow for both usage, the proxy will
allocate 32 chunks of 128KB, and then allocate it either by 1 chunk for
p2p, or by larger contiguous sets of chunks for ring.

Once that was done, the tree algorithm has been modified to use both
connIndexes, implementing one tree on each. Each SM is cut into 4 sets
of threads, to manage both directions (up and down) on both trees. This
is better for performance as it balances the workload better on
channels. Previously, one channel would have a lot of reductions to
perform (as it would be a node in the tree), which the other would have
nothing to do (as it would be a leaf in the tree).

### Signoff list

Author : Sylvain Jeaugey
</details>
 
<details>
<summary><h2>Coding</h2></summary>
 
[MR
12345](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/12345)
</details>
 
<details>
<summary><h2>Testing</h2></summary>
 
### Introduction

#### Test strategy

NCCL testing is done with three main suites:

- API tests is a suite of tests based on Google Tests, and testing all
  NCCL functions from the outside. They are supposed to test boundaries,
  error cases, and functionality. API tests are currently limited to run
  as a single process, which is a significant limitation.
- Unit tests are individual tests compiled against a subpart of NCCL.
  Unfortunately, most of the NCCL code requires a parallel environment
  to run, but some units are designed to be extracted and compiled
  outside of NCCL to be tested in isolation.
- Performance tests, also available on Github (without some features)
  serve both as stress test, functional tests and performance tests.
  They can run at any scale, and rely on MPI to launch parallel
  processes.

#### Test environment

Test platforms include a variety of systems, including DGX servers, PCI
platforms, and cloud platforms.

The complete list of NCCL test platforms is set here for each release:
https://docs.google.com/spreadsheets/d/1hWOScxCJmj2HpuvBpcUE0g4mSMgWJQvc4dePX5TIJ3w/edit?usp=sharing

#### API tests

No new tests.

#### Unit tests

No new tests.

#### Functional tests

No new tests, leveraging existing allreduce tests.

#### Performance tests

No new tests, leveraging existing allreduce tests. Performance should
improve on allreduce on 2 DGX H100.

### Signoff list

Author : Sylvain Jeaugey
</details>
 
