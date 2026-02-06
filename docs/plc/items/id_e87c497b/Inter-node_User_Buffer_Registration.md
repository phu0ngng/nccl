# Inter-node User Buffer Registration
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->
Currently, NCCL does not widely support net-based user buffer registration for collectives, which limits the NCCL's
UB benefits. To solve this issue, inter-node user buffer registration is a feature that registers user buffers into
the corresponding NICs so that NICs can directly access the user buffers without extra copy. The key benefits of the
feature are reducing the memory pressure, consumption and improving performance.
<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets
https://nvbugspro.nvidia.com/bug/4641364
### User Experience
Users do not need to change their implementations. Inter-node UB will be automatically applied.
### Assumptions, constraints and dependencies
NA
### Use Cases
- IB SHARP
- Net send/recv
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
The feature targets user buffer registration for collnet and net transports and allows data to be directly
written into the user buffer through the NIC without extra copy to/from intermediate net buffers. To register
the buffers, the main thread provides the buffer information (buffer address and size) to proxy service thread,
and the proxy thread retrives the buffer information as well as the corresponding net commmunicator to register
the buffer. The buffer registration supports DMA buffer registration as well as the regular nvidia_peermem,
but DMA buffer registration is always used if available. After registration, the proxy thread returns the
memory handle back to the main thread. The main thread passes the memory handle into the proxyOp which will
be accessed during proxy progress.

For collectives except collnet, we need to register sendbuff and recvbuff to sendPeer and recvbuff to recvPeer.
For sendrecv, we only register sendbuff to sendPeer and recvbuff to recvPeer. Collectives need one more registration
for recvbuff because the sender side needs to access recvbuff to send the intermediate data from other ranks.
All registration, tuning and algorithm info (such as chunkSize, reg handles, ring topology and so on) are passed into
newly created algorithm objects (e.g., RingARAlgorithm). The algorithm object is used during proxyProgress and
returns the send or recv buffer address, size and memory handle based on the current step. In addition, the buffer size
is retrieved through connFifo, and connFifo is filled by kernel during waitPeer stage; conFifo is also a signal that
kernel side is ready so that proxy thread can start to access the user buffers.

We keep kernel and proxy progress side in sync. The kernel always posts a message through connFiFo to proxy side to
indicate the user buffer is ready, and proxy thread then starts to post send and recv for each chunk. Kernel
will wait and post the step like non-UB case, only chunk size is tweaked when net UB is enabled in 1PPN case. In this way, it
is very easy to drive the algorithms and signal data ready through kernel and send/recv data from user buffers through
proxy thread.

The inter-node UB supported the following collectives and algorithms:

- (Allreduce, NVLS+COLLNET)
- (Allreduce, RING)
- (Allreduce, COLLNET)
- (Allgather, RING)
- (Allgather, COLLNET)
- (Reducescatter, COLLNET)
- (Broadcast, RING)

For 1ppn inter-node UB case without reduce op, we only use 1 SM to drive algorithms from the kernel and offload most of work
to the network.

The send/recv UB is refactored to adapt to the new design. All send/recv buffers are registered during enqueue stage
instead of progress stage now.

For more implementation details, please see `Coding Section`.

<!-- note: the following HTML code is also valid -->
<!-- <img src="images/example.png" width="900" height="500" /> -->

### Interface Architecture
NA
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

### Collnet UB Implementation
Allreduce + NVLS collnet UB needs both send and recv buffer to be registered to the same NIC/collnet communicator.
In `registerCollBuffers()`, we use `ncclCollnetLocalRegisterBuffer()` or `ncclCollnetGraphRegisterBuffer()` to
perform a proxy call, let proxy thread registers buffers and returns handles back. Once buffers are registered in collnet,
we set devwork.netRegUsed as 1 and pass it to kernel. When the kernel sees `netRegUsed == 1`, we change the order of data
processing for ranks and channels. When `netRegUsed == 0`, we iterate through channels, and NVLS headRanks in each channel
process the data in sequence; when `netRegUsed == 1`, we iterate through headRanks, and channels in each headRanks process
the data in sequence. This change is necessary because collnet requires data to be channel-wise contiguous, so data stored
into recv buffers under UB must be contiguous. The related changes are shown as follows:

```
for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
  ssize_t offset = work->regUsed && work->netRegUsed ? gridOffset + (nvls->headRank * nChannels + bid) * chunkSize
                                                     : gridOffset + (bid * nvls->nHeads + nvls->headRank) * chunkSize;
  int nelem = min(chunkSize, size - offset);
  prims.directRecvDirectSend(offset, offset, nelem);
}
```

For AG + CollnetDirect, we only register sendbuff since AG direct algorithm directly sends data
through sendbuff to NIC and gather data back as the first step. The reason we cannot utilize the
recvbuff is because the data received through NIC is not contiguous so we cannot pass the recvbuff
directly to collnet iallgather API. Therefore, we have to use intermediate buffer to receive data
from NIC and broadcast to local peers. For RS + CollnetDirect, it is the same as AG, but only
recvbuff is registered and used to receive the results from network.

### SendRecv UB Refactorization
SendRecv net registration is refactored. Now it follows the same logic as collective registration; that
is we register the send and recv buffers during enqueue stage through proxy thread. In addition, even if
buffers are registered at the NICs, we don't set proxyOps->nstep as 1 but divUp(Bytes, NCCL_MAX_NET_SIZE).
NCCL_MAX_NET_SIZE is set as 1GB for now and can be changed in the future. The kernel side needs to post the
same amount of steps to the proxy side in order to progress the net ops.

### Ring NET UB Implementation
Multi-PPN Ring Net UB supports AR, AG and BC 3 collectives. The critical part of Ring NET UB is to find out all net peers
and corresponding channels, and then register buffers to these peer channels one by one. Based on the peers and
channels, we pass the net handles to the proxy side, which will be used during proxy progress. In addition, 3
collective algorithm classes (i.e., RingARAlgorithm, RingAGAlgorithm, and RingBCAlgorithm) are implemented to help
find out send or recv address, size, and mem handle for each step. Once the proxy receives the signal from
connFifo which indicates the user buffer is ready and the corresponding head/tail is set correctly, it can start
to issue isend/irecv through the NICs.

In addition, 1PPN Ring Net UB is a bit different from multi-PPN Ring NET UB. For AR, the 1PPN Ring Net UB would just
run as multi-PPN Ring. However, for AG and BC, the 1PPN Ring NET UB would set nChannels to 1 and completely offload
the data communication to network and chunkSize is set as NCCL_MAX_NET_SIZE. In the kernel, we use 1 warp to drive
algorithm, and the rest of warps perform the local copy if the AG or BC is not a in-place collective.

### Kernel-side UB Framework
On kernel-side, UB flag setting starts from `loadRecvConn` and `loadSendConn` when the primitive object is created.
In these two functions, there are 3 flags which determine whether we should use UB or not. The first one is `Direct`,
which is a template parameter and passed when primitive object is created. For all primitives that need to use UB,
this flag must be set as 1. The reason we need this flag is we need a way to control UB enablement at algorithm level.
For example, AR COLLNET_DIRECT should not need UB for scatter op on H100 since it needs intermediate buffer to hold
data from all on-node peers; even if we register the send and recv buffer in this case, we should set `Direct` as 0.
The second is `p2pWork->send/recvIpcReg` or `collWork->regUsed`, which indicates whether the send/recv buffer has been
registered for intra-node communication (i.e., through NVlink or PCIe). The third is `p2pWork->send/recvNetReg` or
`collWork->netRegUsed`, which indicates the send/recv buffer has been registered through NICs for inter-node
communication (i.e., collnet or net).

The existing UB flag setting code snippet looks like this (list `loadSendConn` as an example):

```
if (Direct) {
  if (ipcRegFlag) {
    // User buffers have been registered
    if (conn->flags & (NCCL_P2P_WRITE | NCCL_P2P_READ)) {
      if (P2p) {
        flags |= conn->flags & NCCL_P2P_WRITE ? DirectWrite : DirectRead;
      } else if (connIndex == 1 && direct) {
        flags |= DirectRead;  // scatter-reduce use direct pull
      } else {
        flags |= direct & NCCL_P2P_READ ? DirectRead : DirectWrite;
      }
    } else if ((conn->flags & NCCL_NVLS_MIN_POLL)) {
      /* NVLS direct */
      flags |= DirectWrite;
    }
  }
  if (netRegFlag) {
    if (conn->flags & NCCL_DIRECT_NIC) {
      flags |= NetRegMode;
    }
  }
}
```

For intra-node communication UB, when `ipcRegFlag` is true, `conn->flags` is used to distinguish
different types of conections; NCCL_P2P_WRITE and NCCL_P2P_READ means the connection is intra-node
P2P connection, so we need to set P2P UB related flags. Historacially, we have flag DirectWrite and
DirectRead defined in the primitive. There are 3 type of cases to set the flags:

1. When `P2p` is true, it means it is a sendrecv connection. `conn->flags` decides whether
it is write- or read-based UB. If it is NCCL_P2P_WRITE, we set it write based flag `DirectWrite`;
otherwise, `DirectRead` is used.

2. When `connIndex == 1 && direct` is true, it means it is collnet scatter op. This is historical
flag setting for scatter direct read in AR collnet. If input `direct` is set to any value, it
means the primitive is collnet operation.

3. The rest of flag settings go into the last case. This case includes directGather in collnet when
input `direct` is set. If input `direct` is set, the flag is set based on the `direct` value;
if input `direct` is not set (which is 0), then they are non-collnet collectives which always
use `DirectWrite` flag for now.

Besides P2P connection, when `conn->flags & NCCL_NVLS_MIN_POLL` is true, it means NVLS conection.
In this case, we need to bitwise-or the flag with `DirectWrite`. Except this, NVLS UB logic is the
same as P2P UB during the data movement.

For inter-node communication UB, when `netRegFlag` is true and `conn->flags` contains NCCL_DIRECT_NIC,
it means the connection is a net connection, and it supports GPU Direct RDMA. In this case, we bitwise-or
the flag with NetRegMode to indicate the send peer op can use registered net buffer directly.

After the flag is set properly during `loadSendConn`, we will exchange corresponding imported buffer
ptrs in `setDataPtrs()` function, which is only for intra-node communication. For `DirectWrite` connection,
recv peer needs to pass its remotely imported buffer address to the sender, which will be used later by sender
during send primitive ops. Similarly, for `DirectRead` connection, send peer needs to pass its remotely imported
buffer address to the receiver, which will be used later by receiver during recv primitive ops. The details
about buffer import/export in intra-node communication UB can be found in intra-node UB PLC in NCCL 2.23 PLC.

After primitive init, we will call different primitive functions to send and recv data between peers. No
matter what functions we call on algorithm level, it finally translates to a bunch of send to and recv from
peers in primitive layer. During waitPeer(), we will check flag for each peer and see whether some UB flag is marked.

For inter-node UB, if `NetRegMode` is set, it means the peer is net peer and we should use net UB. For
sendrecv based net peer, we should set ptrs as NULL for both send and recv since no copy is needed. For
collective based net send peer, if there is no recv peer in the primitive function, we need to set ptrs
as NULL since the src should be input buffer and the corresponding primitive functions should be
`directSend` or `directCopySend`. Otherwise, we always set ptrs as output buffer plus offset. For recv
peer, since it only operates on output buffer, so it should always set ptrs as output buffer plus offset.
The code is shown as below:

```
if (flags & NetRegMode) {
  if (P2p) {
    ptrs[index] = NULL;
  } else {
    if (isSendNotRecv) {
      if (!Recv)
        ptrs[index] = NULL;
      else
        ptrs[index] = (T*)ncclShmem.groups[group].userOutput + dstIx + offset;
    } else {
      ptrs[index] = (T*)ncclShmem.groups[group].userOutput + srcIx + offset;
    }
  }
}
```

For intra-node UB, we have 3 different cases and we use UB send as an example here. When flag contains
`DirectWrite`, it means it can directly write data into the dst buffers; when flag contains `DirectRead`,
it means receiver should directly read the data, so sender does not need to do anything and set ptrs
as NULL; when no UB flag is found, we just fall back to intermediate buffers. The code snippet is like this:

```
if (flags & DirectWrite) {
  ptrs[index] = directBuff + dstIx + offset;
} else if (flags & DirectRead) {  // empty send
  ptrs[index] = nullptr;
} else {
  ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
}
```

### CPU-side UB Framework
Currently, we have the following UB registration cases:

1. NVLink SHARP
2. Collnet and Net
3. Intra-node and MNNVL IPC

All CPU side collective registration logic is in `ncclRegisterCollBuffers` function. For all supported registration,
we support both local and graph registration. Local registration is always tried first; if local registration succeeds,
then we continue the rest of enqueue work; otherwise, we try graph registration. If both fail, UB is disabled and
NCCL can still run operations without UB.

The registration starts with the algorithm checking. If it is NCCL_ALGO_NVLS or NCCL_ALGO_NVLS_TREE, we start to
register NVLS buffers. For NCCL_ALGO_NVLS, we can register buffers for intra-node NVLS communication as well as
collnet communication if the number of nodes is >= 2. For NCCL_ALGO_NVLS_TREE, we only register buffers for intra-node
NVLS communication.

For the rest of algorithms with Simple protocol, we apply IPC and Collnet/Net registration. The major idea of
registration behinds these algorithms is very similar, and I use Ring algorithm as an example. For Ring algorithm,
we first iterate through all channels and find the send and recv peers along with each channel. Then we check whether
the peers are net peer or ipc peer based on conn.flags. If it is net peer, we add the corresponding net
connector into send/recvNetConns array, which will be used to find proxy and corresponding net communicator for that
channel; if it is ipc peer, we add the peer rank to peerRanks array, which will be used to find out their proxy
connection and register buffers through these peers' proxy threads. After obtaining all these information, we can
register buffers through IPC and Net registration API `ncclIpcLocal/GraphRegisterBuffer` and `ncclNetLocal/GraphRegisterBuffer`.
If IPC registration succeeds, we bitwise-or regBufType with NCCL_IPC_REG_BUFFER; if Net registration succeeds,
we bitwise-or regBufType with NCCL_NET_REG_BUFFER. These flags will be used later in enqueue to determined whether
we should set `regUsed` and `netRegUsed` flag in `struct ncclDevWorkColl devWork`.

For sendrecv registration, it follows the same logic as collectives but just registers buffers with one send or
recv peer.

We don't support direct buffer access anymore since it does not increase the reference count of buffers from
both side, so if users call abort on any of communicators and free the send and recv buffer, it can cause illegal
memory access on the other side of the GPU, which is accessing the user buffers.

### Commit list or MR
https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/602
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
The performance should be equal or better than the baseline.
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
