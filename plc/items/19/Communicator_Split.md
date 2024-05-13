# Communicator Split
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

None/\$TBD

### Use Cases

In some cases, it is more convenient and efficient to express
communication patterns with the subset of initially-created
communicators, so communicator split functionality is needed to support
this type of use cases.

### Functional Requirements

1\. Split communicators 2. Share resources of the parent communicators

### System Requirements

None/\$TBD -- perf, scalability

### Interface Requirements

Need a new split API interface: ncclResult_t ncclCommSplit(ncclComm_t
comm, int color, int key, ncclComm_t \*newcomm, ncclConfig_t \*config);

### KPI Requirements

None/\$TBD -- perf

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
</details>
 
<details>
<summary><h2>Design</h2></summary>
 
### Proposed Design

NCCL communicator split includes two major parts: 1. split parent
communicators into child communicators; 2. child communicators are
allowed to share resources of parent communicators. For (1), the
communicators which own the same color are in the same subgroup; the
ranks in the same subgroup are connected through parent bootstrap, and
the subgroup bootstrap are created at the same time. Once the bootstrap
is built up, the rest of work is the same as ncclCommInitRank(). For
(2), the purpose of sharing is to save memory resources. Therefore, the
major communication resources to be shared are

      /* shared resources as long as splitShare is set */
      struct ncclChannelPeer* peers[MAXCHANNELS];
      struct ncclDevChannelPeer* devPeers[MAXCHANNELS];
      struct ncclStrongStream deviceStream, hostStream;
      struct ncclProxyState* proxyState;

      /* sharable NVLS and Collnet peers */
      struct ncclChannelPeer* collnetPeers;
      struct ncclDevChannelPeer* collnetDevPeers;
      struct ncclChannelPeer* nvlsPeers;
      struct ncclDevChannelPeer* nvlsDevPeers;
      struct ncclProxySharedCollNet* collNetSharedRes;
      struct nvlsResources* nvlsResources;

      /* always shared resources whenever split */
      volatile uint32_t *abortFlag;

Peer and devPeers stores the connection buffers except NVLS and Collnet,
and child communicators need to create an peer array and map its
peer/devPeer to the parent's. Since all connections are shared, we
cannot bind any communicators to the connections as previous design
"peers-\>send-\>comm". Instead, we delete these bindings and pass the
communicator to the related functions where communicator reference is
necessary. deviceStream and hostStream are shared CUDA stream to avoid
buffer contention among parent and child communicators; proxyState is
shared proxy resources for proxy call. Among them, proxyState is the
most chanllenging one to share since it not only shares the proxy memory
resources, but also all proxy-related operations and requests; in
addition, the proxy service and progress thread are only created once
for the top parent communicators and shared among child communicators.
Therefore, we need to refactor proxy code to adapt to this requirement.
The first change is flattening child communicators. As Fig. 1 shows,
communicators can be split multiple times and the parent of a child
communicator might not be the top parent. Since proxy are based on top
parent communicators, we have to transform the parent of each child
communicator to top parent (i.e. flattening) and perform proxy
operations based on top parent's information (e.g. rank and localRank).

![](images/split-flatten.png) The second change is decoulping
communicator and proxyState. Since child communicators can share proxy
resources, we cannot bind proxyState to any communicators. To this end,
we need to copy all needed information from communication to proxyState
before proxy thread is launched. For example, ncclNet is added to
proxyState in order to perform network operations on proxy side. NVLS
resources are shared when splitShare is set, parent NVLS is enabled, and
for each node the number of local ranks in child communicator is equal
to parent's, the child can share the NVLS resources (i.e. NVLS peers and
buffers) of the parent. Similarly, Collnet resources are shared when
splitShare is set, parent collnet is enabled, and the heads building
each collnet communicator are equal to the parent's, the child can share
the collnet resources of the parent.

### Interface Architecture

### System KPIs & Metrics

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
 

</details>
 
<details>
<summary><h2>Testing</h2></summary>
 
### Objectives and Timeline

#### Code Coverage Goal Defined

None.

#### KPI Coverage Goals Defined (performance, stress, stability, throughput, latency)

No change.

#### Requirement Coverage Goal Defined

TBD.

#### Quality Thresholds Defined

No error.

#### Test Timeline

#### SW Verification and Test Plan

### Test plan

#### Requirements Tests

TBD

#### Interface Tests

TBD

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
</details>
 
