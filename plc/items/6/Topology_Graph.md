# Topology Graph
<details>
<summary><h2>Requirements</h2></summary>

### Project Context and Purpose

NCCL provides fast inter-GPU communication for parallel applications.

### NVbugs / Jira Tickets

[Github \#179](https://github.com/NVIDIA/nccl/issues/179)

[NVBug 2313751](http://nvbugs.nvidia.com/2313751)

[Jira NCCL-827](https://jirasw.nvidia.com/browse/NCCL-827)

### User Experience

NCCL is used through deep learning frameworks or other HPC applications
to accelerate GPU-to-GPU communication. It is supposed to be transparent
to the user.

### Assumptions, constraints and dependencies

None

### Use Cases

DL training on GPU platforms, in particular DGX systems, potentially
within a large cluster. HPC applications

### Functional Requirements

Fix the few cases which didn't work so far (see nvbugs). Provide trees
with different in/out GPUs when possible. Provide trees flowing in a
unique direction intra-node when possible.

### System Requirements

Provide the best performance based on the node topology.

### Interface Requirements

No new API.

### KPI Requirements

Performance should remain the same or better. DGX systems should get
peak performance when using a regular subpart of the node (e.g. GPUs
0,1,2,3 or 4,5,6,7 of a 8 GPU node). DGX systems should be functional
and get reasonable performance when using a random subpart of the node
(e.g. GPUs 0,4,6,7 or an 8 GPU node).

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

We replace the old ring creation system by a new system based on graphs.

#### Ring creation system up to NCCL 2.4

The ring creation system was hierarchical, with a large part being in
the transports (P2P, SHM, NET).

The typical scenario was :

- For each rank, transports filled a "connect" value for each other
  rank. 0 meant the transport could not connect to the other rank, and
  we would try P2P, SHM then NET which should never return 0. The value
  above 0 was transport specific and had different meaning, P2P encoding
  the number of Direct NVLinks, NVSwitch NVLinks or the PCI distance,
  while NET was encoding the distance with each NIC.
- Init exchanged the value so that each rank would have the full NxN
  matrix.
- The main ring creation algorithm would ask the NET transport to
  connect nodes together providing the whole set of net values from
  before.
- It would then ask SHM to connect ranks within the rank node (only),
  honoring previous connections made by NET if any.
- Finally it would ask P2P to connect the ranks that are not yet
  connected already, using NVLink/NCswitch or PCI.

The main issue with this design is that once the NET transport has
connected nodes together choosing ranks from two nodes, those ranks
cannot change. So there can be cases where there is no good path (e.g.
with NVLink) to connect the rank where data enters the node to the rank
where data exits the node, going through all other ranks in between.

Another problem is that it is focused on creating rings, connecting
nodes with NET, then SHM domains, then individual GPUs through P2P.
Simple trees could be created from rings, using only the inbound GPU,
but more complex trees which enter and exit the node from different GPUs
cannot be built from rings and need a dedicated tree creation algorithm.

#### New design

##### Topology graph

We replace the NxN matrix of {transports, values} by a graph
representing the intra-node topology of NVLink, PCI and CPUs. This graph
has nodes (CPUs, PCI switches, NVswitches, GPUs) and links (QPI, PCI or
NVLink). This no longer involves transports, and only involves the
networks to get the /sys path of network devices.

##### Ring/Tree creation

Ring/Tree creation consists in a search throuh the graph to find
multiple paths through all the GPUs, potentially entering/exiting
through NICs.

This is a single algorithm which is no longer hierarchical nor involving
transports. The main issue is the convergence. As it is no longer
hierarchical, the graph has all components, making searches slower.

##### Inter-node connections

Once the intra-node paths are computed, the entering/exiting ranks for
rings and trees are exchanged with other ranks and a final step connects
rings and trees between nodes.

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

Author : Sylvain Jeaugey
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

