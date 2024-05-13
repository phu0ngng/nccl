# Indirect NVLink communication (NVB)
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

Point-to-point communication (send/receive) between GPUs which do not
have a direct NVLink connection, but both have an NVLink connection to a
third GPU.

In particular, alltoall on DGX-1 cubemesh topology.

### Functional Requirements

None.

### System Requirements

DGX-1 or similar.

### Interface Requirements

None.

### KPI Requirements

Improved alltoall performance on DGX-1.

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

GPUs which do not have a direct NVLink on DGX-1 need to communicate
through system memory, i.e. through the CPU. This is extremely slow
since we have many combinations which need to use that channel.

Instead, we want to use NVLinks which would have a much higher
bandwidth, albeit shared with other direct flows needed for the alltoall
operation. This is quite simple to implement in NCCL since we only need
to have the intermediate FIFO used to communicate between two GPUs to be
located on an intermediate GPU which both other GPUs can directly access
(one GPU writing to it, one reading from it).

Also, we only want to go through GPUs in the NCCL communicator, since we
would not want to use resources from a GPU used by another user, in
particular as it could cause an error if the GPU is configured in
exclusive mode.

The first change for that feature is to change the topology path
exploration, and allow communication between two GPUs to go through
another GPU, as long as it is a single hop. Doing so, we add a new PATH
level, NVB (as NVlink Bridge, similar to PXB). That makes sure a further
path search for collectives will accurately reflect the expected
bandwidth, going through the two NVLinks and counting the bandwidth used
accordingly.

Then, when creating the P2P connection between the two ranks, if the
path type is NVB, we find the intermediate GPU and allocate memory from
it, then map it on both ranks using direct P2P or CUDA IPCs. A new
service thread is therefore created, which listens to a socket and
allocates memory for other GPUs, returning CUDA IPC handles. The thread
exits when the communicator is destroyed and all buffers have been
released.

### Interface Architecture

None.

### System KPIs & Metrics

Alltoall performance on DGX-1V is now above 40GB/s, compared to 10GB/s
with NCCL 2.7.

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

No change.

#### Quality Thresholds Defined

No error.

#### Test Timeline

Part of NCCL 2.8 QA.

#### SW Verification and Test Plan

Part of NCCL 2.8 QA.

### Test plan

NCCL 2.8 test plan.

#### Requirements Tests

None.

#### Interface Tests

None.

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
 
