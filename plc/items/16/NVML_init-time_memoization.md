# NVML init-time memoization
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

\$TBD

### Functional Requirements

None/\$TBD -- new functionalities

### System Requirements

None/\$TBD -- perf, scalability

### Interface Requirements

None/\$TBD -- specific API

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

It was discovered that NVML, especially on high GPU count systems like
DGX-2, can have a high cost to query basic things (~1ms). Also, NVML
scales poorly due to some internal system-wide lock which prevents
parallelism of multiple processes invoking NVML. With the advent of P2P
down-detection where we pessimistically query NVML for every GPU-GPU
connection this was bottlenecking our initialization time.

The fix was to memoize (cache in a table) at the process level all
expected NVML queries during initialization time, ensuring a process
never makes these expected calls into NVML redundantly. The wrapper
function (ncclNvmlFoo etc) consult these tables automatically. Because
the memoization's scope is limited to the process, multi-process runs
still experience the redundancy of every process building the same
table, which is unfortunately serialized by NVML internals. As future
work, we could fix this by having a single process per node build the
tables, then broadcast them using the bootstrap API. This would incur
the complication of an API change since the NVML APIs use process-local
handles (pointer to nvmlDevice_t) to refer to devices, and so our
wrapper API would need to diverge from the NVML API to use a
process-invariant device numbering.

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
 
