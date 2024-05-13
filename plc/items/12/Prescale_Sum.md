# Prescale Sum
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

The goal is to allow users to pre-multiply inputs to summation
reductions by an arbitrary per-rank scalar. The API must be capable of
accepting a pointer-to-scalar that lives either on the device or host,
the important case being on the device since that will be the most
performant. We also want to accomplish all of this without changing any
existing NCCL function signatures.

We accomplished all of these by introducing a "constructor" for
ncclRedOp_t values that will take the scaling factor and return a
handle-like ncclRedOp_t for the user to pass to collectives. When the
user is done with the handle it can be reclaimed by NCCL with a
"destroy" function.

The new operator handles are local to the communicator they are created
with. The benefit of this design is it allows us to allocate/deallocate
handles using thread-unsafe code. It is permissible for the
implementation to use the same handle integer for ops created under
different communicators, but it is avoided by the implementation to the
greatest extent possible.

Internal changes consist of creating a new ncclDevRedOp_t to index the
different device side kernels, and a ncclDevRedOpFull struct which holds
that op value and the data which must be shipped up to the kernel (which
is either a scalar immediate value or device side pointer to scalar).
"src/enqueue.cc" contains the new API functions, and the logic for
mapping ncclRedOp_t to ncclDevRedOpFull. ncclDevRedOp_t contains the
four baked-in ops {sum,prod,min,max} and adds {ncclDevPreMulSum,
ncclDevSumPostDiv}. ncclAvg is mapped to ncclDevPreMulSum for floating
point types and ncclDevSumPostDiv for integral. ncclDevSumPostDiv is not
exposed to the user so its device code is only instantiated for integral
types.

### Interface Architecture

New function ncclRedOpCreatePreMulSum(\*op, \*scalar, datatype,
residence, comm)

New function ncclRedOpDestroy(op, comm)

### System KPIs & Metrics

SOL performance is that the extra multiply instructions come at no added
cost. This was verified to be true on Luna for all cases.

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

New "mulsum" op added to tests/perf.

#### Interface Tests

New "tests/apitests/ncclRedOpCreatePreMulSum.cu" apitest added. Covers
both the host-side and device-side scalar cases.

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
 
