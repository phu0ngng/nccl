# User buffer registration for send recv operations
<details>
<summary><h2>Requirements</h2></summary>

### Introduction

#### Overview

NCCL is used in deep learning training to accelerate the training of
models on multiple GPUs. To compute in parallel, frameworks use NCCL to
synchronize the different GPUs on a regular basis. Those synchronization
operations take time and slow down the computing tasks, which degrades
the scalability of the training. This feature aims at improving
scalability, by improving how we can run NCCL operations concurrently
with compute operations.

The NCCL API require users to provide input and output buffers, which
NCCL needs to copy to its own intermediate buffers, which have been
registered to the RDMA network adapter, so that they can be used to send
data directly from one GPU to another. Those extra copies cause
significant extra memory pressure for the GPU, which impacts negatively
the overlap of NCCL operations with compute operations.

This feature is allowing users to register their buffers with NCCL,
which allows NCCL to pre-register them with the network adapter, so that
for some operations like send/recv operations, we can skip the extra
copies.

This should then be integrated by Deep Learning frameworks, which should
call registration functions on buffers they would use with NCCL.

This feature has also been delivered to Meta as a beta functionality.
They have been using it for various purposes, including the
implementation of allgather with direct alltoall or recursive doubling.

#### Assumptions

We assume frameworks will be able to register their communication
buffers. Ideally they should allocate a single large buffer at the
beginning and register it, rather than registering a multitude of
smaller buffers.

#### Constraints

No major constraint identified.

#### Dependencies

The initial implementation is done using the internal IB plugin. This
work will need to be replicated to the external IB SHARP plugin for
productization on some platforms. Other network plugin providers like
AWS may want to also adapt their plugins to leverage that functionality.

#### References

[Jira NCCL-1502](https://jirasw.nvidia.com/browse/NCCL-1502)

### Use cases

#### Registered Alltoall

When running with registered buffers and PXN disabled, we may want to
overlap an alltoall operation with a compute operation.

#### Send/recv based Allgather

An allgather algorithm based on send/recv (recursive doubling or direct)
may also be overlapped with a compute operation.

### Requirements

#### REQ-1 - Functional - User buffer send/recv

Send receive operations involving user buffers which have been
registered should not perform additional copies when possible. One
exception is on the sender side, when PXN has not been disabled, as PXN
performs an extra copy to an intermediate GPU to aggregate operations
and provide better rail locality.

### Signoff list

Author : Sylvain Jeaugey
</details>

<details>
<summary><h2>Design</h2></summary>

### Design

#### Memory registration cache

The buffer registration system works at two levels: the upper layer
(entry point in NCCL) and the bottom layer (the network plugin).

When a buffer is registered in NCCL, a loopback network comm object will
be created and NCCL will register the buffer onto that object.

When a communication operation is started and contains a proxy progress
phase, the proxy operation will start by re-registering the buffer on
the network comm object for that operation. It is assumed that this
re-registration is a very lightweight operation which will only go
through the registration cache and return the memory handle (increasing
a refcount).

At the top layer, the ncclCommRegister code has been adapted to also
handle network registration on top of NVLS registration.

At the bottom layer, the plugin cache has been optimized for the case of
re-registration of a sub-part of a buffer, making sure we would not add
expensive ibv_reg_mr calls to the communication path, and re-use the
existing registration instead (mr).

#### Truncated sends

When buffers are registered, the goal is to send the entirety of the
buffer in a single RDMA operation. But we need to also properly handle
cases where only one side of a network connection is using registered
buffers. This means we may want to send the whole buffer but the
receiver will want to receive in chunks, or the opposite: we'll send in
chunks but the received will try to receive everything directly in the
user buffer.

In the proxy progress functions, we need to therefore check the sizes we
really sent/received and if necessary, add more progress steps until
everything has been sent/received.

In addition to that, the network plugin semantics had to be adapted to
allow for truncated sends. That way, if the sender tries to send the
whole buffer but the receiver only wants to receive one chunk, it should
not longer cause an error, and instead the sender will send only part of
the buffer and continue to send the rest in chunks until all done.

#### Multi-receive sizes

On the receiver side, in the case of a multi-receive, the previous
implementation would not correctly set the size field in ncclNetTest().
Instead of returning the real sizes, the plugin would only set them to 0
or 1; 1 indicating that we had received something, hence we needed a
flush. The size wasn't used for more than that. Given we now need to
know exactly how many bytes we received to keep track of where we are, a
new FIFO was added on the receiver side with the sizes which are sent.
This FIFO is written to by the sender, using the
RDMA_WRITE_WITH_IMMEDIATE operation at the end of each multi-send.

### Signoff list

Author : Sylvain Jeaugey
</details>

<details>
<summary><h2>Coding</h2></summary>

[MR 335 - User buffer reg for
p2p](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/335)
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

A new sendrecv test with registered buffers should be added:
ncclSendRecv_test/Registered. We may want to add registration tests for
all patterns, including scatter, gather, etc.

#### Unit tests

New unit tests for the plugin should be added, to test message
truncation on the receiver side, as well as registration time if
regIsGlobal is set.

#### Functional tests

The NCCL perf tests -R option has been extended to not only support 0 or
1, but also 's', 'r', and 'a', meaning respectively "sender", "receiver"
and "all". This allows to register buffers only on the sender side, only
on the receiver side, or on both sides. The 'a' value is equivalent to
'1'. The test plan has been extended to test alltoall_perf with "-R 1".

#### Performance tests

The perf tests run with -R 1 are part of the non-regression testing.

### Signoff list

Author : Sylvain Jeaugey
</details>

