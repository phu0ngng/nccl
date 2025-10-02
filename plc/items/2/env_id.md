# env_id
<details>
<summary><h2>Requirements</h2></summary>

### Project Context and Purpose

NCCL provides fast inter-GPU communication for parallel applications.

### NVbugs / Jira Tickets

[Jira NCCL-385](https://jirasw.nvidia.com/browse/NCCL-385)

### User Experience

NCCL is used through deep learning frameworks or other HPC applications
to accelerate GPU-to-GPU communication. It is supposed to be transparent
to the user.

### Assumptions, constraints and dependencies

None/\$TBD

### Use Cases

For use in large-scale multi-node scenarios, where many processes can
simultaneously get the ncclUniqueID by looking into the content of a
specified environment variable. They can hence connect to the root
process without waiting for the root process to broadcast the ID. This
can eliminate the reliance on MPI.

### Functional Requirements

Allow peer processes to create ncclUniqueID from an environment
variable, with which the peer processes can connect to the root process.
The design should support hostname as well as IP addresses (both IPv4
and IPv6).

### System Requirements

None

### Interface Requirements

New verbs can be added to provide this functionality, or current verbs
behavior can be extended.

### KPI Requirements

None

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

Author : Ke Wen
</details>

<details>
<summary><h2>Design</h2></summary>

### Proposed Design

#### High level design

An environment variable NCCL_COMM_ID is added to hold the network
information needed to connect to the root process. This network
information includes an IP address and an IP port.

Depending on the protocol, NCCL_COMM_ID supports the following formats:

For IPv4:

      NCCL_COMM_ID=ipv4:port

For IPv6 Global scope, a pair of braket is needed:

      NCCL_COMM_ID=[ipv6]:port

For IPv6 Link scope, an interface name is needed:

      NCCL_COMM_ID=[ipv6%ifName]:port

For host name:

      NCCL_COMM_ID=hostname:port

In the IPv6 Link-scope case, the interface name is used to distinguish
the scope (i.e. subnet). The reason is that two Link-type addresses can
have the same subnet address when they are not in the same subnet. By
IPv6 convention, the interface name (e.g. eth0) is used to distinguish
the scope.

In the old implementation, the root process would call:

      ncclGetUniqueId(&id);

to start the bootStrapRoot routine.

Whereas, the peer processes would call:

      ncclGetUniqueIdFromEnv(&id);

to connect to the root.

A new implementation mitigates the above requirement of an additional
API. In the new implementation, both the root and peer processes call
the same API:

      ncclGetUniqueId(&id);

The ncclGetUniqueId API now consists of two main steps: 1) create the
network handle (Unique ID) based on the environment variable (if
specified); 2) listen on the created network handle, if that is
successful, create the bootStrapRoot thread, otherwise, return.

All processes will go through step 1 without a problem.

In step 2, the processes on peer nodes would realize that they do not
own the specified IP, hence the listen would fail, and they would just
return with the acquired ID.

For the proesses on the root node, there are two cases:

i\) The first process that binds a listening socket to the specified
IP + port would become the root process, and then create the
bootstrapRoot thread;

ii\) All other later-coming processes would fail to bind to the IP +
port as it is already in use. These processes would just return with the
acquired ID.

The above arrangement guarantees that only one process would become the
root. Indeed, competing to bind a socket has been used as a method for
inter-process locking.

#### Low level design

The second part of the design involves how the peer processes contact
the root process. Conventionally, each process would use a default
interface (interface 0) to connect to the root. In the new design, each
process would search over all its interfaces and find one that is in the
same subnet as the root interface.

To deal with this, the meaning of the dev argument is extended for the
following APIs:

      int ncclSocketListen(int dev, void* opaqueHandle, void** listenComm);
      int ncclSocketConnect(int dev, void* opaqueHandle, void** sendComm);

An enum type is defined for the dev field:

    /* Socket Interface Selection type */
    typedef enum { findSubnetIf   = -1,
                   dontCareIf     = -2,
                   defaultIf      = 0 } ncclSocketIfSl_t;

For example:

    ncclSocketListen(dontCareIf, opaqueHandle, listenComm);

would directly listen on the opaqueHandle.

    ncclSocketConnect(findSubnetIf, opaqueHandle, sendComm);

would find an interface that is in the same subnet as the opaqueHandle
to connect to the opaqueHandle.

Otherwise, the use of the dev field is the same as before, i.e. it
specifies a physical device when it is non-negative.

To enable this extension, the bootstrap part is now changed to use only
the ncclSocket APIs.

### Interface Architecture

All processes would call:

    ncclGetUniqueId(&id);

### System KPIs & Metrics

N/A

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

Author : Ke Wen
</details>

<details>
<summary><h2>Coding</h2></summary>


- nccl@a2c0c76d47815226556cb74bbe2c0789af3e0d6c
- nccl@726101e6dfbf9417fb6d85bd60c4b7b18da14a8d
- nccl@83fe0deac65efdb13c707b921cbf2ccbf0c9d7e0
- nccl@78c23927054fa8d644afe08235c88a1faf5a8448
- nccl@68b137ee39b15d5d14da4dfd4ae97021e876e0d5
- nccl@ee6a938281fe131aefb600d936d48a4ab192074a
- nccl@04e90347d693b458e3f72766bf80a85a5bc4b32b
- nccl@4dc4d1618ff144898fae29a8111f82ae60628a0a
- nccl@76e0a8adb3babeab8ca1acc921a2ab9d5627ee67
- nccl@a6fc14abb11bd01857d44ba1749feabb9950aedc
- nccl@3c5737487de63c1435b91837296c25747b8f6935
- nccl@bca129a3eae737b604a3ddaf0d18285d58e6e563
- nccl@e6adc383f038dee3d592e8993c10c69adaef7c56
- nccl@572d286aeef87f7425535d6127fb740ba878631a
- nccl@de2353b2a4a9557dcdaf05cb11271bd24f9142e8
- nccl@65ca071d345e24a8534f6613c10b04c324d5e1b7
- nccl@5132b3b1fde6d9baf5ed76e97e4ea430de9cb16a
- nccl@6eb942f7cb57f543809c23e3db2dbeb5a6b6bb40
- nccl@a1927524ed55cc5bc738d7c7a96139c0028a4c75
- nccl@d3328dd18858ff5bc159b23a513a09c0a0ae3a3e
- nccl@00319ee6cd510722497157c038a124c73558224c
- nccl@c9b80063c614017ba9292022f87584138c3ea9c9
- nccl@2e2e4a8e8cbb29cfa3a65396cc5120c28c5a405f
- nccl@4f68cd654043057c95152c837f325296958775e4
- nccl@9141021efc156d45351dd3f4d41127a1bd4dc01e
- nccl@6702ed814077eb2d726324da959d53aa33d94c60
- nccl@50116359f3e066785581474688f4fa47dfcf1c96
- nccl@540570d80d05387fcc3b9c7ce611f73253278a17
- nccl@8679a1a97decc797dc63f5886596e2c7a533cc60
- nccl@7cdf5f2a56b98c1e83f2e639d77562f5dbdcd45a
- nccl@addf167a606e935c1ed90adb16e003fe95249d5d
</details>

<details>
<summary><h2>Testing</h2></summary>

### Objectives and Timeline

#### Code Coverage Goal Defined

None.

#### KPI Coverage Goals Defined (performance, stress, stability, throughput, latency)

No change.

#### Requirement Coverage Goal Defined

Add testing for bootstrap using

    NCCL_COMM_ID

#### Quality Thresholds Defined

No error.

#### Test Timeline

#### SW Verification and Test Plan

### Test plan

#### Requirements Tests

A new test has been added to test/examples : multiproc_env_id

Sample run (two nodes, 16 ranks):

    NCCL_COMM_ID=eris-ub14-t039:16010 salloc -p gpu-verbs -n16 -N2 --exclusive mpirun -np 16 -npernode 8 --oversubscribe ./build/test/examples/multiproc_env_id

#### Interface Tests

A new test has been added to

    test/apitest/ncclGetUniqueId_test.cu

to test

    ncclGetUniqueID

with

    NCCL_COMM_ID

set.

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

Author : Ke Wen
</details>

