# Fault Tolerance NCCL
<details>
<summary><h2>Requirements</h2></summary>
 
### Project Context and Purpose

NCCL provides fast inter-GPU communication for parallel applications.

### User Experience

Users may want DL frameworks to be able to continue running when for
example one node from a cluster fails, or a sub-part of that node, like
a GPU, NIC, network link, etc .... This is a common issue as jobs scale
to large numbers of GPUs, especially given NCCL relies on almost every
component on the node to operate: CPUs, GPUs, PCI switches, and NICs.

A user launching a DL training on 1024 GPUs may want the training to
continue running on 1016 GPUs if one node fails albeit slightly slower.

New or spare nodes could also be added to the job to allow the DL
framework to recreate a NCCL communicator with additional workers to
replace or extend the current ones.

### Assumptions, constraints and dependencies

None.

### Use Cases

DL training at scale, using DL frameworks which have fault tolerance
capabilities, including parallel process monitoring, network monitoring
and inter-process error propagation.

### Functional Requirements

Applications need to be able to stop any NCCL function at any time, then
clean up any created object, and recreate a working communicator on a
functional set of GPUs/nodes.

### System Requirements

None.

### Interface Requirements

New API are needed.

### KPI Requirements

None.

### Platform Requirements

N/A

### Security Requirements

NCCL is a user library and benefits from the user-mode security.

### Legal and Standards Requirements

N/A

### Telemetry Requirements

N/A

### Backward Compatibility Requirements

Changes need to be backward compatible.

### Virtualization Requirements

N/A

### Signoff list

Author : Kaiming Ouyang
</details>
 
<details>
<summary><h2>Design</h2></summary>
 
### Proposed Design

#### User perspective design

Fault tolerance functionality is added into NCCL. When any fault, such
as network failure, node failure, process failure, soft error and so on,
happen and users decide to restart NCCL, users can call ncclCommAbort to
abort operations that are running even if they are initialization or
finalization and then restart NCCL without aborting the whole
application.

3 APIs are added to support the fault tolerance functionality; they are
ncclCommInitRankConfig(), ncclCommFinalize() and structure ncclConfig_t;
in addition API ncclCommGetAsyncError() is extended to query states of
communicators. The state is indicated by ncclResult_t, ncclInProgress
code is added to describe a communicator with ongoing NCCL operations. A
configuration can be passed to initialize a communicator using
ncclCommInitRankConfig(); it must be initialized with macro
"NCCL_CONFIG_INITIALIZER" before usage. For now, we only support
"blocking" attribute. ncclCommInitRankConfig(ncclComm_t\* comm, int
nranks, ncclUniqueId commId, int rank, ncclConfig_t\* config) is not
different from ncclCommInitRank() but supports initializing
communicators with a user-created attribute. ncclCommFinalize(ncclComm_t
comm) allows users to finalize a communicator where synchronization
among ranks can happen; when communicator is nonblocking, it launches a
thread to flush the issued nccl operations and free network-related
resources and main thread will return immediately. After successful
return, the communicator is in ncclInProgress state. In the following,
we only consider the cases where communicators are marked as
nonblocking.

When ncclCommInitRankConfig() return successfully (i.e. return with
ncclSuccess), it only indicates NCCL initialization starts successfully
but does not mean initialization is complete, and the communicator state
will be ncclInProgress. Users can constantly query state by calling
ncclCommGetAsyncError() until communication state turns into
ncclSuccess. When error happens, the communicator state becomes one of
corresponding NCCL error codes, and error code can be retrieved through
ncclCommGetAsyncError(). If users do not query state but call NCCL
operations (e.g. ncclAllReduce(), ncclCommFinalize() and
ncclCommDestroy()) right away, the NCCL operations can be blocked until
the communicator is fully initialized (i.e., until state ==
ncclSuccess). The NCCL operations except ncclCommAbort() can return the
error code of initialization if initialization fails and communicator
state will turn into the error accordingly; otherwise, NCCL operations
return their own error code.

Considering launching NCCL operations (e.g., ncclBroadcast()), after
returning from NCCL operations, the state of the communicator becomes
ncclInProgress, users need to wait until the state becomes ncclSuccess
to guarantee the operation is actually launched into CUDA stream. If
ncclGroupStart() and ncclGroupEnd() warp a bunch of NCCL operations,
users should check the states of all communicators after ncclGroupEnd()
until they become ncclSuccess. In addition, launching NCCL operations in
a group with both blocking and nonblocking communicators is forbidden,
and doing this can lead to the undefined behavior or deadlock. If one
thread manages multiple nonblocking communicators, it is still required
to launch NCCL operations with these communicators in a group operation.

Similarly, ncclCommFinalize() becomes a nonblocking funtion unless state
== ncclInProgress when entering. Successful call of nonblocking
ncclCommFinalize() only indicates the successful start of communicator
finalization, and communicator state changes into ncclInProgress. users
can query state using ncclCommGetAsyncError() until it becomes
ncclSuccess and then call ncclCommDestroy(). ncclCommDestroy() only free
the rest of local resources after ncclCommFinalize() and is nonblocking
call. No NCCL operation is allowed after ncclCommFinalize(), and the
communicator is not available after ncclCommDestroy() so that no error
code can be retrieved. Directly calling ncclCommDestroy() without
calling ncclCommFinalize() before will results in the internal call of
ncclCommFinalize().

#### Developer perspective design

To support nonblocking initialization and finalization functions, the
first step is to change underlying socket functions (e.g. accept and
connect) into nonblocking ones so that it allows us to return when users
request abort. One thing to notice that the high-level socket functions
in socket.cc file are still blocking, so the changes will not affect
upper layer. Then we add abortFlag to socket struct and check this flag
while checking the socket status in the loop; whenever the flag is set
to 1, we can return from the socket function and proceed to abort.

On the other hand, initialization stage also involves network plugins
initialization where socket-like operations such as ncclNetAccept(),
ncclNetConnect(), ncclNetIsend(), ncclNetIrecv() and so on are called to
build up connection and transmit data. Since network plugin layer does
not involve the abort of NCCL layer, we need to change these functions
nonblocking and determine whether to abort in NCCL layer. Therefore, we
add asyncFlag to socket struct and if asyncFlag is set to 1, it
indicates all high-level socket functions in socket.cc flie should
return immediately no matter whether the socket is connected, accpeted
or data is completely sent or received. In the corresponding functions
(especially ncclNetAccept() and ncclNetConnect()) in network plugin, the
state of the connection is stored, return NULL communicator to the upper
layer and expect users to call the function again to complete the rest
of operations. The high-level implementation of nonblocking
ncclNetConnect() function and corresponding usage in NCCL layer are
shown as follows:

      /* ib network plugin. */
      ncclResult_t ncclIbConnect(int dev, void* opaqueHandle, void** sendComm) {
        static __thread struct ncclNetIbStage stage;
        restoreStage(&stage);
        if (stage == ibstart) goto start; /* user first call. */
        if (stage == ibconnect) goto connect; /* connect fails last time, call connect again. */
        if (stage == ibconnect_check) goto connect_check; /* connection is not complete last time, check connection again. */

      start:
        dataInit(...);
      connect:
        setupStage(&stage, connect);
        if (ncclSocketConnect(...) == fail) {
          *sendComm = NULL; /* expect another call. */
          return ncclSuccess;
        }
      connect_check:
        setupStage(&stage, connect_check);
        if (ncclGetSocketState(...) == ncclSocketConnecting) {
          *sendComm = NULL; /* connection is under initialization, expect another call. */
          return ncclSuccess;
        }
        /* connection succeeds, start to process ib initialization part. */
      }

      /* NCCL layer (i.e. users) which calls network plugin functions. */
      do {
        if (abortFlag == 1) return error;
        /* when sComm return NULL, it means connection is not done. We need 
         * to call ncclNetConnect again. */
        ret = ncclNetConnect(dev, &handle, &sComm);
      } while(sComm == NULL);

After calling ncclCommInitRankConfig(..., config) with a nonblocking
setting, the communicator state is set to ncclInProgress and a
background init thread is spawned for the initialization; when
initialization is done, the thread sets state as ncclSuccess and exits.
If an error happens, main thread sets state as ncclCommError and set
fatalError which can be queried by ncclCommGetAsyncError().

If users call ncclCommFinalize(), the communicator state is set to
ncclInProgress and the local resources should be freed when calling
ncclCommDestroy(); on the contrary, if users do not call
ncclCommFinalize() but just call ncclCommDestroy(), we will internally
call ncclCommFinalize(). Main thread will not be blocked when calling
ncclCommDestroy() until the call to the last communicator a process
manages for intraprocess resource reclaimation; however, calling to
ncclCommAbort() will never be blocked since everything is guaranteed to
abort.

### Interface Architecture

If users want to prevent failure from initialization stage,
ncclCommInitRank() or ncclCommInitAll() is called as usually and then
users can constantly check the state of the communicator by calling
ncclCommGetAsyncError() and set a timer at the same time; if by any
chance the state turns into error or time is out, users can call
ncclCommAbort() to abort communicator and restart NCCL. The example code
is shown as follows (note: users can check "ret" for return code):

      ncclGetUniqueId(&id);
      MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

      ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
      config.blocking = 0;
      NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, myRank, &config));
      do {
        ret = ncclCommGetAsyncError(comm, &state);
      } while(state == ncclInProgress && !timeout());

      if (initTimeout() == true || state != ncclSuccess) {
        abortFlag = true;
      }
      broadcastError(&abortFlag, &globalFlag...);

      if (globalFlag) {
        /* time is out or initialization fails, just abort and restart. */
        ncclCommAbort(comm);
        /* restart NCCL. */
        restartNCCL(&comm);
      }
      /* application workload. */

Similarly, if users want the same functionality for finalization stage,
it can be done as follows:

      ret = ncclCommFinalize(comm);
      do {
        ret = ncclCommGetAsyncError(comm, &state);
      } while(state == ncclInProgress && !timeout());
      
      if (initTimeout() == true || state == ncclCommError) {
        abortFlag = true;
      }
      broadcastError(&abortFlag, &globalFlag...);

      if (globalFlag) {
        /* time is out or initialization fails, just abort and restart. */
        ncclCommAbort(comm);
      } else {
        /* free local resources; since state must be ncclSuccess, the successfuly local free 
         * is guaranteed. */
        ret = ncclCommDestroy(comm);
      }

In addition, users are able to call ncclCommAbort() at any place if
users believe a fault has happened and wants to restart NCCL.

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

Author : Kaiming Ouyang
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

Add ncclCommGetState(), ncclCommStateCallback(),
ncclCommInitRankConfig(), and ncclCommFinalize() API tests and fault
tolerance functionality tests.

#### Quality Thresholds Defined

No error.

#### Test Timeline

#### SW Verification and Test Plan

### Test plan

#### Requirements Tests

One test is added to unit test:

    ft_test

Four tests are added to api test:

    ncclCommGetAsyncError_test

,

    ncclCommInitRankConfig_test 

,

    ncclConfig_test 

,

    ncclCommFinalize_test

,

    ncclCommDestroy_test

#### Interface Tests

ncclCommGetAsyncError_test tests include parameters checking test, basic
API usage test such as querying state after calling
ncclCommInitRankConfig() or ncclCommFinalize(). ncclCommFinalize_test
tests include parameters checking test, user-called finalize and wait
test, user-called finalize test, no-user-called finalize test, one gpu
per thread finalize test, grouped-finalize test, double finalize test.

#### Fault-injection Tests

In ft_test, there are four stages to run and it will sleep 500ms, 1s, 4s
and 8s respectively for each stage after calling
ncclCommInitRankConfig(). First three stages we assume fatal fault
happens so that after return from sleep, we just call ncclCommAbort()
and restart NCCL. Last stage assumes no error happens and will wait
until all communicators successfully initialize; then, it issues
ncclAllReduce(), checks results, calls ncclCommFinalize() and
ncclCommDestroy(). If all operations succeed, the ft_test succeeds.

#### Resource Usage Tests

None.

#### Design Coverage Testing

None.

#### Boundary Tests

None.

#### Certification Tests

None.

#### Stress Tests

In ncclCommGetAsyncError_test, 10 iterations are run for each test.

#### Stability Tests

None.

#### Perf and Power KPI Tests

None.

#### Usability & OOBE Tests

None.

#### Manufacturing Diagnostic(Factory) Tests

None.

### Signoff list

Author : Kaiming Ouyang
</details>
 
