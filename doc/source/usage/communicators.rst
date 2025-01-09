.. _communicator-label:

***********************
Creating a Communicator
***********************

When creating a communicator, a unique rank between 0 and n-1 has to be assigned to each of the n CUDA devices which
are part of the communicator. Using the same CUDA device multiple times as different ranks of the same NCCL
communicator is not supported and may lead to hangs.

Given a static mapping of ranks to CUDA devices, the :c:func:`ncclCommInitRank`, :c:func:`ncclCommInitRankConfig` and
:c:func:`ncclCommInitAll` functions will create communicator objects, each communicator object being associated to a
fixed rank and CUDA device. Those objects will then be used to launch communication operations.

Before calling :c:func:`ncclCommInitRank`, you need to first create a unique object which will be used by all processes
and threads to synchronize and understand they are part of the same communicator. This is done by calling the
:c:func:`ncclGetUniqueId` function.

The :c:func:`ncclGetUniqueId` function returns an ID which has to be broadcast to all participating threads and
processes using any CPU communication system, for example, passing the ID pointer to multiple threads, or broadcasting
it to other processes using MPI or another parallel environment using, for example, sockets.

You can also call the ncclCommInitAll operation to create n communicator objects at once within a single process. As it
is limited to a single process, this function does not permit inter-node communication. ncclCommInitAll is equivalent
to calling a combination of ncclGetUniqueId and ncclCommInitRank.

The following sample code is a simplified implementation of ncclCommInitAll.

.. code:: C

 ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
   ncclUniqueId Id;
   ncclGetUniqueId(&Id);
   ncclGroupStart();
   for (int i=0; i<ndev; i++) {
     cudaSetDevice(devlist[i]);
     ncclCommInitRank(comm+i, ndev, Id, i);
   }
   ncclGroupEnd();
 }

Related links:

 * :c:func:`ncclCommInitAll`
 * :c:func:`ncclGetUniqueId`
 * :c:func:`ncclCommInitRank`

.. _init-rank-config:

Creating a communicator with options
-------------------------------------

The :c:func:`ncclCommInitRankConfig` function allows to create a NCCL communicator with specific options.

The config parameters NCCL supports are listed here :ref:`ncclconfig`.

For example, "blocking" can be set to 0 to ask NCCL to never block in any NCCL call, and at the same time
other config parameters can be set as well to more precisely define communicator behavior. A simple example
code is shown below:

.. code:: C

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;
  config.minCTAs = 4;
  config.maxCTAs = 16;
  config.cgaClusterSize = 2;
  config.netName = "Socket";
  CHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));
  do {
    CHECK(ncclCommGetAsyncError(comm, &state));
    // Handle outside events, timeouts, progress, ...
  } while(state == ncclInProgress);

Related link: :c:func:`ncclCommGetAsyncError`

Creating a communicator using multiple ncclUniqueIds
----------------------------------------------------

The :c:func:`ncclCommInitRankScalable` function enables the creation of a NCCL communicator using many ncclUniqueIds.
All NCCL ranks have to provide the same array of ncclUniqueIds (same ncclUniqueIds, and in with the same order).
For the best performance, we recommend distributing the ncclUniqueIds as evenly as possible amongst the NCCL ranks.

Internally, NCCL ranks will mostly communicate with a single ncclUniqueId.
Therefore, to obtain the best results, we recommend to evenly distribute ncclUniqueIds accross the ranks.

The following function can be used to decide if a NCCL rank should create a ncclUniqueIds:

.. code:: C

 bool rankHasRoot(const int rank, const int nRanks, const int nIds) {
   const int rmr = nRanks % nIds;
   const int rpr = nRanks / nIds;
   const int rlim = rmr * (rpr+1);
   if (rank < rlim) {
     return !(rank % (rpr + 1));
   } else {
     return !((rank - rlim) % rpr);
   }
 }

For example, if 3 ncclUniqueIds are to be distributed accross 7 NCCL ranks, the first ncclUniqueId will be associated to
ranks 0-2, while the others will be associated to ranks 3-4, and 5-6.
This function will therefore return true on rank 0, 3, and 5, and false otherwise.

Note: only the first ncclUniqueId will be used to create the communicator hash id, which is used to identify the
communicator in the log file and in the replay tool.

Creating more communicators
---------------------------

The ncclCommSplit function can be used to create communicators based on an existing one. This allows to split an existing
communicator into multiple sub-partitions, duplicate an existing communicator, or even create a single communicator with
fewer ranks.

The ncclCommSplit function needs to be called by all ranks in the original communicator. If some ranks will not be part
of any sub-group, they still need to call ncclCommSplit with color being NCCL_SPLIT_NOCOLOR.

Newly created communicators will inherit the parent communicator configuration (e.g. non-blocking).
If the parent communicator operates in non-blocking mode, a ncclCommSplit operation may be stopped by calling ncclCommAbort
on the parent communicator, then on any new communicator returned. This is because a hang could happen during
operations on any of the two communicators.

The following code duplicates an existing communicator:

.. code:: C

 int rank;
 ncclCommUserRank(comm, &rank);
 ncclCommSplit(comm, 0, rank, &newcomm, NULL);

This splits a communicator in two halves:

.. code:: C

 int rank, nranks;
 ncclCommUserRank(comm, &rank);
 ncclCommCount(comm, &nranks);
 ncclCommSplit(comm, rank/(nranks/2), rank%(nranks/2), &newcomm, NULL);

This creates a communicator with only the first 2 ranks:

.. code:: C

 int rank;
 ncclCommUserRank(comm, &rank);
 ncclCommSplit(comm, rank<2 ? 0 : NCCL_SPLIT_NOCOLOR, rank, &newcomm, NULL);


Related links:

 * :c:func:`ncclCommSplit`

Using multiple NCCL communicators concurrently
----------------------------------------------

Using multiple NCCL communicators requires careful synchronization, or can lead to deadlocks.

NCCL kernels are blocking (waiting for data to arrive), and any CUDA operation can cause a device synchronization,
meaning it will wait for all NCCL kernels to complete. This can quickly lead to deadlocks since NCCL operations perform
CUDA calls themselves.

Operations on different communicators should therefore be used at different epochs with a locking mechanism, and
applications should ensure operations are submitted in the same order across ranks.

Launching multiple communication operations (on different streams) might work provided they can fit within the GPU, but
could break at any time if NCCL were to use more CUDA blocks per operation, or if some calls used inside NCCL
collectives were to perform a device synchronization (e.g. allocate some CUDA memory dynamically).

Finalizing a communicator
-------------------------

ncclCommFinalize will transition a communicator from the *ncclSuccess* state to the *ncclInProgress* state, start
completing all operations in the background and synchronize with other ranks which may be using resources for their
communications with other ranks.
All uncompleted operations and network-related resources associated to a communicator will be flushed and freed with
ncclCommFinalize.
Once all NCCL operations are complete, the communicator will transition to the *ncclSuccess* state. Users can
query that state with ncclCommGetAsyncError.
If a communicator is marked as nonblocking, this operation is nonblocking; otherwise, it is blocking.

Related link: :c:func:`ncclCommFinalize`

Destroying a communicator
-------------------------

Once a communicator has been finalized, the next step is to free all resources, including the communicator itself.
Local resources associated to a communicator can be destroyed with ncclCommDestroy. If the state of a communicator
is *ncclSuccess* when calling ncclCommDestroy, the call is guaranteed to be nonblocking; otherwise
ncclCommDestroy might block.
In all cases, ncclCommDestroy call will free the resources of the communicator and return, and
the communicator should no longer be accessed after ncclCommDestroy returns.

Related link: :c:func:`ncclCommDestroy`

*************************************
Error handling and communicator abort
*************************************

All NCCL calls return a NCCL error code which is sumarized in the table below. If a NCCL call returns an error code
different from ncclSuccess and ncclInternalError, and if NCCL_DEBUG is set to WARN, NCCL will print a human-readable
message explaining what happened.
If NCCL_DEBUG is set to INFO, NCCL will also print the call stack which led to the error.
This message is intended to help the user fix the problem.

The table below summarizes how different errors should be understood and handled. Each case is explained in details
in the following sections.

.. list-table:: NCCL Errors
   :widths: 20 50 10 10 10
   :header-rows: 1

   * - Error
     - Description
     - Resolution
     - Error handling
     - Group behavior
   * - ncclSuccess
     - No error
     - None
     - None
     - None
   * - ncclUnhandledCudaError
     - Error during a CUDA call (1)
     - CUDA configuration / usage (1)
     - Communicator abort (5)
     - Global (6)
   * - ncclSystemError
     - Error during a system call (1)
     - System configuration / usage (1)
     - Communicator abort (5)
     - Global (6)
   * - ncclInternalError
     - Error inside NCCL (2)
     - Fix in NCCL (2)
     - Communicator abort (5)
     - Global (6)
   * - ncclInvalidArgument
     - An argument to a NCCL call is invalid (3)
     - Fix in the application (3)
     - None (3)
     - Individual (3)
   * - ncclInvalidUsage
     - The usage of NCCL calls is invalid (4)
     - Fix in the application (4)
     - Communicator abort (5)
     - Global (6)
   * - ncclInProgress
     - The NCCL call is still in progress
     - Poll for completion using ncclCommGetAsyncError
     - None
     - None


(1) ncclUnhandledCudaError and ncclSystemError indicate that a call NCCL made to an external component failed,
which caused the NCCL operation to fail. The error message should explain which component the user should look
at and try to fix, potentially with the help of the administrators of the system.

(2) ncclInternalError denotes a NCCL bug. It might not report a message with NCCL_DEBUG=WARN since it requires a
fix in the NCCL source code. NCCL_DEBUG=INFO will print the back trace which led to the error.

(3) ncclInvalidArgument indicates an argument value is incorrect, like a NULL pointer or an out-of-bounds value.
When this error is returned, the NCCL call had no effect. The group state remains unchanged, the communicator is
still functioning normally. The application can call ncclCommAbort or continue as if the call did not happen.
This error will be returned immediately for a call happening within a group and applies to that specific NCCL
call. It will not be returned by ncclGroupEnd since ncclGroupEnd takes no argument.

(4) ncclInvalidUsage is returned when a dynamic condition causes a failure, which denotes an incorrect usage of
the NCCL API.

(5) These errors are fatal for the communicator. To recover, the application needs to call ncclCommAbort on the
communicator and re-create it.

(6) Dynamic errors for operations within a group are always reported by ncclGroupEnd and apply to all operations
within the group, which may or may not have completed. The application must call ncclCommAbort on all communicators
within the group.

Asynchronous errors and error handling
--------------------------------------

Some communication errors, and in particular network errors, are reported through the ncclCommGetAsyncError function.
Operations experiencing an asynchronous error will usually not progress and never complete. When an asynchronous error
happens, the operation should be aborted and the communicator destroyed using ncclCommAbort.
When waiting for NCCL operations to complete, applications should call ncclCommGetAsyncError and destroy the
communicator when an error happens.

The following code shows how to wait on NCCL operations and poll for asynchronous errors, instead of using
cudaStreamSynchronize.

.. code:: C

 int ncclStreamSynchronize(cudaStream_t stream, ncclComm_t comm) {
   cudaError_t cudaErr;
   ncclResult_t ncclErr, ncclAsyncErr;
   while (1) {
    cudaErr = cudaStreamQuery(stream);
    if (cudaErr == cudaSuccess)
      return 0;

    if (cudaErr != cudaErrorNotReady) {
      printf("CUDA Error : cudaStreamQuery returned %d\n", cudaErr);
      return 1;
    }

    ncclErr = ncclCommGetAsyncError(comm, &ncclAsyncErr);
    if (ncclErr != ncclSuccess) {
      printf("NCCL Error : ncclCommGetAsyncError returned %d\n", ncclErr);
      return 1;
    }

    if (ncclAsyncErr != ncclSuccess) {
      // An asynchronous error happened. Stop the operation and destroy
      // the communicator
      ncclErr = ncclCommAbort(comm);
      if (ncclErr != ncclSuccess)
        printf("NCCL Error : ncclCommDestroy returned %d\n", ncclErr);
      // Caller may abort or try to create a new communicator.
      return 2;
    }

    // We might want to let other threads (including NCCL threads) use the CPU.
    sched_yield();
   }
 }

Related links:

 * :c:func:`ncclCommGetAsyncError`
 * :c:func:`ncclCommAbort`

.. _ft:

***************
Fault Tolerance
***************

NCCL provides a set of features to allow applications to recover from fatal errors such as a network failure,
a node failure, or a process failure. When such an error happens, the application should be able to call *ncclCommAbort*
on the communicator to free all resources, then create a new communicator to continue.

In order to abort NCCL communicators safely, NCCL requires applications to set communicators as nonblocking and make sure
no thread is calling any NCCL operations while calling *ncclCommAbort*. After nonblocking is set, all NCCL calls
(except *ncclCommDestroy/Abort*) become nonblocking so that *ncclCommAbort* can be called at any point, during initialization,
communication or finalizing the communicator. If NCCL communicators are set blocking, the thread can possibly get stuck inside
NCCL calls due to network errors; in this case, NCCL communicators might hang forever.

To correctly abort, when any rank in a communicator fails (e.g., due to a segmentation fault), all other ranks need to
call *ncclCommAbort* to abort their own NCCL communicator.
Users can implement methods to decide when and whether to abort the communicators and restart the NCCL operation.
Here is an example showing how to initialize and split a communicator in a non-blocking manner, allowing for an abort at any point:

.. code:: C

  bool globalFlag;
  bool abortFlag = false;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  /* set communicator as nonblocking */
  config.blocking = 0;
  CHECK(ncclCommInitRankConfig(&comm, nRanks, id, myRank, &config));
  do {
    CHECK(ncclCommGetAsyncError(comm, &state));
  } while(state == ncclInProgress && checkTimeout() != true);

  if (checkTimeout() == true || state != ncclSuccess) abortFlag = true;

  /* sync abortFlag among all healthy ranks. */
  reportErrorGlobally(abortFlag, &globalFlag);

  if (globalFlag) {
    /* time is out or initialization failed: every rank needs to abort and restart. */
    ncclCommAbort(comm);
    /* restart NCCL; this is a user implemented function, it might include
     * resource cleanup and ncclCommInitRankConfig() to create new communicators. */
    restartNCCL(&comm);
  }

  /* nonblocking communicator split. */
  CHECK(ncclCommSplit(comm, color, key, &childComm, &config));
  do {
    CHECK(ncclCommGetAsyncError(comm, &state));
  } while(state == ncclInProgress && checkTimeout() != true);

  if (checkTimeout() == true || state != ncclSuccess) abortFlag = true;

  /* sync abortFlag among all healthy ranks. */
  reportErrorGlobally(abortFlag, &globalFlag);

  if (globalFlag) {
    ncclCommAbort(comm);
    /* if chilComm is not NCCL_COMM_NULL, user should abort child communicator
     * here as well for resource reclamation. */
    if (childComm != NCCL_COMM_NULL) ncclCommAbort(childComm);
    restartNCCL(&comm);
  }
  /* application workload */

The *checkTimeout* function needs to be provided by users to determine what is the longest time the application should wait for
NCCL initialization; likewise, users can apply other methods to detect errors besides a timeout function. Similar methods can be applied
to NCCL finalization as well.

******************
Quality of Service
******************

Applications which overlap communication may benefit from network Quality of
Service (QoS) features. NCCL allows an application to assign a traffic class (TC) to
each communicator to identify the communication requirements of the
communicator. All network operations on a communicator will use the assigned
TC.

The meaning of TC is specific to the network plugin in use by the communicator
(e.g. IB networks use service level, RoCE networks use type of service). TCs are defined
by the system configuration. Applications must understand the TCs available on
a system and their relative behavior in order to use them effectively.

TC is specified during communicator creation using :ref:`ncclconfig`.

.. code:: C

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.trafficClass = 1;
  CHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));

Infiniband networks support QoS through the use of Service Levels (SL). Each IB SL 
is mapped to Virtual Lane (VL), which defines the relative priority of traffic. SL
behavior is defined within the subnet manager, such as OpenSM. Refer to subnet
manager documentation for more detail. An example configuration is shown below.

.. code:: C

  ...
  qos_max_vls 2
  qos_high_limit 255
  qos_vlarb_high 1:4
  qos_vlarb_low 0:1,1:4
  qos_sl2vl 0,1

  max_op_vls 2
  ....

The example defines one low priority and one high priority VL which 
are mapped to SL 0 and 1, respectively. The high priority SL will be 
given a larger share of network bandwidth at each port. In NCCL, the 
communicator's traffic class corresponds to the SL on IB networks. Using 
this configuration, applications can assign TC 0 to low-priority communicators 
and TC 1 to high-priority ones.

On RoCE networks, the NCCL communicator trafficClass is interpreted as an IP
Type of Service (ToS). Refer to network management tools to understand how to
configure QoS for a given workload.
