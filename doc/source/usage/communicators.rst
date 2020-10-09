.. _communicator-label:

***********************
Creating a Communicator
***********************

When creating a communicator, a unique rank between 0 and n-1 has to be assigned to each of the n CUDA devices which are
part of the communicator. Using the same CUDA device multiple times as different ranks of the same NCCL communicator is
not supported and may lead to hangs.

Given a static mapping of ranks to CUDA devices, the ncclCommInitRank and ncclCommInitAll functions will create
communicator objects, each communicator object being associated to a fixed rank. Those objects will then be used to
launch communication operations.

Note: Before calling ncclCommInitRank, you need to first create a unique object which will be used by all processes and
threads to synchronize and understand they are part of the same communicator. This is done by calling the
ncclGetUniqueId function.

The ncclGetUniqueId function returns an ID which has to be broadcast to all participating threads and processes using
any CPU communication system, for example, passing the ID pointer to multiple threads, or broadcasting it to other
processes using MPI or another parallel environment using, for example, sockets.

You can also call the ncclCommInitAll operation to create n communicator objects at once within a single process. As it
is limited to a single process, this function does not permit inter-node communication. ncclCommInitAll is equivalent to
calling a combination of ncclGetUniqueId and ncclCommInitRank.

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

Destroying a communicator
-------------------------

Resources associated to a communicator can be destroyed with ncclCommDestroy. This operation will wait for operations
to complete but will not synchronize with other ranks. There is therefore no need to use group semantics with
ncclCommDestroy.

Related link: :c:func:`ncclCommDestroy`

*************************************
Error handling and communicator abort
*************************************

All NCCL calls return a NCCL error code which is sumarized in the table below. If a NCCL call returns an error code
different from ncclSuccess and ncclInternalError, NCCL will print a human-readable message explaining what happened
if NCCL_DEBUG is set to WARN. If NCCL_DEBUG is set to INFO, it will also print the call stack which lead to the error.
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


(1) ncclUnhandledCudaError and ncclSystemError indicate that a call NCCL made to an external component failed,
which caused the NCCL operation to fail. The error message should explain which component the user should look
at and try to fix, potentially with the help of the administrators of the system.

(2) ncclInternalError denotes a NCCL bug. It might not report a message with NCCL_DEBUG=WARN since it requires a
fix in the NCCL source code. NCCL_DEBUG=INFO will print the back trace which lead to the error.

(3) ncclInvalidArgument indicates an argument value is incorrect, like a NULL pointer, or an out-of-bounds value.
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
      // Caller may abort or try to re-create a new communicator.
      return 2;
    }

    // We might want to let other threads (including NCCL threads) use the CPU.
    pthread_yield();
   }
 }

Related links:

 * :c:func:`ncclCommGetAsyncError`
 * :c:func:`ncclCommAbort`
