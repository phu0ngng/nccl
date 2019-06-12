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

*******************************************
Error handling and communicator destruction
*******************************************

Normal termination
------------------

Resources associated to a communicator can be destroyed with ncclCommDestroy. This operation will wait for operations
to complete but will not synchronize with other ranks. There is therefore no need to use group semantics with
ncclCommDestroy.

Related link: :c:func:`ncclCommDestroy`

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
