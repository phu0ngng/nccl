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
