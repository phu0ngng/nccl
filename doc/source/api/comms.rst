**********************************************
Communicator Creation and Management Functions
**********************************************

The following functions are public APIs exposed by NCCL to create and manage the collective communication operations.

ncclGetLastError
----------------

.. c:function:: const char* ncclGetLastError(ncclComm_t comm)
Returns a human-readable string of the last error that occurred in NCCL.
Note: The error is not cleared by calling this function.
The *comm* argument is currently unused and can be set to NULL.


ncclGetVersion
--------------

.. c:function:: ncclResult_t  ncclGetVersion(int* version)

The ncclGetVersion function returns the version number of the currently linked NCCL library.
The NCCL version number is returned in *version* and encoded as an integer which includes the
:c:macro:`NCCL_MAJOR`, :c:macro:`NCCL_MINOR` and :c:macro:`NCCL_PATCH` levels.
The version number returned will be the same as the :c:macro:`NCCL_VERSION_CODE` defined in *nccl.h*.
NCCL version numbers can be compared using the supplied macro; :c:macro:`NCCL_VERSION(MAJOR,MINOR,PATCH)`


ncclGetUniqueId
---------------

.. c:function:: ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId)

Generates an Id to be used in ncclCommInitRank. ncclGetUniqueId should be
called once when creating a communicator and the Id should be distributed to all ranks in the
communicator before calling ncclCommInitRank. *uniqueId* should point to a ncclUniqueId object allocated by the user.

ncclCommInitRank
----------------

.. c:function:: ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank)

Creates a new communicator (multi thread/process version).
*rank* must be between 0 and *nranks*-1 and unique within a communicator clique.
Each rank is associated to a CUDA device, which has to be set before calling
ncclCommInitRank.
ncclCommInitRank implicitly synchronizes with other ranks, hence it must be
called by different threads/processes or use ncclGroupStart/ncclGroupEnd.

ncclCommInitAll
---------------

.. c:function:: ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist)

Creates a clique of communicators (single process version).
This is a convenience function to create a single-process communicator clique.
Returns an array of *ndev* newly initialized communicators in *comms*.
*comms* should be pre-allocated with size at least ndev*sizeof(:c:type:`ncclComm_t`).
*devlist* defines the CUDA devices associated with each rank. If *devlist* is NULL,
the first *ndev* CUDA devices are used, in order.

ncclCommDestroy
---------------

.. c:function:: ncclResult_t ncclCommDestroy(ncclComm_t comm)

Frees resources that are allocated to a communicator object *comm*. Waits for any uncompleted
operations before destroying the communicator.

ncclCommAbort
---------------

.. c:function:: ncclResult_t ncclCommAbort(ncclComm_t comm)

Frees resources that are allocated to a communicator object *comm*. Will abort any uncompleted
operations before destroying the communicator.

ncclCommGetAsyncError
---------------------

.. c:function:: ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError)

Queries whether the communicator has encountered any asynchronous errors. If there
has been an error on the communicator, user should destroy the communicator with :c:func:`ncclCommAbort`.
If an error occurs on the communicator, nothing can be assumed about the completion or correctness
of operations enqueued on that communicator.

ncclCommCount
-------------

.. c:function:: ncclResult_t ncclCommCount(const ncclComm_t comm, int* count)

Returns in *count* the number of ranks in the NCCL communicator *comm*.

ncclCommCuDevice
----------------

.. c:function:: ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device)

Returns in *device* the CUDA device associated with the NCCL communicator *comm*. 

ncclCommUserRank
----------------

.. c:function:: ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank)

Returns in *rank* the rank of the NCCL communicator *comm*.
