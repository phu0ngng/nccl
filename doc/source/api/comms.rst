**********************************************
Communicator Creation and Management Functions
**********************************************

The following functions are public APIs exposed by NCCL to create and manage the collective communication operations.

ncclGetLastError
----------------

.. c:function:: const char* ncclGetLastError(ncclComm_t comm)

Returns a human-readable string of the last error that occurred in NCCL.
Note: The error is not cleared by calling this function.
Please note that the log from ncclGetLastError could be unrelated to the current call
and can be a result of a previously launched asynchronous operations, if any.

ncclGetErrorString
------------------

.. c:function:: const char* ncclGetErrorString(ncclResult_t result)

Returns a string for each error code.

ncclGetVersion
--------------

.. c:function:: ncclResult_t ncclGetVersion(int* version)

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

Creates a clique of communicators (single process version) in a blocking way.
This is a convenience function to create a single-process communicator clique.
Returns an array of *ndev* newly initialized communicators in *comms*.
*comms* should be pre-allocated with size at least ndev*sizeof(:c:type:`ncclComm_t`).
*devlist* defines the CUDA devices associated with each rank. If *devlist* is NULL,
the first *ndev* CUDA devices are used, in order.

ncclCommInitRankConfig
----------------------

.. c:function:: ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config)

This function works the same way as *ncclCommInitRank* but accepts a configuration argument of extra attributes for
the communicator. If config is passed as NULL, the communicator will have the default behavior, as if ncclCommInitRank
was called.

See the :ref:`init-rank-config` section for details on configuration options.

ncclCommSplit
-------------

.. c:function:: ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t* config)

The *ncclCommSplit* function creates a set of new communicators from an existing one. Ranks which are passed
the same *color* value will be part of the same group, and a color must be a non-negative value. If it is 
passed as *NCCL_SPLIT_NOCOLOR*, it means that the rank will not be part of any group, therefore returning NULL 
as newcomm.
The value of key will determine the rank order, and the smaller key means the smaller rank in new communicator.
If keys are equal between ranks, then the rank in the original communicator will be used to order ranks.
If the new communicator needs to have a special configuration, it can be passed as *config*, otherwise setting
config to NULL will make the new communicator inherit the original communicator's configuration.
When split, there should not be any outstanding NCCL operations on the *comm*. Otherwise, it might cause 
deadlock.


ncclCommFinalize
----------------

.. c:function:: ncclResult_t ncclCommFinalize(ncclComm_t comm)

Finalize a communicator object *comm*. When the communicator is marked as nonblocking, *ncclCommFinalize* is a 
nonblocking function. Successful return from it will set communicator state as *ncclInProgress* and indicates 
the communicator is under finalization where all uncompleted operations and the network-related resources are 
being flushed and freed. 
Once all NCCL operations are complete, the communicator will transition to the *ncclSuccess* state. Users 
can query that state with *ncclCommGetAsyncError*.

ncclCommDestroy
---------------

.. c:function:: ncclResult_t ncclCommDestroy(ncclComm_t comm)

Destroy a communicator object *comm*.
*ncclCommDestroy* only frees the local resources that are allocated to the communicator object *comm* if *ncclCommFinalize* 
was previously called on the communicator; otherwise, *ncclCommDestroy* will call ncclCommFinalize internally. 
If *ncclCommFinalize* is called by users, users should guarantee that the state of the communicator become *ncclSuccess* before 
calling *ncclCommDestroy*. 
In all cases, the communicators should no longer be accessed after ncclCommDestroy returns. It is recommended that 
user call *ncclCommFinalize* and then *ncclCommDestroy*.
This function is an intra-node collective call, which all ranks on the same node should call to avoid hang.

ncclCommAbort
-------------

.. c:function:: ncclResult_t ncclCommAbort(ncclComm_t comm)

Frees resources that are allocated to a communicator object *comm*. Will abort any uncompleted
operations before destroying the communicator. This function is an intra-node collective call,
which all ranks on the same node should call to avoid hang.

ncclCommGetAsyncError
---------------------

.. c:function:: ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError)

Queries the progress and potential errors of asynchronous NCCL operations.
Operations which do not require a stream argument (e.g. ncclCommFinalize) can be considered complete as soon
as the function returns *ncclSuccess*; operations with a stream argument (e.g. ncclAllReduce) will return
*ncclSuccess* as soon as the operation is posted on the stream but may also report errors through
ncclCommGetAsyncError() until they are completed. If return code of any NCCL functions is *ncclInProgress*,
it means the operation is in the process of being enqueued in the background, and users must query the states
of the communicators until the all states become *ncclSuccess* before calling next NCCL function. Before the
states change into *ncclSuccess*, users are not allowed to issue CUDA kernel to the streams being used by NCCL.
If there has been an error on the communicator, user should destroy the communicator with :c:func:`ncclCommAbort`.
If an error occurs on the communicator, nothing can be assumed about the completion or correctness of operations
enqueued on that communicator.

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

ncclCommRegister
----------------

.. c:function:: ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle)

Register buffer with *size* under communicator *comm* for zero-copy communication, and *handle* is
returned for future deregistration. See *buff* and *size* requirements (:ref:`user_buffer_reg`).

ncclCommDeregister
------------------

.. c:function:: ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle)

Deregister buffer represented by *handle* under communicator *comm*.

ncclMemAlloc
------------

.. c:function:: ncclResult_t ncclMemAlloc(void **ptr, size_t size)

Allocate a GPU buffer with *size*. Allocated buffer head address will be returned by *ptr*,
and the actual allocated size can be larger than requested because of the buffer granularity 
requirements from all types of NCCL optimizations.

ncclMemFree
-----------

.. c:function:: ncclResult_t ncclMemFree(void *ptr)

Free memory allocated by *ncclMemAlloc()*.
