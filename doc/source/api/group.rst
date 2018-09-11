***********
Group Calls
***********

Group primitives define the behavior of the current thread to avoid blocking. They can therefore be used from multiple threads independently.

Related links: :ref:`group-calls`.

ncclGroupStart
--------------

.. c:function:: ncclResult_t ncclGroupStart()
 
 Start a group call. 
 
 All subsequent calls to NCCL may not block due to inter-CPU synchronization.

ncclGroupEnd
------------

.. c:function:: ncclResult_t ncclGroupEnd()
 
 End a group call. 
 
 Returns when all operations since ncclGroupStart have been processed. This means communication primitives
 have been enqueued to the provided streams, but are not necessary complete. 
 
 When used with the ncclCommInitRank call, the ncclGroupEnd call waits for all communicators to be initialized.

