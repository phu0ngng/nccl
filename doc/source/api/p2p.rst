**************************************
Point To Point Communication Functions
**************************************

(Since NCCL 2.7) Point-to-point communication primitives need to be used when ranks need to send and
receive arbitrary data from each other, which cannot be expressed as a broadcast or allgather, i.e.
when all data sent and received is different.

ncclSend
--------

.. c:function:: ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream)
 
 Send data from ``sendbuff`` to rank ``peer``.
 
 Rank ``peer`` needs to call ncclRecv with the same ``datatype`` and the same ``count`` from this rank.
 
 This operation is blocking for the GPU. If multiple :c:func:`ncclSend` and :c:func:`ncclRecv` operations
 need to progress concurrently to complete, they must be fused within a :c:func:`ncclGroupStart`/
 :c:func:`ncclGroupEnd` section.

Related links: :ref:`point-to-point`.

ncclRecv
--------

.. c:function:: ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream)
 
 Receive data from rank ``peer`` into ``recvbuff``.
 
 Rank ``peer`` needs to call ncclSend with the same ``datatype`` and the same ``count`` to this rank.
 
 This operation is blocking for the GPU. If multiple :c:func:`ncclSend` and :c:func:`ncclRecv` operations
 need to progress concurrently to complete, they must be fused within a :c:func:`ncclGroupStart`/
 :c:func:`ncclGroupEnd` section.

Related links: :ref:`point-to-point`.
