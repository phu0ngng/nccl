**************************************
Point To Point Communication Functions
**************************************

NCCL provides two types of point-to-point communication primitives: two-sided operations and one-sided operations.

Two-Sided Point-to-Point Operations
====================================

(Since NCCL 2.7) Two-sided point-to-point communication primitives need to be used when ranks need to send and
receive arbitrary data from each other, which cannot be expressed as a broadcast or allgather, i.e.
when all data sent and received is different. Both sender and receiver must explicitly participate.

ncclSend
--------

.. c:function:: ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream)

 Send data from ``sendbuff`` to rank ``peer``.

 Rank ``peer`` needs to call ncclRecv with the same ``datatype`` and the same ``count`` as this rank.

 This operation is blocking for the GPU. If multiple :c:func:`ncclSend` and :c:func:`ncclRecv` operations
 need to progress concurrently to complete, they must be fused within a :c:func:`ncclGroupStart`/
 :c:func:`ncclGroupEnd` section.

Related links: :ref:`point-to-point`.

ncclRecv
--------

.. c:function:: ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream)

 Receive data from rank ``peer`` into ``recvbuff``.

 Rank ``peer`` needs to call ncclSend with the same ``datatype`` and the same ``count`` as this rank.

 This operation is blocking for the GPU. If multiple :c:func:`ncclSend` and :c:func:`ncclRecv` operations
 need to progress concurrently to complete, they must be fused within a :c:func:`ncclGroupStart`/
 :c:func:`ncclGroupEnd` section.

Related links: :ref:`point-to-point`.

One-Sided Point-to-Point Operations (RMA)
==========================================

One-sided Remote Memory Access (RMA) operations enable ranks to directly access remote memory without
explicit participation from the target process. These operations require the target memory to be
pre-registered within a symmetric memory window using :c:func:`ncclCommWindowRegister`.

ncclPutSignal
-------------

.. c:function:: ncclResult_t ncclPutSignal(const void* localbuff, size_t count, ncclDataType_t datatype, int peer, ncclWindow_t peerWin, size_t peerWinOffset, ncclSignalMode_t signalMode, int ctx, ncclComm_t comm, cudaStream_t stream)

 Write data from ``localbuff`` to rank ``peer``'s registered memory window at offset ``peerWinOffset``.

 The target memory window ``peerWin`` must be registered using :c:func:`ncclCommWindowRegister`.
 The ``signalMode`` controls signaling behavior (see :c:type:`ncclSignalMode_t`).
 The ``ctx`` is the context identifier for the operation. It must be set to 0 for now.

ncclSignal
----------

.. c:function:: ncclResult_t ncclSignal(int peer, ncclSignalMode_t signalMode, int ctx, ncclComm_t comm, cudaStream_t stream)

 Send a signal to rank ``peer`` without transferring data.

 The ``signalMode`` controls signaling behavior (see :c:type:`ncclSignalMode_t`).
 ``NCCL_SIGNAL_NONE`` is not valid for this operation.
 The ``ctx`` is the context identifier for the operation. It must be set to 0 for now.

ncclWaitSignal
--------------

.. c:function:: ncclResult_t ncclWaitSignal(int npeers, int* peers, int* nsignals, ncclSignalMode_t signalMode, int ctx, ncclComm_t comm, cudaStream_t stream)

 Wait for signals from multiple peers.

 Wait for ``nsignals[i]`` number of signals from rank ``peers[i]`` for each peer.
 The ``signalMode`` controls signaling behavior (see :c:type:`ncclSignalMode_t`).
 ``NCCL_SIGNAL_NONE`` is not valid for this operation.
 The ``ctx`` is the context identifier for the operation. It must be set to 0 for now.
