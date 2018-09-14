**********************************
Collective Communication Functions
**********************************


The following NCCL APIs provide some commonly used collective operations.

ncclAllReduce
-------------

.. c:function:: ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
 
 Reduce data arrays of length ``count`` in ``sendbuff`` using ``op`` operation and leaves identical copies of the result on each ``recvbuff``.
 
 In-place operation will happen if ``sendbuff == recvbuff``.

Related links: :ref:`allreduce`.


ncclBroadcast
-------------

.. c:function:: ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
 
 Copies ``count`` elements from ``sendbuff`` on the ``root` rank to all ranks' ``recvbuff``.
 ``sendbuff`` is only used on rank ``root`` and ignored for other ranks.
 
 In-place operation will happen if ``sendbuff == recvbuff``.
 

.. c:function:: ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
 
 Legacy in-place version of ``ncclBroadcast`` in a similar fashion to MPI_Bcast. A call to ::
  
  ncclBcast(buff, count, datatype, root, comm, stream)
 
 is equivalent to ::
  
  ncclBroadcast(buff, buff, count, datatype, root, comm, stream)

Related links: :ref:`broadcast`

ncclReduce
----------

.. c:function:: ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
 
 Reduce data arrays of length ``count`` in ``sendbuff`` into ``recvbuff`` on the ``root`` rank using the ``op`` operation.
 ``recvbuff`` is only used on rank ``root`` and ignored for other ranks.
 
 In-place operation will happen if ``sendbuff == recvbuff``.

Related links: :ref:`reduce`.

ncclAllGather
-------------

.. c:function:: ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)
 
 Gather ``sendcount`` values from all GPUs into ``recvbuff``, receiving data from rank ``i`` at offset ``i*sendcount``. 
 
 Note : This assumes the receive count is equal to ``nranks*sendcount``, which means that ``recvbuff`` should have a size of at least ``nranks*sendcount`` elements.
 
 In-place operation will happen if ``sendbuff == recvbuff + rank * sendcount``.

Related links: :ref:`allgather`.

ncclReduceScatter
-----------------

.. c:function:: ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
 
 Reduce data in ``sendbuff`` from all GPUs using the ``op`` operation and leave the reduced result scattered over the devices so that the ``recvbuff`` on
 rank ``i`` will contain the i-th block of the result.
 
 Note:  This assumes the send count is equal to ``nranks*recvcount``, which means that ``sendbuff`` should have a size of at least ``nranks*recvcount`` elements.

Related links: :ref:`reducescatter`.
