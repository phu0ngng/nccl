.. _user_buffer_reg:

************************
User Buffer Registration
************************

User Buffer Registration is a feature that allows NCCL to directly send/receive/operate data through the user buffer without extra internal copy (zero-copy).
It can accelerate collectives and greatly reduce the resource usage (e.g. #channel usage). NCCL provides two ways to register user buffers; one is *CUDA Graph*
registration, and the other is *Local* registration. NCCL requires that for all NCCL communication function calls (e.g., allreduce, sendrecv, and so on), if any
rank in a communicator passes registered buffers to a NCCL communication function, all other ranks in the same communicator must pass their registered buffers;
otherwise, mixing registered and non-registered buffers can result in undefined behavior.

NVLink Sharp Buffer Registration
--------------------------------

Since 2.19.x, NCCL supports user buffer registration for NVLink Sharp (NVLS); any NCCL collectives (e.g., allreduce) that support NVLS algorithm can utilize this feature.

To enable the *CUDA Graph* based buffer registration for NVLS, users have to comply with several requirements:

 * The buffer is allocated through :c:func:`ncclMemAlloc` or a qualified allocator (see :ref:`mem_allocator`).
 * The NCCL operation is launched on a stream captured by a CUDA graph for each rank.
 * Offset to the head address of the buffer is the same in collectives for each rank.

Registered buffers will be deregistered when the CUDA graph is destroyed. Here is a CUDA graph based buffer registration example:

.. code:: C

  void* sendbuff;
  void* recvbuff;
  size_t count = 1 << 25;
  CHECK(ncclMemAlloc(&sendbuff, count * sizeof(float)));
  CHECK(ncclMemAlloc(&recvbuff, count * sizeof(float)));

  cudaGraph_t graph;
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CHECK(ncclAllReduce(sendbuff, recvbuff, 1024, ncclFloat, ncclSum, comm, stream));
  // Same offset to the sendbuff and recvbuff head address for each rank
  CHECK(ncclAllReduce((void*)((float*)sendbuff + 1024), (void*)((float*)recvbuff + 2048), 1024, ncclFloat, ncclSum, comm, stream));
  CHECK(cudaStreamEndCapture(stream, &graph));

  cudaGraphExec_t instance;
  CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
  CHECK(cudaGraphLaunch(instance, stream));
  CHECK(cudaStreamSynchronize(stream));
  CHECK(cudaGraphExecDestroy(instance));
  CHECK(cudaGraphDestroy(graph));

  CHECK(ncclMemFree(sendbuff));
  CHECK(ncclMemFree(recvbuff));

On the other hand, to enable the *Local* based buffer registration for NVLS, users have to comply with the following requirements:

 * The buffer is allocated through :c:func:`ncclMemAlloc` or a qualified allocator (see :ref:`mem_allocator`).
 * Register buffer with :c:func:`ncclCommRegister` before calling collectives for each rank.
 * Call NCCL collectives as usual but similarly keep the offset to the head address of the buffer the same for each rank.

Registered buffers will be deregistered when users explicitly call :c:func:`ncclCommDeregister`. Here is a local based buffer registration example:

.. code:: C

  void* sendbuff;
  void* recvbuff;
  size_t count = 1 << 25;
  void* sendRegHandle;
  void* recvRegHandle;
  CHECK(ncclMemAlloc(&sendbuff, count * sizeof(float)));
  CHECK(ncclMemAlloc(&recvbuff, count * sizeof(float)));

  CHECK(ncclCommRegister(comm, sendbuff, count * sizeof(float), &sendRegHandle));
  CHECK(ncclCommRegister(comm, recvbuff, count * sizeof(float), &recvRegHandle));

  CHECK(ncclAllReduce(sendbuff, recvbuff, 1024, ncclFloat, ncclSum, comm, stream));
  CHECK(ncclAllReduce((void*)((float*)sendbuff + 1024), (void*)((float*)recvbuff + 2048), 1024, ncclFloat, ncclSum, comm, stream));
  CHECK(cudaStreamSynchronize(stream));

  CHECK(ncclCommDeregister(comm, sendRegHandle));
  CHECK(ncclCommDeregister(comm, recvRegHandle));

  CHECK(ncclMemFree(sendbuff));
  CHECK(ncclMemFree(recvbuff));

For local based registration, users can register the buffer once at the beginning of the program and reuse the buffer multiple times to utilize
registration benefits.

To save the memory, it is also valid to allocate a large chunk of buffer and register it once. `sendbuff` and `recvbuff` can be further
allocated through the big chunk for zero-copy NCCL operations as long as `sendbuff` and `recvbuff` satisfy the offset requirements. The following
example shows a use case:

.. code:: C

  void* buffer;
  void* handle;
  void* sendbuff;
  void* recvbuff;
  size_t size = 1 << 29;

  CHECK(ncclMemAlloc(&buffer, size));
  CHECK(ncclCommRegister(comm, buffer, size, &handle));

  // assign buffer chunk to sendbuff and recvbuff
  sendbuff = buffer;
  recvbuff = (void*)((uint8_t*)buffer + (1 << 20));

  CHECK(ncclAllReduce(sendbuff, recvbuff, 1024, ncclFloat, ncclSum, comm, stream));
  CHECK(cudaStreamSynchronize(stream));

  CHECK(ncclCommDeregister(comm, handle));

  CHECK(ncclMemFree(sendbuff));

IB Sharp Buffer Registration
----------------------------

NCCL 2.21.x supports IB Sharp buffer registration, any NCCL collectives that support IB Sharp algorithm can benefit from the feature such as allreduce,
reducescatter, and allgather. Currently, NCCL only supports IB Sharp buffer registration for the communicators which contain 1 rank per node, and the
registration can reduce the number of NCCL SM usage down to 1.

To enable IB Sharp buffer registration by CUDA graph:

 * Allocate send and recv buffer with any CUDA allcator (e.g., cudaMalloc/ncclMemAlloc)
 * Launch NCCL collectives with CUDA graph

To enable IB Sharp buffer registration by local registration:

 * Allocate send and recv buffer with any CUDA allcator (e.g., cudaMalloc/ncclMemAlloc)
 * Register send and recv buffer for each rank in the communicator with `ncclCommRegister`
 * Launch NCCL collectives

.. _mem_allocator:

Memory Allocator
----------------

For convenience, NCCL provides `ncclMemAlloc` function to help users to allocate buffers through VMM API, which can be used for NCCL registration later. It is only designed for NCCL so that it is not recommended to use `ncclMemAlloc` allocated buffers everywhere in the applications. For advanced users, if you want to create your own memory allocator for NVLS buffer registration, the allocator needs to satisfy the following requirements:

 * Allocate buffer with shared flag `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`
 * Buffer size is multiple of multicast recommended granularity (i.e. cuMulticastGetGranularity(..., `CU_MULTICAST_GRANULARITY_RECOMMENDED`))
 * Buffer head address is at least aligned to multicast minimal granularity (i.e. cuMulticastGetGranularity(..., `CU_MULTICAST_GRANULARITY_MINIMUM`))
