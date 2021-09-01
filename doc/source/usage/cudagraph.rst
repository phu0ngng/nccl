.. _using-nccl-with-cuda-graphs:

***************************
Using NCCL with CUDA Graphs
***************************

Starting with NCCL 2.9, NCCL operations can be captured by CUDA Graphs.

CUDA Graphs provide a way to define workflows as graphs rather than single operations. They may reduce overhead by launching multiple GPU operations through a single CPU operation. More details about CUDA Graphs can be found in the `CUDA Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs>`_.

NCCL's collective, P2P and group operations all support CUDA Graph captures. This support requires a minimum CUDA version of 11.3.

The following sample code shows how to capture computational kernels and NCCL operations in a CUDA Graph: ::

  cudaGraph_t graph;
  cudaStreamBeginCapture(stream);
  kernel_A<<< ..., stream >>>(...);
  kernel_B<<< ..., stream >>>(...);
  ncclAllreduce(..., stream);
  kernel_C<<< ..., stream >>>(...);
  cudaStreamEndCapture(stream, &graph);

  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);

Capture modes
-------------

By default, CUDA stream capture uses the ``cudaStreamCaptureModeGlobal`` mode if no flag is given to the ``cudaStreamBeginCapture`` call. This mode is compatible with NCCL except two scenarios:

(i) If you are using NCCL in multi-thread mode, i.e. a process has multiple threads each of which is attached to a different GPU, then you would need to add the ``cudaStreamCaptureModeThreadLocal`` flag to the ``cudaStreamBeginCapture`` call.

(ii) If you are capturing NCCL P2P calls (``ncclSend`` and ``ncclRecv``) without any previous P2P calls to the same peer(s), you would also need to use the ``cudaStreamCaptureModeThreadLocal`` mode.

A comparison between ``cudaStreamCaptureModeGlobal`` and ``cudaStreamCaptureModeThreadLocal`` can be found `here <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85>`_.
