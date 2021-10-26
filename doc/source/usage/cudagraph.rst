.. _using-nccl-with-cuda-graphs:

***************************
Using NCCL with CUDA Graphs
***************************

Starting with NCCL 2.9, NCCL operations can be captured by CUDA Graphs.

CUDA Graphs provide a way to define workflows as graphs rather than single operations. They may reduce overhead by launching multiple GPU operations through a single CPU operation. More details about CUDA Graphs can be found at https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs.

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

Note: if you are using NCCL in multi-thread mode, consideration can be given to adding the ``cudaStreamCaptureModeThreadLocal`` flag to the ``cudaStreamBeginCapture`` call.
