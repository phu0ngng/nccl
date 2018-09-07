# NCCL

Optimized primitives for collective multi-GPU communication.

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard collective communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, and reduce-scatter. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using Infiniband Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.
[This blog post](https://devblogs.nvidia.com/parallelforall/fast-multi-gpu-collectives-nccl/) provides details on NCCL functionality, goals, and performance.

## What's inside

At present, the library implements the following collectives:
- all-reduce
- all-gather
- reduce-scatter
- reduce
- broadcast

These collectives are implemented using ring algorithms and have been optimized for throughput and latency. For best performance, small collectives can be either batched into larger operations or aggregated through aggregated colletive API.

## Requirements

NCCL requires at least CUDA 7.0 and Kepler or newer GPUs. For PCIe based platforms, best performance is achieved when all GPUs are located on a common PCIe root complex, but multi-socket configurations are also supported.

## Build & Install

To build NCCL library.

```shell
$ cd nccl
$ make CUDA_HOME=<cuda install path> src.build
```

To install, run `make PREFIX=<install dir> install` and add `<instal dir>/lib` to your `LD_LIBRARY_PATH`.

## Usage

NCCL follows the MPI collectives API fairly closely. Before any collectives can be called, a communicator object must be initialized on each GPU. On a single-process machine, all GPUs can be conveniently initialized using `ncclCommInitAll`. For multi-process applications (e.g., with MPI), `ncclCommInitRank` must be called for each GPU. Internally `ncclCommInitRank` invokes a synchronization among all GPUs, so these calls must be invoked in different host threads (or processes) for each GPU. A brief single-process example follows. More examples including multi-process cases are available at https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#examples. For details about the API see nccl.h.

```c
#include <nccl.h>

typedef struct {
  double* sendBuff;
  double* recvBuff;
  int size;
  cudaStream_t stream;
} PerThreadData;

int main(int argc, char* argv[])
{
  int nGPUs;
  cudaGetDeviceCount(&nGPUs);
  ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nGPUs);
  ncclCommInitAll(comms, nGPUs); // initialize communicator
                                // One communicator per process

  PerThreadData* data;

  ... // Allocate data and issue work to each GPU's
      // perDevStream to populate the sendBuffs.

  for(int i=0; i<nGPUs; ++i) {
    cudaSetDevice(i); // Correct device must be set
                      // prior to each collective call.
    ncclAllReduce(data[i].sendBuff, data[i].recvBuff, size,
        ncclDouble, ncclSum, comms[i], data[i].stream);
  }

  ... // Issue work into data[*].stream to consume buffers, etc.
}
```

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
