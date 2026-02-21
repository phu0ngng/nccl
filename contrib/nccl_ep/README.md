# NCCL EP (Expert Parallelism) API

A high-performance API extension to NCCL (NVIDIA Collective Communications Library) for efficient Mixture-of-Experts (MoE) operations, providing optimized dispatch and combine primitives for expert parallelism across distributed GPU systems.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Building](#building)
- [Running](#running)
- [Key Features](#key-features)
- [Core Concepts](#core-concepts)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Execution Modes](#execution-modes)
- [Usage Examples](#usage-examples)

## Overview

The NCCL EP API extends NCCL with native support for efficient Mixture-of-Experts communication patterns. It provides optimized implementations for token dispatch and expert output combining operations, which are an important components of modern large language models employing sparse mixture-of-experts architectures.

The API supports two distinct algorithms (via the same functions) tailored for different workload characteristics:

- **High Throughput (HT)**: Optimized for training and inference prefilling with large batch sizes
- **Low Latency (LL)**: Optimized for inference decoding with small batch sizes and latency-sensitive applications

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA | 13+ | Required |
| NCCL | 2.29+ | With Device API and GIN support |
| MPI | Any (OpenMPI, MPICH, etc.) | Required for multi-process launch |
| GPU | Hopper (H100) or Blackwell | Tested configurations |

## Building

### Step 1: Build NCCL with Device API Support

```bash
cd /path/to/nccl
make -j src.build

# Optional: build into a custom directory instead of ./build
# make -j src.build BUILDDIR=$PWD/build_rel
```

This creates the NCCL build artifacts in `BUILDDIR` (`./build` by default):
- `BUILDDIR/lib/libnccl.so` - NCCL library with EP support
- `BUILDDIR/include/` - Header files

### Step 2: Build NCCL EP Library and Test

```bash
# If NCCL was built into ./build (default):
make -C contrib/nccl_ep

# If NCCL was built into a custom directory, pass BUILDDIR as an absolute path.
# Example:
# make -C contrib/nccl_ep BUILDDIR=$PWD/build_rel
```

For custom NCCL build directories, use an absolute `BUILDDIR` path when invoking `make -C contrib/nccl_ep` (for example, `$PWD/build_rel`).

This creates:
- `BUILDDIR/lib/libnccl_ep.a` - Static library
- `BUILDDIR/lib/libnccl_ep.so` - Shared library (for Python bindings)
- `BUILDDIR/include/nccl_ep.h` - C API header
- `BUILDDIR/test/nccl_ep/ep_test` - Test application for both Low-Latency and High-Throughput modes

## Running

### Environment Setup

```bash
# Set paths
export MPI=1
export CUDA_HOME=/path/to/cuda
export MPI_HOME=/path/to/openmpi
export NCCL_HOME=/path/to/nccl/build

export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${NCCL_HOME}/lib:$LD_LIBRARY_PATH"
export PATH="${CUDA_HOME}/bin:${NCCL_HOME}/bin:$PATH"

# For multi-node RDMA (recommended)
export NCCL_GIN_TYPE=3  # GDAKI - GPU Direct Async Kernel-Initiated
```

### Running ep_test

The `ep_test` application (`ep_test.cu`) is a comprehensive working example that demonstrates both Low-Latency and High-Throughput modes. Use it as a reference implementation for integrating NCCL EP into your application.

```bash
# Low-Latency mode (default)
mpirun -np 8 ./build/test/nccl_ep/ep_test -a ll -t 128 -d 7168

# High-Throughput mode
mpirun -np 8 ./build/test/nccl_ep/ep_test -a ht -t 4096 -d 7168

```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-a <ll\|ht>` | Algorithm mode: `ll` (Low-Latency) or `ht` (High-Throughput) | `ll` |
| `-t <num>` | Number of tokens | 50 |
| `-d <num>` | Hidden dimension size | 7168 |
| `-m` | Disable max_tokens_per_rank (HT mode only) | disabled |
| `-s <mode>` | Send-only mode: `none`, `dispatch`, `combine`, `both` | `none` |
| `-c` | Enable cached mode (HT only) | disabled |
| `-r` | Enable random mode (random topk_idx) | disabled |

## Key Features

- **Two Algorithm Modes**: Choose between high-throughput and low-latency optimization profiles
- **Staged Execution** (LL mode only): Enable computation-communication overlap through send-only operations
- **Automatic Tuning**: Let the API auto-tune buffer sizes, queue pairs, and channels
- **Type Conversion**: Automatic scaling support when dispatching/combining with different data types

## Core Concepts

### Key Data Structures

#### `ncclNDTensor_t` - Multi-dimensional Tensor Descriptor

Encapsulates tensor metadata and data layout information:

```c
typedef struct {
    unsigned int version;           // Structure version (set to 1)
    unsigned int ndim;              // Number of dimensions
    unsigned int* sizes;            // Dimension sizes [ndim]
    unsigned int* strides;          // Strides in elements [ndim]
    ncclDataType_t datatype;        // Element data type
    void* data;                     // Pointer to tensor data
    unsigned int tag;               // Tensor identification tag
    ncclEpTensorFlags_t flags;     // Tensor flags (set to 0)
} ncclNDTensor_t;
```

#### `ncclEpGroup_t` - EP Group Configuration

Created from an NCCL communicator, manages the distributed EP configuration across all ranks in the group:

```c
typedef struct {
    unsigned int version;           // Structure version (set to 1)
    ncclEpAlgorithm_t algorithm;   // HT or LL mode
    unsigned int num_experts;       // Total experts across all ranks
    unsigned int max_tokens_per_rank;  // Max tokens per rank
    unsigned int token_size_bytes;  // Maximum token size
    unsigned int rdma_buffer_size;  // RDMA buffer size (0=auto)
    unsigned int num_qp_per_rank;   // Queue pairs per rank (0=auto)
    unsigned int num_channels;      // Channels per rank (0=auto)
} ncclEpGroupConfig_t;
```

#### `ncclEpHandle_t` - Operation Handle

Maintains state for a sequence of related MoE operations (forward and backward passes). The handle encapsulates routing metadata and buffers for communication.

### Algorithms

**High Throughput (HT)**:
- Output tokens are in 2D format: `[num_recv_tokens x hidden]` where `num_recv_tokens = num_ranks * max_tokens_per_rank`
- Optimized for large batch sizes and training workloads
- Supports dynamic `max_tokens_per_rank` (set to `NCCL_EP_AUTO`)
- Better bandwidth utilization through larger chunks

**Low Latency (LL)**:
- Output tokens in 3D format: `[num_experts x max_tokens x hidden]`
- Expert-major data layout for efficient expert processing
- Supports `send_only` parameter for overlapping computation with communication
- Better latency for small batch sizes
- Requires `max_tokens_per_rank` to be set correctly

### Custom Allocators

The API supports custom memory allocators for internal buffer management. This enables integration with memory pools, custom allocation strategies, or framework-specific allocators.

#### Function Signatures

```c
typedef cudaError_t (*ncclEpAllocFn_t)(void** ptr, size_t size);
typedef cudaError_t (*ncclEpFreeFn_t)(void* ptr);
```

Allocators must match the `cudaMalloc`/`cudaFree` signatures and return `cudaSuccess` on success.

#### Example

```c
// Custom allocator using a memory pool
cudaError_t my_alloc(void** ptr, size_t size) {
    *ptr = myMemoryPool.allocate(size);
    return (*ptr != nullptr) ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t my_free(void* ptr) {
    myMemoryPool.deallocate(ptr);
    return cudaSuccess;
}

// Pass to ncclEpCreateGroup
ncclEpCreateGroup(&ep_group, comm, &config, stream, my_alloc, my_free);
```

If `NULL` is passed for both allocator functions, the default `cudaMalloc`/`cudaFree` are used.

## Quick Start

### Basic Setup (High Throughput Mode)

```c
#include "nccl_ep.h"
#include "cuda_runtime.h"

// 1. Initialize NCCL communicator (standard NCCL initialization)
ncclComm_t nccl_comm;
ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank);

// 2. Create MoE group with configuration
ncclEpGroupConfig_t config;
config.version = 1;
config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
config.num_experts = 64;
config.max_tokens_per_rank = 1024;
config.token_size_bytes = 8192;
config.rdma_buffer_size = NCCL_EP_AUTO;     // Auto-size
config.num_qp_per_rank = NCCL_EP_AUTO;      // Auto-size
config.num_channels = NCCL_EP_AUTO;         // Auto-size

ncclEpGroup_t ep_group;
ncclEpCreateGroup(&ep_group, nccl_comm, &config, stream, my_alloc, my_free);

// 3. Create EP handle with routing information
ncclNDTensor_t topk_idx = {...};  // [num_tokens x top_k]
ncclEpHandle_t handle;
ncclEpCreateHandle(&handle, ep_group, &topk_idx, NULL, 0, NULL, stream);

// 4. Perform dispatch operation
ncclNDTensor_t* inputs[...] = {...};
ncclNDTensor_t* outputs[...] = {...};
ncclEpDispatch(handle, inputs, num_inputs, outputs, num_outputs,
                NULL, 0, 0, NULL, stream);

// 5. Compute expert forward passes...

// 6. Perform combine operation
ncclNDTensor_t* combine_inputs[...] = {...};
ncclNDTensor_t* combine_outputs[...] = {...};
ncclEpCombine(handle, combine_inputs, num_combine_inputs,
               combine_outputs, num_combine_outputs,
               NULL, 0, 0, NULL, stream);

// 7. Cleanup
ncclEpHandleDestroy(handle);
ncclEpGroupDestroy(ep_group, stream);
```

## API Reference

### Group Management

#### `ncclEpCreateGroup()`

Create an EP group from an NCCL communicator.

```c
ncclResult_t ncclEpCreateGroup(
    ncclEpGroup_t* ep_group,
    ncclComm_t comm,
    const ncclEpGroupConfig_t* config,
    cudaStream_t stream,
    ncclEpAllocFn_t alloc_fn,
    ncclEpFreeFn_t free_fn
);
```

**Arguments:**
- `ep_group` [OUT]: Pointer to newly created EP group
- `comm` [IN]: Existing NCCL communicator
- `config` [IN]: Pointer to EP configuration structure
- `stream` [IN]: CUDA stream for group creation
- `alloc_fn` [IN]: Custom allocator function (NULL for default cudaMalloc)
- `free_fn` [IN]: Custom free function (NULL for default cudaFree)

**Notes:**
- This is a collective call: must be called by all ranks
- Returns `ncclResult_t` error code

#### `ncclEpGroupDestroy()`

Destroy an EP group and release associated resources.

```c
ncclResult_t ncclEpGroupDestroy(
    ncclEpGroup_t ep_group,
    cudaStream_t stream
);
```

### Tensor Management

#### `ncclEpTensorCreate()`

Create a contiguous tensor using the group's allocator.

```c
ncclResult_t ncclEpTensorCreate(
    ncclEpGroup_t ep_group,
    ncclNDTensor_t* tensor,
    unsigned int ndim,
    ncclDataType_t datatype,
    ncclEpTensorTag_t tag,
    unsigned int size0,
    unsigned int size1 = 1,
    unsigned int size2 = 1,
    unsigned int size3 = 1,
    unsigned int size4 = 1
);
```

**Arguments:**
- `ep_group` [IN]: Valid EP group from `ncclEpCreateGroup()`
- `tensor` [OUT]: Pointer to tensor structure to initialize
- `ndim` [IN]: Number of dimensions (1-5)
- `datatype` [IN]: Element data type (e.g., `ncclBfloat16`, `ncclFloat32`, `ncclInt64`)
- `tag` [IN]: Tensor identification tag (e.g., `NCCL_EP_TENSOR_TAG_DISPATCH_INPUT_TOKENS`)
- `size0..size4` [IN]: Dimension sizes (unused dimensions default to 1)

**Notes:**
- Allocates memory using the group's allocator (custom or default cudaMalloc)
- Automatically sets up strides for contiguous layout

#### `ncclEpTensorDestroy()`

Destroy a tensor and free its memory using the group's allocator.

```c
ncclResult_t ncclEpTensorDestroy(
    ncclEpGroup_t ep_group,
    ncclNDTensor_t* tensor
);
```

### Handle Management

#### `ncclEpCreateHandle()`

Create and initialize a EP handle with dispatch setup and optional metadata exchange.

```c
ncclResult_t ncclEpCreateHandle(
    ncclEpHandle_t* handle,
    ncclEpGroup_t ep_group,
    const ncclNDTensor_t* topk_idx,
    ncclNDTensor_t* const* local_tensors,
    unsigned int num_local_tensors,
    const ncclEpHandleConfig_t* config,
    cudaStream_t stream
);
```

**Arguments:**
- `handle` [OUT]: Pointer to newly created handle
- `ep_group` [IN]: Valid EP group from `ncclEpCreateGroup()`
- `topk_idx` [IN]: Tensor with top-K expert indices for routing
- `local_tensors` [IN/OUT]: Array of pointers to local tensors (optional)
  - HT mode accepts 1 optional tensor: `NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST` or `NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE`
  - LL mode does not accept local tensors
- `num_local_tensors` [IN]: Number of local tensors
- `config` [IN]: Reserved for future use, set to `NULL`
- `stream` [IN]: CUDA stream

**Notes:**
- This is a collective operation (all ranks must call it)
- In HT mode, if `max_tokens_per_rank` is 0, this call will block the host until host gets the information from remote ranks.
- The same handle can be used for both forward and backward passes (saves a sync between the group)

#### `ncclEpHandleDestroy()`

Destroy a EP handle and release resources.

```c
ncclResult_t ncclEpHandleDestroy(ncclEpHandle_t handle);
```

#### `ncclEpHandleGetNumRecvTokens()` (HT mode only)

Query the number of tokens that will be received after a call to EP dispatch.

```c
ncclResult_t ncclEpHandleGetNumRecvTokens(
    ncclEpHandle_t handle,
    unsigned int* num_recv_tokens
);
```

**Arguments:**
- `handle` [IN]: A valid EP handle
- `num_recv_tokens` [OUT]: Pointer to unsigned int, will be set to the actual number of tokens expected to be received on this rank

**Notes:**
- This API is only supported in HIGH_THROUGHPUT (HT) mode
- Call after `ncclEpCreateHandle()` to determine buffer sizes for output tensors
- Returns `ncclInvalidArgument` if called in LL mode

### Communication Operations

#### `ncclEpDispatch()`

Perform EP dispatch: send tokens to experts according to routing decisions.

```c
ncclResult_t ncclEpDispatch(
    ncclEpHandle_t handle,
    const ncclNDTensor_t* const* inputs,
    unsigned int num_inputs,
    ncclNDTensor_t* const* outputs,
    unsigned int num_outputs,
    ncclNDTensor_t* const* local_tensors,
    unsigned int num_local_tensors,
    unsigned int send_only,
    const ncclEpDispatchConfig_t* dispatch_config,
    cudaStream_t stream
);
```

**Arguments:**
- `handle` [IN,OUT]: EP handle (updated with metadata)
- `inputs` [IN]: Array of input tensor pointers (all 2D: `[num_tokens x data_size]`)
- `num_inputs` [IN]: Number of input tensors
- `outputs` [IN,OUT]: Array of pre-allocated output tensor pointers
  - HT mode: 2D `[num_recv_tokens x data_size]` where `num_recv_tokens = num_ranks * max_tokens_per_rank`
  - LL mode: 3D `[num_experts x num_recv_tokens x data_size]` where `num_recv_tokens = num_ranks * max_tokens_per_rank`
- `local_tensors` [IN]: Local state tensors (LL mode only)
- `num_local_tensors` [IN]: Number of local tensors
- `send_only` [IN]: Boolean flag (LL mode only, 0 for HT)
- `dispatch_config` [IN]: Dispatch configuration (usually NULL)
- `stream` [IN]: CUDA stream

**Notes:**
- All input tensors must have the same number of tokens
- Data sizes may vary between tensors
- If output datatype differs from input datatype, scaling factors must be provided

#### `ncclEpCombine()`

Perform EP combine: gather expert outputs and return in original token order.

```c
ncclResult_t ncclEpCombine(
    ncclEpHandle_t handle,
    const ncclNDTensor_t* const* inputs,
    unsigned int num_inputs,
    ncclNDTensor_t* const* outputs,
    unsigned int num_outputs,
    ncclNDTensor_t* const* local_tensors,
    unsigned int num_local_tensors,
    unsigned int send_only,
    const ncclEpDispatchConfig_t* combine_config,
    cudaStream_t stream
);
```

**Arguments:** Similar to `ncclEpDispatch()`
- `inputs` [IN,OUT]: Array of input tensor pointers (expert outputs)
  - HT mode: 2D `[num_recv_tokens x data_size]` where `num_recv_tokens = num_ranks * max_tokens_per_rank`
  - LL mode: 3D `[num_experts x num_recv_tokens x data_size]` where `num_recv_tokens = num_ranks * max_tokens_per_rank`
- `outputs` [IN]: Array of pre-allocated output tensor pointers (all 2D: `[num_tokens x data_size]`, original token order)



**Notes:**
- In HT mode: uses top-k weights from dispatch to combine outputs
- In LL mode: requires `local_tensors` including token counts and weights
- `send_only` enables staged execution for computation overlap

#### `ncclEpComplete()` (LL mode only)

Wait for completion of send-only dispatch or combine operations.

```c
ncclResult_t ncclEpComplete(
    ncclEpHandle_t handle,
    const ncclEpCompleteConfig_t* config,
    cudaStream_t stream
);
```

## Execution Modes

### Synchronized Mode (Default)

Functions return when communication kernels are scheduled to the GPU, with kernels executing synchronously:

```c
ncclEpDispatch(handle, inputs, num_inputs, outputs, num_outputs,
                NULL, 0, 0, NULL, stream);  // send_only = 0 (false)
// Dispatch is complete when this returns
```

### Staged Mode (Low Latency Only)

Split communication into send and receive phases for computation overlap:

```c
// Stage 1: Post send requests without waiting for completion
ncclEpDispatch(handle, inputs, num_inputs, outputs, num_outputs,
                NULL, 0, 1, NULL, stream);  // send_only = 1 (true)
// Returns immediately after posting requests

// Stage 2: Continue other computations...

// Stage 3: Wait for actual completion
ncclEpCompleteConfig_t continue_config;
ncclEpComplete(handle, &continue_config, stream);
// Now all data is actually sent/received
```

This mode enables computation-communication overlap by allowing work between posting and completion, particularly beneficial for inference with multiple micro-batches.

## Usage Examples

> **Note:** For a complete working example, see `ep_test.cu` which demonstrates both LL and HT modes with all API calls.

### Example 1: High Throughput Mode - Forward and Backward Pass

```c
#include "nccl.h"
#include "nccl_ep.h"
#include "cuda_runtime.h"

// Initialize NCCL communicator
ncclComm_t comm;
ncclCommInitRank(&comm, nRanks, id, myRank);

cudaStream_t stream;
cudaStreamCreate(&stream);

unsigned int top_k = 8;
unsigned int hidden = 4096;

// Configure for High Throughput mode
ncclEpGroupConfig_t config;
config.version = 1;
config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
config.num_experts = 256;
config.max_tokens_per_rank = 4;  // Or NCCL_EP_AUTO for dynamic sizing
config.token_size_bytes = hidden * 2;  // bfloat16
config.rdma_buffer_size = NCCL_EP_AUTO;     // Auto-size
config.num_qp_per_rank = NCCL_EP_AUTO;      // Auto-size
config.num_channels = NCCL_EP_AUTO;         // Auto-size

ncclEpGroup_t ep_group;
ncclEpCreateGroup(&ep_group, comm, &config, stream, my_alloc, my_free);

ncclNDTensor_t topk_idx;
ncclEpTensorCreate(ep_group, &topk_idx, 2, ncclInt64,
                    NCCL_EP_TENSOR_TAG_TOPK_IDX_HANDLE,
                    num_tokens, top_k);

// Create recv_expert_counter local tensor for ncclEpCreateHandle (optional, for HT mode)
// This tensor will receive the number of tokens per expert after metadata exchange
ncclNDTensor_t recv_expert_counter;
ncclNDTensor_t* local_tensors[1] = {nullptr};
unsigned int num_local_tensors = 0;
if (config.max_tokens_per_rank == NCCL_EP_AUTO) {
    recv_expert_counter.ndim = 1;
    recv_expert_counter.datatype = ncclInt32;
    recv_expert_counter.strides = new unsigned int[1];
    recv_expert_counter.strides[0] = 1;
    recv_expert_counter.tag = NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST;
    recv_expert_counter.flags = NCCL_EP_TENSOR_FLAG_NONE;
    recv_expert_counter.sizes = new unsigned int[1];
    recv_expert_counter.sizes[0] = num_local_experts;
    cudaHostAlloc(&recv_expert_counter.data, num_local_experts * sizeof(int), cudaHostAllocMapped);
    local_tensors[0] = &recv_expert_counter;
    num_local_tensors = 1;
}

// Create EP handle (can be reused for forward and backward)
ncclEpHandle_t handle;
ncclEpCreateHandle(&handle, ep_group, &topk_idx, local_tensors, num_local_tensors, NULL, stream);

// max_tokens_per_rank is the per-rank dispatch count.
// num_recv_tokens is the max tokens this rank can receive (nRanks * max_tokens_per_rank).
unsigned int num_recv_tokens;
if (config.max_tokens_per_rank == NCCL_EP_AUTO) {
    ncclEpHandleGetNumRecvTokens(handle, &num_recv_tokens);
} else {
    num_recv_tokens = config.max_tokens_per_rank * nRanks;
}

// === FORWARD PASS ===

// Create input tensors (HT mode uses 3 inputs)
ncclNDTensor_t input_tokens;
ncclEpTensorCreate(ep_group, &input_tokens, 2, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_DISPATCH_INPUT_TOKENS,
                    num_tokens, hidden);

ncclNDTensor_t topk_weights;
ncclEpTensorCreate(ep_group, &topk_weights, 2, ncclFloat32,
                    NCCL_EP_TENSOR_TAG_DISPATCH_INPUT_TOPK_WEIGHTS,
                    num_tokens, top_k);

topk_idx.tag = NCCL_EP_TENSOR_TAG_TOPK_IDX_DISPATCH;

ncclNDTensor_t* forward_inputs[3] = {&input_tokens, &topk_weights, &topk_idx};

// Create output tensors (HT mode: 3 outputs, all 2D)
ncclNDTensor_t output_tokens;
ncclEpTensorCreate(ep_group, &output_tokens, 2, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_DISPATCH_OUTPUT_TOKENS,
                    num_recv_tokens, hidden);

ncclNDTensor_t recv_topk_weights;
ncclEpTensorCreate(ep_group, &recv_topk_weights, 2, ncclFloat32,
                    NCCL_EP_TENSOR_TAG_DISPATCH_OUTPUT_TOPK_WEIGHTS,
                    num_recv_tokens, top_k);

ncclNDTensor_t recv_topk_idx;
ncclEpTensorCreate(ep_group, &recv_topk_idx, 2, ncclInt64,
                    NCCL_EP_TENSOR_TAG_DISPATCH_OUTPUT_TOPK_IDX,
                    num_recv_tokens, top_k);

ncclNDTensor_t* forward_outputs[3] = {&output_tokens, &recv_topk_weights, &recv_topk_idx};

// Local tensors for dispatch
unsigned int num_local_experts = config.num_experts / nRanks;
ncclNDTensor_t tokens_per_expert;
ncclEpTensorCreate(ep_group, &tokens_per_expert, 1, ncclInt32,
                    NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
                    num_local_experts);

ncclNDTensor_t* dispatch_local_tensors[1] = {&tokens_per_expert};

// Dispatch tokens to experts
ncclEpDispatchConfig_t dispatch_config;
ncclEpDispatch(handle, forward_inputs, 3, forward_outputs, 3,
                dispatch_local_tensors, 1, 0, &dispatch_config, stream);

// Expert forward computation
// ... process output_tokens using tokens_per_expert counts ...

// Create expert output tensor
ncclNDTensor_t expert_outputs;
ncclEpTensorCreate(ep_group, &expert_outputs, 2, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_COMBINE_INPUT_TOKENS,
                    num_recv_tokens, hidden);

ncclNDTensor_t combined_output;
ncclEpTensorCreate(ep_group, &combined_output, 2, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_COMBINE_OUTPUT_TOKENS,
                    num_tokens, hidden);

ncclNDTensor_t* combine_inputs[1] = {&expert_outputs};
ncclNDTensor_t* combine_outputs[1] = {&combined_output};

ncclEpCombine(handle, combine_inputs, 1, combine_outputs, 1,
               nullptr, 0, 0, nullptr, stream);

// === BACKWARD PASS ===
// Use the same handle - routing information is reused

ncclNDTensor_t* backward_dispatch_inputs[1] = {&grad_combined};
ncclNDTensor_t* backward_dispatch_outputs[1] = {&grad_at_experts};

ncclEpDispatch(handle, backward_dispatch_inputs, 1, backward_dispatch_outputs, 1,
                nullptr, 0, 0, &dispatch_config, stream);

// Expert backward computation
// ... compute gradients for each expert ...

// Combine gradients
ncclNDTensor_t* backward_combine_inputs[2] = {&grad_expert_outputs, combine_topk_weights_input};
ncclNDTensor_t* backward_combine_outputs[2] = {&grad_tokens, combine_topk_weights_output};

ncclEpCombine(handle, backward_combine_inputs, 2, backward_combine_outputs, 2,
               nullptr, 0, 0, nullptr, stream);

// Cleanup
ncclEpHandleDestroy(handle);
ncclEpGroupDestroy(ep_group, stream);
ncclCommDestroy(comm);
cudaStreamDestroy(stream);
```

### Example 2: Low Latency Mode - Forward Pass

```c
#include "nccl.h"
#include "nccl_ep.h"
#include "cuda_runtime.h"

// Initialize NCCL communicator
ncclComm_t comm;
ncclCommInitRank(&comm, nRanks, id, myRank);

cudaStream_t stream;
cudaStreamCreate(&stream);

unsigned int top_k = 8;
unsigned int hidden = 4096;
unsigned int num_local_experts = config.num_experts / nRanks;

// Configure for Low Latency mode
ncclEpGroupConfig_t config;
config.version = 1;
config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
config.num_experts = 256;
config.max_tokens_per_rank = 128;  // Must be set for LL mode
config.token_size_bytes = hidden * 2;  // bfloat16
config.rdma_buffer_size = NCCL_EP_AUTO;     // Auto-size
config.num_qp_per_rank = NCCL_EP_AUTO;      // Auto-size (or specify for LL)
config.num_channels = NCCL_EP_AUTO;         // Auto-size

ncclEpGroup_t ep_group;
ncclEpCreateGroup(&ep_group, comm, &config, stream, my_alloc, my_free);

// Create routing tensor (topk_idx)
ncclNDTensor_t topk_idx;
ncclEpTensorCreate(ep_group, &topk_idx, 2, ncclInt64,
                    NCCL_EP_TENSOR_TAG_TOPK_IDX_HANDLE,
                    num_tokens, top_k);

// Create EP handle
ncclEpHandle_t handle;
ncclEpCreateHandle(&handle, ep_group, &topk_idx, NULL, 0, NULL, stream);

// === FORWARD PASS ===

// Create input tensor (LL mode uses 1 input)
ncclNDTensor_t input_tokens;
ncclEpTensorCreate(ep_group, &input_tokens, 2, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_DISPATCH_INPUT_TOKENS,
                    num_tokens, hidden);

ncclNDTensor_t* dispatch_inputs[1] = {&input_tokens};

// Create output tensor (LL mode: 3D format [num_local_experts, nRanks * max_tokens, hidden])
ncclNDTensor_t output_tokens;
ncclEpTensorCreate(ep_group, &output_tokens, 3, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_DISPATCH_OUTPUT_TOKENS,
                    num_local_experts, nRanks * config.max_tokens_per_rank, hidden);

ncclNDTensor_t* dispatch_outputs[1] = {&output_tokens};

// Create local tensors for LL mode
ncclNDTensor_t tokens_per_expert;
ncclEpTensorCreate(ep_group, &tokens_per_expert, 1, ncclInt32,
                    NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE,
                    num_local_experts);

ncclNDTensor_t* local_tensors[1] = {&tokens_per_expert};

// Dispatch tokens to experts (staged execution for overlap)
ncclEpDispatchConfig_t dispatch_config;
dispatch_config.round_scales = 0;

ncclEpDispatch(handle, dispatch_inputs, 1, dispatch_outputs, 1,
                local_tensors, 1, 1 /* send_only */, &dispatch_config, stream);

// Overlap with other computation...
// doOtherWork(stream);

// Wait for dispatch to complete
ncclEpComplete(handle, nullptr, stream);
cudaStreamSynchronize(stream);

// Expert forward computation
// Process output_tokens in 3D layout [experts x tokens x hidden]
// Use tokens_per_expert to know how many valid tokens per expert
// ... expertCompute(output_tokens, expert_outputs, tokens_per_expert, stream) ...

// Create expert output tensor (also 3D in LL mode)
ncclNDTensor_t expert_outputs;
ncclEpTensorCreate(ep_group, &expert_outputs, 3, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_COMBINE_INPUT_TOKENS,
                    num_local_experts, nRanks * config.max_tokens_per_rank, hidden);

// Create topk_weights for combine
ncclNDTensor_t topk_weights;
ncclEpTensorCreate(ep_group, &topk_weights, 2, ncclFloat32,
                    NCCL_EP_TENSOR_TAG_COMBINE_INPUT_TOPK_WEIGHTS,
                    num_tokens, top_k);

ncclNDTensor_t* combine_local_tensors[1] = {&topk_weights};

// Combine expert outputs back to original token order
ncclNDTensor_t combined_output;
ncclEpTensorCreate(ep_group, &combined_output, 2, ncclBfloat16,
                    NCCL_EP_TENSOR_TAG_COMBINE_OUTPUT_TOKENS,
                    num_tokens, hidden);

ncclNDTensor_t* combine_inputs[1] = {&expert_outputs};
ncclNDTensor_t* combine_outputs[1] = {&combined_output};

ncclEpCombine(handle, combine_inputs, 1, combine_outputs, 1,
               combine_local_tensors, 1, 0 /* send_only */, nullptr, stream);

ncclEpComplete(handle, nullptr, stream);
cudaStreamSynchronize(stream);

// Cleanup
ncclEpHandleDestroy(handle);
ncclEpGroupDestroy(ep_group, stream);
ncclCommDestroy(comm);
cudaStreamDestroy(stream);
```
