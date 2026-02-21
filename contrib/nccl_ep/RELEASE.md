# NCCL EP (Expert Parallelism) - Release Notes

NCCL EP is a high-performance API extension to NCCL for efficient Mixture-of-Experts (MoE) operations. It provides optimized dispatch and combine primitives for expert parallelism across distributed GPU systems using NCCL's Device API: Load-Store Accessible (LSA) and GPU-Initiated Networking(GIN) operations.

## Overview

NCCL EP brings the performance benefits of modern device-initiated MoE libraries into the NCCL ecosystem with a unified API. Unlike existing solutions that expose separate interfaces for different operational modes, NCCL EP provides unified `ncclEpDispatch` and `ncclEpCombine` primitives that allow selecting the appropriate algorithm based on workload characteristics.

### Algorithm Implementations

- **Low-Latency (LL) Kernels** employ full all-to-all mesh connectivity with per-expert signal coordination, relying on direct token-to-expert communication.

- **High-Throughput (HT) Kernels**: implement hierarchical communication patterns, relying on NVLink for intra-node aggregation, and on RDMA for inter-node communication. The implementation leverages Hopper architecture features, including warp-specialized pipelines, and TMA (Tensor Memory Accelerator) operations.

Both implementations rely on NCCL Device API directly, performing GIN `put`/`signal` operations for RDMA and LSA operations over NVLink, eliminating CPU involvement in the critical path while inheriting NCCL's topology detection and network plugin architecture.

## Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| CXX  | 17+ | Requied |
| CUDA | 12.9+ | Required |
| NCCL | 2.29 | With Device API and GIN support |
| MPI | Any | For multi-process launch |
| GPU | Hopper (H100) or Blackwell | Tested configurations |

## Build Instructions

### Step 1: Build NCCL with Device API Support

```bash
cd /path/to/nccl
make -j src.build
```

This creates the build artifacts in `./build/`:
- `build/lib/libnccl.so` - NCCL library with EP support
- `build/include/` - Header files

### Step 2: Build NCCL EP Library and Test

```bash
make -C contrib/nccl_ep MPI=1
```

This creates:
- `build/lib/libnccl_ep.a` - Static library
- `build/lib/libnccl_ep.so` - Shared library (for Python bindings)
- `build/include/nccl_ep.h` - C API header
- `build/test/nccl_ep/ep_test` - Test application for both Low-Latency and High-Throughput modes
- `build/test/nccl_ep/ep_bench` - Benchmark application for both Low-Latency and High-Throughput modes


## Environment Setup

```bash
# Set paths
export MPI=1
export CUDA_HOME=/path/to/cuda
export MPI_HOME=/path/to/openmpi
export NCCL_HOME=/path/to/nccl/build

export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${NCCL_HOME}/lib:$LD_LIBRARY_PATH"
export PATH="${CUDA_HOME}/bin:${NCCL_HOME}/bin:$PATH"

# GIN configuration (recommended for RDMA)
export NCCL_GIN_TYPE=3  # GDAKI - GPU Direct Async Kernel-Initiated
```

## Running Tests

### C/C++ Test (ep_test)

```bash
# Low-Latency mode (default)
mpirun -np 8 ./build/test/nccl_ep/ep_test -a ll -t 128 -d 7168

# High-Throughput mode
mpirun -np 8 ./build/test/nccl_ep/ep_test -a ht -t 4096 -d 7168

```

#### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-a <ll\|ht>` | Algorithm mode: `ll` (Low-Latency) or `ht` (High-Throughput) | `ll` |
| `-t <num>` | Number of tokens | 50 |
| `-d <num>` | Hidden dimension size | 7168 |
| `-m` | Use `NCCL_EP_AUTO` for `max_tokens_per_rank` (HT only, **not yet supported**) | disabled |
| `-s <mode>` | Send-only mode: `none`, `dispatch`, `combine`, `both` | `none` |
| `-c` | Enable cached mode (HT only) | disabled |
| `-r` | Enable random mode (random topk_idx) | disabled |

#### Multi-Node Execution

##### Using MPI

```bash
# 2 nodes, 8 GPUs per node
mpirun -np 16 \
  --map-by ppr:8:node \
  -x NCCL_GIN_TYPE=3 \
  -x LD_LIBRARY_PATH \
  ./build/test/nccl_ep/ep_test -a ll -t 128 -d 7168
```

## Algorithm Modes

### Low-Latency (LL) Mode

Optimized for small batch sizes and latency-sensitive applications (i.e., LLM inference):
- Output tokens in 3D format: `[num_experts x max_tokens x hidden]`
- Expert-major data layout for efficient processing
- Supports staged execution (`send_only`) for computation-communication overlap


### High-Throughput (HT) Mode

Optimized for training and inference prefilling with large batch sizes:
- Output tokens in 2D format: `[num_recv_tokens x hidden]` where `num_recv_tokens = num_ranks * max_tokens_per_rank`
- Better bandwidth utilization through larger chunks


## API Quick Reference

### C API

```c
// Group management
ncclEpCreateGroup(&ep_group, comm, &config, stream, alloc_fn, free_fn);
ncclEpGroupDestroy(ep_group, stream);

// Handle management
ncclEpCreateHandle(&handle, ep_group, &topk_idx, local_tensors, num_local, config, stream);
ncclEpHandleDestroy(handle);

// Communication operations
ncclEpDispatch(handle, inputs, num_in, outputs, num_out, local, num_local, send_only, config, stream);
ncclEpCombine(handle, inputs, num_in, outputs, num_out, local, num_local, send_only, config, stream);
ncclEpComplete(handle, config, stream);  // LL mode only
```

### Python API

```bash
# Install Python bindings
pip install -e nccl_ep/python
```

```python
from nccl_ep import NCCLLibrary, NCCL_EP_ALGO_LOW_LATENCY

nccl_lib = NCCLLibrary()
# Use nccl_lib.ncclEpDispatch, ncclEpCombine, etc.
```

## Known Limitations

This release has the following limitations:

| Limitation | Description |
|------------|-------------|
| **No FP8 support** | FP8 data types are not currently supported |
| **8 ranks per node** | Fixed to 8 GPUs per node |
| **max_tokens_per_rank required** | `NCCL_EP_AUTO` is not yet supported; `max_tokens_per_rank` must be set to the per-rank batch size (max tokens any single rank will dispatch) |
| **Up to 8 nodes** | Maximum of 64 GPUs (8 nodes × 8 GPUs) supported |
| **Max Num Tokens Supported**| Currently implementation of HT kernels is using a define varible configured in `common.h` as `MAX_SUPPORTED_TOKENS_PER_RANK` |

### Debug Environment Variables

```bash
export NCCL_DEBUG=INFO        # Enable NCCL debug output
export NCCL_DEBUG_SUBSYS=ALL  # All subsystems
```

## References

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [GPU-Initiated Networking Paper](https://arxiv.org/abs/2511.15076)
- [NCCL EP API Reference](README.md)

## License

See LICENSE.txt for license information.
