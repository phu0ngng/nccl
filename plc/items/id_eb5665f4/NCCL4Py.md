# NCCL4Py

## Abstract

NCCL4Py provides Python bindings for NCCL and offers a Pythonic interface to NCCL's C API, enabling seamless interoperability with Pytorch, CuPy and other frameworks that implement DLPack and CUDA Array Interface standards.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

https://nvbugspro.nvidia.com/bug/5435544

### User Experience

Provide an official NCCL python bindings for python users. Users who need to use NCCL can install NCCL4Py through pip or conda, and don't have to write bindings by themselves or use third-party bindings.

### Assumptions, constraints and dependencies

- Requires NCCL 2.28 or newer
- Requires Python 3.10 or newer
- Requires CUDA 12 or 13
- CUDA-compatible GPUs and CUDA toolkit
- Cython for building bindings
- Optional: MPI for multi-process communicators

### Use Cases

- Python-based ML frameworks requiring NCCL collective operations
- Python HPC applications needing GPU-to-GPU communication
- Prototyping distributed GPU algorithms without C/C++

### Platform Requirements

Linux x86_64 / ARM, NVIDIA GPUs. Supports all NCCL transports (PCIe, NVLink, NVSwitch, IB, TCP/IP).

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed Design

**Architecture:**
1. `nccl.bindings`: Cython wrappers for NCCL C APIs
2. `nccl.core`: Pythonic layer with `Communicator`, `Buffer`, type-safe enums, group semantics, stream management

**Buffer System:** Accepts PyTorch tensor, CuPy array, cuda.core Buffer, and objects that implement DLPack or CUDA Array Interface.

**Resource Management:**
- Python garbage collection handles cleanup for buffer allocated by NCCL.
- Communicator must be destroyed explictly, it won't be handled by garbage collection as it's a collective operation.
- When a communicator is being destroyed, resources belong to the communicator (registered buffer, registered window, custom redop) are released automatically.

### Interface Architecture


**Architecture Layers:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Python Application                       │
│              (PyTorch, CuPy, JAX, etc.)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    nccl.core                                │
│              (High-Level Python API)                        │
│  ┌──────────────┬──────────────┬──────────────┬──────────┐  │
│  │ Communicator │   Buffer     │  Constants   │  Group   │  │
│  │    Memory    │   Resources  │    Typing    │  Interop │  │
│  └──────────────┴──────────────┴──────────────┴──────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               nccl.bindings.nccl                            │
│          (High-Level Cython Bindings)                       │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │  UniqueId    │  Collectives │  Comm Management         │ │
│  │  Classes     │  Operations  │  Error Handling          │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              nccl.bindings.cynccl                           │
│            (C-level Wrapper Functions)                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ncclAllReduce, ncclBroadcast, ncclCommInit, etc.      │ │
│  │  Thin cdef wrappers calling _internal functions        │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│            nccl.bindings._internal                          │
│          (Platform-Specific Dynamic Loading)                │
│  ┌──────────────────────┬─────────────────────────────────┐ │
│  │  nccl_linux.pyx      │  utils.pyx                      │ │
│  │  dlopen/dlsym        │  Buffer handling                │ │
│  │  libnccl.so.2 loader │  Resource management            │ │
│  └──────────────────────┴─────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  libnccl.so.2                               │
│            (NCCL C Library - Official)                      │
└─────────────────────────────────────────────────────────────┘
```

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Coding</h2></summary>
<!-- ============================================================================================-->

### Commit list or MR

https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/1431

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation

**Test Coverage:**
- Unit tests (`tests/unit_tests/`): Type system, buffer handling, API logic without NCCL library dependency
- API tests (`tests/apitest/`): Full integration with NCCL for all collectives, P2P operations, and resource management with mathematical correctness validation


#### Where to run?

#### What to run?

#### Expected output?

<!-- #### Code Coverage Goal Defined -->
<!-- #### KPI Coverage Goals Defined (performance, stress, stability, throughput, latency) -->
<!-- #### Requirement Coverage Goal Defined -->
<!-- #### Quality Thresholds Defined -->
<!-- #### Test Timeline -->
<!-- #### SW Verification and Test Plan -->
<!-- ### Test plan -->
<!-- #### Requirements Tests -->
<!-- #### Interface Tests -->
<!-- #### Fault-injection Tests -->
<!-- #### Resource Usage Tests -->
<!-- #### Design Coverage Testing -->
<!-- #### Boundary Tests -->
<!-- #### Certification Tests -->
<!-- #### Stress Tests -->
<!-- #### Stability Tests -->
<!-- #### Perf and Power KPI Tests -->
<!-- #### Usability & OOBE Tests -->
<!-- #### Manufacturing Diagnostic(Factory) Tests -->

### Performance

#### What is measured?

#### Results


</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Xiakun Lu

</details>
