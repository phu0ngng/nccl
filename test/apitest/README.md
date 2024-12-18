# NCCL API Tests

## Introduction

API tests exercise all NCCL APIs, verifying functional and non-functional cases. They currently don't support multi-process, hence they are launched as a single process which will use all visible GPUs on the system.

## Build

To build the test :

```shell
$ cd nccl
$ make -C test/apitest
```

## Run

To run the test

```shell
$ LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH ./build/test/apitest/apitest
```

Note that the test can be run under 3 conditions:

 - "P2P": no env vars set, will use CUDA P2P on NVLink systems.
 - "SHM": `NCCL_P2P_DISABLE=1`, will use shared memory
 - "NET": `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, will go through the network for all communication.

## Copyright

All source code and accompanying documentation is copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
