#!/bin/bash
ls /workspace/nccl
cd /workspace/nccl/
ls /usr/lib/x86_64-linux-gnu/libnccl.*
rm /usr/lib/x86_64-linux-gnu/libnccl.*
cp /workspace/nccl/build-dlfw/lib/libnccl* /usr/lib/x86_64-linux-gnu
ls /usr/lib/x86_64-linux-gnu/libnccl.*
rm /opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
cd /workspace