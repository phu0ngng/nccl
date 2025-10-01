#!/bin/bash
ls /workspace/nccl
cd /workspace/nccl/
echo "Removing old NCCL libraries"
echo "--------------------------------"
ls /usr/lib/x86_64-linux-gnu/libnccl.*
rm /usr/lib/x86_64-linux-gnu/libnccl.*
cp /workspace/nccl/build-dlfw/lib/libnccl* /usr/lib/x86_64-linux-gnu
ls /usr/lib/x86_64-linux-gnu/libnccl.*
echo "--------------------------------"
echo "Installing new NCCL libraries"
echo "--------------------------------"
ls /workspace/nccl/build-dlfw/lib/libnccl*
echo "--------------------------------"
echo "Done"
echo "--------------------------------"
cd /workspace
