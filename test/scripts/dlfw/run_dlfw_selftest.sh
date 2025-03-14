#!/bin/bash
set -x
set +e
chmod +x /workspace/nccl/test/scripts/dlfw/install_nccl_pytorch.sh
chmod +x /workspace/nccl/test/scripts/dlfw/nccl_pytorch_distributed_test.sh
/workspace/nccl/test/scripts/dlfw/install_nccl_pytorch.sh
chmod +x /opt/pytorch/qa/self_test_common.sh
chmod +x /opt/pytorch/qa/L0_self_test_distributed/test.sh
cd /opt/pytorch/qa/L0_self_test_distributed/
cp /workspace/nccl/test/scripts/dlfw/nccl_pytorch_distributed_test.sh .
export NCCL_DEBUG=WARN
echo "Checking pytorch version:"
export |grep NVIDIA
./nccl_pytorch_distributed_test.sh
