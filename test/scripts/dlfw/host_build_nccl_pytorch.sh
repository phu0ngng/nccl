#!/bin/bash
echo "$(id -u):$(id -g)"
# -e USER_ID=$(id -u) -e USER_GROUP=$(id -g)
docker run -v $(pwd):/workspace/nccl $CONTAINER /workspace/nccl/test/scripts/dlfw/build_nccl_pytorch.sh