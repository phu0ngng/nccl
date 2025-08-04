#!/bin/bash -x
# use inside nccl build tools container
export CUDA_HOME=/usr/local/cuda
export MPI_HOME=/usr/local/openmpi
export LD_LIBRARY_PATH=${MPI_HOME}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda/bin:${PATH}

# clean up previous run if clean was requested
cd /nccl
if [ -d "build.old" ]; then
  rm -rf build.old &
  make clean
fi

# Create build workspace on local volume inside container
nccl_build_workspace=$(mktemp -d)/nccl
if [ "$CI_BUILD" -eq 1 ]; then
  # Local clone to carry over tracked files only when building in CI
  # Add file:// before /nccl to get --depth to work
  git clone --depth 1 file:///nccl $nccl_build_workspace
else
  # rsync instead of clone to allow untracked files when used during development
  # Add --ignore-missing-args to avoid the benign but noisy "file has vanished" errors
  rsync -a --ignore-missing-args --exclude=".git" /nccl/ $nccl_build_workspace/
fi

export NCCL_HOME="$nccl_build_workspace/build"

pushd $nccl_build_workspace

# figure out a fair job number
jobs=$(eval "$NUM_BUILD_PROCS")

echo "$(date +%T) : make starting"
start=$(date +%s%N)
make -j$jobs test.build MPI=1 TRACE=$NCCL_BUILD_TRACE
# make -j$jobs pkg.build

build_status=$?

if [ $build_status -eq 0 ]; then
    echo "INFO: Make exited successfully"
else
    echo "ERROR: Make exited with $build_status"
fi
end=$(date +%s%N)
let runtime=$((end - start))/1000000000
echo "$(date +%T) : make took $runtime s"

# Copy back built artifacts to host directory from container volume
if [ "$CI_BUILD" -eq 1 ]; then
  # Exclude .o files in CI as they are not used in subsequent jobs anyways
  rsync -a --exclude="*.o" build /nccl/
else
  rsync -a build /nccl/
fi

# propagate exit status
exit $build_status
