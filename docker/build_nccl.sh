#!/bin/bash -x
# use inside nccl build tools container

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --enable-ccache     Enable ccache for compilation"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  ENABLE_CCACHE       Set to 1 to enable ccache (overridden by --enable-ccache)"
    exit 1
}

# Parse command-line arguments
ENABLE_CCACHE=${ENABLE_CCACHE:-0}  # Default to 0 if not set

while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-ccache)
            ENABLE_CCACHE=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

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
tempdir=$(mktemp -d)
nccl_build_workspace=$tempdir/nccl
if [ "$CI_BUILD" -eq 1 ]; then
  # Local clone to carry over tracked files only when building in CI
  # Add file:// before /nccl to get --depth to work
  git clone --depth 1 file:///nccl $nccl_build_workspace
else
  # rsync instead of clone to allow untracked files when used during development
  # Add --ignore-missing-args to avoid the benign but noisy "file has vanished" errors
  rsync -a --ignore-missing-args --exclude=".git" /nccl/ $nccl_build_workspace/
fi

# Setup ccache
if [[ "$ENABLE_CCACHE" -eq 1 && -x "$(command -v ccache)" ]]; then
  echo "INFO: Enabling ccache"
  echo "INFO: ccache version: $(ccache --version)"
  # Create directory to store ccache symlinks
  mkdir -p $tempdir/bin
  ln -s $(which ccache) $tempdir/bin/gcc
  ln -s $(which ccache) $tempdir/bin/g++
  ln -s $(which ccache) $tempdir/bin/nvcc
  # Set variables for make
  export CC=$tempdir/bin/gcc
  export CXX=$tempdir/bin/g++
  export NVCC=$tempdir/bin/nvcc
  # Add temp bin to PATH
  export PATH=$tempdir/bin:$PATH
  # Set ccache variables
  source docker/ccache-vars.sh
  export CCACHE_BASEDIR=$nccl_build_workspace
  export CCACHE_DIR=$tempdir/ccache
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

# Print ccache stats if available
ccache --show-stats 2> /dev/null || true

# propagate exit status
exit $build_status
