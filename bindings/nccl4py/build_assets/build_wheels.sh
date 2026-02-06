#!/bin/bash
# Build wheels for nccl4py
# This script is intended for CI use with uv and cibuildwheel
#
# Requires: CUDA_HOME environment variable set

set -e

# Check CUDA_HOME is set
if [ -z "$CUDA_HOME" ]; then
    echo "Error: CUDA_HOME is not set"
    exit 1
fi

# Check uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install from: https://docs.astral.sh/uv/"
    exit 1
fi

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL4PY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$NCCL4PY_DIR/.." && pwd)"
BUILDDIR="${BUILDDIR:-$REPO_ROOT/build}"
DIST_DIR="$BUILDDIR/dist"

mkdir -p "$DIST_DIR"

echo "========================================="
echo "Building nccl4py wheels"
echo "========================================="
echo "CUDA_HOME: $CUDA_HOME"
echo "Output directory: $DIST_DIR"
echo ""

cd "$NCCL4PY_DIR"

# Build wheels using cibuildwheel
export CIBW_ENVIRONMENT="CUDA_HOME=$CUDA_HOME"
export CIBW_CONTAINER_ENGINE="docker; create_args: -v $CUDA_HOME:$CUDA_HOME:ro"

uv tool run cibuildwheel --output-dir "$DIST_DIR" --platform linux .

echo ""
echo "========================================="
echo "Build completed successfully!"
echo "========================================="
echo "Output directory: $DIST_DIR"
echo ""
echo "Built packages:"
find "$DIST_DIR" -type f -name "*.whl" | sort

