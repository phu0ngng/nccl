#!/bin/bash -e

if [ ! -d "nccl4py" ]; then
  echo "Error: Please launch from the top of NCCL source tree"
  exit 1
fi

if [ -z "${NCCL4PY_HOME}" ]; then
    echo "Error: NCCL4PY_HOME is not set."
    exit 1
fi
echo "Using NCCL4PY_HOME=$NCCL4PY_HOME"

if [ -z "${NCCL4PY_CUDA_TARGET}" ]; then
    echo "Error: NCCL4PY_CUDA_TARGET is not set."
    exit 1
fi
echo "Using NCCL4PY_CUDA_TARGET=$NCCL4PY_CUDA_TARGET"

if [ -z "${NCCL4PY_PYTHON_TARGETS}" ]; then
    echo "Error: NCCL4PY_PYTHON_TARGETS is not set."
    exit 1
fi
echo "Using NCCL4PY_PYTHON_TARGETS=$NCCL4PY_PYTHON_TARGETS"

# Configure NCCL4PY venvs
mkdir -p $NCCL4PY_HOME

module load uv
export UV_PYTHON_PREFERENCE="only-managed"

echo "--- Starting NCCL4py Setup ---"

IFS=',' read -r -a python_targets <<< "$NCCL4PY_PYTHON_TARGETS"

# Initialize venvs using uv with the appropriate Python version
for python_target in "${python_targets[@]}"; do
    echo "--- Setting up $python_target ---"

    nccl4py_venv_home=$NCCL4PY_HOME/$python_target

    venv=$nccl4py_venv_home/nccl4py_${python_target}_venv
    set -x
    uv venv --no-project --seed --python=$python_target $venv
    set +x
    echo "Initialized venv at: $venv"

    # Locate the nccl4py wheel based on the python target
    nccl4py_whl=$(find "build/dist" -type f -name "*$python_target*" 2>/dev/null)
    echo "Found wheel at: $nccl4py_whl"

    # Install wheel into the venv
    set -x
    source $venv/bin/activate
    pip install "$nccl4py_whl[$NCCL4PY_CUDA_TARGET]"
    pip list
    deactivate
    set +x
    echo "Installed nccl4py wheel in venv for $python_target"
    echo "--- Done! ---"
done
