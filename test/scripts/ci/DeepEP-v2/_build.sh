#!/bin/bash
#
# Builds DeepEP into $DEEP_EP_INSTALL_DIR. Invoked by _alloc.sh via
# `srun -N 1 -n 1`, so it runs on a single compute node — arch matches the
# GPUs (ARM/x86), which makes uv/pip/wheels correct for runtime.
#
# Required env: NCCL_INSTALL_DIR, DEEP_EP_INSTALL_DIR, SCRIPT_DIR,
# UV_INSTALL_DIR, CUDA_HOME, TORCH_INDEX_URL.
set -e

mkdir -p "$DEEP_EP_INSTALL_DIR"
export PATH="$UV_INSTALL_DIR:$PATH"

echo "Creating venv with Python 3.11 ..."
uv venv --python 3.11 --seed --clear "$DEEP_EP_INSTALL_DIR/venv"
source "$DEEP_EP_INSTALL_DIR/venv/bin/activate"

if [ -n "${TORCH_INDEX_URL:-}" ]; then
    # Pre-install torch from the cluster-specific index; `pip install -r` below
    # then treats it as already satisfied. (e.g. on EOS we grab cu129 here, then
    # requirements.txt's `torch==2.11.0` is satisfied by `2.11.0+cu129`.)
    TORCH_PIN=$(grep -E '^torch==' "$SCRIPT_DIR/requirements.txt")
    echo "Pre-installing torch from $TORCH_INDEX_URL ($TORCH_PIN) ..."
    pip install --quiet "$TORCH_PIN" --index-url "$TORCH_INDEX_URL"
else
    echo "Using PyPI default torch wheel (cu13) — installed via requirements.txt below."
fi

echo "Installing requirements ..."
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

echo "Cloning DeepEP (main) ..."
cd "$DEEP_EP_INSTALL_DIR"
rm -rf DeepEP   # idempotent: wipe any prior clone
git clone --quiet https://github.com/deepseek-ai/DeepEP.git
cd DeepEP

# NCCL is the diff under test. NVSHMEM has to come from the toolkit because
# DeepEP's setup.py links `-l:libnvshmem_host.so` (unversioned symlink) which
# the pip wheel doesn't ship — only `libnvshmem_host.so.3`.
export EP_NCCL_ROOT_DIR="$NCCL_INSTALL_DIR"
export EP_NVSHMEM_ROOT_DIR="$NVSHMEM_INSTALL_DIR"
export LIBRARY_PATH="$NVSHMEM_INSTALL_DIR/lib:$NCCL_INSTALL_DIR/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NVSHMEM_INSTALL_DIR/lib:$NCCL_INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"

echo "Running develop.sh ..."
bash develop.sh
