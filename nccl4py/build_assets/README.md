# Build Assets for NCCL4Py

> **⚠️ Internal Use Only**: This directory is **not released** to the public NCCL GitHub repository.

## Purpose

This directory contains tools and templates for generating Cython bindings for NCCL4Py. The generated bindings (`.pyx`, `.pxd` files) are checked into the repository under `<repo_root>/nccl4py/nccl/bindings/`, but the generation tooling remains internal.

## Why Not Public?

The binding generation relies on [cybind](https://gitlab-master.nvidia.com/leof/cybind), an internal NVIDIA tool that is not publicly available. To ensure users can build NCCL4Py without requiring access to internal tooling, we:

1. Generate bindings manually using `generate_cython.py` when NCCL APIs change
2. Commit the generated Cython source files to the repository
3. Exclude this `build_assets/` directory from public releases

## Prerequisites

### For `generate_header.py`

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Install system dependencies (for graph visualization) if it's not installed:
```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# RHEL/CentOS
sudo yum install graphviz
```

### For `generate_cython.py`

Requires:
- CUDA installation (set `CUDA_HOME` or `CUDA_PATH` environment variable)
- `uv` package manager (https://docs.astral.sh/uv/)

## Usage

### Generate Device API Header

Generate a C-compatible header from NCCL device API headers for pycparser/Cybind compatibility:

```bash
python3 generate_header.py
```

This script:
- Parses NCCL device headers using libclang
- Analyzes type dependencies for device API functions
- Generates a flattened C header with normalized types
- Outputs to `../../build/include/nccl_device_expanded.h` by default

Run with default paths:
```bash
python3 generate_header.py -I<path_to_cuda_include> -I<path_to_nccl_include> -I<path_to_stdbool_dir>
```

See `generate_header.py --help` for all options.

### Generate Cython Bindings

Run when NCCL headers are updated or binding templates are modified:

```bash
python3 generate_cython.py --verbose
```

This script:
- Clones cybind from internal NVIDIA repository
- Configures cybind with NCCL-specific templates and configuration
- Runs cybind to generate Cython bindings (.pyx, .pxd files)
- Copies generated bindings to `../nccl/bindings/` by default
- Requires CUDA_HOME or CUDA_PATH environment variable

Run with default paths:
```bash
python3 generate_cython.py
```

See `generate_cython.py --help` for all options.

## Contents

- `generate_header.py` - Generates C-compatible header from NCCL device API headers
- `generate_cython.py` - Main script for generating Cython bindings using cybind
- `cybind/` - Configuration and templates for cybind
  - `config_nccl.py` - cybind configuration for NCCL
  - `templates/` - Cython templates and static files

