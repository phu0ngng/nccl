# Build Assets for NCCL4Py

> **⚠️ Internal Use Only**: This directory is **not released** to the public NCCL GitHub repository.

## Purpose

This directory contains tools and templates for generating Cython bindings for NCCL4Py. The generated bindings (`.pyx`, `.pxd` files) are checked into the repository under `<repo_root>/nccl4py/nccl/bindings/`, but the generation tooling remains internal.

## Why Not Public?

The binding generation relies on [cybind](https://gitlab-master.nvidia.com/leof/cybind), an internal NVIDIA tool that is not publicly available. To ensure users can build NCCL4Py without requiring access to internal tooling, we:

1. Generate bindings manually using `generate_cython.py` when NCCL APIs change
2. Commit the generated Cython source files to the repository
3. Exclude this `build_assets/` directory from public releases

## Usage

### Generate Bindings

Run when NCCL headers are updated or binding templates are modified:

```bash
python3 generate_cython.py --verbose
```

See `generate_cython.py --help` for additional options.

## Contents

- `generate_cython.py` - Main script for generating Cython bindings using cybind
- `cybind/` - Configuration and templates for cybind
  - `config_nccl.py` - cybind configuration for NCCL
  - `templates/` - Cython templates and static files

