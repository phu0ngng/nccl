# NCCL4Py tests

This directory contains two test suites:

- **Unit tests ([./unit_tests/](./unit_tests/))**: Pure Python tests that validate typing, buffer/spec handling, and high-level API logic **without** requiring the NCCL library (libnccl.so).
- **API tests ([./apitest/](./apitest/))**: Integration tests that exercise NCCL4Py end-to-end against the NCCL library (communicator init, collectives, P2P, and resource management).

## Quick start

Dependencies can be installed by installing `nccl4py` with the `test` dependency group:

```bash
cd <repo_reoot>/nccl4py
pip install -e . --group test
```

PyTorch and CuPy need to be installed manually, otherwise the corresponding tests will be skipped.

### Run unit tests (`libnccl.so` is not required)

```bash
python -m pytest -q tests/unit_tests
```

### Run API tests (requires NCCL + CUDA + MPI)

```bash
mpirun -n 2 python -m pytest -q -m mpi tests/apitest
```

See per-suite READMEs for detailed requirements and options.

