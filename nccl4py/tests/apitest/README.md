# NCCL4Py API (integration) tests

These tests exercise NCCL4Py against a real NCCL library and real devices. Most tests are marked with `@pytest.mark.mpi` and are intended to run under MPI.

## Requirements

- **CUDA-capable GPUs** visible to the process (multi-rank tests typically require >= `-n` GPUs, or a setup that maps ranks to devices).
- **NCCL runtime library** available at runtime (e.g., `libnccl.so` on Linux).
- **MPI runtime** (e.g., OpenMPI / MPICH).
- **Python dependencies**:
  - `nccl4py`
  - `ml-dtypes`
  - `pytest`
  - `pytest-mpi`
  - `mpi4py`

## How to run

### Run all API tests with at least 2 ranks

```bash
mpirun -n 2 python -m pytest -q --with-mpi nccl4py/tests/apitest
```

### Run a single file

```bash
mpirun -n 2 python -m pytest -q --with-mpi nccl4py/tests/apitest/test_communicator_api.py
```

### Run a single test

```bash
mpirun -n 2 python -m pytest -q --with-mpi nccl4py/tests/apitest/test_communicator_api.py::test_init
```

## Useful environment variables

- **NCCL_DEBUG**: set to `INFO`/`WARN` to get NCCL diagnostics.
- **NCCL_DEBUG_SUBSYS**: restrict debug subsystems.
- **CUDA_VISIBLE_DEVICES**: control which GPUs ranks can see.

Note: some tests intentionally manipulate environment variables (see `conftest.py`) to keep behavior stable across systems.

