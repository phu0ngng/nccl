# NCCL4Py unit tests

These are **pure Python** tests that validate NCCL4Py’s Python-layer behavior without requiring the NCCL runtime library.

They focus on:

- typing/type system
- buffer/spec validation
- API logic and error paths
- lightweight mocks for bindings/CUDA

## Requirements

- Python dependencies:
  - `nccl4py`
  - `ml-dtypes`
  - `pytest`

No MPI runtime and no NCCL runtime library are required for this suite.

## How to run

### Run all unit tests

```bash
python -m pytest -q nccl4py/tests/unit_tests
```

### Run a single file

```bash
python -m pytest -q nccl4py/tests/unit_tests/test_utils_unit.py
```

### Run a single test (nodeid)

```bash
python -m pytest -q nccl4py/tests/unit_tests/test_utils_unit.py::test_ndarray_mutation_reflects_in_bytes
```

