# DeepEP-v2 CI

Build + run DeepEP sanity tests in one SLURM allocation. Same script for CI and manual repro.

## Run

```bash
NCCL_INSTALL_DIR=/abs/path/to/nccl-build \
DEEP_EP_INSTALL_DIR=/abs/path/for/venv \
./run.sh <cluster> [test ...]
```

`<cluster>` is `eos`, `theia`, or `ptyche`. Omit test names to run the cluster's full set.

Set `SKIP_BUILD=1` to reuse an existing venv + DeepEP at `$DEEP_EP_INSTALL_DIR` (useful for iterative manual debug — skips uv/pip/clone/`develop.sh`).

## Files

| File | Role |
|---|---|
| `run.sh`     | Entry point. Validates env, computes test set, `salloc → bash _alloc.sh`. |
| `_alloc.sh`  | In-allocation driver (runs on compute master). Builds, then loops tests. |
| `_build.sh`  | Per-build body. Invoked via `srun -N1`. |
| `_test.sh`   | Per-task test body. Invoked via `srun --mpi=pmix`. |
| `clusters.sh`| Per-cluster config + per-(cluster, test) overrides + skip list. |
| `tests.sh`   | Test catalog: `ALL_TESTS`, `GLOBAL_SKIP_TESTS`, `set_test_config`, `test_set_env`. |
| `_pretty.sh` | banner / section / pass / fail / summary helpers. |

## How to edit

| To... | Edit | Where |
|---|---|---|
| Add a cluster                       | `clusters.sh` | new branch in `apply_cluster_config` |
| Skip a test on a cluster            | `clusters.sh` | that cluster's `SKIP_TESTS=(...)` |
| Skip a test everywhere              | `tests.sh`    | add to `GLOBAL_SKIP_TESTS=(...)` |
| Add a per-(cluster, test) override  | `clusters.sh` | new branch in `apply_overrides` |
| Add a test                          | `tests.sh`    | append to `ALL_TESTS` + new branch in `set_test_config` |
| Add a per-test env var              | `tests.sh`    | one `test_set_env VAR=value` line in that test's branch |
| Add a cluster-wide env var          | `clusters.sh` | one `cluster_set_env VAR=value` line in that cluster's branch |

Use `test_set_env` for per-test env (auto-cleared between tests, surfaced in the test's `Extra env:` line) and `cluster_set_env` for cluster-wide env (set once per allocation, surfaced in the banner's `Cluster env:` line). Both export the var **and** record it for the log header — don't `export` directly.

## Per-cluster env (test runtime)

| Cluster | `NCCL_IB_HCA` | `NCCL_IB_DATA_DIRECT` | OMPI / PMIX |
|---|---|---|---|
| eos    | (unset)                                | (unset) | `OMPI_MCA_coll=^hcoll`, `OMPI_MCA_btl=tcp,self`, `PMIX_MCA_gds=hash` |
| theia  | `mlx5_0,mlx5_1,mlx5_2,mlx5_3`          | `0`     | same as eos |
| ptyche | `mlx5_0,mlx5_1,mlx5_4,mlx5_5`          | (unset) | same as eos |

Build-time pins (CUDA toolkit + torch wheel):

| Cluster | `CUDA_HOME` | `TORCH_INDEX_URL` |
|---|---|---|
| eos    | `cuda-12.9` | `https://download.pytorch.org/whl/cu129` |
| theia  | `cuda-13.2` | (PyPI default — cu13) |
| ptyche | `cuda-13.2` | (PyPI default — cu13) |

## Per-test env (per (cluster, test))

Test-level env (effective env at `srun` time, *on top of* the cluster-wide env above). Sources:
1. **Test default** — `test_set_env` calls inside `tests.sh` `set_test_config`, applied on every cluster.
2. **Cluster override** — `test_set_env` calls inside `clusters.sh` `apply_overrides`, applied for a specific `(cluster, test)` pair.

`(none)` = no test-level env beyond cluster-wide. Empty rows are placeholders for future tests/overrides.

| Cluster × Test | sanity_ep | sanity_engram† |
|---|---|---|
| eos    | (none) | `NCCL_CROSS_NIC=0` |
| theia  | (none) | `NCCL_CROSS_NIC=0` |
| ptyche | (none) | `NCCL_CROSS_NIC=0` |

† globally skipped today (see `GLOBAL_SKIP_TESTS` in `tests.sh`); row reflects what would apply if re-enabled.
