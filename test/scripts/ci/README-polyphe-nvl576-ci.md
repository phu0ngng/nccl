# Polyphe NVL576 CI (cross-clique & scale)

## Native rack cliques

On NVL576, `NCCL_MNNVL_CLIQUE_ID=-2` uses rack ID as clique boundary (NVL36 racks). Cross-clique P2P requires `NCCL_MNNVL_CROSS_CLIQUE=1` and **more than one rack in the job** (typically **>36 GPUs** in one NVLD).

The Theia jobs use `cross-clique-wrapper.sh` to *emulate* cliques; Polyphe jobs use **no wrapper** — real fabric topology.

## Jobs (see `.gitlab-ci.yml`)

| Job | When | Placement | GPUs |
|-----|------|-----------|------|
| `perftest-polyphe-nvl576-cross-clique-16n` | **pre-submit MR** (every commit), plus nightly/weekly / `RUN_POLYPHE_NVL576_CI=1` | `--segment=8` | 64 |
| `perftest-polyphe-nvl576-multi-nvld-emulated-16n` | **pre-submit MR** (every commit), plus nightly/weekly / `RUN_POLYPHE_NVL576_CI=1` | `--segment=8` | 64 (2×32 emulated NVLDs) |
| `perftest-polyphe-nvl576-cross-clique-512` | weekly or `RUN_POLYPHE_NVL576_MEGA=1` | `--segment=8` | 512 |
| `perftest-polyphe-nvl576-cross-clique-576` | weekly or `RUN_POLYPHE_NVL576_MEGA=1` | `--segment=9` | 576 |
| `perftest-polyphe-nvl576-multi-nvld-weekly` | `POLYPHE_NVL576_MULTI_NVLD_WEEKLY=1` + weekly | site-specific | 256 (default) |

## Schedules

- **Nightly** pipelines use `TRIGGER_PIPELINE=nightly` (GitLab default schedule).
- **Weekly** scale jobs (512/576): set `TRIGGER_PIPELINE=weekly` on a **weekly** schedule, or run a pipeline with that variable.
- **512/576 on the default nightly schedule:** set project variable `POLYPHE_NVL576_SCALE_ON_NIGHTLY=1` for the scheduled pipeline (runs both 512 and 576 jobs each night — use only if the cluster can sustain that load).
- **Ad-hoc mega:** `RUN_POLYPHE_NVL576_MEGA=1` on any pipeline (long runtime, large allocation).

## Multi-NVLD (multiple NVLink domains)

Single NVLD = one fabric UUID across all ranks. **Multi-NVLD** = job spans pods with different UUIDs; cross-NVLD traffic uses IB.

### Emulated multi-NVLD (per-commit)

**`perftest-polyphe-nvl576-multi-nvld-emulated-16n`** uses `cross-clique-wrapper.sh` to assign `NCCL_MNNVL_UUID` per node, emulating 2 NVLDs on the same 16-node allocation. This runs the `CROSS_CLIQUE_EMULATED` section of `gitlab-runner-perf.sh` at **4× Theia's multi-NVLD scale** (64 GPUs vs 16).

| Variable | Default | Effect |
|----------|---------|--------|
| `NVLDS` | `2` | Number of emulated NVLink domains |
| `CLIQUE_SIZE` | `2` | Nodes per clique (4 cliques per NVLD with 8 nodes/NVLD) |
| `MAX` | `8G` | Max message size for device-path sweeps |

Cross-NVLD traffic routes through the network plugin (not `NCCL_NET_PLUGIN=none`); intra-NVLD cross-clique traffic uses NVLink P2P.

### True multi-NVLD (weekly, requires Slurm support)

GitLab cannot express arbitrary Slurm hetero jobs portably. Enable **`perftest-polyphe-nvl576-multi-nvld-weekly`** only after defining a Slurm allocation that places ranks on **≥2 NVLDs** (e.g. hetjob, multi-constraint). Override via CI variables:

- `POLYPHE_NVL576_MULTI_NVLD_WEEKLY=1`
- Optional: `POLYPHE_MULTI_NVLD_NNODES`, `POLYPHE_MULTI_NVLD_SEGMENT`, `SLURM_EXTRA_SALLOC_FLAGS` (if your runner exports custom salloc)

Until then, leave `POLYPHE_NVL576_MULTI_NVLD_WEEKLY` unset; the job stays skipped.

## `--segment`

Polyphe uses `--segment` so node placement aligns with rack boundaries (`8` or `9` per site policy). Adjust in `.gitlab-ci.yml` if the batch system changes.

## Perf matrix (16n full suite)

- **Slurm wall:** `SLURM_TIME=60` (minutes) on `perftest-polyphe-nvl576-cross-clique-16n` (raise in CI if queue + runtime exceeds that).
- **Device API / NVLS path:** sweep `-b 8` … **`MAX`** (default in CI **`8G`**) with `-f 2`; override via job variable `MAX` (e.g. `32G` for a heavier sweep).
- **CE (`-G 0`) collectives:** `-b 128` … **`CROSS_CLIQUE_CE_MAX`** (default `32G` for 16n, `gitlab-runner-perf.sh` `CROSS_CLIQUE_NATIVE` path).
- **512/576 quick jobs** (`POLYPHE_NVL576_QUICK=1`): unchanged smaller caps (1G / 2G) for wall time.
