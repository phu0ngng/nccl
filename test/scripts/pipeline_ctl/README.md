# pipeline_ctl

Interactive CLI for triggering NCCL GitLab CI/CD pipelines. Replaces the old bash-based `pipeline_ctl` with a Python TUI that walks you through branch selection, pipeline mode, test/cluster configuration, and confirmation before launching.

## Quick Start

```bash
# From the NCCL repo root
./test/scripts/pipeline_ctl/run.sh
```

The `run.sh` entrypoint handles everything automatically:
1. Creates a Python virtual environment (`.venv/`) on first run
2. Installs dependencies (`requests`, `simple-term-menu`)
3. Launches the interactive tool

Subsequent runs reuse the existing venv and skip installation.

## Prerequisites

- **Python 3.10+**
- **`GITLAB_API_TOKEN`** environment variable set to a [GitLab personal access token](https://gitlab-master.nvidia.com/-/user_settings/personal_access_tokens) with `api` scope

## Pipeline Modes

| Mode | Description |
|------|-------------|
| **Custom** | Interactively select individual tests and clusters to run |
| **Pre-submit** | Standard validation suite run before merging |
| **Post-submit** | Extended suite run after merging to master |
| **Nightly** | Full nightly regression suite |
| **Weekly** | Comprehensive weekly sweep |

For all modes except **Custom**, the tool sets `TRIGGER_PIPELINE` to the selected mode and lets the CI configuration handle test/cluster selection.

## Custom Mode

When **Custom** is selected, interactive multi-select menus let you pick specific tests and clusters:

### Available Tests

| Menu Label | CI Variable |
|------------|-------------|
| Unit Tests | `RUN_UNIT_TESTS` |
| API Tests | `RUN_API_TESTS` |
| Perf Single | `RUN_PERF_TESTS_SINGLE` |
| Perf Multi | `RUN_PERF_TESTS_MULTIPLE` |
| Perf Regression | `RUN_PERF_REGRESSION` |
| Resiliency | `RUN_RESILIENCY_TESTS` |
| DLFW | `RUN_DLFW_TESTS` |
| Profiler | `RUN_PROFILER_TESTS` |
| Inspector | `RUN_INSPECTOR_TESTS` |
| NCCL EP | `RUN_NCCL_EP_CI` |
| NCCL4PY | `RUN_NCCL4PY_CI` |

### Available Clusters

| Menu Label | CI Variable |
|------------|-------------|
| IPP6 | `RUN_ON_IPP6` |
| EOS | `RUN_ON_EOS` |
| Pre-Nyx | `RUN_ON_PRE_NYX` |
| Pre-Tyche | `RUN_ON_PRE_TYCHE` |
| Theia | `RUN_ON_THEIA` |
| AWS | `RUN_ON_AWS` |
| DGX Spark | `RUN_ON_DGXSPARK` |
| Draco OCI | `RUN_ON_DRACO_OCI` |
| BIA | `RUN_ON_BIA` |
| Polyphe | `RUN_ON_POLYPHE` |
| Windows | `RUN_ON_WINDOWS` |
| P100 (GC) | `RUN_ON_P100` |
| P40 (GC) | `RUN_ON_P40` |
| GP100 (GC) | `RUN_ON_GP100` |

Use **Space** to toggle selections, **Enter** to confirm, or select **All** at the top to enable everything.

## Menu Controls

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate |
| `Space` | Toggle selection (multi-select menus) |
| `Enter` | Confirm |
| `q` | Quit |

## Confirmation

Before launching, a summary is displayed with the selected branch, mode, tests, and clusters. You can then:
- **Launch pipeline** -- triggers the pipeline via the GitLab API
- **Dry run** -- prints the JSON payload that would be sent without launching
- **Back to start** -- re-enter selections from the beginning
- **Quit** -- exit without launching

## File Structure

```
pipeline_ctl/
├── README.md           # This file
├── run.sh              # Bash entrypoint (venv setup + launch)
├── pipeline_ctl.py     # Python TUI tool
└── requirements.txt    # Python dependencies
```
