# NCCL Inspector Kibana Integration Wrapper (Internal)

This is an internal wrapper that extends the public NCCL Inspector Performance Summary Exporter with internal integrations:

- **pyslurm**: For retrieving SLURM job details
- **nvdataflow**: For uploading data to Kibana/Elasticsearch

## Architecture

The wrapper follows a clean separation of concerns:

1. **Core Functionality**: All data analysis, visualization, and processing is handled by the public exporter
2. **Internal Integrations**: This wrapper adds SLURM and Kibana capabilities on top
3. **Reusability**: The public exporter can be used independently without internal dependencies

## File Structure

```
ext-profiler/inspector-internal/elastic/
├── kibana_wrapper.py          # Main wrapper script
├── requirements.txt           # Internal dependencies
├── kibana_config.ini         # Internal configuration
└── README.md                 # This file
```

## Dependencies

### Public Dependencies
- pandas, tqdm, duckdb, matplotlib, pyarrow, numpy

### Internal Dependencies (Internal Only)
- pyslurm: For SLURM job information
- nvdataflow: For Kibana data upload

## Installation

### 1. Install Public Dependencies

Create Virtual Env:

```bash
# Create virtual environment
python3 -m venv venv
# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
```

### 2. Install Internal Dependencies

```bash
pip install -r requirements.txt
```

**Note**: pyslurm and nvdataflow are internal NVIDIA dependencies and are not available in public repositories.

#### 2.2 Install PySLURM

Install pyslurm from [pyslurm-prebuilds](https://gitlab-master.nvidia.com/fact/pyslurm-prebuilds)

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/fact/pyslurm-prebuilds.git
pip3 install ./pyslurm-prebuilds/pyslurm-23.2.2-cp310-cp310-linux_x86_64.whl
```

#### 2.3 Install nvdataflow

Install nvdataflow from NVIDIA artifactory:

```bash
python3 -m pip install --index-url=https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-gpuwa-pypi/simple nvdataflow
```

### 2.3 Verify Installation

Verify core dependencies:

```bash
python -c "import pandas, tqdm, duckdb, matplotlib, pyslurm, nvdataflow; print('All dependencies installed successfully')"
```

## Usage

### Basic Usage

```bash
python kibana_wrapper.py --jobid <job_id> --upload
```

### Command Line Arguments

- `--jobid <job_id>`: **Required**. SLURM job ID to process
- `--upload`: **Optional**. Upload results to Kibana
- `--config <path>`: **Optional**. Path to configuration file (default: uses internal config)

### Example

```bash
# Process job 12345 and upload to Kibana
python kibana_wrapper.py --jobid 12345 --upload

# Process job 12345 without uploading (for testing)
python kibana_wrapper.py --jobid 12345
```

## How It Works

1. **Import Public Exporter**: The wrapper imports functions from the public exporter
2. **Extend with Internal Features**: Adds pyslurm and nvdataflow capabilities
3. **Reuse Core Logic**: All data processing, visualization, and analysis uses the public exporter
4. **Add Internal Integrations**: SLURM job details and Kibana uploads are handled separately

## Configuration

The wrapper uses an internal configuration file (`kibana_config.ini`) that includes:

- Log root directory path
- Kibana project names for detailed and summary data

## Output

The wrapper produces the same output as the public exporter, plus:

- Kibana data uploads (when `--upload` is specified)
- Enhanced logging with SLURM job information

## Error Handling

- **Missing Dependencies**: Clear error messages if pyslurm or nvdataflow are not available
- **SLURM Failures**: Graceful fallback to default values if SLURM queries fail
- **Kibana Failures**: Logged errors but processing continues

## Development

### Adding New Internal Features

1. **Keep Core Logic in Public Exporter**: All data processing should remain in the public version
2. **Extend in Wrapper**: Add internal integrations here
3. **Maintain Separation**: Don't mix public and internal functionality

### Testing

```bash
# Test without internal dependencies
cd ext-profiler/inspector/exporter/elastic
python perf_summary_exporter_public.py --jobid <test_job>

# Test with internal dependencies
cd ext-profiler/inspector-internal/elastic
python kibana_wrapper.py --jobid <test_job> --upload
```

## Security Notes

- This wrapper contains internal configuration and dependencies
- Do not commit this directory to public repositories
- Keep Kibana project names and internal paths internal

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the public exporter is available and in the Python path
2. **Missing Internal Dependencies**: Install pyslurm and nvdataflow from internal sources
3. **Configuration Issues**: Check that `kibana_config.ini` exists and is properly formatted

### Debug Mode

Add logging to see detailed information about the wrapper's operation:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Support

For issues with the internal integrations, create an nvbug and please post on Slack channel #nccl-inspector.
For issues with core functionality, refer to the public exporter documentation.
