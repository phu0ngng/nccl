#!/bin/bash

set -e  # Exit on error
source test/scripts/ci/ci-utils.sh

load_test_ci_variables
source $CLUSTER_CONFIG
load_cluster_ci_variables
get_slurm_planned_time

# Use gcperf-tools venv
source ${GCPERF_TOOLS_PATH}/venv/bin/activate

# Set gcperf-tools variables
GOLDEN_BRANCH="master"
CURRENT_BRANCH="${CI_COMMIT_BRANCH//\//.}"
OUTDIR="perfregression"
SBATCH_FILE="perfregression.sbatch"
SYSTEMS_TOML="test/scripts/ci/gcperf-tools/systems.toml"
USER_TOML="test/scripts/ci/gcperf-tools/gitlab-runner.toml"
TESTSET_TOML="test/scripts/ci/gcperf-tools/testsuite.toml"

# Set nodes based on pipeline type
NNODES=4
if [[ $TRIGGER_PIPELINE == "weekly" ]]; then
    NNODES=64
fi
echo "Running perf regression with ${NNODES} nodes"

# Set results directory
RESULTS_DIR=${GCPERF_TOOLS_PATH}/nightly_results/${GOLDEN_BRANCH//\//.}/${NNODES}_node

EXTRA_SLURM_ARGS=""
if [[ $CLUSTER_NAME == "PreTyche" ]]; then
    if [[ $NNODES -ge 16 ]]; then
        EXTRA_SLURM_ARGS="--segment 16"
    elif [[ $NNODES -ge 4 ]]; then
        EXTRA_SLURM_ARGS="--segment 4"
    elif [[ $NNODES -ge 2 ]]; then
        EXTRA_SLURM_ARGS="--segment 2"
    fi
    # Single node (NNODES == 1) gets no segment arg
fi

mkdir -p ${OUTDIR}

# Generate the performance regression job script
echo "Generating job script..."
gcperf-tools generate-job-script \
    --systems-toml ${SYSTEMS_TOML} \
    --user-toml ${USER_TOML} \
    --testset-toml ${TESTSET_TOML} \
    --system ${CLUSTER_NAME} \
    -o ${OUTDIR}/${SBATCH_FILE}

# Submit the job
cd perfregression
echo "Submitting job script..."
sbatch --wait -N ${NNODES} ${EXTRA_SLURM_ARGS} -J "${SLURM_ACCOUNT}-cicd.perf-regression.${CURRENT_BRANCH}" -t ${SLURM_TIME} ${SBATCH_FILE}

# Convert results to CSV
echo "Converting results to CSV..."
RESULTS_TARBALL=$(ls -1 *.tar.gz)
gcperf-tools convert --input tarball:${RESULTS_TARBALL} --output results.csv --continue-on-error

# Generate PDF report of results
echo "Generating PDF report..."
gcperf-tools report -i results.csv -o report.pdf

# Upload results to Postgres DB
echo "Uploading results to Postgres DB..."
gcperf-tools upload -i tarball:${RESULTS_TARBALL} \
    --db-url postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@swgpu-gpucomms-dev-rw.db.nvidia.com:5432/gpu_comms \
    --db-schema perf_regression

set +e
# Grab the most recent result from RESULTS_DIR if it exists
PREVIOUS_RESULT=""
REGRESSION_CODE=0
if [ -d "${RESULTS_DIR}" ] && [ "$(ls -A ${RESULTS_DIR}/*.csv 2>/dev/null)" ]; then
    # Get the most recent CSV file (sorted by modification time)
    PREVIOUS_RESULT=$(ls -t ${RESULTS_DIR}/*.csv 2>/dev/null | head -1)
    echo "Most recent previous result: ${PREVIOUS_RESULT}"
    # Run comparison using gcperf-tools
    echo "Running regression check..."
    gcperf-tools regression-check \
        --output regression_report.xlsx \
        --baseline ${PREVIOUS_RESULT} \
        --regression-config ../test/scripts/ci/gcperf-tools/regression.toml \
        results.csv
    REGRESSION_CODE=$?
else
    echo "No previous results found in ${RESULTS_DIR}"
    echo "Skipping regression check"
fi

if [[ $CURRENT_BRANCH == $GOLDEN_BRANCH ]] && [[ $TRIGGER_PIPELINE != "pre-submit" ]]; then
    echo "Detected on golden branch, copying results to ${RESULTS_DIR}"
    # Copy results.csv to RESULTS_DIR with date format
    # Get current date in MM_DD_YY format
    DATE_SUFFIX=$(date +%m_%d_%y)
    # Ensure RESULTS_DIR exists
    mkdir -p ${RESULTS_DIR}
    # Copy and rename results.csv to RESULTS_DIR
    echo "Copying results.csv to ${RESULTS_DIR}/${DATE_SUFFIX}.csv"
    cp results.csv ${RESULTS_DIR}/${DATE_SUFFIX}.csv
fi

echo "Performance regression check completed with exit code: $REGRESSION_CODE"
echo "See job artifacts for more detailed results"

# Handle different exit codes from gcperf-tools regression-check
# Exit with 1 if REGRESSION_CODE is in [1, 66, 67], otherwise exit with 0
# 
# Exit codes from gcperf-tools regression-check:
# 0  - SUCCESS: No regressions detected
# 1  - ERROR: No matching test configurations found between baseline and candidate datasets
# 64 - LOW_REGRESSIONS: Low severity regressions
# 65 - MEDIUM_REGRESSIONS: Medium severity regressions
# 66 - HIGH_REGRESSIONS: High severity regressions
# 67 - CRITICAL_REGRESSIONS: Critical performance degradations
if [[ $REGRESSION_CODE -eq 1 || $REGRESSION_CODE -eq 66 || $REGRESSION_CODE -eq 67 ]]; then
    exit 1
else
    exit 0
fi
