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
OUTDIR="perfregression"
SBATCH_FILE="perfregression.sbatch"
SYSTEMS_TOML="${GCPERF_TOOLS_PATH}/configs/systems.toml"
USER_TOML="test/scripts/ci/gcperf-tools/gitlab-runner.toml"
TESTSET_TOML="test/scripts/ci/gcperf-tools/testsuite.toml"
RESULTS_DIR=${GCPERF_TOOLS_PATH}/nightly_results/${CI_COMMIT_BRANCH//\//.}

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
sbatch --wait -J "${SLURM_ACCOUNT}-cicd.perf-regression.${CI_COMMIT_BRANCH//\//.}" -t ${SLURM_TIME} ${SBATCH_FILE}

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
    gcperf-tools regression-check --output regression_report.xlsx --baseline ${PREVIOUS_RESULT} results.csv
    REGRESSION_CODE=$?
else
    echo "No previous results found in ${RESULTS_DIR}"
    echo "Skipping regression check"
fi

# Copy results.csv to RESULTS_DIR with date format
# Get current date in MM_DD_YY format
DATE_SUFFIX=$(date +%m_%d_%y)
# Ensure RESULTS_DIR exists
mkdir -p ${RESULTS_DIR}
# Copy and rename results.csv to RESULTS_DIR
echo "Copying results.csv to ${RESULTS_DIR}/${DATE_SUFFIX}.csv"
cp results.csv ${RESULTS_DIR}/${DATE_SUFFIX}.csv

echo "Performance regression check completed with exit code: $REGRESSION_CODE"
echo "See job artifacts for more detailed results"

# TODO: Handle different exit codes from gcperf-tools regression-check
# Different codes will correspond to different levels of regressions
# exit $REGRESSION_CODE
exit 0
