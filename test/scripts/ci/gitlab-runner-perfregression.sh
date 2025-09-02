#!/bin/bash

set -e  # Exit on error
source test/scripts/ci/ci-utils.sh

load_test_ci_variables
source $CLUSTER_CONFIG
load_cluster_ci_variables
get_slurm_planned_time

# Use gcperf-tools venv
source ${GCPERF_TOOLS_PATH}/venv/bin/activate

mkdir -p perfregression

# Generate the performance regression job script
echo "Generating job script..."
gcperf-tools generate-job-script \
    --systems-toml ${GCPERF_TOOLS_PATH}/configs/systems.toml \
    --user-toml test/scripts/ci/gcperf-tools/gitlab-runner.toml \
    --testset-toml test/scripts/ci/gcperf-tools/testsuite.toml \
    --system ${CLUSTER_NAME} \
    -o perfregression/test.sbatch

# Submit the job
cd perfregression
echo "Submitting job script..."
sbatch --wait -J "${SLURM_ACCOUNT}-cicd.perf-regression.${CI_COMMIT_BRANCH//\//.}" -t ${SLURM_TIME} test.sbatch
# Convert results to CSV
echo "Converting results to CSV..."
RESULTS_TARBALL=$(ls -1 *.tar.gz)
gcperf-tools convert --input tarball:${RESULTS_TARBALL} --output results.csv --continue-on-error

# TODO: Compare results.csv to baseline.csv
# Report results, then make results.csv the baseline

exit 0
