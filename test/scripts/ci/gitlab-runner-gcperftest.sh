#!/bin/bash
#
# gcperf-tools Performance Test Runner
# Uses gcperf-tools framework for job generation, execution, and result upload.
#
# Supports two modes:
#   - regression (default): Run perf regression tests with comparison and NVBug creation
#   - comprehensive_sweep: Run comprehensive sweeps and upload to dedicated schema
#
# Set PERF_TEST_MODE environment variable to switch modes.
#

set -e
source test/scripts/ci/ci-utils.sh

load_test_ci_variables
source $CLUSTER_CONFIG
load_cluster_ci_variables
get_slurm_planned_time

# Export variables for gcperf-tools variable substitution in TOML configs
# The TOML uses ${INSTALL_DIR}, ${NCCL_HOME}, and ${CUDA_HOME}

# Debug: Show raw values from CI before any modification
echo "Raw CI variables:"
echo "  INSTALL_DIR (from CI): '${INSTALL_DIR}'"
echo "  NCCL_HOME (from CI): '${NCCL_HOME}'"
echo "  CUDA_HOME (from CI): '${CUDA_HOME}'"

# If INSTALL_DIR isn't set, derive it from NCCL_HOME (strip /build suffix)
if [[ -z "$INSTALL_DIR" && -n "$NCCL_HOME" ]]; then
    export INSTALL_DIR="${NCCL_HOME%/build}"
    echo "Derived INSTALL_DIR from NCCL_HOME: ${INSTALL_DIR}"
else
    export INSTALL_DIR=${INSTALL_DIR:-}
    echo "Using INSTALL_DIR from CI: ${INSTALL_DIR}"
fi

# Ensure NCCL_HOME and CUDA_HOME are exported for gcperf-tools
export NCCL_HOME=${NCCL_HOME:-}
export CUDA_HOME=${CUDA_HOME:-}

echo "Variables for gcperf-tools:"
echo "  INSTALL_DIR: ${INSTALL_DIR}"
echo "  NCCL_HOME: ${NCCL_HOME}"
echo "  CUDA_HOME: ${CUDA_HOME}"

run_regression_check() {
    local results_dir="$1"
    local output_file="$2"
    if [ -d "${results_dir}" ] && [ "$(ls -A ${results_dir}/*.csv 2>/dev/null)" ]; then
        # Get the most recent CSV file (sorted by modification time)
        PREVIOUS_RESULT=$(ls -t ${results_dir}/*.csv 2>/dev/null | head -1)
        echo "Most recent previous result: ${PREVIOUS_RESULT}"
        # Run comparison using gcperf-tools
        echo "Running regression check..."
        gcperf-tools regression-check \
            --output ${output_file} \
            --baseline ${PREVIOUS_RESULT} \
            --regression-config ../test/scripts/ci/gcperf-tools/regression.toml \
            results.csv
        return $?
    else
        echo "No previous results found in ${results_dir}"
        echo "Skipping regression check"
        return 0
    fi
}

save_results() {
    local results_dir="$1"
    # Copy results.csv to RESULTS_DIR with date format
    # Get current date in MM_DD_YY format
    DATE_SUFFIX=$(date +%m_%d_%y)
    # Ensure RESULTS_DIR exists
    mkdir -p ${results_dir}
    # Copy and rename results.csv to RESULTS_DIR
    echo "Copying results.csv to ${results_dir}/${DATE_SUFFIX}.csv"
    cp results.csv ${results_dir}/${DATE_SUFFIX}.csv
}

# Use gcperf-tools venv
# TODO: Switch from /venv-next/ to /venv/ once gcperf-tools release is updated with
# https://gitlab-master.nvidia.com/gpucomms/perf-data-tools/-/commits/rc_and_err_improvements
# venv-next is using gcperf-tools manually built from the branch above on each cluster
source ${GCPERF_TOOLS_PATH}/venv-next/bin/activate

# Set gcperf-tools variables
CURRENT_BRANCH="${CI_COMMIT_BRANCH:-${CI_MERGE_REQUEST_SOURCE_BRANCH_NAME:-UNKNOWN}}"
CURRENT_BRANCH="${CURRENT_BRANCH//\//.}"

# Determine run mode: "regression" (default) or "comprehensive_sweep"
PERF_TEST_MODE=${PERF_TEST_MODE:-regression}

# Common paths
SYSTEMS_TOML="test/scripts/ci/gcperf-tools/systems.toml"
USER_TOML="test/scripts/ci/gcperf-tools/gitlab-runner.toml"

# Mode-specific configuration
if [[ $PERF_TEST_MODE == "comprehensive_sweep" ]]; then
    # Comprehensive sweep mode
    OUTDIR="comprehensive_sweep"
    SBATCH_FILE="sweep_${NNODES}n.sbatch"
    # Use TESTSUITE_TOML from environment (required for sweep mode)
    TESTSET_TOML=${TESTSUITE_TOML:?TESTSUITE_TOML must be set for comprehensive_sweep mode}
    DB_SCHEMA="comprehensive_sweep"
    JOB_NAME="${SLURM_ACCOUNT}-cicd.comprehensive-sweep.${CURRENT_BRANCH}.${NNODES}n"
    REPORT_FILE="report_${NNODES}n.pdf"

    # Calculate segment for GB platforms if MAX_SEGMENT is set
    EXTRA_SLURM_ARGS=""
    if [[ -n $MAX_SEGMENT ]]; then
        # For inter-NVLD jobs, ensure at least 2 domains by capping segment at NNODES/2
        MAX_FOR_INTER=$((NNODES / 2))
        if [[ $MAX_FOR_INTER -lt 1 ]]; then
            MAX_FOR_INTER=1
        fi
        EFFECTIVE_MAX=$((MAX_SEGMENT < MAX_FOR_INTER ? MAX_SEGMENT : MAX_FOR_INTER))
        
        # Find largest divisor of NNODES up to EFFECTIVE_MAX
        for S in 18 16 9 8 6 4 3 2 1; do
            if [[ $S -le $EFFECTIVE_MAX ]] && [[ $((NNODES % S)) -eq 0 ]]; then
                NVLD_SIZE="$S"
                break
            fi
        done
        # Fallback if no divisor found
        NVLD_SIZE="${NVLD_SIZE:-1}"
        echo "NNODES=$NNODES, NVLD_SIZE=$NVLD_SIZE (${NNODES}/${NVLD_SIZE}=$((NNODES/NVLD_SIZE)) domains)"
        EXTRA_SLURM_ARGS="--segment=${NVLD_SIZE}"
    fi
else
    # Regression mode (default)
    OUTDIR="perfregression"
    SBATCH_FILE="perfregression.sbatch"
    TESTSET_TOML=${TESTSUITE_TOML:-test/scripts/ci/gcperf-tools/testsuites/testsuite.toml}
    DB_SCHEMA="perf_regression"
    JOB_NAME="${SLURM_ACCOUNT}-cicd.perf-regression.${CURRENT_BRANCH}"
    REPORT_FILE="report.pdf"

    # CI_MERGE_REQUEST_TARGET_BRANCH_NAME will be empty post-merge, so use CURRENT_BRANCH
    TARGET_BRANCH="${CI_MERGE_REQUEST_TARGET_BRANCH_NAME//\//.}"
    COMPARISON_BRANCH="${TARGET_BRANCH:-${CURRENT_BRANCH}}"
    BASELINE_BRANCH="${CI_DEFAULT_BRANCH:-master}"

    # Default values for pipeline variables
    CREATE_NVBUG_ON_FAILURE=${CREATE_NVBUG_ON_FAILURE:-0}
    COMPARE_TO_BASELINE=${COMPARE_TO_BASELINE:-0}

    if [[ $CI_PIPELINE_SOURCE == 'schedule' ]]; then
        # Always create a bug for schedule pipeline failures
        CREATE_NVBUG_ON_FAILURE=1
        # If this is a release branch, also compare to baseline branch
        if [[ $CURRENT_BRANCH =~ ^v[0-9]+\.[0-9]+$ ]]; then
            COMPARE_TO_BASELINE=1
        fi
        # Expand to 64 nodes for weekly pipelines
        if [[ $TRIGGER_PIPELINE == "weekly" ]]; then
            NNODES=${NNODES:-64}
        fi
    fi

    # Check if comparison branch is also baseline branch
    if [[ $COMPARE_TO_BASELINE -eq 1 ]] && [[ $COMPARISON_BRANCH == $BASELINE_BRANCH ]]; then
        echo "Comparison branch is also baseline branch, disabling baseline comparison"
        COMPARE_TO_BASELINE=0
    fi

    # Set results directories
    COMPARISON_RESULTS_DIR=${GCPERF_TOOLS_PATH}/nightly_results/${COMPARISON_BRANCH}/${NNODES}_node
    BASELINE_RESULTS_DIR=${GCPERF_TOOLS_PATH}/nightly_results/${BASELINE_BRANCH}/${NNODES}_node

    # Calculate segment for GB platforms (regression mode only)
    EXTRA_SLURM_ARGS=""
    if [[ $CLUSTER_NAME =~ "PreTyche|Lyris|Bia" ]]; then
        export NVLD_SIZE="1"
        if [[ $NNODES -ge 16 ]]; then
            export NVLD_SIZE="16"
        elif [[ $NNODES -ge 4 ]]; then
            export NVLD_SIZE="4"
        elif [[ $NNODES -ge 2 ]]; then
            export NVLD_SIZE="2"
        fi
        EXTRA_SLURM_ARGS="--segment=${NVLD_SIZE}"
    fi
fi

# Add partition if specified
if [[ -n $SLURM_PARTITION ]]; then
    EXTRA_SLURM_ARGS="${EXTRA_SLURM_ARGS} -p ${SLURM_PARTITION}"
fi

# Default NNODES if not set
NNODES=${NNODES:-4}

echo "================================================"
echo "Performance Test Variables:"
echo "================================================"
echo "PERF_TEST_MODE: ${PERF_TEST_MODE}"
echo "CURRENT_BRANCH: ${CURRENT_BRANCH}"
echo "CLUSTER_NAME: ${CLUSTER_NAME}"
echo "NNODES: ${NNODES}"
echo "TESTSET_TOML: ${TESTSET_TOML}"
echo "DB_SCHEMA: ${DB_SCHEMA}"
echo "EXTRA_SLURM_ARGS: ${EXTRA_SLURM_ARGS}"
if [[ $PERF_TEST_MODE == "regression" ]]; then
    echo "TRIGGER_PIPELINE: ${TRIGGER_PIPELINE}"
    echo "COMPARISON_BRANCH: ${COMPARISON_BRANCH}"
    echo "COMPARISON_RESULTS_DIR: ${COMPARISON_RESULTS_DIR}"
    echo "COMPARE_TO_BASELINE: ${COMPARE_TO_BASELINE}"
    if [[ $COMPARE_TO_BASELINE -eq 1 ]]; then
        echo "BASELINE_BRANCH: ${BASELINE_BRANCH}"
        echo "BASELINE_RESULTS_DIR: ${BASELINE_RESULTS_DIR}"
    fi
fi
echo "================================================"

mkdir -p ${OUTDIR}

set -x

# Generate the job script
echo "Generating job script..."
echo "DEBUG: INSTALL_DIR=${INSTALL_DIR}"
echo "DEBUG: NCCL_HOME=${NCCL_HOME}"
echo "DEBUG: CUDA_HOME=${CUDA_HOME}"
gcperf-tools generate-job-script \
    --systems-toml ${SYSTEMS_TOML} \
    --user-toml ${USER_TOML} \
    --testset-toml ${TESTSET_TOML} \
    --system ${CLUSTER_NAME} \
    -o ${OUTDIR}/${SBATCH_FILE}

# Submit the job
cd ${OUTDIR}
echo "Submitting job script..."
set +e
sbatch --wait --export=ALL --exclusive -N ${NNODES} ${EXTRA_SLURM_ARGS} -J "${JOB_NAME}" -t ${SLURM_TIME} ${SBATCH_FILE}
JOB_EXIT_CODE=$?
if [[ $JOB_EXIT_CODE -ne 0 ]]; then
    echo "Job submission failed with exit code: $JOB_EXIT_CODE"
    cat *.out 2>/dev/null || true  # Dump logs for debugging
    exit 33  # Custom exit code for job submission failure
fi
set -e

# Convert results to CSV
echo "Converting results to CSV..."
RESULTS_TARBALL=$(ls -1 *.tar.gz)
gcperf-tools convert --input tarball:${RESULTS_TARBALL} --output results.csv --continue-on-error

# Generate PDF report of results
echo "Generating PDF report..."
gcperf-tools report -i results.csv -o ${REPORT_FILE}

# Upload results to Postgres DB
echo "Uploading results to Postgres DB (schema: ${DB_SCHEMA})..."
set +e  # Don't fail on upload errors
gcperf-tools -vv upload -P -i tarball:${RESULTS_TARBALL} \
    --db-url postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@swgpu-gpucomms-dev-rw.db.nvidia.com:5432/gpu_comms \
    --db-schema ${DB_SCHEMA} \
    --continue-on-error
UPLOAD_EXIT_CODE=$?
set -e

if [[ $UPLOAD_EXIT_CODE -ne 0 ]]; then
    echo "WARNING: Upload had errors (exit code: $UPLOAD_EXIT_CODE), but continuing..."
fi

# Skip regression checks for comprehensive_sweep mode
if [[ $PERF_TEST_MODE == "comprehensive_sweep" ]]; then
    echo "================================================"
    echo "Comprehensive sweep completed successfully!"
    if [[ $UPLOAD_EXIT_CODE -ne 0 ]]; then
        echo "Note: Some results may not have been uploaded due to parsing errors"
    fi
    echo "Results uploaded to schema: ${DB_SCHEMA}"
    echo "================================================"
    exit 0
fi

# Regression mode: run comparison checks
set +e

echo "Comparing performance to ${COMPARISON_BRANCH}..."
run_regression_check ${COMPARISON_RESULTS_DIR} ${COMPARISON_BRANCH}_comparison.xlsx
REGRESSION_CODE=$?
echo "${COMPARISON_BRANCH} performance regression check completed with exit code: $REGRESSION_CODE"

if [[ $CURRENT_BRANCH == $COMPARISON_BRANCH ]] && [[ $CI_PIPELINE_SOURCE == 'schedule' ]]; then
    echo "Detected on comparison branch, copying results to ${COMPARISON_RESULTS_DIR}"
    save_results ${COMPARISON_RESULTS_DIR}
fi

if [[ $COMPARE_TO_BASELINE -eq 1 ]]; then
    echo "Comparing performance to ${BASELINE_BRANCH}..."
    run_regression_check ${BASELINE_RESULTS_DIR} ${BASELINE_BRANCH}_comparison.xlsx
    BASELINE_REGRESSION_CODE=$?
    echo "${BASELINE_BRANCH} performance regression check completed with exit code: $BASELINE_REGRESSION_CODE"
fi

echo "See job artifacts for detailed reports."

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

create_nvbug() {
    local target_branch=$1

    echo "Creating NvBug for regression compared to ${target_branch}..."
    DATE_STRING=$(date +%m/%d/%y)
    DESCRIPTION="Performance regression detected on $DATE_STRING <br />\n Cluster: $CLUSTER_NAME <br />\n Nodes: $NNODES <br />\n Branch: $CURRENT_BRANCH <br />\n Commit: $CI_COMMIT_SHA <br />\n Compared to: ${target_branch} <br />\n <a href=\\\"$CI_JOB_URL\\\">See job artifacts for more details</a>"
    BUG_JSON="{
        \"BugId\": 0,
        \"BugAction\": {
            \"Value\": \"Dev - Open - To fix\"
        },
        \"Disposition\": {
            \"Value\": \"Open issue\"
        },
        \"IsRestrictedAccess\": 0,
        \"ApplicationDivisionID\": 1,
        \"BugTypeID\": 6,
        \"BugType\": \"Software\",
        \"Priority\": {
            \"Value\": \"Unprioritized\"
        },
        \"Severity\": {
            \"Value\": \"5-Performance\"
        },
        \"Synopsis\": \"NCCL Performance Regression on $CURRENT_BRANCH compared to ${target_branch} - $CLUSTER_NAME - $DATE_STRING\",
        \"Description\": \"$DESCRIPTION\",
        \"ModuleInfo\": {\"Value\": \"GPUComms_DevOps\"},
        \"Origin\": \"Engineering\",
        \"BusinessUnits\": \"Tesla\",
        \"GeographicOrigin\": \"US, CA, Santa Clara\",
        \"Engineer\": \"$GITLAB_USER_EMAIL\",
        \"ARB\": [{\"Value\": \"$GITLAB_USER_EMAIL\"}]
    }"
    RESPONSE=$(curl -s -X POST "https://nvbugsapi.nvidia.com/nvbugswebserviceapi/api/Bug/SaveBug" \
    -H "Authorization: Bearer $NVAUTH_TOKEN" \
    -H "Content-Type: application/json" \
    -d "$BUG_JSON")
    echo "RESPONSE: $RESPONSE"
    BUG_NUMBER=$(echo "$RESPONSE" | grep -o '"ReturnValue":[0-9]*' | cut -d':' -f2)
    echo "Successfully created NvBug: https://nvbugspro.nvidia.com/bug/$BUG_NUMBER"
}

EXIT_CODE=0

# Check target branch regression
if [[ $REGRESSION_CODE -eq 1 || $REGRESSION_CODE -eq 65 || $REGRESSION_CODE -eq 66 || $REGRESSION_CODE -eq 67 ]]; then
    EXIT_CODE=1
    if [[ $CREATE_NVBUG_ON_FAILURE -eq 1 ]]; then
        create_nvbug "$COMPARISON_BRANCH"
    fi
fi

# Check baseline branch regression
if [[ $COMPARE_TO_BASELINE -eq 1 ]]; then
    if [[ $BASELINE_REGRESSION_CODE -eq 1 || $BASELINE_REGRESSION_CODE -eq 65 || $BASELINE_REGRESSION_CODE -eq 66 || $BASELINE_REGRESSION_CODE -eq 67 ]]; then
        EXIT_CODE=1
        if [[ $CREATE_NVBUG_ON_FAILURE -eq 1 ]]; then
            create_nvbug "$BASELINE_BRANCH"
        fi
    fi
fi

exit $EXIT_CODE
