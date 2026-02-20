#!/bin/bash

# Source CI utilities
source test/scripts/ci/ci-utils.sh

# Load CI environment variables
load_test_ci_variables
source $CLUSTER_CONFIG
load_cluster_ci_variables
init_junit_file
get_slurm_planned_time

# Set test name for logging
export TEST_NAME="examples-validation"

# Function to run a single example test
function run_example_test() {
    local example_path="$1"

    # Build the example
    make -C "${example_path}" clean > /dev/null 2>&1

    logfile="$(mktemp -p ${PWD})"
    runfile="$(mktemp -p ${PWD})"

    # Use run_command to build on the compute node since their architecture does not always match login node
    # Use separate script to avoid parallel builds on multiple machines
    echo "#/bin/bash" > "${runfile}"
    echo "if \[ \${SLURM_PROCID:-0} -eq 0 \]; then make -C ${example_path} | tee ${logfile}; fi" >> "${runfile}"
    run_command "build_${example_path/\//_}" "$RUN_MODE" 1 "" \
                "CUDA_HOME=${CUDA_HOME} MPI_HOME=${MPI_HOME} NCCL_HOME=${NCCL_HOME}" \
                bash "${runfile}"
    example_name=$(awk '/Built target /{print $3}' "${logfile}" | head -n1)
    rm -f "${logfile}" "${runfile}"

    # Run the example

    # Adjust GPU count based on example requirements
    case "$example_name" in
        "one_device_per_process_mpi")
            # MPI example - skip if not in MPI mode
            if [[ "$RUN_MODE" != *"MPI"* ]]; then
                echo "SKIP: MPI example requires MPI mode"
                return 0
            fi
            ;;
        *device_api)
            # Advanced examples need at least 2 GPUs
            if [[ $NGPUS -lt 2 ]]; then
                echo "SKIP: Example requires at least 2 GPUs"
                return 0
            fi
            ;;
    esac

    # Use run_command from ci-utils.sh, just like profiler/perf/unit tests
    # run_command "label" "run_mode" "ppn" "test_mpi_flags" "test_env_vars" "binary" "args"
    local example_binary="$(pwd)/$example_path/$example_name"
    local label="example_${example_path/\//_}"
    local ppn=1  # Processes per node

    # Adjust for MPI example
    if [[ "$example_name" == "one_device_per_process_mpi" ]]; then
        ppn=$NGPUS  # One process per GPU for MPI example
    fi

    # Call run_command which handles all the salloc/mpirun logic
    run_command "$label" "$RUN_MODE" $ppn "" "" "$example_binary" ""
}

# Convert NCCL_HOME to absolute path if it's relative
if [[ ! "$NCCL_HOME" = /* ]]; then
    NCCL_HOME="$(pwd)/$NCCL_HOME"
fi

# Check if we should run only basic tests
example_dirs=(
    "01_communicators/01_multiple_devices_single_process"
    "02_point_to_point/01_ring_pattern"
    "03_collectives/01_allreduce"
)

if [[ "${EXAMPLES_TESTS_BASIC_ONLY}" != "1" ]]; then
    example_dirs+=(
        "01_communicators/02_one_device_per_pthread"
        "01_communicators/03_one_device_per_process_mpi"
        "04_user_buffer_registration/01_allreduce"
    )
    if [ "$SYMMETRIC" == "1" ];
    then
        example_dirs+=(
            "05_symmetric_memory/01_allreduce"
            "06_device_api/01_allreduce_lsa"
            "06_device_api/02_alltoall_gin"
            "06_device_api/03_alltoall_hybrid"
        )
    fi
fi

# Run each example
pushd docs/examples > /dev/null
for example_dir in "${example_dirs[@]}"; do
    if [[ -d "$example_dir" ]]; then
        run_example_test "$example_dir"
    else
        echo "WARNING: Example directory not found: $example_dir"
    fi
done

print_failed_commands
end_junit_file
ci_exit
