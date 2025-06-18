#!/bin/bash

# Associate array
declare -A commands
declare -A command_times
declare -A command_stdouts
declare -A command_retcodes
declare -A failed_commands

function function_exists() {
    declare -F "$1" > /dev/null
    return $?
}

function get_failed_dir() {
    label="$1"
    echo "failed/$CI_JOB_NAME/$label"
}

# Transform a space-separated list of env vars ("k1=v1 k2=v2...") into format needed by target command:
# - For mpirun: "-x k1=v1 -x k2=v2..."
# - For srun: "k1=v1,k2=v2,..."
function transform_env_vars() {
    local env_vars="$1"
    local target="$2"

    local result=""

    OLD_IFS=$IFS
    IFS=' '

    if [ "$target" = "mpirun" ]; then
        for var in $env_vars; do
            if [ -z "$result" ]; then
                result="-x $var"  # No space before the first element
            else
                result+=" -x $var"  # Add a space before subsequent elements
            fi
        done
    elif [ "$target" = "srun" ]; then
        for var in $env_vars; do
            if [ -z "$result" ]; then
                result="$var"  # No comma before the first element
            else
                result+=",${var}"  # Add a comma before subsequent elements
            fi
        done
    elif [ "$target" = "cmd" ]; then
        for var in $env_vars; do
            if [ -z "$result" ]; then
                result="env ${var}"  # No space before the first element
            else
                result+=" env ${var}"  # Add a space before subsequent elements
            fi
        done
    fi

    IFS=$OLD_IFS

    echo "$result"
}

# Options=CMD,SALLOC_MPI,MPI,SRUN_MPI
function make_run_command() {
    run_mode="$1"
    ppn="$2"
    test_mpi_flags="$3"
    test_env_vars="$4"

    local mpi_flags="$MPI_PARAMS $test_mpi_flags"
    local env_vars="$test_env_vars LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

    if [ "$run_mode" = "SALLOC_MPI" ]; then
        run_mode_cmd="salloc -N ${NNODES} --ntasks-per-node ${NGPUS} -t ${SLURM_TIME} --exclusive"
        if [ "$SLURM_PARTITION" != "" ]; then
            run_mode_cmd+=" -p $SLURM_PARTITION"
        fi
        if [ "$SLURM_HOSTS" != "" ]; then
            run_mode_cmd+=" -w $SLURM_HOSTS"
        fi
        if [ "$MPIRUN_SKIP_PPN" != "1" ] || [ "$ppn" == "1" ]; then
            mpi_flags+=" --map-by ppr:$ppn:node"
        fi
        transformed_env_vars=$(transform_env_vars "$env_vars" "mpirun")
        run_mode_cmd+=" mpirun $mpi_flags $transformed_env_vars"
        echo $run_mode_cmd
    elif [ "$run_mode" = "MPI" ]; then
        if [ "$MPIRUN_SKIP_PPN" != "1" ] || [ "$ppn" == "1" ]; then
            mpi_flags+=" --map-by ppr:$ppn:node"
        fi
        transformed_env_vars=$(transform_env_vars "$env_vars" "mpirun")
        run_mode_cmd="mpirun $mpi_flags $transformed_env_vars"
        echo $run_mode_cmd
    elif [ "$run_mode" = "SRUN_MPI" ]; then
        # Deliberately ignore any MPI flags passed as they should have been set in the test env already
        transformed_env_vars=$(transform_env_vars "$env_vars" "srun")
        run_mode_cmd="srun --export=ALL,$transformed_env_vars --ntasks-per-node=$ppn --mpi=pmix"
        echo $run_mode_cmd
    elif [ "$run_mode" = "CMD" ]; then
        # Directly pass through space-delimited env_vars
        transformed_env_vars=$(transform_env_vars "$env_vars" "cmd")
        run_mode_cmd="$transformed_env_vars"
        echo $run_mode_cmd
    fi
}

to_seconds() {
    IFS=: read -r h m s <<< "$1"
    echo $(( 10#$h*3600 + 10#$m*60 + 10#$s ))
}

# Init junit command
init_junit_command() {
    : ${JUNIT:="junit_results.xml"}
    local label=$1
    echo "<testcase name=\"$label\" start_time=\"$(date +%s%N)\"" >> "$JUNIT"
}

get_slurm_planned_time() {
    if [ "$PLANNED_RESERVED" != "Skip" ] && [ ! -z "$SLURM_JOB_ID" ]; then
        output=$(sacct -X -j $SLURM_JOB_ID --format "JobName%100,User%10,Partition%15,NNodes,Timelimit,$PLANNED_RESERVED")
        runtime=$(to_seconds $(sacct -X -j $SLURM_JOB_ID --format "$PLANNED_RESERVED" | awk 'NR==3 {print $1}'))
        label="slurm_planned_time"
        init_junit_command $label
        command_times+=(["$label"]="$runtime")
        commands+=(["$label"]="sacct -X -j $SLURM_JOB_ID --format \"JobName%100,User%10,Partition%15,NNodes,Timelimit,$PLANNED_RESERVED\"")
        command_retcodes+=(["$label"]="0")
        command_stdouts+=(["$label"]="$output")
        complete_junit_command $label
        echo "$PLANNED_RESERVED for $runtime seconds"
    else
        echo "Skipping get_slurm_planned_time()"
    fi
}

complete_junit_command() {
    local label=$1
    local command="${commands[$label]}"
    local time="${command_times[$label]}"
    local return_code="${command_retcodes[$label]}"
    local stdout="${command_stdouts[$label]}"
    local failure=""
    local status=""
    : ${JUNIT:="junit_results.xml"}

    # Determine test case status
    if [ "$return_code" -eq 0 ]; then
        status="passed"
    else
        status="failed"
        failed_dir="$(get_failed_dir $label)"
        first_failed_log=$(ls -t $failed_dir/*.out | head -n 1)
        raw_log_output=$(tail -n 500 "$first_failed_log")
        failure="<failure>$(echo "$raw_log_output" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g; s/"/\&quot;/g; s/'"'"'/\&#39;/g')</failure>"
    fi
    system_out="<system-out>$(echo "$stdout" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g; s/"/\&quot;/g; s/'"'"'/\&#39;/g')</system-out>"
    # Complete the test case
    sed -i '$s|$| time="'"$time"'" status="'"$status"'">|' $JUNIT
    echo "$system_out" >> $JUNIT
    echo "$failure"    >> $JUNIT
    echo "</testcase>"   >> $JUNIT
}

init_junit_file() {
    : ${JUNIT:="junit_results.xml"}
    rm $JUNIT
    # Create the XML file with header
    cat << EOF > "$JUNIT"
<?xml version="1.0" encoding="UTF-8"?>
<testsuite>
EOF
}

detect_malformed_testcase_tags() {
    : ${JUNIT:="junit_results.xml"}
    local error_count=0

    # Use grep to find lines with <testcase but without >
    while IFS= read -r line_info; do
        local line_num=$(echo "$line_info" | cut -d: -f1)
        local line_content=$(echo "$line_info" | cut -d: -f2-)
        echo "Error: Malformed testcase element found at line $line_num:"
        echo "$line_content"
        ((error_count++))
    done < <(grep -n "<testcase[^>]*$" "$JUNIT")

    if [ $error_count -eq 0 ]; then
        echo "All testcase elements are properly formed."
        return 0
    else
        echo "Total malformed testcase elements found: $error_count"
        return 1
    fi
}

end_junit_file() {
    : ${JUNIT:="junit_results.xml"}
    local total_time=0
    local failed_count=0
    local total_tests=0

    if [ ! -f "$JUNIT" ] || [ ! -r "$JUNIT" ]; then
        echo "Error: File does not exist or is not readable JUNIT=$JUNIT"
        return 1
    fi

    while IFS= read -r line; do
        if [[ $line =~ \<testcase[[:space:]] ]]; then
            # Extract time if it exists
            if [[ $line =~ time=\"([0-9]+(\.[0-9]+)?)\" ]]; then
                time="${BASH_REMATCH[1]}"
                total_time=$(echo "$total_time + $time" | bc)
            fi

            # Check status if it exists
            if [[ $line =~ status=\"failed\" ]]; then
                ((failed_count++))
            fi

            ((total_tests++))
        fi
    done < "$JUNIT"

    # Update testsuite attributes and close the tag
    sed -i 's/.*<testsuite>/<testsuite name="'"$CI_JOB_NAME"'" tests="'"$total_tests"'" failures="'"$failed_count"'" time="'"$total_time"'">/' $JUNIT
    echo '</testsuite>' >> "$JUNIT"
    echo "JUnit XML file created: $JUNIT"
}

has_testsuite_closing_tag() {
    : ${JUNIT:="junit_results.xml"}

    # Check if file exists and is readable
    if [ ! -f "$JUNIT" ] || [ ! -r "$JUNIT" ]; then
        echo "Error: File does not exist or is not readable"
        return 1
    fi

    # Use grep to search for </testsuite> tag
    if grep -q "</testsuite>" "$JUNIT"; then
        echo "The XML file contains the </testsuite> closing tag."
        return 0
    else
        echo "The XML file does not contain the </testsuite> closing tag."
        return 1
    fi
}

function move_repro_script() {
    echo "  Made reproducer script for $label"
    failed_dir="$(get_failed_dir $label)"
    mkdir -p $failed_dir
    mv repro.sh $failed_dir
    chmod +x $failed_dir/repro.sh
    if [ "$SALLOC" != "" ]; then
        mv run.sh $failed_dir
        chmod +x $failed_dir/run.sh
    fi
}

function generate_repro_script() {
    label=$1
    rm repro.sh
    echo "#!/bin/bash" > repro.sh
    run_file=repro.sh
    if [ "$SALLOC" != "" ]; then
        rm run.sh
        failed_dir="$(get_failed_dir $label)"
        echo "$SALLOC $failed_dir/run.sh" >> repro.sh
        run_file=run.sh
        echo "#!/bin/bash" >> $run_file
    fi

    echo "set -x" >> $run_file
    echo "set +e" >> $run_file
    echo "NCCL_HOME=$NCCL_HOME" >> $run_file
    echo "NNODES=$NNODES" >> $run_file
    echo "NGPUS=$NGPUS" >> $run_file
    echo "SLURM_TIME=$SLURM_TIME" >> $run_file
    echo "SLURM_PARTITION=$SLURM_PARTITION" >> $run_file
    echo "RUN_MODE=$RUN_MODE" >> $run_file
    echo "export NCCL_DEBUG=WARN" >> $run_file
    echo "CLUSTER_CONFIG=$CLUSTER_CONFIG" >> $run_file
    echo "source $CLUSTER_CONFIG" >> $run_file
    echo "source test/scripts/ci/ci-utils.sh" >> $run_file
    echo "load_cluster_ci_variables" >> $run_file
    echo "load_test_ci_variables" >> $run_file
    echo "configure_test_env" >> $run_file
    echo "echo \"Running $label\"" >> $run_file
    echo "${commands[$label]}" >> $run_file
}

function parse_store_coredumps() {
    binary=$1
    label=$2
    libnccl=$NCCL_HOME/lib/libnccl.so
    find . -type f -name "core.*" -not -path "./failed/*" -not -path "./src/*" -not -path "./ext-net/*" | while read -r file; do
        echo "Processing file: $file"
        failed_dir="$(get_failed_dir $label)"
        mkdir -p $failed_dir/cores
        base_name=$(basename $file)
        # Run once on the binary and again on libnccl in case the binary is the problem
        cuda-gdb -batch -ex "info threads" -ex "thread apply all bt" $binary $file > $failed_dir/cores/bin.$base_name.txt
        cuda-gdb -batch -ex "info threads" -ex "thread apply all bt" $libnccl $file > $failed_dir/cores/libnccl.$base_name.txt
        mv $file $failed_dir/cores
    done
}

# Add a label and command line
function run_command() {
    label="$1"
    run_mode="$2"
    ppn="$3"
    test_mpi_flags="$4"
    test_env_vars="$5"
    binary="$6"
    args="$7"
    cmd="$binary $args"
    if [ "$ONLY_RUN_LABEL" != "" ] && [ "$label" != "$ONLY_RUN_LABEL" ]; then
            echo "Skipping label: $label (Matching for $ONLY_RUN_LABEL)"
    else
        run_mode_cmd=$(make_run_command $run_mode $ppn "$test_mpi_flags" "$test_env_vars")
        echo "$(date +%T) : "=============================== $label "==============================="
        echo "$(date +%T) : $run_mode_cmd $cmd"
        init_junit_command $label
        commands+=(["$label"]="$run_mode_cmd $cmd")
        generate_repro_script $label
        start=$(date +%s%N)
        stdout=$($run_mode_cmd $cmd 2>&1)
        ret=$?
        end=$(date +%s%N)
        echo "$stdout"
        let runtime=$((end - start))/1000000000
        echo "Command took $runtime s with retcode $ret"
        if [ $ret -ne 0 ]; then
            failed_commands+=(["$label"]="$run_mode_cmd $cmd")
            rerun_failed_command_with_logging $label "$run_mode_cmd $cmd"
            parse_store_coredumps $binary $label
            move_repro_script $label
        fi
        command_times+=(["$label"]="$runtime")
        command_retcodes+=(["$label"]="$ret")
        command_stdouts+=(["$label"]="$stdout")
        complete_junit_command $label
    fi
}

function print_command_times() {
    if [ ${#command_times[@]} -ne 0 ]; then
        echo "Command Times:"
        for label in "${!command_times[@]}"; do
            echo "  $label: ${command_times[$label]} ms"
        done
    fi
}

function print_failed_commands() {
    if [ ${#failed_commands[@]} -ne 0 ]; then
        echo "Repros:"
        for label in "${!failed_commands[@]}"; do
            echo "  $label: ${failed_commands[$label]}"
            failed_dir="$(get_failed_dir $label)"
            echo "    Repro: bash -c ${failed_dir}/repro.sh"
        done
    fi

    echo "To download artifacts: ./test/scripts/ci/get_pipeline_artifacts.sh $CI_PIPELINE_ID"
}

function rerun_failed_command_with_logging() {
    label=$1
    cmd=$2
    ORIG_DEBUG=$NCCL_DEBUG
    ORIG_SUBSYS=$NCCL_DEBUG_SUBSYS
    ORIG_FILE=$NCCL_DEBUG_FILE
    failed_dir="$(get_failed_dir $label)"
    mkdir -p $failed_dir
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=ALL
    export NCCL_DEBUG_FILE="${label}.%h.%p.out"
    echo "Rerunning $label with logging: $cmd"
    output=$($cmd)
    echo "$output"
    mv *.out $failed_dir
    echo "Moved logs $NCCL_DEBUG_FILE to $failed_dir"
    if [ "$ORIG_DEBUG" != "" ]; then
      export NCCL_DEBUG=$ORIG_DEBUG
    else
      unset NCCL_DEBUG
    fi
    if [ "$ORIG_SUBSYS" != "" ]; then
      export NCCL_DEBUG_SUBSYS=$ORIG_SUBSYS
    else
      unset NCCL_DEBUG_SUBSYS
    fi
    if [ "$ORIG_FILE" != "" ]; then
      export NCCL_DEBUG_FILE=$ORIG_FILE
    else
      unset NCCL_DEBUG_FILE
    fi
}

function load_cluster_ci_variables() {
    configure_test_env
    CUDA_HOME=$(get_cuda_home)
    MPI_HOME=$(get_openmpi_home)
    LD_LIBRARY_PATH=$(get_extra_ld_library_path):${LD_LIBRARY_PATH}
    MPI_PARAMS=$(get_mpi_params)
    PATH=$(get_extra_path):$PATH

    # Optional functions
    if function_exists get_nccl_socket_ifname; then
        export NCCL_SOCKET_IFNAME=$(get_nccl_socket_ifname)
    fi
    if function_exists get_slurm_account; then
        SLURM_ACCOUNT=$(get_slurm_account)
    fi
    if function_exists get_nccl_ib_sl; then
        export NCCL_IB_SL=$(get_nccl_ib_sl)
    fi
    if function_exists get_planned_reserved; then
        PLANNED_RESERVED=$(get_planned_reserved)
    fi

    export NCCL_PROFILER_PLUGIN=none

    echo "CUDA_HOME=$CUDA_HOME"
    echo "PLANNED_RESERVED=$PLANNED_RESERVED"
    echo "MPI_HOME=$MPI_HOME"
    echo "Cluster LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "MPI_PARAMS=$MPI_PARAMS"
    echo "PATH=$PATH"
    echo "MPIRUN_SKIP_PPN=$MPIRUN_SKIP_PPN"
}

function ci_exit() {
    exit ${#failed_commands[@]}
}

function load_test_ci_variables() {
    echo "CLUSTER_CONFIG: $CLUSTER_CONFIG"
    echo "NCCL_HOME: $NCCL_HOME"
    echo "NNODES: $NNODES"
    echo "NP: $NP"
    echo "NGPUS: $NGPUS"
    echo "SLURM_TIME: $SLURM_TIME"
    echo "SLURM_PARTITION: $SLURM_PARTITION"
    echo "NCCL_DEBUG: $NCCL_DEBUG"
    echo "RUN_MODE: $RUN_MODE"
    echo "SALLOC: $SALLOC"
    LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
    echo "Test LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    export NCCL_DEBUG=$NCCL_DEBUG
}

function build-pytorch-dlfw-container() {
    docker build -f docker/dockerfiles/Dockerfile.dlfw.pytorch -t pytorch-image .
}
