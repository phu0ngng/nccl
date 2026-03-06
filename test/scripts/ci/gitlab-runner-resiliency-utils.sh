#!/bin/bash

# ============================== Description ===================================
# Common utilities for NCCL resiliency tests (failover and recovery).
# This file is sourced by the test scripts and provides shared functions.
# ==============================================================================

# Enable strict error handling:
#   -e: Exit immediately if a command exits with a non-zero status
#   -u: Treat unset variables as an error
#   -o pipefail: Return value of a pipeline is the status of the last command
#                to exit with a non-zero status, or zero if no command exited
#                with a non-zero status
set -eo pipefail

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CURRENT_DIR/ci-utils.sh"

# Validate required environment variables
# Arguments:
#   $@ - Additional variable names to validate (optional)
#        Example: 
#        validate_environment VAR1 VAR2 - checks "base" vars and both additional 
#        vars VAR1 and VAR2
# Returns 0 if all required variables are set, exits with 1 otherwise
validate_environment() {
    local required_vars=("PORT_FAILURE_EMU_BIN" "RUN_MODE")
    # Append any additional variable names passed as arguments
    required_vars+=("$@")
    local missing=0
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo "ERROR: $var is not set"
            missing=1
        fi
    done
    
    if [ $missing -eq 1 ]; then
        exit 1
    fi
}

# Get the first node from SLURM_NODELIST for running port failure emulation
get_port_failure_emu_node() {
    scontrol show hostnames ${SLURM_NODELIST} | head -1
}

# Setup common NCCL parameters for resiliency tests
# Arguments:
#   $1 - Enable port recovery (1 = yes, 0 = no)
# Returns: NCCL parameters string via stdout
setup_nccl_common_params() {
    local enable_recovery=${1:-0}
    local params=""
    
    params+="NCCL_DEBUG=${NCCL_DEBUG:-warn} "
    params+="NCCL_DEBUG_SUBSYS=NET "
    # To force the use of network transport for all communications
    params+="NCCL_P2P_DISABLE=1 "
    params+="NCCL_SHM_DISABLE=1 "
    params+="NCCL_IB_ADAPTIVE_ROUTING=1 "
    if [ -n "${NCCL_IB_HCA:-}" ]; then
        params+="NCCL_IB_HCA=${NCCL_IB_HCA} "
        params+="NCCL_NET_FORCE_MERGE=${NCCL_NET_FORCE_MERGE} "
    fi
    # Enable the prerequisites for port failover
    params+="NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=1 "
    params+="NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1 "
    # To reduce the time NCCL waits before detecting a failure
    params+="NCCL_IB_TIMEOUT=1 "
    params+="NCCL_IB_RETRY_CNT=1 "
    # Enable port failover feature
    params+="NCCL_IB_RESILIENCY_PORT_FAILOVER=1 "
    
    # Add port recovery parameters if enabled
    if [ "$enable_recovery" -eq 1 ]; then
        params+="NCCL_IB_RESILIENCY_PORT_RECOVERY=1 "
        local SEC=1000
        params+="NCCL_IB_RESILIENCY_PORT_RECOVERY_ATTEMPTS_MAX=1000 "
        params+="NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_INTERVAL=100 "
        params+="NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_TIMEOUT=$((1*SEC)) "
    fi
    
    echo "$params"
}

# Setup extra NCCL parameters based on AR threshold
# Arguments:
#   $1 - AR threshold value
# Returns: Extra NCCL parameters string via stdout
setup_nccl_ar_params() {
    local ar_threshold=$1
    local params=""
    
    params+="NCCL_IB_AR_THRESHOLD=$ar_threshold "
    if [ "$ar_threshold" -eq 2147483647 ]; then
        params+="NCCL_PROTO=simple "
        params+="NCCL_IB_QPS_PER_CONNECTION=4 "
    fi
    
    # Force NCCL to use the internal plugin that supports port failover
    # and port recovery. So, for example NCCL SHARP plugin which is loaded by
    # HPCX on some clusters is not used.
    params+="NCCL_NET_PLUGIN=none "

    echo "$params"
}

# Setup common NCCL test parameters
# Arguments:
#   $1 - Number of run cycles
# Returns: NCCL test parameters string via stdout
setup_nccl_test_params() {
    local run_cycles=${1:-100}
    local params=""
    
    params+="--minbytes 32MB " # -b
    params+="--maxbytes 32MB " # -e
    params+="--run_cycles $run_cycles " # -N
    params+="--stepfactor 2 " # -f
    params+="--warmup_iters 0 " # -w
    params+="--check 1 " # -c
    #params+="--datatype all " # -d
    #params+="--iters 5 " # -n
    #params+="--nthreads 1 " # -t
    #params+="--ngpus 1 " # -g
    echo "$params"
}

# Start NCCL perf test in background
# Arguments:
#   $1 - Function name (e.g., all_reduce, alltoall)
#   $2 - NCCL parameters
#   $3 - NCCL test parameters
#   $4 - Log file path
#   $5 - Name of variable to store PID (nameref)
# Note: Uses nameref to avoid subshell issues with wait
start_nccl_perf_test() {
    local func=$1
    local nccl_params=$2
    local test_params=$3
    local log_file=$4
    local -n pid_ref=$5
    local run_mode=${RUN_MODE:-""}
    local ppn=${NGPUS:-1}

    local cmd=""
    # $run_mode = SRUN_MPI
    # $ppn 
    # "$test_mpi_flags" 
    # "$test_env_vars"
    cmd+=$(make_run_command $run_mode $ppn "$mpi_params" "$nccl_params")
    cmd+=" ${NCCL_HOME}/test/perf/${func}_perf ${test_params} "
    echo "Running: ${cmd} > $log_file 2>&1 &"
    ${cmd} 2>&1 | tee "$log_file" &
    pid_ref=$!
    echo "NCCL perf test PID: $pid_ref"
}

# Wait for NCCL test to run for a specified number of cycles
# Arguments:
#   $1 - Log file path to monitor
#   $2 - Number of cycles to wait for (default: 5)
#   $3 - Timeout in seconds (default: 30)
#   $4 - Timeout in NCCL cycles. If NCCL completed this number of cycles - exit (default: 50).
wait_for_nccl_cycles() {
    local log_file=$1
    local num_cycles=${2:-5}
    local timeout=${3:-30}
    local cycle_timeout=${4:-50}
    local elapsed=0

    # Validate that cycle_timeout is larger than num_cycles
    if [ "$cycle_timeout" -le "$num_cycles" ]; then
        echo "ERROR: cycle_timeout ($cycle_timeout) must be larger than num_cycles ($num_cycles)"
        exit 1
    fi

    echo "Waiting for NCCL test to run for $num_cycles cycles before launching port failure emulation"
    while true; do
        local count
        # Match lines that look like NCCL perftest result rows (start with number, end with number)
        count=$(grep -E '^\s*[0-9]+.*[0-9]+\s*$' "$log_file" 2>/dev/null | wc -l) || count=0
        if [ "$count" -gt "$cycle_timeout" ]; then
            echo "ERROR: NCCL test already finished $count cycles (timeout: $cycle_timeout)"
            exit 1
        fi
        if [[ "$count" -lt "$num_cycles" ]]; then
            echo "NCCL test running, cycles finished: $count (<$num_cycles). Waiting to start port failure emulation."
            sleep 1
            elapsed=$((elapsed + 1))
            if [[ "$elapsed" -ge "$timeout" ]]; then
                echo "ERROR: Timeout waiting for NCCL test to reach $num_cycles cycles"
                exit 1
            fi
            continue
        fi
        echo "NCCL test running, cycles finished: $count (>=$num_cycles). Proceeding to launch port failure emulation."
        break
    done
}

# Wait for NCCL test process to finish
# Arguments:
#   $1 - Process ID to wait for
#   $2 - Timeout in seconds (default: 120)
# Returns: Process exit code, or 124 if timeout reached
wait_for_nccl_finish() {
    local pid=$1
    local timeout=${2:-120}  # default 120 seconds
    local elapsed=0

    while kill -0 "$pid" 2>/dev/null; do
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "Timeout reached waiting for process $pid. Killing process."
            kill -TERM "$pid" 2>/dev/null
            # Optionally, wait a bit and force kill if still alive
            sleep 2
            kill -KILL "$pid" 2>/dev/null
            return 124
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    wait "$pid"
    return $?
}

# Start port failure emulation on the specified node
# Arguments:
#   $1 - Node to run on
#   $2 - NIC to fail
#   $3 - Log file path
#   $4 - Name of variable to store local SSH PID (nameref)
# Note: Uses nameref to avoid subshell issues with wait
# Note: Uses SSH instead of srun to avoid resource contention with concurrent srun commands
# Exits with 1 if binary not found
start_port_failure_emulation() {
    local node=$1
    local nic=$2
    local log_file=$3
    local -n ssh_pid_ref=$4
    
    # Check if port failure emulation binary exists
    if [ ! -f "${PORT_FAILURE_EMU_BIN}" ]; then
        echo "ERROR: Port failure emulation binary not found"
        exit 1
    fi
    
    echo "Launching port failure emulation binary via SSH on node ($node)"
    
    # Use SSH to launch port failure emulation binary directly
    # This avoids SLURM resource contention when running concurrent srun commands
    local ssh_cmd="ssh $node ${PORT_FAILURE_EMU_BIN} -d ${nic}"
    echo "Running: $ssh_cmd"
    $ssh_cmd > "$log_file" 2>&1 &
    ssh_pid_ref=$!
    echo "Port failure emulation PID: $ssh_pid_ref"
    
    # Let the port failure emulation binary start up
    sleep 2
}

# Verify port failure emulation is running on the specified node
# Arguments:
#   $1 - Node to check
#   $2 - PID of the process
#   $3 - Max retries (default: 3)
# Returns: 0 if running, exits with 1 if not running after retries
verify_port_failure_emulation() {
    local node=$1
    local ssh_pid=$2
    local max_retries=${3:-3}
    local binary_name=$(basename "${PORT_FAILURE_EMU_BIN}")
    
    echo "Verifying port failure emulation is running on node ($node)"
    
    for ((i=1; i<=max_retries; i++)); do
        # First check if the local SSH PID is still alive - if not, no point checking remote
        if ! kill -0 ${ssh_pid} 2>/dev/null; then
            echo "ERROR: SSH process (PID: $ssh_pid) has died"
            exit 1
        fi
        
        # Use pgrep to get PID - if we get a valid PID, process is running
        local emu_pid=$(ssh $node "pgrep ${binary_name}" 2>/dev/null) || emu_pid=""
        if [ -n "${emu_pid}" ]; then
            echo "Port failure emulation process found running on node (Remote PID: $emu_pid)"
            return 0
        fi
        
        if [ $i -lt $max_retries ]; then
            echo "Attempt $i/$max_retries: Process not found yet, retrying in 1 second..."
            sleep 1
        fi
    done
    
    echo "ERROR: Port failure emulation process not found on node after $max_retries attempts"
    exit 1
}

# Stop port failure emulation process on the specified node
# Arguments:
#   $1 - Node where process is running
#   $2 - Local SSH PID to wait for
# Returns: 0 on success, exits with 1 if process not found when expected
stop_port_failure_emulation() {
    local node=$1
    local ssh_pid=$2
    local expect_running=${3:-1}  # 1 = expect process to be running, 0 = ok if not
    
    local binary_name=$(basename "${PORT_FAILURE_EMU_BIN}")
    
    echo "Sending TERM signal to port failure emulation process on node: $node"
    
    # Use pgrep to check if process is running and get remote PID
    local emu_pid=$(ssh $node "pgrep ${binary_name}" 2>/dev/null) || emu_pid=""
    if [ -n "$emu_pid" ]; then
        echo "Port failure emulation PID on node $node: $emu_pid. Sending TERM signal"
        ssh $node "kill -TERM ${emu_pid}" 2>/dev/null
        echo "Sent TERM signal to ${binary_name}"
        # Give the process time to handle the TERM signal gracefully
        sleep 2
        
        # Verify the process has terminated using kill -0 on the known PID
        if ssh $node "kill -0 ${emu_pid}" 2>/dev/null; then
            echo "WARNING: Port failure emulation process is still running after TERM signal (PID: $emu_pid)"
            echo "Attempting force kill with SIGKILL..."
            ssh $node "kill -KILL ${emu_pid}" 2>/dev/null
            sleep 1
            
            # Verify SIGKILL worked
            if ssh $node "kill -0 ${emu_pid}" 2>/dev/null; then
                echo "ERROR: Failed to kill port failure emulation process (PID: $emu_pid) even with SIGKILL"
                exit 1
            fi
            echo "Port failure emulation process killed with SIGKILL"
        else
            echo "Port failure emulation process has terminated gracefully"
        fi
    else
        echo "Port failure emulation process already terminated or not found"
        if [ "$expect_running" -eq 1 ]; then
            exit 1
        fi
    fi
    
    echo "Waiting for port failure emulation local SSH process (PID=${ssh_pid}) to exit."
    wait $ssh_pid
    echo "Port failure emulation local SSH process (PID=${ssh_pid}) is not running anymore."
}

# Print test logs
# Arguments:
#   $1 - NCCL test log file
#   $2 - Port failure emulation log file
print_test_logs() {
    local nccl_log=$1
    local emu_log=$2
    
    echo ">>>>>>>>>>>>>>>>>>>>>> NCCL test log >>>>>>>>>>>>>>>>>>>>>>"
    cat "$nccl_log"
    
    echo ">>>>>>>>>>>>>>>>>>>>>> Port failure emulation log >>>>>>>>>>>>>>>>>>>>>>"
    cat "$emu_log"

    echo ">>>>>>>>>>>>>>>>>>>>>> End of logs >>>>>>>>>>>>>>>>>>>>>>"
}

# Print test summary
# Arguments:
#   $1 - Failure count
#   $@ - Array of failure names (passed as remaining arguments)
print_test_summary() {
    local failure_count=$1
    shift
    local failure_names=("$@")
    
    echo "======== Summary ========"
    
    for str in "${failure_names[@]}"; do
        echo "Failed Test: $str"
    done
    
    echo "$failure_count tests failed"
}

# Record test result
# Arguments:
#   $1 - Test result (exit code)
#   $2 - Function name
#   $3 - Test type (failover/recovery)
#   $4 - AR threshold
#   $5 - Reference to failure_count variable name
#   $6 - Reference to failure_names array name
record_test_result() {
    local result=$1
    local func=$2
    local test_type=$3
    local ar_threshold=$4
    local -n count_ref=$5
    local -n names_ref=$6
    
    if [ $result -ne 0 ]; then
        ((count_ref++))
        names_ref+=("$func $test_type test (AR threshold: $ar_threshold)")
        echo "NCCL perf $func test failed (AR threshold: $ar_threshold)"
    else
        echo "NCCL perf test completed successfully"
        echo "Test passed"
    fi
}
