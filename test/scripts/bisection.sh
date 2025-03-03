#!/bin/bash
# Ignore errors
set +e

help () {
    echo "
Easily find an offending regression between an initial and final commit using NCCL perftests using bisection.

Usage:
    ./bisection.sh <initial_commit> <final_commit> <repro> <threshold> <ONLY_FUNCS> <iterations> <NCCL_TMP_DIR> <NCCL_SRC_DIR> <load_checkpoint>

Required Params:
    <initial_commit> - Git ref
    <final_commit>   - Git ref
    <repro>          - A string encapsulating the perftest command to test. It's advised to reduce this to as small of a set of message sizes as possible to reduce false negatives

Optional Params:
    <threshold>    - Percentage, as an int. Default is 15
    <iterations>   - How many outer loop test iterations. Default is 10
    <ONLY_FUNCS>   - Parameter to pass to each make command. Use this to reduce build step times (see NCCL docs.) Default is blank
    <NCCL_TMP_DIR> - Path to staging directory for all builds and perf data. Default is ~. Using a high performance filesystem is recommended. Must be accessible by all mpi workers (/tmp probably won't work for internode runs)
    <NCCL_SRC_DIR> - Path to source NCCL directory, default is ~/nccl
    <load_checkpoint>   - If 1, this script will resume a prior bisection. Useful if a slurm session times out, or if it's easier to get multiple short allocations.

Example:
    - salloc -N8 -t60 -J coreai_libraries_nccl-bisection:8x8 -A coreai_libraries_nccl -pluna ./bisection.sh v2.17.1-1 v2.18.3-1 "alltoall_perf -d half -b4M -e4M -w5 -n20 -f2" 15 10 "SendRecv"
    - sbatch bisection_selene.sub
    - sbatch bisection_preos.sub
"
}

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# CLI args
initial_commit=$1
if [ "$initial_commit" == "" ]; then
    echo "No initial_commit specified, exiting"
    help
    exit 1
else
    echo "initial_commit=$initial_commit"
fi

shift
final_commit=$1
if [ "$final_commit" == "" ]; then
    echo "No final_commit specified, exiting"
    help
    exit 1
else
    echo "final_commit=$final_commit"
fi

shift
repro="$1"
if [ "$repro" == "" ]; then
    echo "No repro specified, exiting"
    help
    exit 1
else
    echo "repro=$repro"
fi

shift
threshold="$1"
if [ "$threshold" == "" ]; then
    threshold=15
fi
echo "threshold=$threshold%"

shift
ONLY_FUNCS="$1"
if [[ "$ONLY_FUNCS" == "" || "$ONLY_FUNCS" == "-" ]]; then
    ONLY_FUNCS=""
fi
echo "ONLY_FUNCS=$ONLY_FUNCS"

shift
iterations=$1
if [[ "$iterations" == "" || "$iterations" == "-" ]]; then
    iterations=10
fi
echo "iterations=$iterations"

# This must be an accessible NFS path
shift
NCCL_TMP_DIR="$1"
if [[ "$NCCL_TMP_DIR" == "" || "$NCCL_TMP_DIR" == "-" ]]; then
    NCCL_TMP_DIR=~
fi
NCCL_TMP_DIR=$NCCL_TMP_DIR/bisection-$initial_commit-to-$final_commit
echo "NCCL_TMP_DIR=$NCCL_TMP_DIR"

shift
NCCL_SRC_DIR="$1"
if [[ "$NCCL_SRC_DIR" == "" || "$NCCL_SRC_DIR" == "-" ]]; then
    NCCL_SRC_DIR=~/nccl
fi
echo "NCCL_SRC_DIR=$NCCL_SRC_DIR"

shift
load_checkpoint=$1
if [[ "$load_checkpoint" == "" || "$load_checkpoint" == "-" ]]; then
    load_checkpoint=0
fi
echo "load_checkpoint=$load_checkpoint"

if [ "$NVCC_GENCODE" == "" ] && command -v nvidia-smi &> /dev/null; then
    gencode_flags=""

    for arch in $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -Vu | sed 's/\.//g'); do
        gencode_flags+="-gencode=arch=compute_${arch},code=sm_${arch} "
    done

    export NVCC_GENCODE=$gencode_flags
fi
echo "NVCC_GENCODE=$NVCC_GENCODE"

# Copy NCCL to staging dir to avoid altering src repo
if [ $load_checkpoint -eq 0 ]; then
    echo "Deleting $NCCL_SRC_DIR/build*"
    rm -rf $NCCL_SRC_DIR/build*
    echo "Deleting $NCCL_TMP_DIR"
    rm -rf $NCCL_TMP_DIR
    echo "Copying $NCCL_SRC_DIR to $NCCL_TMP_DIR"
    mkdir -p $NCCL_TMP_DIR
    cp -r $NCCL_SRC_DIR $NCCL_TMP_DIR
fi
NCCL_TMP_DIR=$NCCL_TMP_DIR/nccl

# Use the tmp nccl repo as a working directory
echo "Moving to $NCCL_TMP_DIR"
pushd $NCCL_TMP_DIR
git reset --hard

git rev-list --first-parent --oneline $initial_commit..$final_commit -- src/* > commits-oneline.txt
echo "Performing a bisection over the following set of commits:"
nl -b a commits-oneline.txt

git rev-list --first-parent --abbrev-commit $initial_commit..$final_commit -- src/* > commits.txt
count=$(git rev-list --first-parent --count $initial_commit..$final_commit -- src/*)
first=0
last=$count

d=build-0-$initial_commit
if [ $load_checkpoint -eq 0 ]; then
    # Build initial
    git checkout $initial_commit 2> /dev/null
    make clean > /dev/null
    echo "$(date) Building $initial_commit: $(git show --oneline -s)"
    make -j test.build BUILDDIR=$d MPI=1 ONLY_FUNCS="$ONLY_FUNCS" > /dev/null 2>&1
    git show --oneline -s > $d/commit.txt
fi

TEST_HOME=$d/test/perf
rm "$d/$repro.txt" 2> /dev/null

# Collect initial perf data (event with load_checkpoint)
if test -f $d/lib/libnccl.so; then
    echo "$(date) Collecting initial_commit performance data"
    echo "$(date) Running $repro on $initial_commit for $iterations iters"
    # Run the test
    for i in $(seq 1 $iterations)
    do
        NCCL_HOME=$d \
        LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH \
        $MPI_HOME/bin/mpirun -q --bind-to numa $TEST_HOME/$repro >> "$d/$repro.txt"
    done
else
    echo "$(date) NCCL build for $initial_commit failed to build. Exiting"
    exit -1
fi

commit=""

if [ $load_checkpoint -eq 0 ]; then
    commit_num_high=$last
    commit_num_low=$first
    last_regression=-1
else
    commit_num_high=$(cat load_checkpoint-high.txt)
    commit_num_low=$(cat load_checkpoint-low.txt)
    last_regression=$(cat load_checkpoint-last.txt)
    echo "Loaded commit_num_low=$commit_num_low commit_num_high=$commit_num_high last_regression=$last_regression from checkpoint"
fi

while [ $commit_num_low -ne $commit_num_high ]; do
    # Get the ceiling mid point (X + Y -1) / Y => (high + low + 2 - 1) / 2 => (high + low + 1) / 2
    let commit_num_mid=($commit_num_high + $commit_num_low + 1)/2

    # Get a good build
    while [ 1 ] ; do
        commit=$(head -$commit_num_mid commits.txt | tail -1 )
        git checkout $commit 2> /dev/null
        d="build-$commit_num_mid-$commit"
        echo "$(date) Checking out and building commit#$commit_num_mid out of $last: $(git show --oneline -s)"
        make -j BUILDDIR=$d ONLY_FUNCS="$ONLY_FUNCS" > /dev/null 2>&1
        git show --oneline -s > $d/commit.txt
        if ! test -f $d/lib/libnccl.so; then
            echo "$(date) NCCL lib not found, failed to build. Trying $commit_num_mid+1"
            let commit_num_mid=($commit_num_mid+1)
        else
            break
        fi
    done

    # Run the test
    echo "$(date) Running $repro on $commit for $iterations iters"
    for i in $(seq 1 $iterations)
    do
        NCCL_HOME=$d \
        LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH \
        $MPI_HOME/bin/mpirun -q --bind-to numa $TEST_HOME/$repro >> "$d/$repro.txt"
    done

    ${BASE_DIR}/perf_regression.py old="build-0-$initial_commit/$repro.txt" new="$d/$repro.txt" threshold=$threshold rmad_threshold=0.2
    regression=$?
    if [ $regression -eq 0 ]; then
        # The final commit is the first line, so to search closer to the present, decrease the number
        # If no regression, select the midpoint between 1 and midpoint and try again
        let commit_num_high=($commit_num_mid-1)
        echo "$(date) No regression detected, low=$commit_num_low high=$commit_num_high"
    fi
    if [ $regression -eq 1 ]; then
        # The initial commit is the one after the last line, so to search closer to the past, increase the number
        # If no regression, select the midpoint between midpoint and final and try again
        last_regression=$commit_num_mid
        commit_num_low=$commit_num_mid
        echo "$(date) Regression detected, low=$commit_num_low high=$commit_num_high"
    fi
    if [ $regression -gt 1 ]; then
        let commit_num_high=($commit_num_mid-1)
        echo "$(date) Mismatched cases. Test failed to run, low=$commit_num_low high=$commit_num_high"
    fi

    # Checkpoint
    echo $last_regression > load_checkpoint-last.txt
    echo $commit_num_low  > load_checkpoint-low.txt
    echo $commit_num_high > load_checkpoint-high.txt
done

# $last_regression is storing the last seen regression
if [ $last_regression -eq -1 ]; then
    echo "No regressions found between $initial_commit and $last_commit"
else
    let commit_num_mid=$last_regression
    commit=$(head -$commit_num_mid commits.txt | tail -1 )
    echo "$(date) Checking out commit#$commit_num_mid out of $last, hash=$commit"
    git checkout $commit 2> /dev/null
    echo "$(date) Suspected regression:"
    git show --oneline -s
    d="build-$commit_num_mid-$commit"
    # Show this regression data one last time
    ${BASE_DIR}/perf_regression.py old="build-0-$initial_commit/$repro.txt" new="$d/$repro.txt" threshold=$threshold rmad_threshold=0.2
fi

popd
