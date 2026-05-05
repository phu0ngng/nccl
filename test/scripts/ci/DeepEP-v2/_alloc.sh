#!/bin/bash
#
# In-allocation driver for the DeepEP-v2 CI. Run by run.sh as
# `salloc ... bash _alloc.sh`, so this body runs on the compute master.
#
# Required env: CLUSTER, SCRIPT_DIR, TESTS_STR, NCCL_INSTALL_DIR,
# DEEP_EP_INSTALL_DIR (+ everything clusters.sh exports).
set -e

source "$SCRIPT_DIR/_pretty.sh"
source "$SCRIPT_DIR/clusters.sh"
source "$SCRIPT_DIR/tests.sh"

# Word-split TESTS_STR back into an array (run.sh passes it as a string
# because arrays don't survive salloc --export=ALL).
read -r -a TESTS <<< "$TESTS_STR"

section_start "BUILD"
if [ -n "${SKIP_BUILD:-}" ]; then
    info "skipped (SKIP_BUILD=$SKIP_BUILD; reusing $DEEP_EP_INSTALL_DIR)"
else
    # --input=none: srun otherwise inherits this script's stdin and may swallow
    # subsequent commands when stdin is a pipe/heredoc.
    srun --input=none --nodes=1 --ntasks=1 "$SCRIPT_DIR/_build.sh"
fi
section_end

failed=()
for TEST in "${TESTS[@]}"; do
    export TEST
    reset_test_env
    set_test_config "$TEST"
    apply_overrides

    # --export=ALL forwards every env var we exported above (cluster_set_env,
    # test_set_env, plus the rendezvous bits in _test.sh) to each task.
    SRUN_CMD="srun --export=ALL --input=none --mpi=pmix -N 2 --ntasks-per-node=1 $SCRIPT_DIR/_test.sh"

    section_start "TEST :: $TEST"
    info "Module:    $TEST_MODULE"
    info "Port:      $MASTER_PORT"
    info "Args:      $TEST_ARGS"
    info "Extra env: ${TEST_ENV_NOTES:-(none)}"
    info "srun:      $SRUN_CMD"
    info "Launches:  python -m $TEST_MODULE $TEST_ARGS"
    echo

    # `set -e` doesn't bail on `$SRUN_CMD` failure here — bash exempts commands
    # used as `if` conditions, so every test in the list runs even if one fails.
    if $SRUN_CMD; then
        pass "$TEST"
    else
        fail "$TEST"
        failed+=("$TEST")
    fi
    section_end
done

summary "${#TESTS[@]}" "${failed[@]}"
exit "${#failed[@]}"
