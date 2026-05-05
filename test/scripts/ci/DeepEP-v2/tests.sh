# Test catalog for the DeepEP-v2 CI. Sourced by _alloc.sh.
# Expects $GPUS_PER_NODE (set in clusters.sh).
#
# Provides:
#   ALL_TESTS, GLOBAL_SKIP_TESTS, reset_test_env, test_set_env VAR=value,
#   set_test_config <name>

ALL_TESTS=(sanity_ep sanity_engram)
# Skipped on every cluster (vs SKIP_TESTS in clusters.sh which is per-cluster).
# Use this for tests that are broken upstream or pending a fix everywhere.
GLOBAL_SKIP_TESTS=(sanity_engram)

# Tracks which env keys test_set_env set this iteration so reset_test_env
# can unset exactly those.
_TEST_ENV_KEYS=()

reset_test_env() {
    local key
    for key in "${_TEST_ENV_KEYS[@]}"; do
        unset "$key"
    done
    _TEST_ENV_KEYS=()
    TEST_ENV_NOTES=""
}

# Exports VAR, tracks it for reset, appends to TEST_ENV_NOTES.
test_set_env() {
    local kv="$1"
    local key="${kv%%=*}"
    export "$kv"
    _TEST_ENV_KEYS+=("$key")
    if [ -z "$TEST_ENV_NOTES" ]; then
        TEST_ENV_NOTES="$kv"
    else
        TEST_ENV_NOTES="$TEST_ENV_NOTES $kv"
    fi
}

# To add a test: append to ALL_TESTS above + add a case branch here.
# Cluster-specific tweaks belong in clusters.sh's apply_overrides.
set_test_config() {
    case "$1" in
        sanity_ep)
            TEST_MODULE=tests.elastic.test_ep
            MASTER_PORT=8361
            TEST_ARGS="--num-processes=$GPUS_PER_NODE --allow-hybrid-mode=0 --num-tokens=16 --hidden=4096 --num-topk=8 --num-experts=256 --num-sms=8 --test-first-only --num-qps 8 --num-allocated-qps 17 --do-cpu-sync=0"
            ;;
        sanity_engram)
            TEST_MODULE=tests.elastic.test_engram
            MASTER_PORT=8364
            TEST_ARGS="--num-processes=$GPUS_PER_NODE"
            test_set_env NCCL_CROSS_NIC=0
            ;;
        *)
            echo "Unsupported test: '$1' (expected: ${ALL_TESTS[*]})" >&2
            exit 1
            ;;
    esac
    export TEST_MODULE MASTER_PORT TEST_ARGS
}
