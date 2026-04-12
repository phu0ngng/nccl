#!/bin/bash
#
# test_nccl4py-api-check.sh
#
# Tests for nccl4py-api-check.sh using a temporary git repo and DRY_RUN mode.
# Run from the repository root: bash bindings/nccl4py/ci/test_nccl4py-api-check.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_UNDER_TEST="${SCRIPT_DIR}/nccl4py-api-check.sh"
TMPDIR_ROOT=$(mktemp -d)
PASS=0
FAIL=0

cleanup() {
  rm -rf "$TMPDIR_ROOT"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Create a fresh test repo with a base commit containing mock headers.
# Sets REPO_DIR and MERGE_BASE_SHA.
setup_repo() {
  REPO_DIR=$(mktemp -d "$TMPDIR_ROOT/repo.XXXXXX")
  cd "$REPO_DIR"
  git init -q
  git config user.email "test@test.com"
  git config user.name "Test"

  # Create monitored header directories
  mkdir -p src
  mkdir -p src/include
  mkdir -p src/include/nccl_device
  mkdir -p src/include/nccl_device/impl

  # Base nccl.h.in with some existing API
  cat > src/nccl.h.in << 'HEADER'
#ifndef NCCL_H_
#define NCCL_H_

#define NCCL_MAJOR 2
#define NCCL_MINOR 30

typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2
} ncclResult_t;

typedef enum {
  ncclInt8 = 0,
  ncclFloat16 = 6,
  ncclFloat32 = 7
} ncclDataType_t;

typedef struct {
  char internal[128];
} ncclUniqueId;

typedef struct ncclComm* ncclComm_t;

typedef struct {
  int splitShare;
  int blocking;
} ncclConfig_t;

ncclResult_t ncclGetVersion(int *version);
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t ncclAllReduceTestOnly(const void* sendbuff, void* recvbuff, size_t count);

#endif
HEADER

  # Base nccl_common.h
  cat > src/include/nccl_common.h << 'HEADER'
#ifndef NCCL_COMMON_H_
#define NCCL_COMMON_H_
typedef enum {
  ncclDebugNone = 0,
  ncclDebugWarn = 1
} ncclDebugLogLevel;
#endif
HEADER

  # Base nccl_device.h
  cat > src/include/nccl_device.h << 'HEADER'
#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_
#include "nccl_device/core.h"
#endif
HEADER

  # Base device sub-header
  cat > src/include/nccl_device/core.h << 'HEADER'
#ifndef NCCL_DEVICE_CORE_H_
#define NCCL_DEVICE_CORE_H_
typedef struct {
  int rank;
  int nRanks;
} ncclDevComm_t;
#endif
HEADER

  # A non-monitored file
  mkdir -p src/internal
  echo "// internal stuff" > src/internal/transport.h

  git add -A
  git commit -q -m "Base commit"
  MERGE_BASE_SHA=$(git rev-parse HEAD)

  # Work on a feature branch
  git checkout -q -b feature
}

# Run the script with DRY_RUN and capture output.
# Sets OUTPUT and EXIT_CODE.
# Callers may override env vars before calling; this sets defaults only.
run_check() {
  export CI_MERGE_REQUEST_IID="${CI_MERGE_REQUEST_IID-999}"
  export CI_MERGE_REQUEST_DIFF_BASE_SHA="$MERGE_BASE_SHA"
  export CI_API_V4_URL="https://gitlab.example.com/api/v4"
  export CI_PROJECT_ID="12345"
  export CI_JOB_TOKEN="fake-token"
  export CI_PIPELINE_URL="https://gitlab.example.com/pipeline/1"
  export DRY_RUN="1"
  export NCCL4PY_REVIEWERS="${NCCL4PY_REVIEWERS-@reviewer1 @reviewer2}"

  EXIT_CODE=0
  OUTPUT=$("$SCRIPT_UNDER_TEST" 2>&1) || EXIT_CODE=$?

  # Reset overrides for next test
  unset CI_MERGE_REQUEST_IID
  unset NCCL4PY_REVIEWERS
}

assert_contains() {
  local label="$1"
  local needle="$2"
  if echo "$OUTPUT" | grep -qF "$needle"; then
    echo "  PASS: $label"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: $label"
    echo "    Expected output to contain: $needle"
    FAIL=$((FAIL + 1))
  fi
}

assert_not_contains() {
  local label="$1"
  local needle="$2"
  if ! echo "$OUTPUT" | grep -qF "$needle"; then
    echo "  PASS: $label"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: $label"
    echo "    Expected output NOT to contain: $needle"
    FAIL=$((FAIL + 1))
  fi
}

assert_exit_code() {
  local label="$1"
  local expected="$2"
  if [[ "$EXIT_CODE" -eq "$expected" ]]; then
    echo "  PASS: $label"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: $label (expected exit $expected, got $EXIT_CODE)"
    FAIL=$((FAIL + 1))
  fi
}

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

test_no_changes() {
  echo "TEST: No API changes"
  setup_repo
  # Commit a change to a non-monitored file
  echo "// new internal code" >> src/internal/transport.h
  git add -A && git commit -q -m "Internal change"
  run_check
  assert_exit_code "exits 0" 0
  assert_contains "reports no changes" "No public API header changes detected"
}

test_new_function() {
  echo "TEST: New function declaration"
  setup_repo
  cat >> src/nccl.h.in << 'ADDITION'
ncclResult_t ncclMemAllocTestOnly(void** ptr, size_t size);
ADDITION
  git add -A && git commit -q -m "Add ncclMemAllocTestOnly"
  run_check
  assert_exit_code "exits 0" 0
  assert_contains "detects new function" "ncclMemAllocTestOnly"
  assert_contains "lists changed file" "src/nccl.h.in"
  assert_contains "shows required actions table" "config_nccl.py"
  assert_contains "includes reviewer mentions" "@reviewer1"
}

test_removed_function() {
  echo "TEST: Removed function declaration"
  setup_repo
  # Remove ncclAllReduceTestOnly
  sed -i '/ncclAllReduceTestOnly/d' src/nccl.h.in
  git add -A && git commit -q -m "Remove ncclAllReduceTestOnly"
  run_check
  assert_exit_code "exits 0" 0
  assert_contains "detects removed function" "ncclAllReduceTestOnly"
  assert_contains "shows removed section" "Removed function declarations"
}

test_new_struct() {
  echo "TEST: New struct definition"
  setup_repo
  cat >> src/nccl.h.in << 'ADDITION'
typedef struct {
  int minCtas;
  int maxCtas;
} ncclSimInfoTestOnly_t;
ADDITION
  git add -A && git commit -q -m "Add ncclSimInfoTestOnly_t"
  run_check
  assert_exit_code "exits 0" 0
  assert_contains "detects new struct" "ncclSimInfoTestOnly_t"
  assert_contains "shows structures section" "Modified/new structures"
}

test_new_enum_value() {
  echo "TEST: New enum value"
  setup_repo
  sed -i '/ncclFloat32 = 7/a\  ncclBfloat16TestOnly = 8' src/nccl.h.in
  git add -A && git commit -q -m "Add ncclBfloat16TestOnly"
  run_check
  assert_exit_code "exits 0" 0
  assert_contains "detects new enum value" "ncclBfloat16TestOnly"
  assert_contains "shows enum section" "New/modified enum values"
}

test_new_macro() {
  echo "TEST: New macro"
  setup_repo
  echo '#define NCCL_VERSION_CODE_TEST_ONLY 23000' >> src/nccl.h.in
  git add -A && git commit -q -m "Add NCCL_VERSION_CODE_TEST_ONLY"
  run_check
  assert_exit_code "exits 0" 0
  assert_contains "detects new macro" "NCCL_VERSION_CODE_TEST_ONLY"
  assert_contains "shows macros section" "Changed macros"
}

test_device_header_change() {
  echo "TEST: Device sub-header change"
  setup_repo
  cat >> src/include/nccl_device/core.h << 'ADDITION'
typedef struct {
  int teamSize;
} ncclTeamTestOnly_t;
ADDITION
  git add -A && git commit -q -m "Add ncclTeamTestOnly_t to device core"
  run_check
  assert_exit_code "exits 0" 0
  assert_contains "lists device header" "src/include/nccl_device/core.h"
  assert_contains "detects struct" "ncclTeamTestOnly_t"
}

test_no_mr_iid() {
  echo "TEST: Missing CI_MERGE_REQUEST_IID"
  setup_repo
  echo "// trivial" >> src/nccl.h.in
  git add -A && git commit -q -m "Trivial"
  export CI_MERGE_REQUEST_IID=""
  run_check
  assert_exit_code "exits 0" 0
  assert_contains "skips gracefully" "Not an MR pipeline"
}

test_no_reviewers() {
  echo "TEST: Missing NCCL4PY_REVIEWERS"
  setup_repo
  cat >> src/nccl.h.in << 'ADDITION'
ncclResult_t ncclMemFreeTestOnly(void* ptr);
ADDITION
  git add -A && git commit -q -m "Add ncclMemFreeTestOnly"
  NCCL4PY_REVIEWERS=""
  run_check
  assert_exit_code "exits 0" 0
  assert_contains "warns about missing reviewers" "WARNING: NCCL4PY_REVIEWERS is not set"
  assert_not_contains "no cc line" "**cc:**"
}

test_multiple_changes() {
  echo "TEST: Multiple change types in one MR"
  setup_repo
  # Add a new function, enum value, and macro
  cat >> src/nccl.h.in << 'ADDITION'
ncclResult_t ncclCommSplitTestOnly(ncclComm_t comm, int color, int key, ncclComm_t *newcomm);
#define NCCL_SPLIT_NOCOLOR_TEST_ONLY (-1)
ADDITION
  sed -i '/ncclFloat32 = 7/a\  ncclFloat64TestOnly = 9' src/nccl.h.in
  git add -A && git commit -q -m "Multiple API changes"
  run_check
  assert_exit_code "exits 0" 0
  assert_contains "detects new function" "ncclCommSplitTestOnly"
  assert_contains "detects new enum" "ncclFloat64TestOnly"
  assert_contains "detects new macro" "NCCL_SPLIT_NOCOLOR_TEST_ONLY"
}

test_comment_has_marker() {
  echo "TEST: Comment includes idempotency marker"
  setup_repo
  echo '#define NCCL_PATCH_TEST_ONLY 1' >> src/nccl.h.in
  git add -A && git commit -q -m "Add NCCL_PATCH_TEST_ONLY"
  run_check
  assert_exit_code "exits 0" 0
  assert_contains "has HTML marker" "<!-- nccl4py-api-check -->"
}

test_comment_has_pipeline_link() {
  echo "TEST: Comment includes pipeline link"
  setup_repo
  echo '#define NCCL_PATCH_TEST_ONLY 2' >> src/nccl.h.in
  git add -A && git commit -q -m "Add NCCL_PATCH_TEST_ONLY"
  run_check
  assert_exit_code "exits 0" 0
  assert_contains "has pipeline link" "https://gitlab.example.com/pipeline/1"
}

# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

test_no_changes
test_new_function
test_removed_function
test_new_struct
test_new_enum_value
test_new_macro
test_device_header_change
test_no_mr_iid
test_no_reviewers
test_multiple_changes
test_comment_has_marker
test_comment_has_pipeline_link

echo ""
echo "=============================="
echo "Results: ${PASS} passed, ${FAIL} failed"
echo "=============================="

if (( FAIL > 0 )); then
  exit 1
fi
