#!/usr/bin/env bash
# *************************************************************************
#  * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  *************************************************************************
#
# Run device_api_test (reduceCopy) once per type; stop on first failure so you see the error.
# Invoke the script with your runner (e.g. srun) so it works the same locally and in job launchers:
#   srun -n 1 --label ./test/apitest/device_api/reduceCopy/run_reducecopy_types.sh [options] [extra_gtest_filter] [binary]
# Or run the script directly (no srun) for local runs.
#
# Usage:
#   ./run_reducecopy_types.sh [-t TYPE] [extra_gtest_filter] [binary]
#
# Options:
#   -t TYPE   Run only the named type (e.g. Float, Half, Bf16).  The type must be one of
#             the tags emitted by generate_tests.py --list-types.
#
# Examples:
#   # Run all types
#   ./run_reducecopy_types.sh
#
#   # Run only Float
#   ./run_reducecopy_types.sh -t Float
#
#   # Run only Half, narrowed to one test shape
#   ./run_reducecopy_types.sh -t Half '*LsaReduceMultimemCopy_Generic_Thread_UNROLL1_count*_gpus3*'
#
#   # Run all types with srun
#   srun -n 1 --label ./test/apitest/device_api/reduceCopy/run_reducecopy_types.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse flags
SINGLE_TYPE=""
SKIP_MULTIMEM=0
while getopts ":t:m" opt; do
  case $opt in
    t) SINGLE_TYPE="$OPTARG" ;;
    m) SKIP_MULTIMEM=1 ;;
    \?) echo "Unknown flag: -$OPTARG" 1>&2; exit 1 ;;
    :)  echo "Flag -$OPTARG requires an argument" 1>&2; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

# Verify that the RST docs and the C++ headers (api_function_traits.h, test_type_traits.h)
# are in sync before running any tests.  A mismatch means someone updated one without the
# other and should be caught early.
python3 "${SCRIPT_DIR}/generate_tests.py" --validate-docs || {
  echo "Error: docs/header drift detected — fix before running tests." 1>&2
  exit 1
}

# Derive the type list from test_type_traits.h via generate_tests.py so that adding
# a type only requires updating the C++ header — the script stays in sync automatically.
TYPES_STR=$(python3 "${SCRIPT_DIR}/generate_tests.py" --list-types 2>&1) || {
  echo "Warning: generate_tests.py --list-types failed; falling back to hardcoded list." 1>&2
  TYPES_STR="Int Uint Int8 Uint8 LongLong ULongLong Float Double Half Bf16 Fp8E4M3 Fp8E5M2"
}
# shellcheck disable=SC2206
TYPES=($TYPES_STR)

# If -t was given, restrict to that one type (validate it first)
if [ -n "$SINGLE_TYPE" ]; then
  found=0
  for t in "${TYPES[@]}"; do
    if [ "$t" = "$SINGLE_TYPE" ]; then found=1; break; fi
  done
  if [ $found -eq 0 ]; then
    echo "Error: unknown type '$SINGLE_TYPE'. Known types: ${TYPES[*]}" 1>&2
    exit 1
  fi
  TYPES=("$SINGLE_TYPE")
fi

# Optional: narrow to a specific test pattern (e.g. one function/coop/count/gpus shape)
EXTRA_FILTER="${1:-}"
BINARY="${2:-./build/test/apitest/device_api/device_api_test}"

if [ ! -x "$BINARY" ]; then
  echo "Error: binary not found or not executable: $BINARY" 1>&2
  exit 1
fi

SKIP_MULTIMEM_SUFFIX=""
if [ "$SKIP_MULTIMEM" = "1" ]; then
  SKIP_MULTIMEM_SUFFIX=":-*Multimem*"
fi

for TYPE in "${TYPES[@]}"; do
  # GTest full name is "TestCaseName.TestName" (dot). Require "." after type so e.g. "Int" does not match "Int8".
  if [ -n "$EXTRA_FILTER" ]; then
    FILTER="*ReduceCopy_${TYPE}.${EXTRA_FILTER}*${SKIP_MULTIMEM_SUFFIX}"
  else
    FILTER="*ReduceCopy_${TYPE}.*${SKIP_MULTIMEM_SUFFIX}"
  fi
  echo ""
  echo "===== Running type: $TYPE (filter: $FILTER) ====="
  # Matrix expected count is driven by NCCL_TEST_REDUCE_COPY_TYPES; set to current type so we only expect this type's tests.
  NCCL_TEST_REDUCE_COPY_TYPES="$TYPE" "$BINARY" \
    --gtest_filter="$FILTER" \
    --gtest_color=yes \
    --gtest_print_time=1 \
    --verbose
  RET=$?
  if [ $RET -ne 0 ]; then
    echo ""
    echo "===== FAILED at type: $TYPE (exit $RET) ====="
    exit $RET
  fi
done
echo ""
echo "===== All types passed ====="
