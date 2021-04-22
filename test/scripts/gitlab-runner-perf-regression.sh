#!/usr/bin/env bash
if [[ ! -d ./test/perf ]]; then
  >&2 echo "Script must be run from top-level nccl directory."
  exit 1
fi
if [[ ! -d ./build/lib ]]; then
  >&2 echo "'./build' must contain built nccl lib and tests."
  exit 1
fi
if [[ -z "$NCCL_REGRESSION_BASELINE" ]]; then
  >&2 echo "NCCL_REGRESSION_BASELINE must be set."
  exit 1
fi

export OPAL_PREFIX=$MPI_HOME
export LD_LIBRARY_PATH=$MPI_HOME/lib:$PWD/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

function absify() {
  python3 -c 'import os; import sys; sys.stdout.write(os.path.abspath(sys.argv[1]))' "$1"
}

build_baseline="build-${NCCL_REGRESSION_BASELINE}"

# Build the baseline version if necessary. We manually check since the makefile
# will use file mtime's which will have been just updated by the preceding
# `git checkout` so will always conclude to rebuild.
if [[ ! -e "$build_baseline"/test/perf/all_reduce_perf ]]; then
  orig_branch=$(git rev-parse --abbrev-ref HEAD) &&
  git checkout "$NCCL_REGRESSION_BASELINE" &&
  make -j src.build test.build BUILDDIR="$build_baseline" MPI=1 MPI_HOME="$MPI_HOME" CUDA_HOME="$CUDA_HOME" &&
  git checkout "$orig_branch"
  ok=$?
else
  ok=0
fi

if [[ $ok == 0 ]]; then
  # run regression suite
  ./test/scripts/perf_regression.py old="$build_baseline" new=build --check 0
fi
