#!/bin/bash
ref=$1
if [ "$ref" == "" ]; then ref="$REGRESSION_BASELINE"; fi
baseline_build_dir="build-${ref}"

shift

orig_branch=$1
if [ "$orig_branch" == "" ]; then orig_branch=$CI_COMMIT_SHORT_SHA; fi

failure_count=0
failure_names=()

echo "CI_MERGE_REQUEST_DIFF_BASE_SHA=$CI_MERGE_REQUEST_DIFF_BASE_SHA"
echo "REGRESSION_BASELINE=$REGRESSION_BASELINE"
echo "ref=$ref"
echo "baseline_build_dir=$baseline_build_dir"

echo "Checking out $ref (originally at $orig_branch)"
git checkout "$ref"
rm -rf $baseline_build_dir
srun -N 1 --exclusive -p luna,interactive -A nccl -J nccl-build:baseline -t 00:20:00 make -j test.build BUILDDIR="$baseline_build_dir"
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Compiling $ref")
echo "Checking out $orig_branch"
git checkout "$orig_branch"
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Checkout $orig_branch")

for str in "${failure_names[@]}"
do
  echo "Failed check: $str"
done

echo "$failure_count failures detected"
exit $failure_count
