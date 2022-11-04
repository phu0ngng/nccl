#!/usr/bin/env bash
# This is a copy of the gitlab runner custom executor script sitting on bewilliams@gpu-comms-nvshmem-jenkins-01:/home/bewilliams/run_exec_wrapper_selene.sh
# More can be read here about custom executors: https://docs.gitlab.com/runner/executors/custom.html
# This script just logs into the selene-login node, then selene-login-03 (CI-only node) and invokes each step
# Gitlab SSH runners can't access selene-login, hence we need this custom executor
# The gitlab-runner executable must be in $PATH for several of the steps to work, notably artifact downloads

# WARNING - This is only a copy for reference. Editing this script does not do anything - the active script is located on gpu-comms-nvshmem-jenkins-01

echo "$HOSTNAME: $@"
script_dir=`dirname $1`

ssh -q selene-ci mkdir -p $script_dir
scp -q ${1} selene-ci:${1}
ssh -q -t selene-ci ssh -q -t selene-login-03 "PATH=~/bin:$PATH && ${1}"