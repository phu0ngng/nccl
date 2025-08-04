#!/usr/bin/env bash
# This is a copy of the gitlab runner custom executor script sitting on bewilliams@gpu-comms-nvshmem-jenkins-01:/home/bewilliams/run_exec_wrapper_draco.sh
# More can be read here about custom executors: https://docs.gitlab.com/runner/executors/custom.html
# This script just logs into the draco-login node via the jumpbox and invokes each step
# Gitlab SSH runners can't access draco-login, hence we need this custom executor
# The gitlab-runner executable must be in $PATH for several of the steps to work, notably artifact downloads

# WARNING - This is only a copy for reference. Editing this script does not do anything - the active script is located on gpu-comms-nvshmem-jenkins-01

echo "$HOSTNAME: $@"
script_dir=`dirname $1`

/usr/bin/ssh draco-ci mkdir -p $script_dir
/usr/bin/scp ${1} draco-ci:${1}
/usr/bin/ssh draco-ci "PATH=~/bin:$PATH && ${1}"
