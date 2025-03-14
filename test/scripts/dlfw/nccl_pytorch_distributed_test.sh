#!/bin/bash

# install required packages
source ../self_test_common.sh

set -o nounset
set -o errexit
set -o pipefail
set -e
set -x

cd /opt/pytorch/pytorch/test

AWK=awk
AWK_SCRIPT=./update.awk
RUN_TST=run_test_new.py

cat <<EOF > $AWK_SCRIPT
  BEGIN{}
  {
      print \$0
  }
  /verbose = '--verbose'/{
      print "    shell('nvidia-smi', test_directory)";
      print "    proc1 = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE)";
      print "    proc2 = subprocess.Popen(['grep', 'python'], stdin=proc1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)";
      print "    proc1.stdout.close()";
      print "    out, err = proc2.communicate()";
      print "    print('out: {0}'.format(out))";
      print "    print('err: {0}'.format(err))";
  }

EOF

$AWK -f $AWK_SCRIPT < run_test.py > $RUN_TST
shard_id=$CI_NODE_INDEX
nshards=$CI_NODE_TOTAL

# distributed/test_c10d_ops_nccl is a known hang in pytorch upstream
EXCLUDE_PYTEST=(
  distributed/test_c10d_ops_nccl
  distributed/rpc/test_tensorpipe_agent
  distributed/rpc/cuda/test_tensorpipe_agent
)

cp $RUN_TST run_test.py

# note(mkozuki): Cheat `run_test.py` so that we can disable "parallel" execution of tests as possible.
export TEST_CONFIG="distributed"

export PYTORCH_TEST_RUN_EVERYTHING_IN_SERIAL=1

pip install --upgrade --force-reinstall zstandard
pip install --upgrade --force-reinstall zstd
export NCCL_DEBUG=WARN
export PYTEST_K_EXPR="nccl"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_test.py --shard ${shard_id} ${nshards} --distributed-tests -v --pipe-logs -x ${EXCLUDE_PYTEST[@]} --continue-through-error -- -k "$PYTEST_K_EXPR"
