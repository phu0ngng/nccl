#!/bin/bash
#
#SBATCH --nodes=<NODES>
#SBATCH --ntasks-per-node=<PROCS>
#SBATCH --time=1:00:00
#SBATCH --job-name=<NAME>
#SBATCH --output=log.txt
#SBATCH --error=log.txt

set -e
#set -x

# Job parameters (this run)

JOB_DIR="$(pwd)"
JOB_DIR_BASE="$(basename "$JOB_DIR")"
JOB_NAME="${SLURM_JOB_NAME}-${JOB_DIR_BASE}"
JOB_DIR_UID=$(stat -c "%u" "$JOB_DIR")
NUM_NODES="$SLURM_JOB_NUM_NODES"
NODE_LIST=($(scontrol show hostname $SLURM_JOB_NODELIST))
GPUS="<GPUS>"
PROCS="<PROCS>"

# Job parameters (all runs)

MPI_FLAGS="
--mca btl_tcp_if_exclude lo,docker0
-x PATH
-x LD_LIBRARY_PATH
-x PYTHONPATH
-x CUDA_DEVICE_ORDER=PCI_BUS_ID
-x NCCL_DEBUG=INFO
"
#-x NCCL_SOCKET_IFNAME=^docker
#-x OMP_NUM_THREADS=10
#-x NCCL_BUFFSIZE=65536
#-x NCCL_IB_DISABLE=1

DOCKER_FLAGS="
--name $JOB_NAME
--privileged
--net=host
--shm-size=1G
--ulimit memlock=-1
--ulimit stack=67108864
--device=/dev/infiniband:/dev/infiniband
-v /mnt/shared:/mnt/shared:ro
-v $JOB_DIR:/jobdir
-v /raid/dldata:/data:ro
"

# Overload the default NCCL2 library
# -v /mnt/shared/daddison/src/DL/lib_2.1.2/libnccl.so.2:/usr/lib/x86_64-linux-gnu/libnccl.so.2

# Luke Yeager's multi-node base container with MPI/mlx5/ibverbs and sshd support
CONTAINER=nvdl.githost.io:4678/dgx/caffe2:17.11-multi-node

# Use the command line arguments as the command to run
CMD=$*

# Pull containers
docker pull $CONTAINER
for NODE in ${NODE_LIST[@]:1}; do
    ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no \
        -o LogLevel=quiet $NODE "nvidia-docker pull $CONTAINER" > /dev/null
done

# From this point on, don't crash on failed commands
set +e
ERRORS=0

# Start sshd on worker nodes
for NODE in ${NODE_LIST[@]:1}; do
    ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o LogLevel=quiet \
        $NODE nvidia-docker run -d $DOCKER_FLAGS $CONTAINER /usr/sbin/sshd -D > /dev/null
    if [[ $? -ne 0 ]]; then
        echo "Failed to create container on $NODE"
        ERRORS=1
    fi
done

# Run mpi on host node
if [[ "$ERRORS" -eq 0 ]]; then
    RANKFILE=$(mktemp)
    RANKCMDS=""
    RANK=0
    for NODE in ${NODE_LIST[@]}; do
	for PROC in `seq 1 <PROCS>`; do
	    if [[ -n "$RANKCMDS" ]]; then
		RANKCMDS="$RANKCMDS :"
	    fi
	    RANKCMDS="${RANKCMDS} -n 1 -H $NODE $MPI_FLAGS $CMD"
	    echo "rank $RANK=$NODE slot=0:*" >> $RANKFILE
	    RANK=$((RANK+1))
	done
    done

    echo "Running commnad: '$CMD' on $NUM_NODES with $PROCS processes per node"
    nvidia-docker run --rm \
	 -v $RANKFILE:/rankfile $DOCKER_FLAGS $CONTAINER \
	mpirun --allow-run-as-root $RANKCMDS
    if [[ $? -ne 0 ]]; then
        ERRORS=1
    fi
    rm -f $RANKFILE
fi

# Shut down worker nodes
for NODE in ${NODE_LIST[@]:1}; do
    ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no \
        -o LogLevel=quiet $NODE "docker stop $JOB_NAME && docker rm $JOB_NAME" > /dev/null
    if [[ $? -ne 0 ]]; then
        echo "Failed to cleanup $JOB_NAME container on $NODE"
        ERRORS=1
    fi
done

# Fix permissions on jobdir
#nvidia-docker run --rm -v $JOB_DIR:/jobdir -w /jobdir $CONTAINER chown -R $JOB_DIR_UID /jobdir > /dev/null
#if [[ $? -ne 0 ]]; then
#    echo "Failed to re-own $JOB_DIR as $JOB_DIR_UID"
#    ERRORS=1
#fi

if [[ $ERRORS -ne 0 ]]; then
    echo "RUN TERMINATED AFTER ERRORS"
fi
exit $ERRORS
