#!/bin/bash
#SBATCH --partition=a3
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --nodes 8
#SBATCH --comment=sysctl-sys.kernel.numa_balancing=0,transparent_hugepage_defrag=never,transparent_hugepage=never
#SBATCH -t 10
# Usage: sbatch run-nccl-tests.sh

set -x
# This should be set to the squashfs file that you created for your application
NVDA_IMAGE=./nvidia+pytorch+23.10-py3.sqsh

# Important TCPX environment variables
UDS_PATH="/run/tcpx-${SLURM_JOB_ID}"
GPU_NIC_TOPOLOGY=/opt/tcpdirect_benchmark/gpu_rxq_configuration.textproto
GPU_NIC_TOPOLOGY_DIR=`dirname ${GPU_NIC_TOPOLOGY}`
if [ ! -f "${GPU_NIC_TOPOLOGY}" ]; then
  echo "GPU_NIC_TOPOLOGY file ${GPU_NIC_TOPOLOGY} must exist!"
  exit 1
fi

# Install the TCPX NCCL Plugin
srun -n ${SLURM_NNODES} --ntasks-per-node=1 \
    docker run --rm -v /var/lib:/var/lib \
    us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpx/nccl-plugin-gpudirecttcpx-dev:v3.1.6_2023_10_06 install --install-nccl

# Tuning (only needs to be done once per node)
srun -n ${SLURM_NNODES} --ntasks-per-node=1 \
  sudo /opt/apps/scripts/a3_tuning_for_bd.sh

# Start TCPX receive-datapath-manager
srun --ntasks-per-node=1 -n ${SLURM_NNODES} \
     docker run \
     --pull=always \
     --detach \
     --rm \
     --name receive-datapath-manager-${SLURM_JOB_ID} \
     --cap-add=NET_ADMIN \
     --network=host \
     --privileged \
     --gpus all \
     --volume /var/lib/nvidia/lib64:/usr/local/nvidia/lib64 \
     --volume ${GPU_NIC_TOPOLOGY_DIR}:${GPU_NIC_TOPOLOGY_DIR} \
     --volume ${UDS_PATH}:${UDS_PATH} \
     --env LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/lib/lib32:/usr/lib/x86_64-linux-gnu/ \
     --entrypoint /tcpgpudmarxd/build/app/tcpgpudmarxd \
     us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpx/tcpgpudmarxd-dev:v2.0.7 \
     --setup_param "--verbose 128 2 0" \
     --gpu_nic_preset manual \
     --gpu_nic_topology ${GPU_NIC_TOPOLOGY} \
     --gpu_shmem_type fd \
     --uds_path ${UDS_PATH}

# Give some time for tcpgpudmarxd to come up
sleep 10s

# Set up NCCL Environment variables
# The following two can be useful for debugging
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING
export USE_TCPX=yes
export NCCL_NET=GPUDirectTCPX_v7
# These network interfaces use Ubuntu's consistent naming scheme. See
# https://manpages.ubuntu.com/manpages/focal/man7/systemd.net-naming-scheme.7.html
export NCCL_SOCKET_IFNAME=enp0s12
export NCCL_GPUDIRECTTCPX_CTRL_DEV=enp0s12
export NCCL_GPUDIRECTTCPX_SOCKET_IFNAME=enp6s0,enp12s0,enp134s0,enp140s0
export NCCL_CROSS_NIC=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=1
export NCCL_MAX_NCHANNELS=12
export NCCL_MIN_NCHANNELS=12
export NCCL_DYNAMIC_CHUNK_SIZE=524288
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_BUFFSIZE=4194304
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_P2P_PXN_LEVEL=0
export NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX=${UDS_PATH}
export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=1000000
export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=1000
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="enp6s0:6-11;enp12s0:32-37;enp134s0:58-63;enp140s0:84-89"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="enp6s0:12-17;enp12s0:38-43;enp134s0:64-69;enp140s0:90-95"

export LD_LIBRARY_PATH=/var/lib/tcpx/lib64:/usr/local/mpi/lib:/usr/local/nvidia/lib64:/usr/lib:/usr/lib/lib32:/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}

# Here we use a trick to grab all the environment variables that need to be
# passed down into the container. Slurm would otherwise only pass these env vars
# to the job environment on the host. Would be nice if slurm respected
# "--export=ALL" for passing to the container as well.
HOST_VARS=`env | grep -e USE_TCPX -e LD_LIBRARY_PATH -e NCCL -e CUDA | awk -F = '{print $1}' | paste -sd "," -`

LOCAL_IMAGE=/mnt/localssd/$(id -u)/${NVDA_IMAGE}
sbcast -f ${NVDA_IMAGE} ${LOCAL_IMAGE}

# Run the workload
srun --mpi=pmi2 \
  --cpu_bind=verbose \
  -n $((SLURM_NNODES*8)) \
  --ntasks-per-node=8 \
  --gpus-per-node=8 \
  --export=ALL \
  --container-image=${LOCAL_IMAGE} \
  --container-name=nccl \
  --container-env=${HOST_VARS} \
  --container-mounts="/var/tmp:/var/tmp,/var/lib/tcpx/lib64:/var/lib/tcpx/lib64,$PWD:/nccl,${GPU_NIC_TOPOLOGY_DIR}:${GPU_NIC_TOPOLOGY_DIR},${UDS_PATH}:${UDS_PATH}" \
  all_reduce_perf_mpi -b 256M -e 8G -f 2 -g 1 -w 5 --iters 20 -c 0

# Shut down the TCPX datapath manager
# Note -- this can also be handled by a Slurm epilog, which may be present.
srun -n ${SLURM_NNODES} --ntasks-per-node=1 rm -f ${LOCAL_IMAGE}
srun -n ${SLURM_NNODES} --ntasks-per-node=1 docker container stop receive-datapath-manager-${SLURM_JOB_ID}
