# Config params for AWS P5 cluster
# Cluster config:
# 16 nodes - 8 H100 80G GPUs / node - EFA NICs
# OS: Ubuntu20.04
# NVIDIA-SMI/Driver: 535.183.01
# CUDA: 12.2 -> 12.8.0
# OpenMPI: 4.1.6
# EFA driver: 121.0
# AWS_OFI_PLUGIN: 1.9.2

# no host-specific code executes in this module to allow it to live elsewhere

# Relevant paths
# AWS_CUDA_HOME="/usr/local/cuda-12.4"
AWS_CUDA_HOME="/shared/toolkits/cuda-12.8.0"
AWS_OPENMPI_HOME="/opt/amazon/openmpi"
AWS_OFI_PLUGIN_HOME="/home/ubuntu/junyum/aws-ofi-nccl"

# Modules to be loaded-in for configuring the EFA driver
AWS_EFA_MODULE="libfabric-aws"
AWS_PLANNED_RESERVED="Planned"
AWS_MPI_PARAMS="--oversubscribe --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none"

# Target configs
function get_cuda_home() {
	echo "$AWS_CUDA_HOME"
}

function get_openmpi_home() {
	echo "$AWS_OPENMPI_HOME"
}

function get_ofi_plugin_home() {
	echo "$AWS_OFI_PLUGIN_HOME"
}

function get_efa_module() {
	echo "$AWS_EFA_MODULE"
}

function configure_efa() {
    # Configure the EFA driver
    module load $(get_efa_module)

    # Set other variables to help NCCL discover EFA as a network interface
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
}

function configure_test_env() {
    configure_efa
}

function get_planned_reserved() {
    echo "$AWS_PLANNED_RESERVED"
}

function get_mpi_params() {
    echo "$AWS_MPI_PARAMS"
}

function get_extra_ld_library_path() {
    # Add OFI plugin to LD_LIBRARY_PATH
    echo "$(get_ofi_plugin_home)/lib:$AWS_CUDA_HOME/lib64:$AWS_OPENMPI_HOME/lib"
}

function get_extra_path() {
    echo "$AWS_CUDA_HOME/bin:$AWS_OPENMPI_HOME/bin"
}
