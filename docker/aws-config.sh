# Config params for AWS P5 cluster
# Cluster config:
# 16 nodes - 8 H100 80G GPUs / node - EFA NICs
# OS: Ubuntu20.04
# NVIDIA-SMI/Driver: 535.183.01
# CUDA: 12.2
# OpenMPI: 4.1.6
# EFA driver: 121.0
# AWS_OFI_PLUGIN: 1.9.2

# no host-specific code executes in this module to allow it to live elsewhere

# Relevant paths
AWS_CUDA_HOME="/usr/local/cuda-12.2"
AWS_OPENMPI_HOME="/opt/amazon/openmpi"
AWS_OFI_PLUGIN_HOME="/opt/aws-ofi-nccl"

# Modules to be loaded-in for configuring the EFA driver
AWS_EFA_MODULE="libfabric-aws"

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

    # Add OFI plugin to LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$(get_ofi_plugin_home)/lib:$LD_LIBRARY_PATH

    # Set other variables to help NCCL discover EFA as a network interface
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
}

function configure_aws_test_env() {
    export CUDA_HOME=$(get_cuda_home)
    export MPI_HOME=$(get_openmpi_home)

    configure_efa
}
