###############
Installing NCCL
###############

************
Requirements
************

Ensure your environment meets the following requirements:

* CUDA 7.0 or higher
* Compute capability 3.0 and higher. Check https://developer.nvidia.com/cuda-gpus to check for the compute capability of all NVIDIA GPUs.

*********
Compiling
*********

Get NCCL source code and compile : ::
 
 $ git clone https://github.com/nvidia/nccl.git
 $ cd nccl
 $ make -j

You might specify CUDA_HOME if CUDA is not installed in ``/usr/local/cuda`` : ::
 
 $ make -j CUDA_HOME=/path/to/cuda
 
**********
Installing
**********

To install NCCL, first create a package, then install it on the system.

Debian/Ubuntu
*************

Build a deb package for Debian/Ubuntu : ::
 
 $ make -j pkg.debian.build
 $ dpkg -i build/pkg/deb/*.deb

RedHat/CentOS/Suse
******************

Build an RPM package for RedHat/CentOS/Suse : ::
 
 $ make -j pkg.redhat.build
 $ rpm -ivh build/pkg/rpm/*.rpm

Others/Agnostic
***************

Build a tar.xz package for extraction : ::
 
 $ make -j pkg.txz.build
 $ sudo tar -C /opt/nccl -xf build/pkg/txz/*.txz

Note you can replace ``/opt/nccl`` with any path where you want to extract NCCL.

