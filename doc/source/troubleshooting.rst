###############
Troubleshooting
###############

Ensure you are familiar with the following known issues and useful debugging strategies. 

******
Errors
******

NCCL calls may return a variety of return codes. Ensure that the return codes are always equal to ncclSuccess. If any call fails and returns a value different from ncclSuccess, setting NCCL_DEBUG to “WARN” will make NCCL print an explicit warning message before returning the error.

Errors are grouped into different categories.
* ncclUnhandledCudaError and ncclSystemError indicate that a call to an external library failed.
* ncclInvalidArgument and ncclInvalidUsage indicates there was a programming error in the application using NCCL.

In either case, refer to the NCCL warning message to understand how to resolve the problem. 

**********
GPU Direct
**********

NCCL heavily relies on GPU Direct for inter-GPU communication. This refers to the ability for a GPU to directly
communicate with another device, such as another GPU or a network card, using direct point-to-point PCI messages.

Direct point-to-point PCI messages can fail or perform poorly for a variety of reasons, like missing components,
a bad configuration of a virtual machine or a container, or some BIOS settings.

GPU-to-GPU communication
------------------------

To make sure GPU-to-GPU communication is working correctly, look for the p2pBandwidthLastencyTest from the CUDA
samples.

.. code::

  cd /usr/local/cuda/samples/1_Utilities/p2pBandwidthLatencyTest
  sudo make
  ./p2pBandwidthLatencyTest

The test should run to completion and report good performance between GPUs.

GPU-to-NIC communication
------------------------

GPUs can also communicate directly with a network card using GPU Direct RDMA. This requires to have a compatible
network card and driver and load an extra kernel module. For Mellanox Infiniband/RoCE cards, the module is
called nv_peer_mem and can be found at https://github.com/Mellanox/nv_peer_memory.

Refer to your vendor's documentation for information on how to install and configure GPU Direct RDMA.

ACS
---

IO virtualization (also known as, VT-d or IOMMU) can interfere with GPU Direct by redirecting all PCI point-to-point
traffic to the CPU root complex, causing a significant performance reduction or even a hang. You can check
whether ACS is enabled on PCI bridges by running:

.. code::

  sudo lspci -vvv | grep ACSCtl

If lines show "SrcValid+", then ACS might be enabled. Looking at the full output of lspci, one can check if
a PCI bridge has ACS enabled.

.. code::

  sudo lspci -vvv

If PCI switches have ACS enabled, it needs to be disabled. On some systems this can be done from the BIOS
by disabling IO virtualization or VT-d. For Broadcom PLX devices, it can be done from the OS but needs to
be done again after each reboot.

Use the command below to find the PCI bus IDs of PLX PCI bridges:

.. code::

  sudo lspci | grep PLX

Next, use setpci to disable ACS with the command below, replacing 03:00.0 by the PCI bus ID of each PCI bridge.

.. code::

  sudo setpci -s 03:00.0 f2a.w=0000

******************
Topology detection
******************

NCCL relies on /sys to discover the PCI topology of GPUs and network cards. When running inside a virtual
machine or container, make sure /sys is properly mounted. Having /sys expose a virtual PCI topology can
result in suboptimal performance.

*****************
Networking issues
*****************

IP Network Interfaces
---------------------

NCCL auto-detects which network interfaces to use for inter-node communication. If some interfaces are in state up, however are not able to communicate between nodes, NCCL may try to use them anyway and therefore fail during the init functions or even hang.

For information about how to specify which interfaces to use, see NCCL Knobs section, particularly the NCCL_SOCKET_IFNAME knob.   

InfiniBand
----------

Before running NCCL on InfiniBand, running low-level InfiniBand tests (and in particular the ib_write_bw test) can help verify which nodes are able to communicate properly.

************
Known Issues
************

Ensure you are familiar with the following known issues:

Sharing Data
------------

In order to share data between ranks, NCCL may require shared system memory for IPC and pinned (page-locked) system memory resources. The operating system’s limits on these resources may need to be increased accordingly. Please see your system’s documentation for details. In particular, Docker containers default to limited shared and pinned memory resources. When using NCCL inside a container, it is recommended that you increase these resources by issuing:
        
    --shm-size=1g --ulimit memlock=-1
 
in the command line to nvidia-docker run.

Concurrency between NCCL and CUDA calls (NCCL up to 2.0.5 or CUDA 8)
--------------------------------------------------------------------

NCCL uses CUDA kernels to perform inter-GPU communication. The NCCL kernels synchronize with each other, therefore, each kernel requires other kernels on other GPUs to be also executed in order to complete. The application should therefore make sure that nothing prevents the NCCL kernels from being executed concurrently on the different devices of a NCCL communicator.

For example, let's say you have a process managing multiple CUDA devices, and, also features a thread which calls CUDA functions asynchronously. In this case, CUDA calls could be executed between the enqueuing of two NCCL kernels. The CUDA call may wait for the first NCCL kernel to complete and prevent the second one from being launched, causing a deadlock since the first kernel will not complete until the second one is executed.  To avoid this issue, one solution is to have a lock around the NCCL launch on multiple devices (around ncclGroupStart and ncclGroupEnd when using a single thread, around the NCCL launch when using multiple threads, using thread synchronization if necessary) and take this lock when calling CUDA from the asynchronous thread.

Starting with NCCL 2.1.0, this issue is no longer present when using CUDA 9, unless Cooperative Group Launch is disabled in the NCCL_LAUNCH_MODE=PARALLEL setting.
