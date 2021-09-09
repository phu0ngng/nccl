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

To make sure GPU-to-GPU communication is working correctly, look for the p2pBandwidthLatencyTest from the CUDA
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

PCI Access Control Services (ACS)
---------------------------------

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

IP Ports
--------

NCCL opens TCP ports to connect processes together and exchange connection information. To restrict the range of ports used by NCCL, one can set the net.ipv4.ip_local_port_range property of the
linux kernel.

This example shows how to restrict NCCL ports to 50000-51000:

.. code:: shell

 echo 50000 51000 > /proc/sys/net/ipv4/ip_local_port_range

Or to make this permanent, add a line to /set/sysctl.conf:

.. code:: shell

 echo "net.ipv4.ip_local_port_range = 50000 51000" >> /etc/sysctl.conf

Restricting the port range can be useful to open a corresponding range in the firewall, for example on Google Cloud:

.. code:: shell

 gcloud compute --project=myproject firewall-rules create ncclnet0-ingress --direction=INGRESS --priority=1 --network=ncclnet --action=ALLOW --rules=tcp:50000-51000,22,1024-1039 --destination-ranges=0.0.0.0/0 --target-tags=ncclnet

InfiniBand
----------

Before running NCCL on InfiniBand, running low-level InfiniBand tests (and in particular the ib_write_bw test) can help verify which nodes are able to communicate properly.

A common issue seen with InfiniBand is the library not being able to register sufficient pinned memory. In such cases you may see an error like:

.. code:: shell

 NCCL WARN Call to ibv_create_qp failed

or

.. code:: shell

 NCCL WARN Call to ibv_reg_mr failed

The solution is to remove the user limits on registering pinned memory. This can be done by adding these lines:

.. code:: shell

 * soft memlock unlimited
 * hard memlock unlimited

To the /etc/security/limits.conf configuration file or equivalent on your Linux distribution.

************
Known Issues
************

Ensure you are familiar with the following known issues:

Sharing Data
------------

In order to share data between ranks, NCCL may require shared system memory for IPC and pinned (page-locked) system memory resources. The operating system’s limits on these resources may need to be increased accordingly. Please see your system’s documentation for details. In particular, Docker containers default to limited shared and pinned memory resources. When using NCCL inside a container, it is recommended that you increase these resources by issuing:
        
    --shm-size=1g --ulimit memlock=-1
 
in the command line to nvidia-docker run.

