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

Another tool for checking GPU-to-GPU performance is called ``nvbandwidth``.
This can be downloaded and built from the code and instructions found here: https://github.com/NVIDIA/nvbandwidth

GPU-to-NIC communication
------------------------

GPUs can also communicate directly with network cards using GPU Direct RDMA (GDRDMA). This requires having a compatible
network cards and drivers, plus loading an extra kernel module called ``nvidia-peermem``.
The ``nvidia-peermem`` module is now supplied with the CUDA drivers, however it must be loaded on each node boot with:

.. code::

 sudo modprobe nvidia-peermem

GDRDMA can also be enabled by using the DMA-BUF feature of recent Linux kernels combined with the Open Source Nvidia GPU driver.
In this case, NCCL will automatically detect and enable DMA-BUF so the nvidia-peermem module will not be necessary.


PCI Access Control Services (ACS)
---------------------------------

IO virtualization (also known as VT-d or IOMMU) can interfere with GPU Direct by redirecting all PCI point-to-point
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

  sudo setpci -s 03:00.0 ECAP_ACS+0x6.w=0000

Or you can use a script similar to this:

.. code::

  for BDF in `lspci -d "*:*:*" | awk '{print $1}'`; do
    # skip if it doesn't support ACS
    sudo setpci -v -s ${BDF} ECAP_ACS+0x6.w > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      continue
    fi
    sudo setpci -v -s ${BDF} ECAP_ACS+0x6.w=0000
  done

******************
Topology detection
******************

NCCL relies on /sys to discover the PCI topology of GPUs and network cards. When running inside a virtual
machine or container, make sure /sys is properly mounted. Having /sys expose a virtual PCI topology can
result in sub-optimal performance.

*************
Shared memory
*************

To communicate between processes and even between threads of a process, NCCL creates shared memory segments
in /dev/shm. The operating system’s limits on these resources may need to be increased accordingly. Please see your
system’s documentation for details.

If insufficient shared memory is available, NCCL will fail to initialize. Running with NCCL_DEBUG=WARN
will show a message similar to this:

.. code::

 NCCL WARN Error: failed to extend /dev/shm/nccl-03v824 to 4194660 bytes

Docker
------

In particular, Docker containers default to limited shared and pinned memory resources. When using NCCL inside a
container, please make sure to adjust the shared memory size inside the container, for example by adding the following
arguments to the docker launch command line:

.. code::

 --shm-size=1g --ulimit memlock=-1

Systemd
-------

When running jobs using mpirun or SLURM, systemd may remove files in shared memory when it detects that the
corresponding user is not logged in, in an attempt to clean up old temporary files. This can cause NCCL to crash
during init with an error like:

.. code::

 NCCL WARN unlink shared memory /dev/shm/nccl-d5rTd0 failed, error: No such file or directory

Given mpirun and SLURM jobs can run on the node without the user being seen as logged in by systemd, system administrators need
to disable that clean-up mechanism, which can be performed by SLURM epilogue scripts instead. To do this, the following
line needs to be set in /etc/systemd/logind.conf:

.. code::

 RemoveIPC=no

Once updated, the daemons should be restarted with:

.. code::

 sudo systemctl restart systemd-logind

*****************
Networking issues
*****************

IP Network Interfaces
---------------------

NCCL auto-detects which network interfaces to use for inter-node communication. If some interfaces are in the UP state but are not able to communicate between nodes, NCCL may try to use them anyway and therefore fail during the init functions or even hang.

For information about how to specify which interfaces to use, see the Environment Variables section, particularly the ``NCCL_SOCKET_IFNAME`` environment variable.

IP Ports
--------

NCCL opens TCP ports to connect processes together and exchange connection information. To restrict the range of ports used by NCCL, one can set the ``net.ipv4.ip_local_port_range`` property of the Linux kernel.

This example shows how to restrict NCCL ports to 50000-51000:

.. code:: shell

 echo 50000 51000 > /proc/sys/net/ipv4/ip_local_port_range

Or to make this permanent, add a line to /etc/sysctl.conf:

.. code:: shell

 echo "net.ipv4.ip_local_port_range = 50000 51000" >> /etc/sysctl.conf

Restricting the port range can be useful to open a corresponding range in the firewall, for example on Google Cloud:

.. code:: shell

 gcloud compute --project=myproject firewall-rules create ncclnet0-ingress --direction=INGRESS --priority=1 --network=ncclnet --action=ALLOW --rules=tcp:50000-51000,22,1024-1039 --destination-ranges=0.0.0.0/0 --target-tags=ncclnet

InfiniBand
----------

Before running NCCL on InfiniBand, running low-level InfiniBand tests (and in particular the ib_write_bw test) can help verify whether the nodes are able to communicate properly.

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

To the ``/etc/security/limits.conf`` configuration file or equivalent on your Linux distribution.


RDMA over Converged Ethernet (RoCE)
-----------------------------------

Before running NCCL on RoCE, running low-level RDMA tests (and in particular the ``ib_write_bw`` test) can help verify whether the nodes are able to communicate properly.

A common issue seen with RoCE is the incorrect GID Index being selected for the RoCE v2 NICs. This can result in the following error:

.. code:: shell

 NCCL WARN Call to ibv_modify_qp failed with error Invalid argument


With NCCL 2.21 and later the GID index is dynamically selected, but with prior versions the user would need to run:

.. code:: shell

 show_gids


And then set ``NCCL_IB_GID_INDEX`` to the GID INDEX for the RoCE v2 VER GID.
With NCCL 2.21 and later releases, this environment variable should *not* be set.

Users may also need to set ``NCCL_IB_TC`` when using RoCE based networks. Refer to your vendor's documentation for the values this should be set to.
